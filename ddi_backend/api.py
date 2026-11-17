# Module docstring: describes the HTTP API, its routes and the prototype/simulated disclaimers.
"""A small HTTP API so a frontend can call the backend later. No user-interface code lives here.

    python -m ddi_backend serve          # then open http://127.0.0.1:8000/docs

Routes: POST /check, GET /drugs/{name}, GET /avoid/{drug}, POST /ask, POST /review-queue/requests,
GET /review-queue, GET /trace/{request_id}, GET /health. Every response says it is a class prototype
and not medical advice; everything about the review queue is labelled simulated.
"""

import os  # reads the DDI_CORS_ORIGINS environment variable
from typing import List, Literal, Optional, Union  # type hints for the request models

from fastapi import FastAPI, HTTPException  # the web framework and its error type
from fastapi.encoders import jsonable_encoder  # turns result objects into JSON-safe data
from fastapi.responses import JSONResponse  # lets a route choose its own status code
from pydantic import BaseModel, Field  # request-body validation

from ddi_backend import composer, llm  # template text for results and the language-model explainer
from ddi_backend.agent import REVIEW_TARGET, REVIEW_TOOL, format_trace  # review-queue names and trace formatting
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE  # standard disclaimer texts
from ddi_backend.service import App  # shared setup that wires the database, checker, agent and state

# Gate and tool outcomes mapped to HTTP statuses. A denial writes nothing.
_STATUS_CODES = {
    "staged_in_simulation": 201,  # success: the item was staged in the simulated queue
    "approval_required": 403,  # user did not approve
    "tool_disabled": 403,  # tool turned off
    "tool_not_allowed": 403,  # tool is not the permitted one
    "out_of_scope": 403,  # target outside the allowed scope
    "duplicate_request_id": 409,  # request id already used
    "missing_evidence": 404,  # no documented, eligible interaction to stage
    "error": 500,  # tool failed
}


class Medication(BaseModel):  # one medication in a request
    name: str = Field(min_length=1)  # drug name, must be non-empty
    dose: Optional[str] = None       # stored and echoed, never recommended or computed
    details: Optional[str] = None  # free-text extra detail, also only echoed


class CheckOptions(BaseModel):  # optional settings for POST /check
    include_current_pairs: bool = False  # also check the current medications against each other
    detail_level: Literal["brief", "standard", "detailed"] = "standard"  # how much text to write
    context_level: Literal["names_only", "full_context"] = "full_context"  # how much patient context the model sees
    explain: bool = False            # ask the language model (if configured) to write the explanation


class CheckRequest(BaseModel):  # body of POST /check
    current: List[Union[str, Medication]] = Field(default_factory=list)  # medications already taken (names or objects)
    new: Union[str, Medication]  # the new medication being checked
    options: CheckOptions = Field(default_factory=CheckOptions)  # settings, defaults when omitted


class AskRequest(BaseModel):  # body of POST /ask
    question: str = Field(min_length=1)  # the free-text question, non-empty


class ReviewRequest(BaseModel):  # body of POST /review-queue/requests
    drug_pair: List[str] = Field(min_length=2, max_length=2)  # exactly two drug names
    request_id: str = Field(min_length=1)  # caller-chosen unique id for tracing and duplicate detection
    approved: bool                   # required: nothing is written without the user's explicit approval
    target: str = REVIEW_TARGET  # which queue to stage into (defaults to the simulated review queue)


def _medication(item):
    # Convert a Medication model to a plain dict (dropping unset fields); plain names pass through unchanged.
    return item.model_dump(exclude_none=True) if isinstance(item, Medication) else item


def create_app(db_path=None, state_path=None, symptom_mapping=True, embeddings="off", in_memory_state=False,
               backend=None, provider="env"):
    # Use a supplied backend (tests do this) or build one; check_same_thread=False because web requests use threads.
    backend = backend or App(db_path=db_path, state_path=state_path, symptom_mapping=symptom_mapping,
                             embeddings=embeddings, in_memory_state=in_memory_state, provider=provider,
                             check_same_thread=False)
    api = FastAPI(title="Drug interaction backend (class prototype, not medical advice)", version="0.1")  # the app
    api.state.backend = backend  # keep the backend reachable from the app object

    # Allowed browser origins come from a comma-separated environment variable; blanks are dropped.
    origins = [o.strip() for o in os.environ.get("DDI_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:  # for a frontend served from another origin, set DDI_CORS_ORIGINS
        from fastapi.middleware.cors import CORSMiddleware  # imported only when needed
        api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])  # allow them

    @api.get("/health")  # reports which tiers are available
    def health():
        with backend.lock:  # one request at a time touches the shared database connection
            return backend.health()  # tier availability summary

    @api.post("/check")  # main route: check a new drug against the current ones
    def check(request: CheckRequest):
        options = request.options  # shorthand for the settings
        with backend.lock:  # serialize access to the shared connection
            # Deterministic lookup: resolve names, query pairs, apply rules and the eligibility filter.
            out = backend.checker.check([_medication(m) for m in request.current], _medication(request.new),
                                        include_current_pairs=options.include_current_pairs)
            for result in out["results"]:  # add the template plain-language text to every pair result
                result["text"] = composer.compose_pair(result, options.detail_level)
            if options.explain:  # optional language-model explanation
                # Build an explainer with the requested settings and the list of known drug names (for validation).
                explainer = llm.Explainer(backend.provider, llm.Settings(options.context_level, options.detail_level),
                                          backend.explainer.known_drug_names)
                out["explanations"] = explainer.explain_check(out)  # validated text, falls back to templates
        # Echo the settings used, and whether a language model was available.
        out["settings"] = {"detail_level": options.detail_level, "context_level": options.context_level,
                           "explain": options.explain, "language_model": backend.provider is not None}
        return jsonable_encoder(out)  # return JSON-safe data

    def _drug_response(out):  # shared reply for the drug and avoid routes
        if out["status"] == "resolved":  # the name was found exactly
            return jsonable_encoder(out)  # normal 200 reply
        code = 409 if out["status"] == "ambiguous_name" else 404  # ambiguous = conflict, otherwise unknown = not found
        return JSONResponse(status_code=code, content=jsonable_encoder(out))  # reply with the error status

    @api.get("/drugs/{name}")  # one drug and the interactions it appears in
    def drug(name: str):
        with backend.lock:
            return _drug_response(backend.checker.drug(name))

    @api.get("/avoid/{drug}")  # drugs with a Major or Moderate edge to this drug
    def avoid(drug: str):
        with backend.lock:
            return _drug_response(backend.checker.avoid(drug))

    @api.post("/ask")  # free-text question
    def ask(request: AskRequest):
        with backend.lock:
            return jsonable_encoder(backend.qa.ask(request.question))  # answered by the question handler

    @api.post("/review-queue/requests")  # ask the guarded agent to stage a review
    def stage_review(request: ReviewRequest):
        with backend.lock:
            # The agent runs its gate checks in order and only then calls the tool.
            outcome = backend.agent.request_review(request.drug_pair, request.request_id,
                                                   approved=request.approved, target=request.target,
                                                   tool=REVIEW_TOOL)
        # Keep only the documented fields of the outcome in the response body.
        body = jsonable_encoder({k: outcome[k] for k in (
            "status", "gate", "executed_tool", "evidence_id", "request_id", "action_occurred",
            "state_before", "state_after", "tool_result", "simulated", "note", "disclaimer")})
        # Map the outcome to an HTTP status; anything unexpected is 400.
        return JSONResponse(status_code=_STATUS_CODES.get(outcome["status"], 400), content=body)

    @api.get("/review-queue")  # show the simulated queue
    def review_queue():
        with backend.lock:
            rows = backend.state.queue_rows()  # staged items from the state file
        return {"simulated": True, "note": SIMULATED_NOTE, "review_queue": rows, "disclaimer": DISCLAIMER}

    @api.get("/trace/{request_id}")  # what happened to one review request
    def trace(request_id: str):
        with backend.lock:
            attempts = backend.state.attempts(request_id)  # logged attempts for this request id
        if not attempts:  # nothing logged for this id
            raise HTTPException(status_code=404, detail=f"No attempts are logged for request_id {request_id!r}.")
        # Return both structured attempts and the readable trace text.
        return {"simulated": True, "request_id": request_id, "attempts": jsonable_encoder(attempts),
                "text": format_trace(request_id, attempts), "disclaimer": DISCLAIMER}

    return api  # the finished FastAPI application

# ---------------------------------------------------------------------------
# FILE SUMMARY: Defines the FastAPI application a future frontend will call. It validates request bodies, turns each
# route into a call on the shared backend (checker, question handler, agent, state), adds template or validated
# language-model text, and maps agent gate outcomes to HTTP status codes. It adds no user-interface code.
# ASSIGNMENT TOOL: Supporting infrastructure (HTTP front door) for the knowledge graph + rules core, the language
# model explain mode, and the guarded agent with its simulated review queue.
# ---------------------------------------------------------------------------
