"""A small HTTP API so a frontend can call the backend later. No user-interface code lives here.

    python -m ddi_backend serve          # then open http://127.0.0.1:8000/docs

Routes: POST /check, GET /drugs/{name}, GET /avoid/{drug}, POST /ask, POST /review-queue/requests,
GET /review-queue, GET /trace/{request_id}, GET /health. Every response says it is a class prototype
and not medical advice; everything about the review queue is labelled simulated.
"""

import os
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ddi_backend import composer, llm
from ddi_backend.agent import REVIEW_TARGET, REVIEW_TOOL, format_trace
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE
from ddi_backend.service import App

# Gate and tool outcomes mapped to HTTP statuses. A denial writes nothing.
_STATUS_CODES = {
    "staged_in_simulation": 201,
    "approval_required": 403,
    "tool_disabled": 403,
    "tool_not_allowed": 403,
    "out_of_scope": 403,
    "duplicate_request_id": 409,
    "missing_evidence": 404,
    "error": 500,
}


class Medication(BaseModel):
    name: str = Field(min_length=1)
    dose: Optional[str] = None       # stored and echoed, never recommended or computed
    details: Optional[str] = None


class CheckOptions(BaseModel):
    include_current_pairs: bool = False
    detail_level: Literal["brief", "standard", "detailed"] = "standard"
    context_level: Literal["names_only", "full_context"] = "full_context"
    explain: bool = False            # ask the language model (if configured) to write the explanation


class CheckRequest(BaseModel):
    current: List[Union[str, Medication]] = Field(default_factory=list)
    new: Union[str, Medication]
    options: CheckOptions = Field(default_factory=CheckOptions)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class ReviewRequest(BaseModel):
    drug_pair: List[str] = Field(min_length=2, max_length=2)
    request_id: str = Field(min_length=1)
    approved: bool                   # required: nothing is written without the user's explicit approval
    target: str = REVIEW_TARGET


def _medication(item):
    return item.model_dump(exclude_none=True) if isinstance(item, Medication) else item


def create_app(db_path=None, state_path=None, symptom_mapping=True, embeddings="off", in_memory_state=False,
               backend=None, provider="env"):
    backend = backend or App(db_path=db_path, state_path=state_path, symptom_mapping=symptom_mapping,
                             embeddings=embeddings, in_memory_state=in_memory_state, provider=provider,
                             check_same_thread=False)
    api = FastAPI(title="Drug interaction backend (class prototype, not medical advice)", version="0.1")
    api.state.backend = backend

    origins = [o.strip() for o in os.environ.get("DDI_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:  # for a frontend served from another origin, set DDI_CORS_ORIGINS
        from fastapi.middleware.cors import CORSMiddleware
        api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @api.get("/health")
    def health():
        with backend.lock:
            return backend.health()

    @api.post("/check")
    def check(request: CheckRequest):
        options = request.options
        with backend.lock:
            out = backend.checker.check([_medication(m) for m in request.current], _medication(request.new),
                                        include_current_pairs=options.include_current_pairs)
            for result in out["results"]:
                result["text"] = composer.compose_pair(result, options.detail_level)
            if options.explain:
                explainer = llm.Explainer(backend.provider, llm.Settings(options.context_level, options.detail_level),
                                          backend.explainer.known_drug_names)
                out["explanations"] = explainer.explain_check(out)
        out["settings"] = {"detail_level": options.detail_level, "context_level": options.context_level,
                           "explain": options.explain, "language_model": backend.provider is not None}
        return jsonable_encoder(out)

    def _drug_response(out):
        if out["status"] == "resolved":
            return jsonable_encoder(out)
        code = 409 if out["status"] == "ambiguous_name" else 404
        return JSONResponse(status_code=code, content=jsonable_encoder(out))

    @api.get("/drugs/{name}")
    def drug(name: str):
        with backend.lock:
            return _drug_response(backend.checker.drug(name))

    @api.get("/avoid/{drug}")
    def avoid(drug: str):
        with backend.lock:
            return _drug_response(backend.checker.avoid(drug))

    @api.post("/ask")
    def ask(request: AskRequest):
        with backend.lock:
            return jsonable_encoder(backend.qa.ask(request.question))

    @api.post("/review-queue/requests")
    def stage_review(request: ReviewRequest):
        with backend.lock:
            outcome = backend.agent.request_review(request.drug_pair, request.request_id,
                                                   approved=request.approved, target=request.target,
                                                   tool=REVIEW_TOOL)
        body = jsonable_encoder({k: outcome[k] for k in (
            "status", "gate", "executed_tool", "evidence_id", "request_id", "action_occurred",
            "state_before", "state_after", "tool_result", "simulated", "note", "disclaimer")})
        return JSONResponse(status_code=_STATUS_CODES.get(outcome["status"], 400), content=body)

    @api.get("/review-queue")
    def review_queue():
        with backend.lock:
            rows = backend.state.queue_rows()
        return {"simulated": True, "note": SIMULATED_NOTE, "review_queue": rows, "disclaimer": DISCLAIMER}

    @api.get("/trace/{request_id}")
    def trace(request_id: str):
        with backend.lock:
            attempts = backend.state.attempts(request_id)
        if not attempts:
            raise HTTPException(status_code=404, detail=f"No attempts are logged for request_id {request_id!r}.")
        return {"simulated": True, "request_id": request_id, "attempts": jsonable_encoder(attempts),
                "text": format_trace(request_id, attempts), "disclaimer": DISCLAIMER}

    return api
