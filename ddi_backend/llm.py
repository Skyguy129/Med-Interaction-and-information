# Module docstring: explains the explain mode and the limits placed on the model.
"""The language model (tier three): a reader of our database and the writer of the plain-language response.

Two modes. The selector (``llm_chat.py``) lets the model pick record IDs from candidates.
The explain mode here lets it write a plain-language response from the eligible records for
the requested pair, under strict limits: no browsing, no tools, only the supplied records, a
structured reply in which every claim points to a record ID, and a validator that rejects any
new fact. A failed check falls back to the template composer. Nothing depends on the model:
with no API key the system runs deterministic-only, and that mode is complete on its own.
The model never decides anything and never triggers a write.
"""

import json  # builds the model request and parses its reply
import os  # reads OPENAI_API_KEY and OPENAI_MODEL
import re  # regular expressions for word and number checks
from dataclasses import dataclass  # makes the small immutable Settings class

from ddi_backend import composer, llm_chat  # template fallback, and the shared OpenAI call
from ddi_backend.common import DISCLAIMER  # the not-medical-advice line attached to every response

CONTEXT_LEVELS = ("names_only", "full_context")  # how much patient context the model may see
CLAIM_FIELDS = ("severity", "drug_a", "drug_b", "mechanism", "effect", "safer_alternative", "rationale", "reference")  # record fields a claim may cite
REQUIRED_CLAIM_FIELDS = ("severity", "mechanism", "effect")  # claims the model must make for the headline record
MAX_SUMMARY_CHARS = {"brief": 280, "standard": 700, "detailed": 1500}  # longest allowed summary per detail level


@dataclass(frozen=True)  # an immutable record of the two settings
class Settings:  # user-chosen response settings
    """context_level: names_only sends just the pair and its records; full_context also sends the doses
    and other medication details the user entered. detail_level: brief, standard or detailed."""
    context_level: str = "full_context"  # default: send doses and other medications too
    detail_level: str = "standard"  # default: medium-length answer

    def __post_init__(self):  # runs right after creation to reject bad values
        if self.context_level not in CONTEXT_LEVELS:  # unknown context level
            raise ValueError(f"context_level must be one of {CONTEXT_LEVELS}")  # fail loudly
        if self.detail_level not in composer.DETAIL_LEVELS:  # unknown detail level
            raise ValueError(f"detail_level must be one of {composer.DETAIL_LEVELS}")  # fail loudly


# --------------------------------------------------------------------------
# The provider interface
# --------------------------------------------------------------------------
class Provider:  # base class for any model provider
    """Anything that turns chat messages into a JSON string. No tools, no browsing."""
    name = "provider"  # label for the provider

    def complete(self, messages):  # subclasses must implement this
        raise NotImplementedError  # the base class does nothing


class OpenAIProvider(Provider):  # the OpenAI implementation
    """OpenAI Chat Completions through urllib, as in llm_chat. The key and model come from the environment."""
    name = "openai"  # label for this provider

    def __init__(self, api_key, model=llm_chat.DEFAULT_MODEL):  # store the key and model
        self.api_key, self.model = api_key, model  # kept on the object, never written anywhere

    def complete(self, messages):  # send the messages and return the reply text
        return llm_chat.get_reply(messages, self.api_key, self.model)  # reuse the selector's web call


def provider_from_env(environ=None):  # choose a provider from environment settings
    """OpenAIProvider when OPENAI_API_KEY is set, otherwise None (deterministic-only)."""
    environ = os.environ if environ is None else environ  # tests can pass a fake environment
    key = environ.get("OPENAI_API_KEY")  # the key, if any
    if not key:  # no key means no model
        return None  # deterministic-only mode
    return OpenAIProvider(key, environ.get("OPENAI_MODEL", llm_chat.DEFAULT_MODEL))  # model name from the environment or the default


# --------------------------------------------------------------------------
# Explain mode
# --------------------------------------------------------------------------
EXPLAIN_SYSTEM = (  # the rules sent to the model for explain mode
    "You write a short plain-language explanation of ONE documented drug interaction for a class "  # task and scope
    "prototype that is not medical advice. You have no tools and no internet, and you must not use "  # no outside knowledge
    "anything you know or remember. The only permitted source is the records in the user message. "  # supplied records only
    "Return a JSON object with exactly these keys: summary (string), claims (list), clarifying_question "  # output shape
    "(string or null). Each claim is an object with exactly the keys record_id, field and value, where "  # claim shape
    "value is copied exactly from that field of that record. Include a claim for severity, mechanism and "  # claims must copy values
    "effect. Write the summary using the records' own wording: add no facts, no names, no numbers, no "  # no new facts
    "doses, no advice and no reassurance (never say a combination is safe). If the user's patient context "  # no advice, never "safe"
    "is missing something you would need, put a question in clarifying_question instead of assuming."  # ask, do not assume
)

# Ordinary connecting words the summary may use even though they are not in the records (one big string, split into a set).
ALLOWED_PLAIN_WORDS = set("""
this that with from have does your their they them than then when which while also both each other some such
into more most been being were will would could about between taking takes take together there these those
only documented interaction interactions data severity record records pair effect effects mechanism source
reference references safer alternative alternatives review pharmacist clinician doctor ask talk speak
medication medications medicine drug drugs patient patients listed lists named says states shows shown
reason because causes cause caused leading leads lead result results resulting
""".split())
_WORD = re.compile(r"[A-Za-z]+")  # matches runs of letters
_NUMBER = re.compile(r"\d+(?:\.\d+)?")  # matches whole or decimal numbers
_SEVERITY_WORDS = {"major", "moderate", "minor"}  # words that name a severity


def _norm(text):  # normalise text for comparison
    return " ".join(str(text or "").casefold().split())  # lower-case, trim and collapse spaces; None becomes empty


def _record_text(records, extra=()):  # gather all text the records contain
    parts = list(extra)  # start with any extra words (the pair names)
    for record in records:  # each record
        for key in ("drug_a", "drug_b", "severity", "mechanism", "effect", "safer_alternative", "rationale", "reference", "id"):  # fields the summary may draw from
            if record.get(key):  # skip empty fields
                parts.append(str(record[key]))  # add the value
    return " ".join(parts)  # one big text used to check the summary's words


def build_explain_messages(result, settings, other_medications=()):  # make the model request
    """Only the pair, its eligible records and the settings go to the model. Nothing else."""
    payload = {  # the data the model is allowed to see
        "pair": result["pair"],  # the two drug names
        "detail_level": settings.detail_level,  # requested size
        "max_summary_characters": MAX_SUMMARY_CHARS[settings.detail_level],  # hard length limit
        "records": [{k: r.get(k) for k in ("id", "severity", "drug_a", "drug_b", "mechanism", "effect",  # each record, with only these fields
                                          "safer_alternative", "rationale", "reference") if r.get(k) is not None}  # and only non-empty values
                    for r in result["records"]],  # every eligible record for the pair
    }
    if settings.context_level == "full_context":  # more context allowed
        payload["doses_as_entered"] = {n: d for n, d in (result.get("doses") or {}).items() if d}  # doses the user typed
        payload["other_medications"] = list(other_medications)  # the user's other drugs
    return [{"role": "system", "content": EXPLAIN_SYSTEM},  # the rules
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]  # the data as JSON


def validate_explanation(reply, result, known_drug_names=(), settings=Settings()):  # check the model reply
    """Check a model reply against the records it was given. Returns (parsed, problems).

    The reply must be a JSON object with exactly summary, claims and clarifying_question. Every claim must
    name a supplied record and copy that record's field exactly. The summary may not state a severity other
    than the records', name a drug other than the pair and the records' alternatives, contain a number that is
    not in the records, or use a content word that is not in the records. Any problem means the reply is
    rejected and the template is used instead.
    """
    problems = []  # every problem found, so all are reported
    try:  # the reply must be JSON
        parsed = json.loads(reply)  # parse it
    except (TypeError, json.JSONDecodeError):  # not JSON
        return None, ["reply is not valid JSON"]  # reject
    if not isinstance(parsed, dict) or set(parsed) != {"summary", "claims", "clarifying_question"}:  # wrong shape
        return None, ["reply must be an object with exactly summary, claims and clarifying_question"]  # reject
    summary, claims, question = parsed["summary"], parsed["claims"], parsed["clarifying_question"]  # pull out the three parts
    if not isinstance(summary, str) or not isinstance(claims, list) or not (question is None or isinstance(question, str)):  # wrong types
        return None, ["summary, claims or clarifying_question has the wrong type"]  # reject

    records = {r["id"]: r for r in result["records"]}  # supplied records by ID
    headline = result["records"][0]  # the record that headlines the pair
    claimed = set()  # (record, field) pairs correctly claimed
    for claim in claims:  # check each claim
        if not isinstance(claim, dict) or set(claim) != {"record_id", "field", "value"}:  # wrong claim shape
            problems.append("a claim must have exactly record_id, field and value")  # note it
            continue  # next claim
        record = records.get(claim["record_id"])  # the cited record, if supplied
        if record is None:  # cited a record we did not send
            problems.append(f"claim cites {claim['record_id']!r}, which was not supplied")  # note it
        elif claim["field"] not in CLAIM_FIELDS:  # unknown field name
            problems.append(f"claim names an unknown field {claim['field']!r}")  # note it
        elif _norm(claim["value"]) != _norm(record.get(claim["field"])):  # value is not copied exactly
            problems.append(f"claim for {claim['record_id']}.{claim['field']} does not match the record")  # note it
        else:  # a valid claim
            claimed.add((claim["record_id"], claim["field"]))  # remember it
    for field in REQUIRED_CLAIM_FIELDS:  # severity, mechanism and effect must be claimed
        if headline.get(field) and (headline["id"], field) not in claimed:  # present in the record but not claimed
            problems.append(f"no valid claim for the headline record's {field}")  # note it

    text = " ".join(t for t in (summary, question) if t)  # everything the model wrote
    corpus = _record_text(result["records"], extra=result["pair"])  # everything the records say
    corpus_words = {w.casefold() for w in _WORD.findall(corpus)}  # the set of words the records use
    allowed_severities = {_norm(r["severity"]) for r in result["records"]}  # severities that really occur
    for word in _WORD.findall(text):  # check each word the model wrote
        lower = word.casefold()  # compare in lower case
        if lower in _SEVERITY_WORDS:  # a severity word
            if lower not in allowed_severities:  # not one of the records' severities
                problems.append(f"summary states severity {word}, which the records do not")  # note it
        elif len(lower) >= 4 and lower not in ALLOWED_PLAIN_WORDS and lower not in corpus_words:  # a content word nobody supplied
            problems.append(f"summary uses {word!r}, which is not in the records (a possible new fact)")  # note it
    for number in _NUMBER.findall(text):  # check each number
        if number not in corpus:  # not in the records
            problems.append(f"summary contains the number {number}, which is not in the records")  # note it
    allowed_drugs = {_norm(n) for n in result["pair"]}  # the pair itself is allowed
    for record in result["records"]:  # plus whatever the records name
        allowed_drugs |= {_norm(record["drug_a"]), _norm(record["drug_b"])}  # both drugs of each record
        allowed_drugs |= {_norm(n) for n in record["alternative"]["drug_names"]}  # and drugs named as alternatives
    for name in known_drug_names:  # every drug we know about
        key = _norm(name)  # normalised name
        if key and key not in allowed_drugs and re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", _norm(text)):  # an outside drug appears as a whole word
            problems.append(f"summary names {name}, which is not part of this pair or its alternatives")  # note it
    if len(summary) > MAX_SUMMARY_CHARS[settings.detail_level]:  # too long
        problems.append("summary is longer than the detail level allows")  # note it
    if question is not None and not question.strip().endswith("?"):  # a "question" that is not one
        problems.append("clarifying_question must be a question")  # note it
    return (None, problems) if problems else (parsed, [])  # any problem rejects the whole reply


class Explainer:  # runs explain mode with a safe fallback
    def __init__(self, provider=None, settings=Settings(), known_drug_names=()):  # store the pieces
        self.provider = provider  # the model provider, or None for deterministic-only
        self.settings = settings  # context and detail settings
        self.known_drug_names = list(known_drug_names)  # drug names used by the validator

    def explain_pair(self, result, other_medications=(), clarifying_question=None):  # response for one pair
        """A plain-language response for one pair result, shown next to its source records."""
        out = {  # the response dictionary
            "pair": result["pair"],  # the two drug names
            "status": result["status"],  # the pair status
            "settings": {"context_level": self.settings.context_level, "detail_level": self.settings.detail_level},  # settings in effect
            "generated_by": "template",  # default; changes only if the model's reply passes
            "fallback_reason": None,  # filled in when the template is used instead
            "records": result.get("records", []),  # the source records, always shown
            "source": result.get("source"),  # citations
            "disclaimer": DISCLAIMER,  # the not-medical-advice line
        }
        template = composer.compose_pair(result, self.settings.detail_level)  # the safe deterministic text
        out["clarifying_question"] = clarifying_question  # any question about missing patient context
        if result["status"] != "documented_interaction" or not result.get("records"):  # nothing documented to explain
            out["text"] = template  # absent evidence: "not documented in our data"; no records, no API call
            out["fallback_reason"] = "no eligible record for this pair, so no language model call was made"  # why
            return out  # done without calling the model
        if self.provider is None:  # no key configured
            out["text"] = template  # use the template
            out["fallback_reason"] = "no language model configured (deterministic-only mode)"  # why
            return out  # done
        try:  # the model call may fail
            reply = self.provider.complete(build_explain_messages(result, self.settings, other_medications))  # ask the model
        except RuntimeError as error:  # network or API problem
            out["text"] = template  # use the template
            out["fallback_reason"] = f"language model unavailable: {error}"  # why
            return out  # done
        parsed, problems = validate_explanation(reply, result, self.known_drug_names, self.settings)  # check the reply
        out["raw_reply"] = reply  # keep the raw reply for inspection
        if problems:  # the check failed
            out["text"] = template  # use the template
            out["fallback_reason"] = "language model reply failed validation: " + "; ".join(problems)  # why
            out["validation_problems"] = problems  # the full list
            return out  # done
        out["generated_by"] = "language_model"  # the reply passed, so mark its origin
        out["claims"] = parsed["claims"]  # the claims that tie the text to records
        out["text"] = parsed["summary"]  # the model's summary
        out["clarifying_question"] = parsed["clarifying_question"] or clarifying_question  # prefer the model's question, else ours
        out["text"] += "\n" + "Source: " + "; ".join(f"{c['id']} ({c['reference']})" for c in result["source"]["citations"])  # always add the cited sources
        return out  # the finished response

    def explain_check(self, check_output, patient_context_question=None):  # response for a whole check
        """Explain every pair in a /check style output. The generated text sits next to the source records."""
        names = [m["name"] for m in check_output["inputs"]["current"] if m.get("name")]  # names of the user's current drugs
        names.append(check_output["inputs"]["new"].get("name"))  # plus the new drug
        return [self.explain_pair(r, other_medications=[n for n in names if n and n not in r["pair"]],  # the drugs not in this pair count as "other"
                                  clarifying_question=patient_context_question)  # pass any context question through
                for r in check_output["results"]]  # one response per pair


# ---------------------------------------------------------------------------
# FILE SUMMARY: Implements the language model's explain mode: it sends only the pair and its eligible records to a
# provider, requires a structured reply whose claims copy record fields exactly, and validates that no new severity,
# drug name, number or content word appears. Any failure, or no API key, falls back to the template composer, so
# the model can never add a fact, decide anything or trigger a write.
# ASSIGNMENT TOOL: The language model (explain mode, the reader of our database and writer of the plain-language
# response, under safeguards); it does not replace the knowledge graph and rules that produce the facts.
# ---------------------------------------------------------------------------
