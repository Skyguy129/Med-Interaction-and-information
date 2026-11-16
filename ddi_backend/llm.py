"""The language model (tier three): a reader of our database and the writer of the plain-language response.

Two modes. The selector (``llm_chat.py``) lets the model pick record IDs from candidates.
The explain mode here lets it write a plain-language response from the eligible records for
the requested pair, under strict limits: no browsing, no tools, only the supplied records, a
structured reply in which every claim points to a record ID, and a validator that rejects any
new fact. A failed check falls back to the template composer. Nothing depends on the model:
with no API key the system runs deterministic-only, and that mode is complete on its own.
The model never decides anything and never triggers a write.
"""

import json
import os
import re
from dataclasses import dataclass

from ddi_backend import composer, llm_chat
from ddi_backend.common import DISCLAIMER

CONTEXT_LEVELS = ("names_only", "full_context")
CLAIM_FIELDS = ("severity", "drug_a", "drug_b", "mechanism", "effect", "safer_alternative", "rationale", "reference")
REQUIRED_CLAIM_FIELDS = ("severity", "mechanism", "effect")
MAX_SUMMARY_CHARS = {"brief": 280, "standard": 700, "detailed": 1500}


@dataclass(frozen=True)
class Settings:
    """context_level: names_only sends just the pair and its records; full_context also sends the doses
    and other medication details the user entered. detail_level: brief, standard or detailed."""
    context_level: str = "full_context"
    detail_level: str = "standard"

    def __post_init__(self):
        if self.context_level not in CONTEXT_LEVELS:
            raise ValueError(f"context_level must be one of {CONTEXT_LEVELS}")
        if self.detail_level not in composer.DETAIL_LEVELS:
            raise ValueError(f"detail_level must be one of {composer.DETAIL_LEVELS}")


# --------------------------------------------------------------------------
# The provider interface
# --------------------------------------------------------------------------
class Provider:
    """Anything that turns chat messages into a JSON string. No tools, no browsing."""
    name = "provider"

    def complete(self, messages):
        raise NotImplementedError


class OpenAIProvider(Provider):
    """OpenAI Chat Completions through urllib, as in llm_chat. The key and model come from the environment."""
    name = "openai"

    def __init__(self, api_key, model=llm_chat.DEFAULT_MODEL):
        self.api_key, self.model = api_key, model

    def complete(self, messages):
        return llm_chat.get_reply(messages, self.api_key, self.model)


def provider_from_env(environ=None):
    """OpenAIProvider when OPENAI_API_KEY is set, otherwise None (deterministic-only)."""
    environ = os.environ if environ is None else environ
    key = environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAIProvider(key, environ.get("OPENAI_MODEL", llm_chat.DEFAULT_MODEL))


# --------------------------------------------------------------------------
# Explain mode
# --------------------------------------------------------------------------
EXPLAIN_SYSTEM = (
    "You write a short plain-language explanation of ONE documented drug interaction for a class "
    "prototype that is not medical advice. You have no tools and no internet, and you must not use "
    "anything you know or remember. The only permitted source is the records in the user message. "
    "Return a JSON object with exactly these keys: summary (string), claims (list), clarifying_question "
    "(string or null). Each claim is an object with exactly the keys record_id, field and value, where "
    "value is copied exactly from that field of that record. Include a claim for severity, mechanism and "
    "effect. Write the summary using the records' own wording: add no facts, no names, no numbers, no "
    "doses, no advice and no reassurance (never say a combination is safe). If the user's patient context "
    "is missing something you would need, put a question in clarifying_question instead of assuming."
)

ALLOWED_PLAIN_WORDS = set("""
this that with from have does your their they them than then when which while also both each other some such
into more most been being were will would could about between taking takes take together there these those
only documented interaction interactions data severity record records pair effect effects mechanism source
reference references safer alternative alternatives review pharmacist clinician doctor ask talk speak
medication medications medicine drug drugs patient patients listed lists named says states shows shown
reason because causes cause caused leading leads lead result results resulting
""".split())
_WORD = re.compile(r"[A-Za-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SEVERITY_WORDS = {"major", "moderate", "minor"}


def _norm(text):
    return " ".join(str(text or "").casefold().split())


def _record_text(records, extra=()):
    parts = list(extra)
    for record in records:
        for key in ("drug_a", "drug_b", "severity", "mechanism", "effect", "safer_alternative", "rationale", "reference", "id"):
            if record.get(key):
                parts.append(str(record[key]))
    return " ".join(parts)


def build_explain_messages(result, settings, other_medications=()):
    """Only the pair, its eligible records and the settings go to the model. Nothing else."""
    payload = {
        "pair": result["pair"],
        "detail_level": settings.detail_level,
        "max_summary_characters": MAX_SUMMARY_CHARS[settings.detail_level],
        "records": [{k: r.get(k) for k in ("id", "severity", "drug_a", "drug_b", "mechanism", "effect",
                                          "safer_alternative", "rationale", "reference") if r.get(k) is not None}
                    for r in result["records"]],
    }
    if settings.context_level == "full_context":
        payload["doses_as_entered"] = {n: d for n, d in (result.get("doses") or {}).items() if d}
        payload["other_medications"] = list(other_medications)
    return [{"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_explanation(reply, result, known_drug_names=(), settings=Settings()):
    """Check a model reply against the records it was given. Returns (parsed, problems).

    The reply must be a JSON object with exactly summary, claims and clarifying_question. Every claim must
    name a supplied record and copy that record's field exactly. The summary may not state a severity other
    than the records', name a drug other than the pair and the records' alternatives, contain a number that is
    not in the records, or use a content word that is not in the records. Any problem means the reply is
    rejected and the template is used instead.
    """
    problems = []
    try:
        parsed = json.loads(reply)
    except (TypeError, json.JSONDecodeError):
        return None, ["reply is not valid JSON"]
    if not isinstance(parsed, dict) or set(parsed) != {"summary", "claims", "clarifying_question"}:
        return None, ["reply must be an object with exactly summary, claims and clarifying_question"]
    summary, claims, question = parsed["summary"], parsed["claims"], parsed["clarifying_question"]
    if not isinstance(summary, str) or not isinstance(claims, list) or not (question is None or isinstance(question, str)):
        return None, ["summary, claims or clarifying_question has the wrong type"]

    records = {r["id"]: r for r in result["records"]}
    headline = result["records"][0]
    claimed = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"record_id", "field", "value"}:
            problems.append("a claim must have exactly record_id, field and value")
            continue
        record = records.get(claim["record_id"])
        if record is None:
            problems.append(f"claim cites {claim['record_id']!r}, which was not supplied")
        elif claim["field"] not in CLAIM_FIELDS:
            problems.append(f"claim names an unknown field {claim['field']!r}")
        elif _norm(claim["value"]) != _norm(record.get(claim["field"])):
            problems.append(f"claim for {claim['record_id']}.{claim['field']} does not match the record")
        else:
            claimed.add((claim["record_id"], claim["field"]))
    for field in REQUIRED_CLAIM_FIELDS:
        if headline.get(field) and (headline["id"], field) not in claimed:
            problems.append(f"no valid claim for the headline record's {field}")

    text = " ".join(t for t in (summary, question) if t)
    corpus = _record_text(result["records"], extra=result["pair"])
    corpus_words = {w.casefold() for w in _WORD.findall(corpus)}
    allowed_severities = {_norm(r["severity"]) for r in result["records"]}
    for word in _WORD.findall(text):
        lower = word.casefold()
        if lower in _SEVERITY_WORDS:
            if lower not in allowed_severities:
                problems.append(f"summary states severity {word}, which the records do not")
        elif len(lower) >= 4 and lower not in ALLOWED_PLAIN_WORDS and lower not in corpus_words:
            problems.append(f"summary uses {word!r}, which is not in the records (a possible new fact)")
    for number in _NUMBER.findall(text):
        if number not in corpus:
            problems.append(f"summary contains the number {number}, which is not in the records")
    allowed_drugs = {_norm(n) for n in result["pair"]}
    for record in result["records"]:
        allowed_drugs |= {_norm(record["drug_a"]), _norm(record["drug_b"])}
        allowed_drugs |= {_norm(n) for n in record["alternative"]["drug_names"]}
    for name in known_drug_names:
        key = _norm(name)
        if key and key not in allowed_drugs and re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", _norm(text)):
            problems.append(f"summary names {name}, which is not part of this pair or its alternatives")
    if len(summary) > MAX_SUMMARY_CHARS[settings.detail_level]:
        problems.append("summary is longer than the detail level allows")
    if question is not None and not question.strip().endswith("?"):
        problems.append("clarifying_question must be a question")
    return (None, problems) if problems else (parsed, [])


class Explainer:
    def __init__(self, provider=None, settings=Settings(), known_drug_names=()):
        self.provider = provider
        self.settings = settings
        self.known_drug_names = list(known_drug_names)

    def explain_pair(self, result, other_medications=(), clarifying_question=None):
        """A plain-language response for one pair result, shown next to its source records."""
        out = {
            "pair": result["pair"],
            "status": result["status"],
            "settings": {"context_level": self.settings.context_level, "detail_level": self.settings.detail_level},
            "generated_by": "template",
            "fallback_reason": None,
            "records": result.get("records", []),
            "source": result.get("source"),
            "disclaimer": DISCLAIMER,
        }
        template = composer.compose_pair(result, self.settings.detail_level)
        out["clarifying_question"] = clarifying_question
        if result["status"] != "documented_interaction" or not result.get("records"):
            out["text"] = template  # absent evidence: "not documented in our data"; no records, no API call
            out["fallback_reason"] = "no eligible record for this pair, so no language model call was made"
            return out
        if self.provider is None:
            out["text"] = template
            out["fallback_reason"] = "no language model configured (deterministic-only mode)"
            return out
        try:
            reply = self.provider.complete(build_explain_messages(result, self.settings, other_medications))
        except RuntimeError as error:
            out["text"] = template
            out["fallback_reason"] = f"language model unavailable: {error}"
            return out
        parsed, problems = validate_explanation(reply, result, self.known_drug_names, self.settings)
        out["raw_reply"] = reply
        if problems:
            out["text"] = template
            out["fallback_reason"] = "language model reply failed validation: " + "; ".join(problems)
            out["validation_problems"] = problems
            return out
        out["generated_by"] = "language_model"
        out["claims"] = parsed["claims"]
        out["text"] = parsed["summary"]
        out["clarifying_question"] = parsed["clarifying_question"] or clarifying_question
        out["text"] += "\n" + "Source: " + "; ".join(f"{c['id']} ({c['reference']})" for c in result["source"]["citations"])
        return out

    def explain_check(self, check_output, patient_context_question=None):
        """Explain every pair in a /check style output. The generated text sits next to the source records."""
        names = [m["name"] for m in check_output["inputs"]["current"] if m.get("name")]
        names.append(check_output["inputs"]["new"].get("name"))
        return [self.explain_pair(r, other_medications=[n for n in names if n and n not in r["pair"]],
                                  clarifying_question=patient_context_question)
                for r in check_output["results"]]
