"""Evaluation harness: the backend and the language model, checked against the dataset.

The dataset is the ground truth. Every system is run on the same pairs in every condition
(change one variable at a time): names-only against full-context, and brief against detailed.
Reported for each condition: missed known interactions, false interactions, wrong severity and
unsupported claims, Major-interaction recall on its own, and a severity confusion matrix. The
filter check reports how often an eligible record would be wrongly excluded.

Failure codes follow the course notebooks: R1 retrieval or knowledge miss (a documented
interaction was not found), G1 unsupported generation (a claim the dataset does not support or
contradicts), C1 citation mismatch, E1 evidence insufficient (nothing is documented, so nothing
is claimed). Model claims are labelled supported, plausible but unsupported, or contradicted.
"""

import json
import random
import re
from collections import Counter

from ddi_backend.common import SEVERITIES
from ddi_backend.eligibility import EligibilityFilter
from ddi_backend.retrieval import QuestionAnswerer

SEVERITY_COLUMNS = list(SEVERITIES) + ["None", "Unparseable"]
SUPPORTED = "supported"
UNSUPPORTED = "plausible but unsupported"
CONTRADICTED = "contradicted"

CONTEXT_LEVELS = ("names_only", "full_context")
DETAIL_LEVELS = ("brief", "detailed")

_WORD = re.compile(r"[a-z]{4,}")


def pair_key(a, b):
    return tuple(sorted((a.casefold(), b.casefold())))


# --------------------------------------------------------------------------
# Ground truth and the pair set
# --------------------------------------------------------------------------
def ground_truth(checker):
    """{pair_key: truth} for every documented pair, from the dataset's own rows (headline = most severe)."""
    from ddi_backend import rules
    truth = {}
    graph = checker.graph
    seen = set()
    for record in graph.records:
        ids = (record["drug_a_id"], record["drug_b_id"])
        if frozenset(ids) in seen:
            continue
        seen.add(frozenset(ids))
        rows, _ = rules.collapse_duplicates(graph.pair(*ids))
        headline, ordered = rules.severity_precedence(rows)
        truth[pair_key(headline["drug_a"], headline["drug_b"])] = {
            "pair": (headline["drug_a"], headline["drug_b"]),
            "severity": headline["severity"],
            "mechanism": headline["mechanism"],
            "effect": headline["effect"],
            "record_ids": [r["id"] for r in ordered],
        }
    return truth


def build_pair_set(checker, truth, n_negative=60, seed=0):
    """All documented pairs plus a seeded sample of pairs of known drugs with no documented row."""
    names = sorted({r["drug_a"] for r in checker.graph.records} | {r["drug_b"] for r in checker.graph.records},
                   key=str.casefold)
    rng = random.Random(seed)
    negatives, seen = [], set()
    attempts = 0
    while len(negatives) < n_negative and attempts < n_negative * 200:
        attempts += 1
        a, b = rng.sample(names, 2)
        key = pair_key(a, b)
        if key in truth or key in seen:
            continue
        seen.add(key)
        negatives.append((a, b))
    documented = [t["pair"] for t in sorted(truth.values(), key=lambda t: t["pair"])]
    return documented, negatives


def synthetic_context(pair, names, seed=0):
    """Synthetic doses and other medications for the full-context condition. Simulated, never real patient data."""
    rng = random.Random(f"{seed}:{pair_key(*pair)}")
    others = rng.sample([n for n in names if n not in pair], 2)
    return {"doses": {pair[0]: f"{rng.choice([5, 10, 20, 40])} mg", pair[1]: f"{rng.choice([50, 100, 200])} mg"},
            "other_medications": others, "synthetic": True}


# --------------------------------------------------------------------------
# The deterministic backend as a system under test
# --------------------------------------------------------------------------
def backend_assessment(checker, pair):
    result = checker.check([pair[0]], pair[1])["results"][0]
    documented = result["status"] == "documented_interaction"
    return {"interaction": documented, "severity": result.get("severity") if documented else None,
            "mechanism": result.get("mechanism") if documented else None,
            "effect": result.get("effect") if documented else None, "status": result["status"]}


# --------------------------------------------------------------------------
# The language model as a system under test (ungrounded: it sees no records)
# --------------------------------------------------------------------------
ASSESS_SYSTEM = (
    "You are being evaluated on drug-drug interaction knowledge for a class prototype (not medical advice). "
    "You have no tools and no internet. Answer from your own knowledge only. Return a JSON object with exactly "
    "these keys: interaction (true or false), severity (\"Major\", \"Moderate\", \"Minor\" or null), mechanism "
    "(string or null), effect (string or null). Use null for severity, mechanism and effect when you say there is "
    "no interaction."
)


def build_assessment_messages(pair, context_level, detail_level, context=None):
    payload = {"drug_a": pair[0], "drug_b": pair[1]}
    if context_level == "full_context" and context:
        payload["doses_as_entered"] = context["doses"]
        payload["other_current_medications"] = context["other_medications"]
        payload["note"] = "Doses and medications are synthetic."
    instruction = ("Give the severity only; set mechanism and effect to null." if detail_level == "brief"
                   else "Give the severity, and one sentence each for the mechanism and the clinical effect.")
    return [{"role": "system", "content": ASSESS_SYSTEM + " " + instruction},
            {"role": "user", "content": json.dumps(payload)}]


def parse_assessment(reply):
    """The model's reply as an assessment dict, or None when it does not follow the format."""
    try:
        data = json.loads(reply)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not {"interaction", "severity"} <= set(data) or not isinstance(data["interaction"], bool):
        return None
    severity = data.get("severity")
    if severity is not None:
        severity = str(severity).strip().capitalize()
        if severity not in SEVERITIES:
            return None
    return {"interaction": data["interaction"], "severity": severity if data["interaction"] else None,
            "mechanism": data.get("mechanism") if isinstance(data.get("mechanism"), str) else None,
            "effect": data.get("effect") if isinstance(data.get("effect"), str) else None}


def model_assessment(provider, pair, context_level, detail_level, context=None):
    try:
        reply = provider.complete(build_assessment_messages(pair, context_level, detail_level, context))
    except RuntimeError as error:
        return {"error": str(error)}
    parsed = parse_assessment(reply)
    return parsed if parsed is not None else {"unparseable": True, "raw": reply}


# --------------------------------------------------------------------------
# Labelling and scoring
# --------------------------------------------------------------------------
def _content_words(text):
    return set(_WORD.findall((text or "").casefold()))


def _text_supported(claim, reference):
    """A mechanism or effect claim is supported when the shorter text's content words all appear in the other."""
    a, b = _content_words(claim), _content_words(reference)
    if not a or not b:
        return False
    return a <= b or b <= a


def label_claims(assessment, truth_entry):
    """Label each claim: supported, plausible but unsupported, or contradicted, with a failure code."""
    labels = []
    documented = truth_entry is not None
    says_yes = assessment["interaction"]
    if says_yes and documented:
        labels.append(("existence", SUPPORTED, None))
    elif says_yes:
        labels.append(("existence", UNSUPPORTED, "G1"))
    elif documented:
        labels.append(("existence", CONTRADICTED, "R1"))
    if says_yes and assessment.get("severity"):
        if not documented:
            labels.append(("severity", UNSUPPORTED, "G1"))
        elif assessment["severity"] == truth_entry["severity"]:
            labels.append(("severity", SUPPORTED, None))
        else:
            labels.append(("severity", CONTRADICTED, "G1"))
    for field in ("mechanism", "effect"):
        text = assessment.get(field)
        if says_yes and text:
            if documented and _text_supported(text, truth_entry[field]):
                labels.append((field, SUPPORTED, None))
            else:
                labels.append((field, UNSUPPORTED, "G1"))
    return labels


def score_condition(assessments, truth):
    """Metrics for one condition. ``assessments`` is [(pair, assessment)]; unscored errors are counted."""
    counts = Counter()
    matrix = {row: Counter() for row in list(SEVERITIES) + ["None"]}
    claim_labels, failures = Counter(), Counter()
    major_total = major_hit = major_any = 0
    for pair, assessment in assessments:
        entry = truth.get(pair_key(*pair))
        row = entry["severity"] if entry else "None"
        if "error" in assessment:
            counts["errors"] += 1
            continue
        if assessment.get("unparseable"):
            counts["unparseable"] += 1
            matrix[row]["Unparseable"] += 1
            if row == "Major":
                major_total += 1
            continue
        counts["scored"] += 1
        counts["documented" if entry else "absent"] += 1
        predicted = (assessment["severity"] or "None") if assessment["interaction"] else "None"
        matrix[row][predicted] += 1
        if entry and not assessment["interaction"]:
            counts["missed_known_interactions"] += 1
        if not entry and assessment["interaction"]:
            counts["false_interactions"] += 1
        if entry and assessment["interaction"]:
            if assessment["severity"] == entry["severity"]:
                counts["correct_severity"] += 1
            else:
                counts["wrong_severity"] += 1
        if row == "Major":
            major_total += 1
            major_any += 1 if assessment["interaction"] else 0
            major_hit += 1 if predicted == "Major" else 0
        for field, label, code in label_claims(assessment, entry):
            claim_labels[label] += 1
            if code:
                failures[code] += 1
    unsupported = claim_labels[UNSUPPORTED] + claim_labels[CONTRADICTED]
    return {
        "counts": dict(counts),
        "major": {"total": major_total, "recall_exact_severity": _ratio(major_hit, major_total),
                  "recall_any_interaction": _ratio(major_any, major_total)},
        "claims": dict(claim_labels),
        "unsupported_claims": unsupported,
        "failure_codes": dict(failures),
        "confusion_matrix": {row: {col: matrix[row].get(col, 0) for col in SEVERITY_COLUMNS} for row in matrix},
    }


def _ratio(num, den):
    return None if den == 0 else round(num / den, 4)


def run_condition(system, pairs, truth, context_for=None):
    """Run one system on every pair. ``system(pair, context) -> assessment``."""
    assessments = []
    for pair in pairs:
        assessments.append((pair, system(pair, context_for(pair) if context_for else None)))
    return score_condition(assessments, truth)


# --------------------------------------------------------------------------
# The eligibility filter: how often does it wrongly exclude an eligible record?
# --------------------------------------------------------------------------
def labelled_filter_suite():
    """(registry, cases): made-up sources with a known right answer for each record.

    Each case is (record, pair_names, expected_eligible). These are test inputs, not real metadata.
    """
    from ddi_backend import registry as registry_module
    seed = {"approved_authority_classes": ["approved_demo"], "sources": {
        "demo_current": {"status": "current", "authority_class": "approved_demo", "scope": "exact_pair"},
        "demo_single_drug": {"status": "current", "authority_class": "approved_demo", "scope": "single_drug"},
        "demo_stale": {"status": "archived", "authority_class": "approved_demo", "scope": "exact_pair"},
        "demo_unknown_status": {"status": "unknown", "authority_class": "approved_demo", "scope": "exact_pair"},
        "demo_no_authority": {"status": "current", "scope": "exact_pair"},
        "demo_unapproved": {"status": "current", "authority_class": "vocabulary", "scope": "exact_pair"},
        "demo_vocabulary": {"status": "current", "authority_class": "approved_demo", "scope": "none"},
    }}
    pair = ("Warfarin", "Ibuprofen")

    def rec(key, a="Warfarin", b="Ibuprofen"):
        return {"id": f"{key}:1", "source_key": key, "drug_a": a, "drug_b": b}

    cases = [
        (rec("demo_current"), pair, True),
        (rec("demo_current", "Ibuprofen", "Warfarin"), pair, True),          # reversed order is the same pair
        (rec("demo_single_drug"), pair, True),                              # names the exact pair
        (rec("demo_single_drug", "Warfarin", "Aspirin"), pair, True),       # context: shares one drug
        (rec("demo_stale"), pair, False),
        (rec("demo_unknown_status"), pair, False),
        (rec("demo_no_authority"), pair, False),
        (rec("demo_unapproved"), pair, False),
        (rec("demo_vocabulary"), pair, False),
        (rec("demo_not_registered"), pair, False),
        (rec("demo_current", "Zinc", "Iron Sulfate"), pair, False),         # a different pair altogether
    ]
    return registry_module.build_registry(seed), cases


def evaluate_filter(checker, registry, suite=None):
    """Wrong-exclusion rate on the real dataset and on a labelled suite of made-up sources.

    On the dataset every record is eligible under the team's designation, so any exclusion is a wrong
    exclusion. The suite (from labelled_filter_suite) reports wrong exclusions (an eligible record dropped)
    and wrong inclusions (an ineligible one kept).
    """
    flt = EligibilityFilter(registry)
    records = checker.graph.records
    # The graph already holds only eligible records, so evaluate the raw rows through the filter again.
    from ddi_backend import ingest
    raw = ingest.load_interactions(checker.conn)
    excluded = [r["id"] for r in raw if not flt.evaluate(r, (r["drug_a"], r["drug_b"])).eligible]
    out = {"dataset_records": len(raw), "eligible_records_wrongly_excluded": len(excluded),
           "wrong_exclusion_rate": _ratio(len(excluded), len(raw)), "excluded_ids": excluded,
           "graph_records": len(records)}
    if suite:
        suite_registry, labelled_suite = suite
        suite_filter = EligibilityFilter(suite_registry)
        wrong_ex = wrong_in = 0
        eligible_total = sum(1 for *_, expected in labelled_suite if expected)
        for record, names, expected in labelled_suite:
            actual = suite_filter.evaluate(record, names).eligible
            wrong_ex += 1 if expected and not actual else 0
            wrong_in += 1 if actual and not expected else 0
        out["labelled_suite"] = {"cases": len(labelled_suite), "eligible_cases": eligible_total,
                                 "eligible_wrongly_excluded": wrong_ex, "ineligible_wrongly_included": wrong_in,
                                 "wrong_exclusion_rate": _ratio(wrong_ex, eligible_total)}
    return out


# --------------------------------------------------------------------------
# The handout and Meeting 8 questions, run through the question answerer
# --------------------------------------------------------------------------
def _ids(answer):
    return [rid for r in answer.get("results", []) for rid in r.get("source", {}).get("record_ids", [])]


HANDOUT_CASES = [
    ("RAG 1 direct question", "Do warfarin and ibuprofen interact, and how severe is it?",
     lambda a: _ids(a) == ["ddi:1"] and a["results"][0]["severity"] == "Major"),
    ("RAG 2 paraphrase", "Is it dangerous to take ibuprofen while on warfarin?", lambda a: _ids(a) == ["ddi:1"]),
    ("RAG 3 severity, effect and mechanism", "What are the severity, clinical effect and mechanism of warfarin with fluconazole?",
     lambda a: _ids(a) == ["ddi:12"] and all(a["results"][0].get(k) for k in ("severity", "effect", "mechanism"))),
    ("RAG 4 distractor: pain reliever and warfarin", "Which pain reliever should I not take with warfarin?",
     lambda a: a.get("intent") == "avoid" and a["avoid"]["avoid"][0]["severity"] == "Major"
     and a["avoid"]["avoid"][0]["drug"] != "Acetaminophen"),
    ("RAG 5 absent pair (drugs not in the data)", "Do cabergoline and rucaparib interact?",
     lambda a: a["results"][0]["status"] == "unknown_drug" and "does not mean the combination is safe" in " ".join(a["results"][0]["notes"])),
    ("RAG 5 variant: known drugs, no documented row", "Do warfarin and docusate interact?",
     lambda a: a["results"][0]["status"] == "no_documented_interaction"),
    ("Meeting 7 reworded query, same record", "Does taking clopidogrel with omeprazole reduce the antiplatelet effect?",
     lambda a: len(_ids(a)) == 2 and a["results"][0]["severity"] == "Major" and a["results"][0]["severity_conflict"]),
    ("Meeting 8 ambiguity: MS", "What does MS interact with?", lambda a: a["status"] == "needs_clarification" and "results" not in a),
    ("Meeting 8 negation", "What should I not take with warfarin?", lambda a: a.get("intent") == "avoid"),
    ("Meeting 8 context: children", "Is warfarin with ibuprofen safe for children?",
     lambda a: "population_context_not_covered" in a["rules_fired"] and a["results"][0]["status"] == "documented_interaction"),
]


def run_handout_cases(question_answerer):
    """Run the fixed cases. Expected behaviour is written in the table above, before any output is read."""
    rows = []
    for label, question, check in HANDOUT_CASES:
        answer = question_answerer.ask(question)
        try:
            passed = bool(check(answer))
        except (KeyError, IndexError, TypeError):
            passed = False
        rows.append({"case": label, "question": question, "passed": passed})
    return rows


# --------------------------------------------------------------------------
# Simulated providers: only for checking the harness itself, never for reporting model quality
# --------------------------------------------------------------------------
class SimulatedProvider:
    """A scripted stand-in for a model. SIMULATED: its numbers say nothing about any real model."""
    name = "simulated"

    def __init__(self, truth, mode="oracle", seed=0):
        self.truth, self.mode, self.rng = truth, mode, random.Random(seed)

    def complete(self, messages):
        payload = json.loads(messages[1]["content"])
        entry = self.truth.get(pair_key(payload["drug_a"], payload["drug_b"]))
        detailed = "one sentence each" in messages[0]["content"]
        if self.mode == "oracle":
            if entry:
                return json.dumps({"interaction": True, "severity": entry["severity"],
                                   "mechanism": entry["mechanism"] if detailed else None,
                                   "effect": entry["effect"] if detailed else None})
            return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})
        # "noisy": misses some Major rows, invents some interactions, mislabels some severities
        roll = self.rng.random()
        if entry and roll < 0.2:
            return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})
        if entry and roll < 0.4:
            wrong = [s for s in SEVERITIES if s != entry["severity"]][0]
            return json.dumps({"interaction": True, "severity": wrong, "mechanism": "unrelated mechanism claim",
                               "effect": entry["effect"] if detailed else None})
        if not entry and roll < 0.3:
            return json.dumps({"interaction": True, "severity": "Moderate",
                               "mechanism": "invented pathway" if detailed else None, "effect": None})
        if entry:
            return json.dumps({"interaction": True, "severity": entry["severity"], "mechanism": None, "effect": None})
        return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})


# --------------------------------------------------------------------------
# The full run and the report
# --------------------------------------------------------------------------
def run_evaluation(checker, registry, provider=None, n_negative=60, seed=0, limit=None, label="no language model"):
    truth = ground_truth(checker)
    documented, negatives = build_pair_set(checker, truth, n_negative, seed)
    pairs = documented + negatives
    if limit:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, min(limit, len(pairs)))
    names = sorted({r["drug_a"] for r in checker.graph.records} | {r["drug_b"] for r in checker.graph.records})
    results = {
        "pairs": {"documented": sum(1 for p in pairs if pair_key(*p) in truth),
                  "absent": sum(1 for p in pairs if pair_key(*p) not in truth), "seed": seed},
        "backend": run_condition(lambda pair, _ctx: backend_assessment(checker, pair), pairs, truth),
        "filter": evaluate_filter(checker, registry, labelled_filter_suite()),
        "handout_cases": run_handout_cases(QuestionAnswerer(checker)),
        "model_label": label,
        "conditions": {},
    }
    if provider is not None:
        for context_level in CONTEXT_LEVELS:
            for detail_level in DETAIL_LEVELS:
                def system(pair, context, cl=context_level, dl=detail_level):
                    return model_assessment(provider, pair, cl, dl, context)
                results["conditions"][f"{context_level} / {detail_level}"] = run_condition(
                    system, pairs, truth,
                    context_for=(lambda pair: synthetic_context(pair, names, seed)) if context_level == "full_context" else None)
    return results


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_report(results):
    lines = ["# Evaluation report", "",
             "Class prototype, not medical advice. The dataset is the ground truth. Doses and other "
             "medications in the full-context condition are synthetic.", "",
             f"Pairs: {results['pairs']['documented']} documented, {results['pairs']['absent']} with no documented "
             f"row (seed {results['pairs']['seed']}). Every condition ran the same pairs.", ""]

    def table(name, score):
        c, m = score["counts"], score["major"]
        out = [f"## {name}", "",
               "| Measure | Value |", "|---|---|",
               f"| Pairs scored | {c.get('scored', 0)} (errors {c.get('errors', 0)}, unparseable {c.get('unparseable', 0)}) |",
               f"| Missed known interactions | {c.get('missed_known_interactions', 0)} of {c.get('documented', 0)} |",
               f"| False interactions | {c.get('false_interactions', 0)} of {c.get('absent', 0)} |",
               f"| Wrong severity | {c.get('wrong_severity', 0)} |",
               f"| Unsupported claims (plausible but unsupported / contradicted) | {score['unsupported_claims']} "
               f"({score['claims'].get('plausible but unsupported', 0)} / {score['claims'].get('contradicted', 0)}) |",
               f"| Major recall, exact severity | {_pct(m['recall_exact_severity'])} of {m['total']} |",
               f"| Major recall, any interaction | {_pct(m['recall_any_interaction'])} |",
               f"| Failure codes | {score['failure_codes'] or 'none'} |", "",
               "Severity confusion matrix (rows: dataset, columns: system):", "",
               "| | " + " | ".join(SEVERITY_COLUMNS) + " |", "|---|" + "---|" * len(SEVERITY_COLUMNS)]
        for row, cells in score["confusion_matrix"].items():
            out.append(f"| {row} | " + " | ".join(str(cells[col]) for col in SEVERITY_COLUMNS) + " |")
        return out + [""]

    lines += table("Backend (deterministic)", results["backend"])
    for name, score in results["conditions"].items():
        lines += table(f"Language model, {name} ({results['model_label']})", score)
    if not results["conditions"]:
        lines += ["## Language model", "", "Not run: no language model was configured. Set OPENAI_API_KEY and run "
                  "`python eval/run_eval.py --llm openai`.", ""]
    lines += ["## Handout and Meeting 8 questions", "", "| Case | Result |", "|---|---|"]
    lines += [f"| {r['case']} | {'pass' if r['passed'] else 'FAIL'} |" for r in results["handout_cases"]]
    lines.append("")
    f = results["filter"]
    lines += ["## Eligibility filter", "",
              f"On the dataset ({f['dataset_records']} records) the filter wrongly excluded "
              f"{f['eligible_records_wrongly_excluded']} eligible records ({_pct(f['wrong_exclusion_rate'])}).", ""]
    if "labelled_suite" in f:
        s = f["labelled_suite"]
        lines.append(f"On the labelled suite of made-up sources ({s['cases']} cases): {s['eligible_wrongly_excluded']} "
                     f"eligible records wrongly excluded, {s['ineligible_wrongly_included']} ineligible records wrongly included.")
    return "\n".join(lines) + "\n"
