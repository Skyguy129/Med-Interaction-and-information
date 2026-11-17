# Module docstring: what the harness measures, the failure codes (R1, G1, C1, E1) and the claim labels.
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

import json  # encodes and decodes the model messages
import random  # seeded sampling so every run uses the same pairs
import re  # word pattern for comparing claim text
from collections import Counter  # tallies counts, claim labels and failure codes

from ddi_backend.common import SEVERITIES  # ("Major", "Moderate", "Minor")
from ddi_backend.eligibility import EligibilityFilter  # the evidence filter being tested
from ddi_backend.retrieval import QuestionAnswerer  # the free-text question handler, for the handout cases

SEVERITY_COLUMNS = list(SEVERITIES) + ["None", "Unparseable"]  # confusion-matrix columns: severities plus two extras
SUPPORTED = "supported"  # claim label: matches the dataset
UNSUPPORTED = "plausible but unsupported"  # claim label: not in the dataset
CONTRADICTED = "contradicted"  # claim label: conflicts with the dataset

CONTEXT_LEVELS = ("names_only", "full_context")  # variable one: how much context the model gets
DETAIL_LEVELS = ("brief", "detailed")  # variable two: how much it is asked to write

_WORD = re.compile(r"[a-z]{4,}")  # words of four or more letters, used as "content words"


def pair_key(a, b):
    # Order-free, case-free key so (A, B) and (b, a) are the same pair.
    return tuple(sorted((a.casefold(), b.casefold())))


# --------------------------------------------------------------------------
# Ground truth and the pair set
# --------------------------------------------------------------------------
def ground_truth(checker):
    # Docstring: builds the answer key from the dataset itself.
    """{pair_key: truth} for every documented pair, from the dataset's own rows (headline = most severe)."""
    from ddi_backend import rules  # imported here to avoid loading it unless needed
    truth = {}  # pair_key -> true severity, mechanism, effect and record ids
    graph = checker.graph  # the knowledge graph of eligible records
    seen = set()  # drug-id pairs already handled (a pair can appear twice)
    for record in graph.records:
        ids = (record["drug_a_id"], record["drug_b_id"])
        if frozenset(ids) in seen:  # skip a pair already processed
            continue
        seen.add(frozenset(ids))
        rows, _ = rules.collapse_duplicates(graph.pair(*ids))  # merge exact repeat rows
        headline, ordered = rules.severity_precedence(rows)  # most severe row leads, others follow
        truth[pair_key(headline["drug_a"], headline["drug_b"])] = {
            "pair": (headline["drug_a"], headline["drug_b"]),  # names as stored
            "severity": headline["severity"],  # the headline (highest) severity
            "mechanism": headline["mechanism"],
            "effect": headline["effect"],
            "record_ids": [r["id"] for r in ordered],  # every row's id, most severe first
        }
    return truth


def build_pair_set(checker, truth, n_negative=60, seed=0):
    # Docstring: documented pairs plus a seeded sample of known-drug pairs with no row.
    """All documented pairs plus a seeded sample of pairs of known drugs with no documented row."""
    # Every drug name in the data, sorted so the sample is repeatable.
    names = sorted({r["drug_a"] for r in checker.graph.records} | {r["drug_b"] for r in checker.graph.records},
                   key=str.casefold)
    rng = random.Random(seed)  # private seeded random generator
    negatives, seen = [], set()  # chosen no-row pairs, and keys already used
    attempts = 0  # guards against looping forever
    while len(negatives) < n_negative and attempts < n_negative * 200:
        attempts += 1
        a, b = rng.sample(names, 2)  # two different drugs at random
        key = pair_key(a, b)
        if key in truth or key in seen:  # skip documented or repeated pairs
            continue
        seen.add(key)
        negatives.append((a, b))  # a known pair with no documented row
    documented = [t["pair"] for t in sorted(truth.values(), key=lambda t: t["pair"])]  # all documented, in order
    return documented, negatives


def synthetic_context(pair, names, seed=0):
    # Docstring: fake patient context for the full-context condition.
    """Synthetic doses and other medications for the full-context condition. Simulated, never real patient data."""
    rng = random.Random(f"{seed}:{pair_key(*pair)}")  # seeded by pair so a pair always gets the same context
    others = rng.sample([n for n in names if n not in pair], 2)  # two other made-up medications
    return {"doses": {pair[0]: f"{rng.choice([5, 10, 20, 40])} mg", pair[1]: f"{rng.choice([50, 100, 200])} mg"},
            "other_medications": others, "synthetic": True}  # flagged synthetic


# --------------------------------------------------------------------------
# The deterministic backend as a system under test
# --------------------------------------------------------------------------
def backend_assessment(checker, pair):
    result = checker.check([pair[0]], pair[1])["results"][0]  # run the real checker on one pair
    documented = result["status"] == "documented_interaction"  # only this status counts as an interaction
    # Convert the checker's result to the same assessment shape used for the model.
    return {"interaction": documented, "severity": result.get("severity") if documented else None,
            "mechanism": result.get("mechanism") if documented else None,
            "effect": result.get("effect") if documented else None, "status": result["status"]}


# --------------------------------------------------------------------------
# The language model as a system under test (ungrounded: it sees no records)
# --------------------------------------------------------------------------
# System prompt for the model under test: own knowledge only, strict JSON keys, null when no interaction.
ASSESS_SYSTEM = (
    "You are being evaluated on drug-drug interaction knowledge for a class prototype (not medical advice). "
    "You have no tools and no internet. Answer from your own knowledge only. Return a JSON object with exactly "
    "these keys: interaction (true or false), severity (\"Major\", \"Moderate\", \"Minor\" or null), mechanism "
    "(string or null), effect (string or null). Use null for severity, mechanism and effect when you say there is "
    "no interaction."
)


def build_assessment_messages(pair, context_level, detail_level, context=None):
    payload = {"drug_a": pair[0], "drug_b": pair[1]}  # the minimum the model always gets
    if context_level == "full_context" and context:  # add synthetic context only in this condition
        payload["doses_as_entered"] = context["doses"]
        payload["other_current_medications"] = context["other_medications"]
        payload["note"] = "Doses and medications are synthetic."  # say so, so it is never read as real
    # The brief condition asks for severity only; the detailed one also asks for mechanism and effect.
    instruction = ("Give the severity only; set mechanism and effect to null." if detail_level == "brief"
                   else "Give the severity, and one sentence each for the mechanism and the clinical effect.")
    return [{"role": "system", "content": ASSESS_SYSTEM + " " + instruction},
            {"role": "user", "content": json.dumps(payload)}]


def parse_assessment(reply):
    # Docstring: strict parser; anything off-format becomes None.
    """The model's reply as an assessment dict, or None when it does not follow the format."""
    try:
        data = json.loads(reply)  # must be JSON
    except (TypeError, json.JSONDecodeError):
        return None
    # Needs a dict with the two required keys and a true/false interaction.
    if not isinstance(data, dict) or not {"interaction", "severity"} <= set(data) or not isinstance(data["interaction"], bool):
        return None
    severity = data.get("severity")
    if severity is not None:
        severity = str(severity).strip().capitalize()  # tolerate case and spaces
        if severity not in SEVERITIES:  # an unknown severity word is a format failure
            return None
    # Severity only counts when the model says there is an interaction; text fields must be strings.
    return {"interaction": data["interaction"], "severity": severity if data["interaction"] else None,
            "mechanism": data.get("mechanism") if isinstance(data.get("mechanism"), str) else None,
            "effect": data.get("effect") if isinstance(data.get("effect"), str) else None}


def model_assessment(provider, pair, context_level, detail_level, context=None):
    try:
        reply = provider.complete(build_assessment_messages(pair, context_level, detail_level, context))  # one call
    except RuntimeError as error:  # network or key problem is recorded, not fatal
        return {"error": str(error)}
    parsed = parse_assessment(reply)
    return parsed if parsed is not None else {"unparseable": True, "raw": reply}  # keep the raw reply for inspection


# --------------------------------------------------------------------------
# Labelling and scoring
# --------------------------------------------------------------------------
def _content_words(text):
    return set(_WORD.findall((text or "").casefold()))  # the set of content words in a text


def _text_supported(claim, reference):
    # Docstring: loose text match used for mechanism and effect claims.
    """A mechanism or effect claim is supported when the shorter text's content words all appear in the other."""
    a, b = _content_words(claim), _content_words(reference)
    if not a or not b:  # nothing to compare
        return False
    return a <= b or b <= a  # one word set contains the other


def label_claims(assessment, truth_entry):
    # Docstring: gives each claim a label and, if wrong, a failure code.
    """Label each claim: supported, plausible but unsupported, or contradicted, with a failure code."""
    labels = []  # (claim name, label, failure code or None)
    documented = truth_entry is not None  # is the pair in the dataset?
    says_yes = assessment["interaction"]  # does the model claim an interaction?
    if says_yes and documented:
        labels.append(("existence", SUPPORTED, None))
    elif says_yes:  # claims an interaction the dataset does not have
        labels.append(("existence", UNSUPPORTED, "G1"))
    elif documented:  # missed a documented interaction
        labels.append(("existence", CONTRADICTED, "R1"))
    if says_yes and assessment.get("severity"):  # a severity claim was made
        if not documented:
            labels.append(("severity", UNSUPPORTED, "G1"))
        elif assessment["severity"] == truth_entry["severity"]:
            labels.append(("severity", SUPPORTED, None))
        else:
            labels.append(("severity", CONTRADICTED, "G1"))
    for field in ("mechanism", "effect"):  # free-text claims
        text = assessment.get(field)
        if says_yes and text:
            if documented and _text_supported(text, truth_entry[field]):
                labels.append((field, SUPPORTED, None))
            else:
                labels.append((field, UNSUPPORTED, "G1"))
    return labels


def score_condition(assessments, truth):
    # Docstring: turns a list of (pair, assessment) into the reported metrics.
    """Metrics for one condition. ``assessments`` is [(pair, assessment)]; unscored errors are counted."""
    counts = Counter()  # scored, missed, false, wrong severity and similar counts
    matrix = {row: Counter() for row in list(SEVERITIES) + ["None"]}  # confusion matrix: dataset row -> system column
    claim_labels, failures = Counter(), Counter()  # claim label tallies and failure-code tallies
    major_total = major_hit = major_any = 0  # Major recall counters
    for pair, assessment in assessments:
        entry = truth.get(pair_key(*pair))  # the true entry, or None when no row exists
        row = entry["severity"] if entry else "None"  # which matrix row this pair belongs to
        if "error" in assessment:  # the call failed: count it and skip scoring
            counts["errors"] += 1
            continue
        if assessment.get("unparseable"):  # badly formatted reply
            counts["unparseable"] += 1
            matrix[row]["Unparseable"] += 1
            if row == "Major":  # an unparseable reply on a Major pair still counts against Major recall
                major_total += 1
            continue
        counts["scored"] += 1
        counts["documented" if entry else "absent"] += 1  # how many scored pairs had a row
        predicted = (assessment["severity"] or "None") if assessment["interaction"] else "None"  # system's column
        matrix[row][predicted] += 1
        if entry and not assessment["interaction"]:  # missed a known interaction
            counts["missed_known_interactions"] += 1
        if not entry and assessment["interaction"]:  # invented an interaction
            counts["false_interactions"] += 1
        if entry and assessment["interaction"]:  # both agree there is one: compare severity
            if assessment["severity"] == entry["severity"]:
                counts["correct_severity"] += 1
            else:
                counts["wrong_severity"] += 1
        if row == "Major":  # Major recall is reported on its own
            major_total += 1
            major_any += 1 if assessment["interaction"] else 0  # found some interaction
            major_hit += 1 if predicted == "Major" else 0  # found it and called it Major
        for field, label, code in label_claims(assessment, entry):  # label each claim
            claim_labels[label] += 1
            if code:  # count failure codes (R1, G1)
                failures[code] += 1
    unsupported = claim_labels[UNSUPPORTED] + claim_labels[CONTRADICTED]  # all claims not backed by the dataset
    return {
        "counts": dict(counts),
        "major": {"total": major_total, "recall_exact_severity": _ratio(major_hit, major_total),
                  "recall_any_interaction": _ratio(major_any, major_total)},
        "claims": dict(claim_labels),
        "unsupported_claims": unsupported,
        "failure_codes": dict(failures),
        # Fill every cell so the matrix is complete, with zeros where nothing happened.
        "confusion_matrix": {row: {col: matrix[row].get(col, 0) for col in SEVERITY_COLUMNS} for row in matrix},
    }


def _ratio(num, den):
    return None if den == 0 else round(num / den, 4)  # None instead of dividing by zero


def run_condition(system, pairs, truth, context_for=None):
    # Docstring: runs one system over every pair and scores it.
    """Run one system on every pair. ``system(pair, context) -> assessment``."""
    assessments = []
    for pair in pairs:
        # Give the system the synthetic context only when a context builder was supplied.
        assessments.append((pair, system(pair, context_for(pair) if context_for else None)))
    return score_condition(assessments, truth)


# --------------------------------------------------------------------------
# The eligibility filter: how often does it wrongly exclude an eligible record?
# --------------------------------------------------------------------------
def labelled_filter_suite():
    # Docstring: explains the made-up test sources and the case format.
    """(registry, cases): made-up sources with a known right answer for each record.

    Each case is (record, pair_names, expected_eligible). These are test inputs, not real metadata.
    """
    from ddi_backend import registry as registry_module  # imported here to build a custom registry
    # A made-up registry covering each way a source can pass or fail the filter.
    seed = {"approved_authority_classes": ["approved_demo"], "sources": {
        "demo_current": {"status": "current", "authority_class": "approved_demo", "scope": "exact_pair"},  # passes
        "demo_single_drug": {"status": "current", "authority_class": "approved_demo", "scope": "single_drug"},  # passes
        "demo_stale": {"status": "archived", "authority_class": "approved_demo", "scope": "exact_pair"},  # not current
        "demo_unknown_status": {"status": "unknown", "authority_class": "approved_demo", "scope": "exact_pair"},
        "demo_no_authority": {"status": "current", "scope": "exact_pair"},  # missing authority is never filled in
        "demo_unapproved": {"status": "current", "authority_class": "vocabulary", "scope": "exact_pair"},
        "demo_vocabulary": {"status": "current", "authority_class": "approved_demo", "scope": "none"},  # scope none
    }}
    pair = ("Warfarin", "Ibuprofen")  # the pair every case is judged against

    def rec(key, a="Warfarin", b="Ibuprofen"):
        # Build a minimal record from a source key and two drug names.
        return {"id": f"{key}:1", "source_key": key, "drug_a": a, "drug_b": b}

    # (record, pair, expected_eligible)
    cases = [
        (rec("demo_current"), pair, True),
        (rec("demo_current", "Ibuprofen", "Warfarin"), pair, True),          # reversed order is the same pair
        (rec("demo_single_drug"), pair, True),                              # names the exact pair
        (rec("demo_single_drug", "Warfarin", "Aspirin"), pair, True),       # context: shares one drug
        (rec("demo_stale"), pair, False),  # archived source
        (rec("demo_unknown_status"), pair, False),  # status unknown
        (rec("demo_no_authority"), pair, False),  # no authority class
        (rec("demo_unapproved"), pair, False),  # authority class not approved
        (rec("demo_vocabulary"), pair, False),  # scope none is never interaction evidence
        (rec("demo_not_registered"), pair, False),  # source not in the registry
        (rec("demo_current", "Zinc", "Iron Sulfate"), pair, False),         # a different pair altogether
    ]
    return registry_module.build_registry(seed), cases


def evaluate_filter(checker, registry, suite=None):
    # Docstring: explains the two checks (real dataset and labelled suite).
    """Wrong-exclusion rate on the real dataset and on a labelled suite of made-up sources.

    On the dataset every record is eligible under the team's designation, so any exclusion is a wrong
    exclusion. The suite (from labelled_filter_suite) reports wrong exclusions (an eligible record dropped)
    and wrong inclusions (an ineligible one kept).
    """
    flt = EligibilityFilter(registry)  # the real filter with the real registry
    records = checker.graph.records  # records that survived filtering into the graph
    # The graph already holds only eligible records, so evaluate the raw rows through the filter again.
    from ddi_backend import ingest  # imported here, only needed for this check
    raw = ingest.load_interactions(checker.conn)  # every row straight from the database
    excluded = [r["id"] for r in raw if not flt.evaluate(r, (r["drug_a"], r["drug_b"])).eligible]  # dropped ids
    out = {"dataset_records": len(raw), "eligible_records_wrongly_excluded": len(excluded),
           "wrong_exclusion_rate": _ratio(len(excluded), len(raw)), "excluded_ids": excluded,
           "graph_records": len(records)}
    if suite:  # also run the labelled made-up cases
        suite_registry, labelled_suite = suite
        suite_filter = EligibilityFilter(suite_registry)  # filter built on the made-up registry
        wrong_ex = wrong_in = 0  # wrongly excluded, wrongly included
        eligible_total = sum(1 for *_, expected in labelled_suite if expected)  # cases that should pass
        for record, names, expected in labelled_suite:
            actual = suite_filter.evaluate(record, names).eligible  # what the filter decided
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
    # Every source record id across the answer's pair results.
    return [rid for r in answer.get("results", []) for rid in r.get("source", {}).get("record_ids", [])]


# Each case is (label, question, check function that returns True when the answer is right).
HANDOUT_CASES = [
    ("RAG 1 direct question", "Do warfarin and ibuprofen interact, and how severe is it?",
     lambda a: _ids(a) == ["ddi:1"] and a["results"][0]["severity"] == "Major"),  # one record, Major
    ("RAG 2 paraphrase", "Is it dangerous to take ibuprofen while on warfarin?", lambda a: _ids(a) == ["ddi:1"]),
    ("RAG 3 severity, effect and mechanism", "What are the severity, clinical effect and mechanism of warfarin with fluconazole?",
     lambda a: _ids(a) == ["ddi:12"] and all(a["results"][0].get(k) for k in ("severity", "effect", "mechanism"))),
    # The "not" question must return the avoid list, not the similarity-ranked acetaminophen.
    ("RAG 4 distractor: pain reliever and warfarin", "Which pain reliever should I not take with warfarin?",
     lambda a: a.get("intent") == "avoid" and a["avoid"]["avoid"][0]["severity"] == "Major"
     and a["avoid"]["avoid"][0]["drug"] != "Acetaminophen"),
    # Drugs missing from the data are unknown, and the note says that is not the same as safe.
    ("RAG 5 absent pair (drugs not in the data)", "Do cabergoline and rucaparib interact?",
     lambda a: a["results"][0]["status"] == "unknown_drug" and "does not mean the combination is safe" in " ".join(a["results"][0]["notes"])),
    ("RAG 5 variant: known drugs, no documented row", "Do warfarin and docusate interact?",
     lambda a: a["results"][0]["status"] == "no_documented_interaction"),
    # A repeated pair returns both rows, headlines Major and flags the conflict.
    ("Meeting 7 reworded query, same record", "Does taking clopidogrel with omeprazole reduce the antiplatelet effect?",
     lambda a: len(_ids(a)) == 2 and a["results"][0]["severity"] == "Major" and a["results"][0]["severity_conflict"]),
    # An ambiguous abbreviation must ask a question and return no results.
    ("Meeting 8 ambiguity: MS", "What does MS interact with?", lambda a: a["status"] == "needs_clarification" and "results" not in a),
    ("Meeting 8 negation", "What should I not take with warfarin?", lambda a: a.get("intent") == "avoid"),
    # A population word triggers the "not covered" rule while the documented pair is still returned.
    ("Meeting 8 context: children", "Is warfarin with ibuprofen safe for children?",
     lambda a: "population_context_not_covered" in a["rules_fired"] and a["results"][0]["status"] == "documented_interaction"),
]


def run_handout_cases(question_answerer):
    # Docstring: states that the expected behaviour was fixed before any output was read.
    """Run the fixed cases. Expected behaviour is written in the table above, before any output is read."""
    rows = []
    for label, question, check in HANDOUT_CASES:
        answer = question_answerer.ask(question)  # run the real question handler
        try:
            passed = bool(check(answer))  # did the answer meet the written expectation?
        except (KeyError, IndexError, TypeError):  # a missing field counts as a failure
            passed = False
        rows.append({"case": label, "question": question, "passed": passed})
    return rows


# --------------------------------------------------------------------------
# Simulated providers: only for checking the harness itself, never for reporting model quality
# --------------------------------------------------------------------------
class SimulatedProvider:
    # Docstring: warns that results from this class say nothing about any real model.
    """A scripted stand-in for a model. SIMULATED: its numbers say nothing about any real model."""
    name = "simulated"  # provider name shown in outputs

    def __init__(self, truth, mode="oracle", seed=0):
        self.truth, self.mode, self.rng = truth, mode, random.Random(seed)  # answer key, behaviour, seeded randomness

    def complete(self, messages):
        payload = json.loads(messages[1]["content"])  # the user message holds the two drug names
        entry = self.truth.get(pair_key(payload["drug_a"], payload["drug_b"]))  # the true entry, or None
        detailed = "one sentence each" in messages[0]["content"]  # was the detailed instruction used?
        if self.mode == "oracle":  # always correct
            if entry:
                return json.dumps({"interaction": True, "severity": entry["severity"],
                                   "mechanism": entry["mechanism"] if detailed else None,
                                   "effect": entry["effect"] if detailed else None})
            return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})
        # "noisy": misses some Major rows, invents some interactions, mislabels some severities
        roll = self.rng.random()  # one random number decides what kind of mistake, if any
        if entry and roll < 0.2:  # miss a real interaction
            return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})
        if entry and roll < 0.4:  # right interaction, wrong severity, made-up mechanism
            wrong = [s for s in SEVERITIES if s != entry["severity"]][0]
            return json.dumps({"interaction": True, "severity": wrong, "mechanism": "unrelated mechanism claim",
                               "effect": entry["effect"] if detailed else None})
        if not entry and roll < 0.3:  # invent an interaction that is not in the data
            return json.dumps({"interaction": True, "severity": "Moderate",
                               "mechanism": "invented pathway" if detailed else None, "effect": None})
        if entry:  # otherwise correct severity only
            return json.dumps({"interaction": True, "severity": entry["severity"], "mechanism": None, "effect": None})
        return json.dumps({"interaction": False, "severity": None, "mechanism": None, "effect": None})  # correctly none


# --------------------------------------------------------------------------
# The full run and the report
# --------------------------------------------------------------------------
def run_evaluation(checker, registry, provider=None, n_negative=60, seed=0, limit=None, label="no language model"):
    truth = ground_truth(checker)  # the answer key
    documented, negatives = build_pair_set(checker, truth, n_negative, seed)  # documented pairs and no-row pairs
    pairs = documented + negatives  # every system runs on this same list
    if limit:  # optional seeded sample, to save API calls
        rng = random.Random(seed)
        pairs = rng.sample(pairs, min(limit, len(pairs)))
    names = sorted({r["drug_a"] for r in checker.graph.records} | {r["drug_b"] for r in checker.graph.records})  # for fake context
    results = {
        "pairs": {"documented": sum(1 for p in pairs if pair_key(*p) in truth),  # how many pairs have a row
                  "absent": sum(1 for p in pairs if pair_key(*p) not in truth), "seed": seed},
        "backend": run_condition(lambda pair, _ctx: backend_assessment(checker, pair), pairs, truth),  # deterministic run
        "filter": evaluate_filter(checker, registry, labelled_filter_suite()),  # filter wrong-exclusion check
        "handout_cases": run_handout_cases(QuestionAnswerer(checker)),  # fixed handout and Meeting 8 questions
        "model_label": label,  # how to describe the model in the report
        "conditions": {},  # filled below when a provider exists
    }
    if provider is not None:  # run the four model conditions, one variable at a time
        for context_level in CONTEXT_LEVELS:
            for detail_level in DETAIL_LEVELS:
                # Default arguments freeze the current loop values for this function.
                def system(pair, context, cl=context_level, dl=detail_level):
                    return model_assessment(provider, pair, cl, dl, context)
                results["conditions"][f"{context_level} / {detail_level}"] = run_condition(
                    system, pairs, truth,
                    # Only the full-context condition receives synthetic doses and other medications.
                    context_for=(lambda pair: synthetic_context(pair, names, seed)) if context_level == "full_context" else None)
    return results


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"  # a ratio as a percentage, or n/a


def format_report(results):
    # Header lines of the markdown report, including the prototype disclaimer and the pair counts.
    lines = ["# Evaluation report", "",
             "Class prototype, not medical advice. The dataset is the ground truth. Doses and other "
             "medications in the full-context condition are synthetic.", "",
             f"Pairs: {results['pairs']['documented']} documented, {results['pairs']['absent']} with no documented "
             f"row (seed {results['pairs']['seed']}). Every condition ran the same pairs.", ""]

    def table(name, score):
        # Build the markdown section for one condition.
        c, m = score["counts"], score["major"]  # counts and Major recall numbers
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
               "| | " + " | ".join(SEVERITY_COLUMNS) + " |", "|---|" + "---|" * len(SEVERITY_COLUMNS)]  # matrix header
        for row, cells in score["confusion_matrix"].items():  # one table row per dataset severity
            out.append(f"| {row} | " + " | ".join(str(cells[col]) for col in SEVERITY_COLUMNS) + " |")
        return out + [""]  # blank line after the table

    lines += table("Backend (deterministic)", results["backend"])  # the backend section always appears
    for name, score in results["conditions"].items():  # one section per model condition
        lines += table(f"Language model, {name} ({results['model_label']})", score)
    if not results["conditions"]:  # no model was run: say so rather than leaving a gap
        lines += ["## Language model", "", "Not run: no language model was configured. Set OPENAI_API_KEY and run "
                  "`python eval/run_eval.py --llm openai`.", ""]
    lines += ["## Handout and Meeting 8 questions", "", "| Case | Result |", "|---|---|"]  # fixed-cases table header
    lines += [f"| {r['case']} | {'pass' if r['passed'] else 'FAIL'} |" for r in results["handout_cases"]]
    lines.append("")
    f = results["filter"]  # filter results
    lines += ["## Eligibility filter", "",
              f"On the dataset ({f['dataset_records']} records) the filter wrongly excluded "
              f"{f['eligible_records_wrongly_excluded']} eligible records ({_pct(f['wrong_exclusion_rate'])}).", ""]
    if "labelled_suite" in f:  # also report the made-up suite
        s = f["labelled_suite"]
        lines.append(f"On the labelled suite of made-up sources ({s['cases']} cases): {s['eligible_wrongly_excluded']} "
                     f"eligible records wrongly excluded, {s['ineligible_wrongly_included']} ineligible records wrongly included.")
    return "\n".join(lines) + "\n"  # the finished markdown text

# ---------------------------------------------------------------------------
# FILE SUMMARY: The evaluation harness. It builds an answer key from the dataset, makes one fixed set of pairs
# (every documented pair plus a seeded sample with no row), runs the deterministic backend and, optionally, a language
# model in four conditions (names-only or full-context, brief or detailed), then scores missed interactions, false
# interactions, wrong severity, unsupported claims, Major recall and a confusion matrix. It also tests the eligibility
# filter, runs the handout questions, offers a scripted SIMULATED provider to test the harness itself, and writes the
# markdown report.
# ASSIGNMENT TOOL: Evaluation harness (it also exercises the knowledge graph + rules core, the eligibility filter and
# the language model modes).
# ---------------------------------------------------------------------------
