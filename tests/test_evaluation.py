# Module docstring: tests the evaluation harness (scoring maths, conditions, the filter check and the report).
"""Tests for the evaluation harness: scoring maths, conditions, the filter check and the report."""

import contextlib  # captures stdout / stderr printed by the eval script
import io  # in-memory text buffers for the captured output
import json  # reads the results file the script writes and builds model replies
import tempfile  # throwaway output folders
import unittest  # standard-library test framework
from pathlib import Path  # path objects for the script and output files

from ddi_backend import evaluation as ev  # the harness under test, shortened to ev
from ddi_backend import registry  # source registry, needed for the filter checks
from ddi_backend.checker import Checker  # the deterministic backend being evaluated
from tests import support  # shared helpers: open the real databases

# Small made-up ground truth: three documented pairs with severity, mechanism, effect and record ID.
TRUTH = {
    # Pair A+B is Major, with a bleeding mechanism and effect.
    ev.pair_key("A", "B"): {"pair": ("A", "B"), "severity": "Major", "mechanism": "Increased bleeding risk",
                            "effect": "Serious GI bleeding", "record_ids": ["ddi:1"]},
    # Pair C+D is Moderate.
    ev.pair_key("C", "D"): {"pair": ("C", "D"), "severity": "Moderate", "mechanism": "Enzyme inhibition",
                            "effect": "Raised levels", "record_ids": ["ddi:2"]},
    # Pair E+F is Minor.
    ev.pair_key("E", "F"): {"pair": ("E", "F"), "severity": "Minor", "mechanism": "Absorption", "effect": "Slower absorption",
                            "record_ids": ["ddi:3"]},
}


def yes(severity, mechanism=None, effect=None):  # helper: a model assessment that says "there is an interaction"
    return {"interaction": True, "severity": severity, "mechanism": mechanism, "effect": effect}


NO = {"interaction": False, "severity": None, "mechanism": None, "effect": None}  # a model assessment saying "no interaction"


class TestScoring(unittest.TestCase):  # tests the counting and recall maths
    def score(self, rows):  # helper: score a list of (pair, assessment) rows against the made-up truth
        return ev.score_condition(rows, TRUTH)

    def test_missed_false_wrong_and_correct_are_counted_separately(self):  # the four outcome types
        score = self.score([
            (("A", "B"), yes("Major")),          # correct
            (("C", "D"), yes("Major")),          # wrong severity
            (("E", "F"), NO),                    # missed known interaction
            (("X", "Y"), yes("Moderate")),       # false interaction
            (("P", "Q"), NO),                    # correct absence
        ])
        c = score["counts"]  # the counters
        # One of each outcome type was produced above.
        self.assertEqual((c["correct_severity"], c["wrong_severity"], c["missed_known_interactions"],
                          c["false_interactions"]), (1, 1, 1, 1))
        self.assertEqual((c["documented"], c["absent"], c["scored"]), (3, 2, 5))  # 3 documented pairs, 2 absent, 5 scored

    def test_major_recall_is_reported_on_its_own(self):  # Major recall is tracked separately
        truth = dict(TRUTH)  # copy the truth so the shared one is not changed
        truth[ev.pair_key("G", "H")] = dict(TRUTH[ev.pair_key("A", "B")], pair=("G", "H"))  # add a second Major pair
        rows = [(("A", "B"), yes("Major")), (("G", "H"), yes("Moderate")), (("C", "D"), NO)]  # one exact, one soft, one miss
        score = ev.score_condition(rows, truth)  # score them
        self.assertEqual(score["major"]["total"], 2)  # two Major pairs in the truth
        self.assertEqual(score["major"]["recall_exact_severity"], 0.5)  # only one got exactly Major
        self.assertEqual(score["major"]["recall_any_interaction"], 1.0)  # both were at least recognised as interactions

    def test_a_missed_major_is_a_miss_even_when_the_model_says_something(self):  # saying "no" to a Major is a failure
        score = self.score([(("A", "B"), NO)])  # model denies the Major pair
        self.assertEqual(score["major"]["recall_exact_severity"], 0.0)  # zero Major recall
        self.assertEqual(score["failure_codes"], {"R1": 1})  # counted as one R1 failure

    def test_confusion_matrix(self):  # rows are truth, columns are the answer
        # Mixed answers: Major right, Major called Minor, Moderate denied, an absent pair called Major.
        score = self.score([(("A", "B"), yes("Major")), (("A", "B"), yes("Minor")), (("C", "D"), NO),
                            (("X", "Y"), yes("Major"))])
        m = score["confusion_matrix"]  # the matrix
        self.assertEqual((m["Major"]["Major"], m["Major"]["Minor"], m["Moderate"]["None"], m["None"]["Major"]), (1, 1, 1, 1))  # one in each cell
        self.assertEqual(set(m), {"Major", "Moderate", "Minor", "None"})  # the four truth rows

    def test_unparseable_and_errors_are_not_silently_scored_as_answers(self):  # bad replies are counted apart
        score = self.score([(("A", "B"), {"unparseable": True, "raw": "?"}), (("C", "D"), {"error": "HTTP 500"})])  # one of each bad kind
        self.assertEqual(score["counts"]["unparseable"], 1)  # the unreadable reply is counted
        self.assertEqual(score["counts"]["errors"], 1)  # the error is counted
        self.assertNotIn("scored", score["counts"])  # neither counts as a scored answer
        self.assertEqual(score["confusion_matrix"]["Major"]["Unparseable"], 1)  # the matrix shows the unreadable reply

    def test_no_division_by_zero_without_major_rows(self):  # recall with nothing to recall is None
        self.assertIsNone(self.score([(("C", "D"), yes("Moderate"))])["major"]["recall_exact_severity"])  # no Major rows scored


class TestClaimLabels(unittest.TestCase):  # tests labelling each claim as supported, unsupported or contradicted
    def labels(self, assessment, key=("A", "B")):  # helper: map each claimed field to (label, failure code)
        return {field: (label, code) for field, label, code in ev.label_claims(assessment, TRUTH.get(ev.pair_key(*key)))}

    def test_supported_unsupported_and_contradicted(self):  # the three labels
        got = self.labels(yes("Major", "Increased bleeding risk from platelets", "Serious GI bleeding"))  # a faithful reply
        self.assertEqual(got["existence"], (ev.SUPPORTED, None))  # existence matches
        self.assertEqual(got["severity"], (ev.SUPPORTED, None))  # severity matches
        self.assertEqual(got["mechanism"], (ev.SUPPORTED, None))  # mechanism contains the recorded wording
        self.assertEqual(got["effect"], (ev.SUPPORTED, None))  # effect matches
        got = self.labels(yes("Minor", "Liver enzyme induction"))  # wrong severity and an invented mechanism
        self.assertEqual(got["severity"], (ev.CONTRADICTED, "G1"))  # wrong severity contradicts the record
        self.assertEqual(got["mechanism"], (ev.UNSUPPORTED, "G1"))  # invented mechanism is not in the record
        self.assertEqual(self.labels(NO)["existence"], (ev.CONTRADICTED, "R1"))  # denying a documented pair is an R1 contradiction

    def test_claims_about_an_undocumented_pair_are_plausible_but_unsupported_not_contradicted(self):  # no record to contradict
        got = self.labels(yes("Moderate", "Some pathway"), key=("X", "Y"))  # claims about a pair with no record
        self.assertTrue(all(label == ev.UNSUPPORTED for label, _ in got.values()))  # all unsupported, none contradicted
        self.assertEqual(self.labels(NO, key=("X", "Y")), {})  # saying "none documented" claims nothing


class TestParseAssessment(unittest.TestCase):  # tests reading the model's JSON reply
    def test_valid_and_invalid_replies(self):  # good and bad inputs
        self.assertEqual(ev.parse_assessment(json.dumps(yes("major")))["severity"], "Major")  # severity is normalised to "Major"
        self.assertIsNone(ev.parse_assessment("They interact."))  # not JSON, so None
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": "yes", "severity": "Major"})))  # "yes" is not a boolean
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": True, "severity": "Catastrophic"})))  # unknown severity
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": False, "severity": "Major"}))["severity"])  # no interaction clears severity


class TestConditions(unittest.TestCase):  # tests running the evaluation conditions on the real data
    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.conn = support.open_real()  # open the real database
        cls.checker = Checker(cls.conn)  # deterministic checker over it
        cls.reg = registry.load_registry()  # the shipped source registry
        cls.truth = ev.ground_truth(cls.checker)  # ground truth built from the dataset

    @classmethod
    def tearDownClass(cls):  # runs once after the class
        cls.conn.close()  # close the shared connection

    def test_ground_truth_is_the_dataset(self):  # the truth matches what we know of the data
        self.assertEqual(len(self.truth), 172)  # 172 distinct documented pairs
        self.assertEqual(sum(1 for t in self.truth.values() if t["severity"] == "Major"), 63)  # 63 of them are Major
        self.assertEqual(self.truth[ev.pair_key("Warfarin", "Ibuprofen")]["record_ids"], ["ddi:1"])  # known record ID
        self.assertEqual(self.truth[ev.pair_key("Omeprazole", "Clopidogrel")]["severity"], "Major")  # highest of the two rows

    def test_pair_set_is_seeded_and_negatives_are_really_absent(self):  # the pairs to test are reproducible
        documented, negatives = ev.build_pair_set(self.checker, self.truth, 25, seed=4)  # all documented pairs plus 25 sampled absent ones
        self.assertEqual((len(documented), len(negatives)), (172, 25))  # expected counts
        self.assertTrue(all(ev.pair_key(*p) not in self.truth for p in negatives))  # no negative is actually documented
        self.assertEqual(ev.build_pair_set(self.checker, self.truth, 25, seed=4)[1], negatives)  # same seed, same sample
        self.assertNotEqual(ev.build_pair_set(self.checker, self.truth, 25, seed=5)[1], negatives)  # different seed, different sample

    def test_the_backend_has_perfect_recall_on_the_dataset(self):  # the deterministic backend misses nothing
        results = ev.run_evaluation(self.checker, self.reg, None, n_negative=40, seed=1)  # no model, backend only
        backend = results["backend"]  # the backend's score block
        self.assertEqual(backend["counts"].get("missed_known_interactions", 0), 0)  # no misses
        self.assertEqual(backend["counts"].get("false_interactions", 0), 0)  # no false interactions
        self.assertEqual(backend["counts"].get("wrong_severity", 0), 0)  # no wrong severities
        self.assertEqual(backend["major"]["recall_exact_severity"], 1.0)  # perfect Major recall
        self.assertEqual(backend["unsupported_claims"], 0)  # no unsupported claims
        self.assertEqual(results["conditions"], {})  # no model conditions were run

    def test_an_oracle_stand_in_scores_perfectly_in_all_four_conditions(self):  # the harness runs all four conditions
        provider = ev.SimulatedProvider(self.truth, "oracle")  # scripted stand-in that always answers correctly
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=20, seed=2, limit=60)  # run with a pair limit
        # The four conditions are the combinations of context level and detail level.
        self.assertEqual(set(results["conditions"]), {"names_only / brief", "names_only / detailed",
                                                      "full_context / brief", "full_context / detailed"})
        for name, score in results["conditions"].items():  # check each condition
            with self.subTest(condition=name):  # report each separately
                self.assertEqual(score["counts"].get("missed_known_interactions", 0), 0)  # nothing missed
                self.assertEqual(score["counts"].get("false_interactions", 0), 0)  # nothing false
                self.assertEqual(score["unsupported_claims"], 0)  # nothing unsupported

    def test_a_noisy_stand_in_is_caught_by_the_harness(self):  # the harness can detect bad model behaviour
        provider = ev.SimulatedProvider(self.truth, "noisy", seed=1)  # stand-in that makes deliberate mistakes
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=40, seed=1)  # run it
        score = results["conditions"]["names_only / detailed"]  # one of the conditions
        self.assertGreater(score["counts"]["missed_known_interactions"], 0)  # misses were detected
        self.assertGreater(score["counts"]["false_interactions"], 0)  # false interactions were detected
        self.assertGreater(score["counts"]["wrong_severity"], 0)  # wrong severities were detected
        self.assertGreater(score["unsupported_claims"], 0)  # unsupported claims were detected
        self.assertLess(score["major"]["recall_exact_severity"], 1.0)  # Major recall dropped below perfect
        self.assertIn("SIMULATED", ev.format_report(dict(results, model_label="SIMULATED noisy stand-in")))  # the report labels it simulated

    def test_every_condition_runs_the_same_pairs(self):  # a fair comparison uses identical pairs
        provider = ev.SimulatedProvider(self.truth, "oracle")  # perfect stand-in
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=15, seed=3, limit=40)  # run with a limit
        scored = {name: s["counts"]["scored"] for name, s in results["conditions"].items()}  # scored count per condition
        self.assertEqual(len(set(scored.values())), 1)  # all conditions scored the same number
        self.assertEqual(scored["names_only / brief"], results["backend"]["counts"]["scored"])  # and the backend did too

    def test_the_conditions_differ_in_exactly_one_variable(self):  # one change at a time
        pair = ("Warfarin", "Ibuprofen")  # the pair to build prompts for
        context = ev.synthetic_context(pair, ["Warfarin", "Ibuprofen", "Zinc", "Insulin", "Aspirin"], seed=0)  # made-up doses and other medications
        self.assertTrue(context["synthetic"])  # the context is labelled synthetic
        base = ev.build_assessment_messages(pair, "names_only", "brief", context)  # baseline prompt
        richer = ev.build_assessment_messages(pair, "full_context", "brief", context)  # only the context level changes
        longer = ev.build_assessment_messages(pair, "names_only", "detailed", context)  # only the detail level changes
        self.assertEqual(base[0], richer[0])                      # same instructions, only the context differs
        self.assertNotIn("doses_as_entered", base[1]["content"])  # names-only has no doses
        self.assertIn("doses_as_entered", richer[1]["content"])  # full context includes the doses
        self.assertEqual(base[1], longer[1])                      # same context, only the detail instruction differs
        self.assertNotEqual(base[0], longer[0])  # the system instructions differ for detail level
        self.assertNotIn("records", base[1]["content"])           # the evaluated model sees no records

    def test_handout_and_meeting_8_cases_pass(self):  # the assignment's example questions all pass
        conn = support.open_real_full()  # real database including HPO tables
        self.addCleanup(conn.close)  # close afterwards
        from ddi_backend.retrieval import QuestionAnswerer  # imported here, only needed by this test
        rows = ev.run_handout_cases(QuestionAnswerer(Checker(conn), hpo_conn=conn))  # run the handout and Meeting 8 questions
        self.assertEqual([r["case"] for r in rows if not r["passed"]], [])  # no case failed
        self.assertEqual(len(rows), 10)  # ten cases in total


class TestFilterEvaluation(unittest.TestCase):  # tests the eligibility-filter evaluation
    def test_no_eligible_record_is_wrongly_excluded(self):  # the filter's wrong-exclusion rate is zero
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        checker = Checker(conn)  # checker over it
        out = ev.evaluate_filter(checker, registry.load_registry(), ev.labelled_filter_suite())  # run the real-data check and the labelled cases
        self.assertEqual(out["eligible_records_wrongly_excluded"], 0)  # no eligible record dropped
        self.assertEqual(out["wrong_exclusion_rate"], 0.0)  # so the rate is zero
        suite = out["labelled_suite"]  # the hand-labelled cases' results
        self.assertEqual((suite["eligible_wrongly_excluded"], suite["ineligible_wrongly_included"]), (0, 0))  # both kinds of error are zero
        self.assertGreaterEqual(suite["cases"], 10)  # at least ten labelled cases

    def test_the_check_is_sensitive_to_a_bad_registry(self):  # the check can actually detect a problem
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        # A registry where the only source is archived, so every real record becomes ineligible.
        stale = registry.build_registry({"approved_authority_classes": ["course_designated_dataset"], "sources": {
            "kaggle_ddi": {"status": "archived", "authority_class": "course_designated_dataset", "scope": "exact_pair"}}})
        out = ev.evaluate_filter(Checker(conn), stale)  # run the check with the bad registry
        self.assertEqual(out["eligible_records_wrongly_excluded"], 180)  # all 180 records are reported as wrongly excluded


class TestReportAndScript(unittest.TestCase):  # tests the report text and the eval/run_eval.py script
    def test_report_says_when_no_model_was_run(self):  # the report is honest about missing model results
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        checker = Checker(conn)  # checker over it
        report = ev.format_report(ev.run_evaluation(checker, registry.load_registry(), None, 10, 0))  # backend-only report
        # Headings and phrases that must be present in the report.
        for heading in ("# Evaluation report", "## Backend (deterministic)", "Severity confusion matrix",
                        "Major recall, exact severity", "## Eligibility filter", "Not run: no language model"):
            self.assertIn(heading, report)  # each one appears

    def test_the_script_writes_the_report(self):  # running the script produces its output files
        import importlib.util  # imported here, only needed by this test
        # Load eval/run_eval.py by path since eval is not a package.
        spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py")
        module = importlib.util.module_from_spec(spec)  # empty module object
        spec.loader.exec_module(module)  # run the script's top level
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):  # output folder, and hide the printing
            code = module.main(["--db", str(support.real_db_path()), "--negatives", "10", "--out", tmp])  # run with 10 negatives
            self.assertEqual(code, 0)  # success exit code
            self.assertTrue((Path(tmp) / "report.md").is_file())  # the report was written
            self.assertTrue(json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))["backend"])  # results.json has backend data

    def test_claude_mode_needs_a_key(self):  # --llm claude without a key must fail clearly
        import importlib.util  # imported here, only needed by this test
        # Load eval/run_eval.py by path, as above.
        spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py")
        module = importlib.util.module_from_spec(spec)  # empty module object
        spec.loader.exec_module(module)  # run the script's top level
        from unittest import mock  # imported here, only needed by this test
        with mock.patch.dict("os.environ", {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as err:  # no key in the environment, capture errors
            code = module.main(["--db", str(support.real_db_path()), "--llm", "claude"])  # request Claude mode
        self.assertEqual(code, 1)  # failure exit code
        self.assertIn("ANTHROPIC_API_KEY", err.getvalue())  # the message names the missing variable


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests ddi_backend/evaluation.py and eval/run_eval.py. It checks the scoring maths (missed, false and
# wrong-severity counts, Major recall on its own, the confusion matrix, unparseable replies), the supported /
# unsupported / contradicted claim labels, ground truth and seeded pair sets, that the backend scores perfectly on the
# dataset, that a scripted stand-in model is scored and a noisy one is caught, that the four language-model conditions
# change one variable at a time, the eligibility-filter check, the handout questions, and the report and script output.
# ASSIGNMENT TOOL: The evaluation harness, which measures the knowledge graph and rule-based backend, the eligibility
# filter and the language model modes.
# ---------------------------------------------------------------------------
