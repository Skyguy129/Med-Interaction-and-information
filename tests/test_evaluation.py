"""Tests for the evaluation harness: scoring maths, conditions, the filter check and the report."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ddi_backend import evaluation as ev
from ddi_backend import registry
from ddi_backend.checker import Checker
from tests import support

TRUTH = {
    ev.pair_key("A", "B"): {"pair": ("A", "B"), "severity": "Major", "mechanism": "Increased bleeding risk",
                            "effect": "Serious GI bleeding", "record_ids": ["ddi:1"]},
    ev.pair_key("C", "D"): {"pair": ("C", "D"), "severity": "Moderate", "mechanism": "Enzyme inhibition",
                            "effect": "Raised levels", "record_ids": ["ddi:2"]},
    ev.pair_key("E", "F"): {"pair": ("E", "F"), "severity": "Minor", "mechanism": "Absorption", "effect": "Slower absorption",
                            "record_ids": ["ddi:3"]},
}


def yes(severity, mechanism=None, effect=None):
    return {"interaction": True, "severity": severity, "mechanism": mechanism, "effect": effect}


NO = {"interaction": False, "severity": None, "mechanism": None, "effect": None}


class TestScoring(unittest.TestCase):
    def score(self, rows):
        return ev.score_condition(rows, TRUTH)

    def test_missed_false_wrong_and_correct_are_counted_separately(self):
        score = self.score([
            (("A", "B"), yes("Major")),          # correct
            (("C", "D"), yes("Major")),          # wrong severity
            (("E", "F"), NO),                    # missed known interaction
            (("X", "Y"), yes("Moderate")),       # false interaction
            (("P", "Q"), NO),                    # correct absence
        ])
        c = score["counts"]
        self.assertEqual((c["correct_severity"], c["wrong_severity"], c["missed_known_interactions"],
                          c["false_interactions"]), (1, 1, 1, 1))
        self.assertEqual((c["documented"], c["absent"], c["scored"]), (3, 2, 5))

    def test_major_recall_is_reported_on_its_own(self):
        truth = dict(TRUTH)
        truth[ev.pair_key("G", "H")] = dict(TRUTH[ev.pair_key("A", "B")], pair=("G", "H"))
        rows = [(("A", "B"), yes("Major")), (("G", "H"), yes("Moderate")), (("C", "D"), NO)]
        score = ev.score_condition(rows, truth)
        self.assertEqual(score["major"]["total"], 2)
        self.assertEqual(score["major"]["recall_exact_severity"], 0.5)
        self.assertEqual(score["major"]["recall_any_interaction"], 1.0)

    def test_a_missed_major_is_a_miss_even_when_the_model_says_something(self):
        score = self.score([(("A", "B"), NO)])
        self.assertEqual(score["major"]["recall_exact_severity"], 0.0)
        self.assertEqual(score["failure_codes"], {"R1": 1})

    def test_confusion_matrix(self):
        score = self.score([(("A", "B"), yes("Major")), (("A", "B"), yes("Minor")), (("C", "D"), NO),
                            (("X", "Y"), yes("Major"))])
        m = score["confusion_matrix"]
        self.assertEqual((m["Major"]["Major"], m["Major"]["Minor"], m["Moderate"]["None"], m["None"]["Major"]), (1, 1, 1, 1))
        self.assertEqual(set(m), {"Major", "Moderate", "Minor", "None"})

    def test_unparseable_and_errors_are_not_silently_scored_as_answers(self):
        score = self.score([(("A", "B"), {"unparseable": True, "raw": "?"}), (("C", "D"), {"error": "HTTP 500"})])
        self.assertEqual(score["counts"]["unparseable"], 1)
        self.assertEqual(score["counts"]["errors"], 1)
        self.assertNotIn("scored", score["counts"])
        self.assertEqual(score["confusion_matrix"]["Major"]["Unparseable"], 1)

    def test_no_division_by_zero_without_major_rows(self):
        self.assertIsNone(self.score([(("C", "D"), yes("Moderate"))])["major"]["recall_exact_severity"])


class TestClaimLabels(unittest.TestCase):
    def labels(self, assessment, key=("A", "B")):
        return {field: (label, code) for field, label, code in ev.label_claims(assessment, TRUTH.get(ev.pair_key(*key)))}

    def test_supported_unsupported_and_contradicted(self):
        got = self.labels(yes("Major", "Increased bleeding risk from platelets", "Serious GI bleeding"))
        self.assertEqual(got["existence"], (ev.SUPPORTED, None))
        self.assertEqual(got["severity"], (ev.SUPPORTED, None))
        self.assertEqual(got["mechanism"], (ev.SUPPORTED, None))
        self.assertEqual(got["effect"], (ev.SUPPORTED, None))
        got = self.labels(yes("Minor", "Liver enzyme induction"))
        self.assertEqual(got["severity"], (ev.CONTRADICTED, "G1"))
        self.assertEqual(got["mechanism"], (ev.UNSUPPORTED, "G1"))
        self.assertEqual(self.labels(NO)["existence"], (ev.CONTRADICTED, "R1"))

    def test_claims_about_an_undocumented_pair_are_plausible_but_unsupported_not_contradicted(self):
        got = self.labels(yes("Moderate", "Some pathway"), key=("X", "Y"))
        self.assertTrue(all(label == ev.UNSUPPORTED for label, _ in got.values()))
        self.assertEqual(self.labels(NO, key=("X", "Y")), {})  # saying "none documented" claims nothing


class TestParseAssessment(unittest.TestCase):
    def test_valid_and_invalid_replies(self):
        self.assertEqual(ev.parse_assessment(json.dumps(yes("major")))["severity"], "Major")
        self.assertIsNone(ev.parse_assessment("They interact."))
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": "yes", "severity": "Major"})))
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": True, "severity": "Catastrophic"})))
        self.assertIsNone(ev.parse_assessment(json.dumps({"interaction": False, "severity": "Major"}))["severity"])


class TestConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real()
        cls.checker = Checker(cls.conn)
        cls.reg = registry.load_registry()
        cls.truth = ev.ground_truth(cls.checker)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_ground_truth_is_the_dataset(self):
        self.assertEqual(len(self.truth), 172)
        self.assertEqual(sum(1 for t in self.truth.values() if t["severity"] == "Major"), 63)
        self.assertEqual(self.truth[ev.pair_key("Warfarin", "Ibuprofen")]["record_ids"], ["ddi:1"])
        self.assertEqual(self.truth[ev.pair_key("Omeprazole", "Clopidogrel")]["severity"], "Major")  # highest of the two rows

    def test_pair_set_is_seeded_and_negatives_are_really_absent(self):
        documented, negatives = ev.build_pair_set(self.checker, self.truth, 25, seed=4)
        self.assertEqual((len(documented), len(negatives)), (172, 25))
        self.assertTrue(all(ev.pair_key(*p) not in self.truth for p in negatives))
        self.assertEqual(ev.build_pair_set(self.checker, self.truth, 25, seed=4)[1], negatives)
        self.assertNotEqual(ev.build_pair_set(self.checker, self.truth, 25, seed=5)[1], negatives)

    def test_the_backend_has_perfect_recall_on_the_dataset(self):
        results = ev.run_evaluation(self.checker, self.reg, None, n_negative=40, seed=1)
        backend = results["backend"]
        self.assertEqual(backend["counts"].get("missed_known_interactions", 0), 0)
        self.assertEqual(backend["counts"].get("false_interactions", 0), 0)
        self.assertEqual(backend["counts"].get("wrong_severity", 0), 0)
        self.assertEqual(backend["major"]["recall_exact_severity"], 1.0)
        self.assertEqual(backend["unsupported_claims"], 0)
        self.assertEqual(results["conditions"], {})

    def test_an_oracle_stand_in_scores_perfectly_in_all_four_conditions(self):
        provider = ev.SimulatedProvider(self.truth, "oracle")
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=20, seed=2, limit=60)
        self.assertEqual(set(results["conditions"]), {"names_only / brief", "names_only / detailed",
                                                      "full_context / brief", "full_context / detailed"})
        for name, score in results["conditions"].items():
            with self.subTest(condition=name):
                self.assertEqual(score["counts"].get("missed_known_interactions", 0), 0)
                self.assertEqual(score["counts"].get("false_interactions", 0), 0)
                self.assertEqual(score["unsupported_claims"], 0)

    def test_a_noisy_stand_in_is_caught_by_the_harness(self):
        provider = ev.SimulatedProvider(self.truth, "noisy", seed=1)
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=40, seed=1)
        score = results["conditions"]["names_only / detailed"]
        self.assertGreater(score["counts"]["missed_known_interactions"], 0)
        self.assertGreater(score["counts"]["false_interactions"], 0)
        self.assertGreater(score["counts"]["wrong_severity"], 0)
        self.assertGreater(score["unsupported_claims"], 0)
        self.assertLess(score["major"]["recall_exact_severity"], 1.0)
        self.assertIn("SIMULATED", ev.format_report(dict(results, model_label="SIMULATED noisy stand-in")))

    def test_every_condition_runs_the_same_pairs(self):
        provider = ev.SimulatedProvider(self.truth, "oracle")
        results = ev.run_evaluation(self.checker, self.reg, provider, n_negative=15, seed=3, limit=40)
        scored = {name: s["counts"]["scored"] for name, s in results["conditions"].items()}
        self.assertEqual(len(set(scored.values())), 1)
        self.assertEqual(scored["names_only / brief"], results["backend"]["counts"]["scored"])

    def test_the_conditions_differ_in_exactly_one_variable(self):
        pair = ("Warfarin", "Ibuprofen")
        context = ev.synthetic_context(pair, ["Warfarin", "Ibuprofen", "Zinc", "Insulin", "Aspirin"], seed=0)
        self.assertTrue(context["synthetic"])
        base = ev.build_assessment_messages(pair, "names_only", "brief", context)
        richer = ev.build_assessment_messages(pair, "full_context", "brief", context)
        longer = ev.build_assessment_messages(pair, "names_only", "detailed", context)
        self.assertEqual(base[0], richer[0])                      # same instructions, only the context differs
        self.assertNotIn("doses_as_entered", base[1]["content"])
        self.assertIn("doses_as_entered", richer[1]["content"])
        self.assertEqual(base[1], longer[1])                      # same context, only the detail instruction differs
        self.assertNotEqual(base[0], longer[0])
        self.assertNotIn("records", base[1]["content"])           # the evaluated model sees no records

    def test_handout_and_meeting_8_cases_pass(self):
        conn = support.open_real_full()
        self.addCleanup(conn.close)
        from ddi_backend.retrieval import QuestionAnswerer
        rows = ev.run_handout_cases(QuestionAnswerer(Checker(conn), hpo_conn=conn))
        self.assertEqual([r["case"] for r in rows if not r["passed"]], [])
        self.assertEqual(len(rows), 10)


class TestFilterEvaluation(unittest.TestCase):
    def test_no_eligible_record_is_wrongly_excluded(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        checker = Checker(conn)
        out = ev.evaluate_filter(checker, registry.load_registry(), ev.labelled_filter_suite())
        self.assertEqual(out["eligible_records_wrongly_excluded"], 0)
        self.assertEqual(out["wrong_exclusion_rate"], 0.0)
        suite = out["labelled_suite"]
        self.assertEqual((suite["eligible_wrongly_excluded"], suite["ineligible_wrongly_included"]), (0, 0))
        self.assertGreaterEqual(suite["cases"], 10)

    def test_the_check_is_sensitive_to_a_bad_registry(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        stale = registry.build_registry({"approved_authority_classes": ["course_designated_dataset"], "sources": {
            "kaggle_ddi": {"status": "archived", "authority_class": "course_designated_dataset", "scope": "exact_pair"}}})
        out = ev.evaluate_filter(Checker(conn), stale)
        self.assertEqual(out["eligible_records_wrongly_excluded"], 180)


class TestReportAndScript(unittest.TestCase):
    def test_report_says_when_no_model_was_run(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        checker = Checker(conn)
        report = ev.format_report(ev.run_evaluation(checker, registry.load_registry(), None, 10, 0))
        for heading in ("# Evaluation report", "## Backend (deterministic)", "Severity confusion matrix",
                        "Major recall, exact severity", "## Eligibility filter", "Not run: no language model"):
            self.assertIn(heading, report)

    def test_the_script_writes_the_report(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            code = module.main(["--db", str(support.real_db_path()), "--negatives", "10", "--out", tmp])
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "report.md").is_file())
            self.assertTrue(json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))["backend"])

    def test_openai_mode_needs_a_key(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).resolve().parent.parent / "eval" / "run_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from unittest import mock
        with mock.patch.dict("os.environ", {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as err:
            code = module.main(["--db", str(support.real_db_path()), "--llm", "openai"])
        self.assertEqual(code, 1)
        self.assertIn("OPENAI_API_KEY", err.getvalue())


if __name__ == "__main__":
    unittest.main()
