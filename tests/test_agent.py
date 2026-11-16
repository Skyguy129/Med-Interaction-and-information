"""Tests for the guarded agent, its gates, the simulated review queue and the trace."""

import ast
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ddi_backend import agent as agent_module
from ddi_backend import ingest, registry
from ddi_backend.agent import REVIEW_TARGET, REVIEW_TOOL, GuardedAgent, describe_attempt, format_trace
from ddi_backend.checker import Checker
from ddi_backend.state import AgentState
from tests import support
from tests.support import ddi_row

# The notebook's cases 1-5 with their expected statuses.
CASES = [
    {"case": "1", "label": "approved in-scope success", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "request-001", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "staged_in_simulation"},
    {"case": "2", "label": "no approval", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "request-002", "approved": False, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "approval_required"},
    {"case": "3", "label": "out-of-scope target", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": "medication_order", "request_id": "request-003", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "out_of_scope"},
    {"case": "4", "label": "duplicate request_id", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "already-used-004", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "duplicate_request_id"},
    {"case": "5a", "label": "missing evidence", "drug_pair": ("Warfarin", "NoSuchDrug"),
     "target": REVIEW_TARGET, "request_id": "request-005a", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "missing_evidence"},
    {"case": "5b", "label": "disabled high-impact tool", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": "medication_order", "request_id": "request-005b", "approved": True, "tool_enabled": False,
     "requested_tool": "apply_clinical_change", "expected_b": "tool_disabled"},
]


class AgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real()
        cls.checker = Checker(cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def setUp(self):
        self.state = AgentState.for_tests()
        self.addCleanup(self.state.close)
        self.agent = GuardedAgent(self.checker, self.state)


class TestNotebookCases(AgentTestCase):
    def test_each_case_reaches_its_expected_status(self):
        for case in CASES:
            with self.subTest(case=case["case"]):
                state = AgentState.for_tests()  # every run starts from a fresh copy of the same state
                self.addCleanup(state.close)
                initial = state.snapshot()
                outcome = GuardedAgent(self.checker, state).run_guarded(case)
                self.assertEqual(outcome["status"], case["expected_b"])
                self.assertEqual(outcome["state_before"], initial)
                if case["expected_b"] == "staged_in_simulation":
                    self.assertEqual(outcome["executed_tool"], REVIEW_TOOL)
                    self.assertEqual(len(outcome["state_after"]["review_queue"]), 1)
                    self.assertEqual(outcome["state_after"]["review_queue"][0]["evidence_id"], "ddi:1")
                    self.assertEqual(len(outcome["tool_trace"]), 1)
                    self.assertTrue(outcome["action_occurred"])
                else:
                    self.assertIsNone(outcome["executed_tool"])
                    self.assertEqual(outcome["state_after"], outcome["state_before"])  # a denial writes nothing
                    self.assertEqual(outcome["tool_trace"], [])
                    self.assertFalse(outcome["action_occurred"])

    def test_baseline_never_writes(self):
        for case in CASES:
            with self.subTest(case=case["case"]):
                baseline = self.agent.run_baseline(case)
                self.assertEqual(baseline["state_after"], baseline["state_before"])
                self.assertIsNone(baseline["executed_tool"])
        self.assertEqual(self.agent.run_baseline(CASES[0])["evidence_id"], "ddi:1")
        self.assertIsNone(self.agent.run_baseline(CASES[4])["evidence_id"])

    def test_initial_state_has_the_preseeded_request_id(self):
        self.assertEqual(self.state.snapshot(), {"review_queue": [], "seen_request_ids": ["already-used-004"]})

    def test_queue_note_has_exactly_the_agreed_fields(self):
        self.agent.run_guarded(CASES[0])
        self.assertEqual(self.state.review_queue, [{
            "evidence_id": "ddi:1", "drug_pair": ["Warfarin", "Ibuprofen"],
            "request_id": "request-001", "purpose": "clinician review only"}])
        self.assertTrue(self.state.queue_rows()[0]["simulated"])

    def test_trace_entry_records_everything_the_notebook_did(self):
        outcome = self.agent.run_guarded(CASES[0])
        entry = outcome["tool_trace"][0]
        self.assertEqual(set(entry), {"tool", "args", "approved", "enabled", "before", "after", "feedback"})
        self.assertEqual(entry["tool"], REVIEW_TOOL)
        self.assertEqual((entry["approved"], entry["enabled"]), (True, True))
        self.assertEqual(entry["before"]["review_queue"], [])
        self.assertEqual(len(entry["after"]["review_queue"]), 1)
        self.assertEqual(entry["feedback"], {"status": "staged_in_simulation", "request_id": "request-001"})
        self.assertEqual(self.state.tool_trace, [entry])

    def test_the_request_id_cannot_be_used_twice(self):
        self.agent.run_guarded(CASES[0])
        again = self.agent.run_guarded(CASES[0])
        self.assertEqual(again["status"], "duplicate_request_id")
        self.assertEqual(len(self.state.review_queue), 1)

    def test_a_denied_request_does_not_burn_its_id(self):
        denied = dict(CASES[0], approved=False)
        self.assertEqual(self.agent.run_guarded(denied)["status"], "approval_required")
        self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "staged_in_simulation")


class TestGateOrder(AgentTestCase):
    def test_gates_fire_in_the_documented_order(self):
        case = dict(CASES[0], drug_pair=("Warfarin", "NoSuchDrug"), tool_enabled=False,
                    requested_tool="other_tool", target="medication_order",
                    approved=False, request_id="already-used-004")
        fixes = [{"drug_pair": ("Warfarin", "Ibuprofen")}, {"tool_enabled": True},
                 {"requested_tool": REVIEW_TOOL}, {"target": REVIEW_TARGET}, {"approved": True},
                 {"request_id": "fresh-1"}]
        seen = [self.agent.run_guarded(case)["status"]]
        for fix in fixes:
            case.update(fix)
            seen.append(self.agent.run_guarded(case)["status"])
        expected = ["missing_evidence", "tool_disabled", "tool_not_allowed", "out_of_scope",
                    "approval_required", "duplicate_request_id", "staged_in_simulation"]
        self.assertEqual(seen, expected)
        self.assertEqual(agent_module.GATES, tuple(expected[:-1]))

    def test_not_allowed_tool_is_refused(self):
        outcome = self.agent.run_guarded(dict(CASES[0], requested_tool="apply_clinical_change"))
        self.assertEqual(outcome["status"], "tool_not_allowed")
        self.assertEqual(self.state.review_queue, [])


class TestToolBoundary(AgentTestCase):
    """The tool repeats its own checks even if the controller is bypassed."""

    def call(self, **overrides):
        args = {"target": REVIEW_TARGET, "request_id": "direct-1", "drug_pair": ["Warfarin", "Ibuprofen"],
                "evidence_id": "ddi:1"}
        args.update(overrides)
        approved = args.pop("approved", True)
        enabled = args.pop("enabled", True)
        return self.agent.stage_interaction_review(args, approved, enabled)[0]

    def test_each_check_is_repeated(self):
        self.assertEqual(self.call(enabled=False)["reason"], "tool_disabled")
        self.assertEqual(self.call(approved=False)["reason"], "approval_required")
        self.assertEqual(self.call(target="medication_order")["reason"], "out_of_scope")
        self.assertEqual(self.call(request_id="already-used-004")["reason"], "duplicate_request_id")
        self.assertEqual(self.call(evidence_id=None)["reason"], "missing_evidence")
        self.assertEqual(self.state.review_queue, [])

    def test_evidence_must_exist_in_our_data_for_that_pair(self):
        self.assertEqual(self.call(evidence_id="ddi:99999")["reason"], "missing_evidence")
        self.assertEqual(self.call(evidence_id="ddi:12")["reason"], "missing_evidence")  # a different pair's row
        self.assertEqual(self.call(drug_pair=["Warfarin", "NoSuchDrug"])["reason"], "missing_evidence")
        self.assertEqual(self.state.review_queue, [])
        self.assertEqual(self.call()["status"], "staged_in_simulation")


class TestErrorPath(AgentTestCase):
    def test_a_failed_write_reports_the_failure_and_changes_nothing(self):
        before = self.state.snapshot()
        with mock.patch.object(AgentState, "stage_note", side_effect=sqlite3.OperationalError("disk I/O error")):
            outcome = self.agent.run_guarded(CASES[0])
        self.assertEqual(outcome["status"], "error")
        self.assertIn("disk I/O error", outcome["tool_result"]["reason"])
        self.assertFalse(outcome["action_occurred"])
        self.assertEqual(self.state.snapshot(), before)
        self.assertNotIn("request-001", self.state.seen_request_ids)

    def test_retry_needs_the_error_fixed_and_approval_given_again(self):
        with mock.patch.object(AgentState, "stage_note", side_effect=sqlite3.OperationalError("disk I/O error")):
            self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "error")
        # The error is fixed, but the earlier approval does not carry over.
        self.assertEqual(self.agent.run_guarded(dict(CASES[0], approved=False))["status"], "approval_required")
        self.assertEqual(self.state.review_queue, [])
        self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "staged_in_simulation")
        self.assertEqual(len(self.state.review_queue), 1)

    def test_the_note_and_the_used_id_are_written_together_or_not_at_all(self):
        # 'already-used-004' is in the used-ID set, so the second statement fails after the first succeeded.
        note = {"evidence_id": "ddi:1", "drug_pair": ["Warfarin", "Ibuprofen"],
                "request_id": "already-used-004", "purpose": "clinician review only"}
        with self.assertRaises(sqlite3.IntegrityError):
            self.state.stage_note(note)
        self.assertEqual(self.state.review_queue, [])  # the first insert was rolled back


class TestEvidenceRules(unittest.TestCase):
    def test_evidence_ineligible_pair_cannot_be_staged_as_documented(self):
        reg = registry.build_registry({"approved_authority_classes": [],
                                       "sources": {"kaggle_ddi": {"status": "archived", "scope": "exact_pair"}}})
        with tempfile.TemporaryDirectory() as tmp:
            conn = ingest.open_readonly(support.build_db(Path(tmp), [ddi_row("A", "B", "Major")]), ingest.CORE_TABLES)
            try:
                state = AgentState.for_tests()
                self.addCleanup(state.close)
                agent = GuardedAgent(Checker(conn, registry=reg), state)
                outcome = agent.run_guarded(dict(CASES[0], drug_pair=("A", "B")))
                self.assertEqual(outcome["status"], "missing_evidence")
                self.assertEqual(state.review_queue, [])
            finally:
                conn.close()


class TestPersistence(unittest.TestCase):
    def test_queue_and_log_live_in_a_separate_file_and_survive_reopening(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        checker = Checker(conn)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_state.db"
            state = AgentState(path)
            GuardedAgent(checker, state).run_guarded(CASES[0])
            GuardedAgent(checker, state).run_guarded(CASES[1])
            state.close()
            reopened = AgentState(path)
            try:
                self.assertEqual([n["request_id"] for n in reopened.review_queue], ["request-001"])
                self.assertEqual(reopened.seen_request_ids, {"request-001"})  # not pre-seeded outside tests
                self.assertEqual(len(reopened.attempts("request-002")), 1)
                self.assertEqual(reopened.attempts("request-002")[0]["gate"], "approval_required")
            finally:
                reopened.close()

    def test_state_is_never_stored_in_medications_db(self):
        with self.assertRaises(ValueError):
            AgentState(Path("somewhere") / "medications.db")

    def test_the_interaction_database_is_never_written(self):
        path = support.real_db_path()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        conn = support.open_real()
        self.addCleanup(conn.close)
        state = AgentState.for_tests()
        self.addCleanup(state.close)
        agent = GuardedAgent(Checker(conn), state)
        for case in CASES:
            agent.run_guarded(case)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_the_orchestrator_imports_no_language_model_code(self):
        tree = ast.parse(Path(agent_module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
                imported |= {f"{node.module}.{a.name}" for a in node.names}
        self.assertEqual([name for name in imported if "llm" in name or "openai" in name or "urllib" in name], [])


class TestTrace(AgentTestCase):
    def test_trace_reports_the_attempt_gate_result_state_change_and_whether_it_happened(self):
        self.agent.run_guarded(CASES[0])
        self.agent.run_guarded(CASES[1])
        staged = format_trace("request-001", self.state.attempts("request-001"))
        self.assertIn("SIMULATED", staged)
        self.assertIn("stage_interaction_review", staged)
        self.assertIn("gate result    : allowed", staged)
        self.assertIn('"status": "staged_in_simulation"', staged)
        self.assertIn("review_queue 0 -> 1", staged)
        self.assertIn("action happened: yes", staged)

        denied = format_trace("request-002", self.state.attempts("request-002"))
        self.assertIn("gate result    : approval_required", denied)
        self.assertIn("not produced", denied)
        self.assertIn("review_queue 1 -> 1", denied)  # request-001 was already in the queue
        self.assertIn("action happened: no", denied)
        self.assertIn("therefore the action did not occur", denied)

    def test_unknown_request_id(self):
        self.assertIn("No attempts", format_trace("nope", self.state.attempts("nope")))

    def test_describe_attempt_matches_the_notebook_sentence(self):
        self.agent.run_guarded(CASES[2])
        sentence = describe_attempt(self.state.attempts("request-003")[0])
        self.assertTrue(sentence.startswith("The agent attempted stage_interaction_review(args="))
        self.assertIn("the gate returned out_of_scope", sentence)
        self.assertIn("therefore the action did not occur.", sentence)


if __name__ == "__main__":
    unittest.main()
