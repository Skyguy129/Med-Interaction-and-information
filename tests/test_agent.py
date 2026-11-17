# Docstring describing the module: tests for the guarded agent, its gates, the review queue and the trace.
"""Tests for the guarded agent, its gates, the simulated review queue and the trace."""

import ast  # used to inspect the agent's source for forbidden imports
import hashlib  # used to prove the interaction database file never changes
import sqlite3  # used for the database error types we simulate
import tempfile  # temporary folders for state files and fake databases
import unittest  # the test framework
from pathlib import Path  # path objects
from unittest import mock  # lets us force failures with patches

from ddi_backend import agent as agent_module  # the agent module itself (for GATES and its file path)
from ddi_backend import ingest, registry  # read-only database opener and the source registry
from ddi_backend.agent import REVIEW_TARGET, REVIEW_TOOL, GuardedAgent, describe_attempt, format_trace  # agent API
from ddi_backend.checker import Checker  # the deterministic interaction checker the agent relies on
from ddi_backend.state import AgentState  # the separate state file holding the queue and the log
from tests import support  # shared database builders
from tests.support import ddi_row  # builds one fake interaction row

# The notebook's cases 1-5 with their expected statuses.
CASES = [  # the six scenarios (1, 2, 3, 4, 5a, 5b) taken from the agent notebook
    # Case 1: everything is valid, so the request should be staged in the simulated queue.
    {"case": "1", "label": "approved in-scope success", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "request-001", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "staged_in_simulation"},
    # Case 2: the user did not approve, so the approval gate must stop it.
    {"case": "2", "label": "no approval", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "request-002", "approved": False, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "approval_required"},
    # Case 3: the target is not the review queue, so the scope gate must stop it.
    {"case": "3", "label": "out-of-scope target", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": "medication_order", "request_id": "request-003", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "out_of_scope"},
    # Case 4: the request id was already used, so the duplicate gate must stop it.
    {"case": "4", "label": "duplicate request_id", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": REVIEW_TARGET, "request_id": "already-used-004", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "duplicate_request_id"},
    # Case 5a: one drug is not in our data, so there is no evidence to stage.
    {"case": "5a", "label": "missing evidence", "drug_pair": ("Warfarin", "NoSuchDrug"),
     "target": REVIEW_TARGET, "request_id": "request-005a", "approved": True, "tool_enabled": True,
     "requested_tool": REVIEW_TOOL, "expected_b": "missing_evidence"},
    # Case 5b: a high-impact tool that is switched off must be refused as disabled.
    {"case": "5b", "label": "disabled high-impact tool", "drug_pair": ("Warfarin", "Ibuprofen"),
     "target": "medication_order", "request_id": "request-005b", "approved": True, "tool_enabled": False,
     "requested_tool": "apply_clinical_change", "expected_b": "tool_disabled"},
]  # end of the case list


# Base class that gives every agent test a real read-only database, a checker, a fresh state and an agent.
class AgentTestCase(unittest.TestCase):
    @classmethod  # runs once for the whole class
    def setUpClass(cls):
        cls.conn = support.open_real()  # open the real interaction database read-only
        cls.checker = Checker(cls.conn)  # one checker shared by all tests in the class

    @classmethod  # runs once after the class's tests
    def tearDownClass(cls):
        cls.conn.close()  # release the database connection

    def setUp(self):  # runs before every test
        self.state = AgentState.for_tests()  # fresh in-memory state, with the pre-seeded used id
        self.addCleanup(self.state.close)  # close it when the test ends
        self.agent = GuardedAgent(self.checker, self.state)  # a new agent bound to that state


# Tests that each notebook case ends in its expected status and writes only when it should.
class TestNotebookCases(AgentTestCase):
    def test_each_case_reaches_its_expected_status(self):
        for case in CASES:  # go through all six scenarios
            with self.subTest(case=case["case"]):  # report each case separately if one fails
                state = AgentState.for_tests()  # every run starts from a fresh copy of the same state
                self.addCleanup(state.close)  # close the state afterwards
                initial = state.snapshot()  # remember what the state looked like before running
                outcome = GuardedAgent(self.checker, state).run_guarded(case)  # run the guarded flow
                self.assertEqual(outcome["status"], case["expected_b"])  # the gate result is the expected one
                self.assertEqual(outcome["state_before"], initial)  # the outcome records the starting state
                if case["expected_b"] == "staged_in_simulation":  # the success case
                    self.assertEqual(outcome["executed_tool"], REVIEW_TOOL)  # the review tool really ran
                    self.assertEqual(len(outcome["state_after"]["review_queue"]), 1)  # one note was queued
                    self.assertEqual(outcome["state_after"]["review_queue"][0]["evidence_id"], "ddi:1")  # for ddi:1
                    self.assertEqual(len(outcome["tool_trace"]), 1)  # exactly one tool call was traced
                    self.assertTrue(outcome["action_occurred"])  # the action is reported as having happened
                else:  # every denial case
                    self.assertIsNone(outcome["executed_tool"])  # no tool ran
                    self.assertEqual(outcome["state_after"], outcome["state_before"])  # a denial writes nothing
                    self.assertEqual(outcome["tool_trace"], [])  # nothing in the tool trace
                    self.assertFalse(outcome["action_occurred"])  # the action did not happen

    def test_baseline_never_writes(self):
        for case in CASES:  # try every scenario
            with self.subTest(case=case["case"]):  # report cases separately
                baseline = self.agent.run_baseline(case)  # the unguarded comparison run, which must stay read-only
                self.assertEqual(baseline["state_after"], baseline["state_before"])  # state is unchanged
                self.assertIsNone(baseline["executed_tool"])  # no tool executed
        self.assertEqual(self.agent.run_baseline(CASES[0])["evidence_id"], "ddi:1")  # it finds ddi:1 for case 1
        self.assertIsNone(self.agent.run_baseline(CASES[4])["evidence_id"])  # and no evidence for the unknown drug

    def test_initial_state_has_the_preseeded_request_id(self):
        # A new test state has an empty queue and only the pre-used id from the notebook.
        self.assertEqual(self.state.snapshot(), {"review_queue": [], "seen_request_ids": ["already-used-004"]})

    def test_queue_note_has_exactly_the_agreed_fields(self):
        self.agent.run_guarded(CASES[0])  # stage the successful case
        # The queue holds exactly one note with these four fields and no others.
        self.assertEqual(self.state.review_queue, [{
            "evidence_id": "ddi:1", "drug_pair": ["Warfarin", "Ibuprofen"],  # the evidence id and the pair
            "request_id": "request-001", "purpose": "clinician review only"}])  # the request id and the purpose
        self.assertTrue(self.state.queue_rows()[0]["simulated"])  # the stored row is flagged as simulated

    def test_trace_entry_records_everything_the_notebook_did(self):
        outcome = self.agent.run_guarded(CASES[0])  # run the success case
        entry = outcome["tool_trace"][0]  # the single trace entry
        # The entry has exactly these keys.
        self.assertEqual(set(entry), {"tool", "args", "approved", "enabled", "before", "after", "feedback"})
        self.assertEqual(entry["tool"], REVIEW_TOOL)  # which tool was called
        self.assertEqual((entry["approved"], entry["enabled"]), (True, True))  # approval and enabled flags recorded
        self.assertEqual(entry["before"]["review_queue"], [])  # queue was empty before
        self.assertEqual(len(entry["after"]["review_queue"]), 1)  # and holds one note after
        self.assertEqual(entry["feedback"], {"status": "staged_in_simulation", "request_id": "request-001"})  # result
        self.assertEqual(self.state.tool_trace, [entry])  # the state keeps the same trace

    def test_the_request_id_cannot_be_used_twice(self):
        self.agent.run_guarded(CASES[0])  # first use succeeds
        again = self.agent.run_guarded(CASES[0])  # reusing the id
        self.assertEqual(again["status"], "duplicate_request_id")  # is refused
        self.assertEqual(len(self.state.review_queue), 1)  # and nothing extra is queued

    def test_a_denied_request_does_not_burn_its_id(self):
        denied = dict(CASES[0], approved=False)  # same request but without approval
        self.assertEqual(self.agent.run_guarded(denied)["status"], "approval_required")  # denied for approval
        # The id was not consumed, so the approved retry succeeds.
        self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "staged_in_simulation")


# Tests the order in which the six gates are checked.
class TestGateOrder(AgentTestCase):
    def test_gates_fire_in_the_documented_order(self):
        # Start with a request that breaks every gate at once.
        case = dict(CASES[0], drug_pair=("Warfarin", "NoSuchDrug"), tool_enabled=False,
                    requested_tool="other_tool", target="medication_order",
                    approved=False, request_id="already-used-004")
        # Fixes applied one at a time, in the documented gate order.
        fixes = [{"drug_pair": ("Warfarin", "Ibuprofen")}, {"tool_enabled": True},
                 {"requested_tool": REVIEW_TOOL}, {"target": REVIEW_TARGET}, {"approved": True},
                 {"request_id": "fresh-1"}]
        seen = [self.agent.run_guarded(case)["status"]]  # the first status with everything wrong
        for fix in fixes:  # repair one problem at a time
            case.update(fix)  # apply the fix
            seen.append(self.agent.run_guarded(case)["status"])  # record the next gate that fires
        # Each fix should expose exactly the next gate, ending in success.
        expected = ["missing_evidence", "tool_disabled", "tool_not_allowed", "out_of_scope",
                    "approval_required", "duplicate_request_id", "staged_in_simulation"]
        self.assertEqual(seen, expected)  # the observed order matches
        self.assertEqual(agent_module.GATES, tuple(expected[:-1]))  # and matches the module's GATES constant

    def test_not_allowed_tool_is_refused(self):
        outcome = self.agent.run_guarded(dict(CASES[0], requested_tool="apply_clinical_change"))  # a tool not allowed
        self.assertEqual(outcome["status"], "tool_not_allowed")  # it is refused
        self.assertEqual(self.state.review_queue, [])  # and nothing is written


# Tests the tool by itself, without the controller in front of it.
class TestToolBoundary(AgentTestCase):
    # Docstring explaining that the tool defends itself even when called directly.
    """The tool repeats its own checks even if the controller is bypassed."""

    def call(self, **overrides):  # helper that calls the tool directly
        args = {"target": REVIEW_TARGET, "request_id": "direct-1", "drug_pair": ["Warfarin", "Ibuprofen"],
                "evidence_id": "ddi:1"}  # a valid default set of arguments
        args.update(overrides)  # let each test break one thing
        approved = args.pop("approved", True)  # approval flag is passed separately, default yes
        enabled = args.pop("enabled", True)  # enabled flag is passed separately, default yes
        return self.agent.stage_interaction_review(args, approved, enabled)[0]  # return just the result dict

    def test_each_check_is_repeated(self):
        self.assertEqual(self.call(enabled=False)["reason"], "tool_disabled")  # disabled tool refuses
        self.assertEqual(self.call(approved=False)["reason"], "approval_required")  # no approval refuses
        self.assertEqual(self.call(target="medication_order")["reason"], "out_of_scope")  # wrong target refuses
        self.assertEqual(self.call(request_id="already-used-004")["reason"], "duplicate_request_id")  # used id refuses
        self.assertEqual(self.call(evidence_id=None)["reason"], "missing_evidence")  # no evidence refuses
        self.assertEqual(self.state.review_queue, [])  # none of the refusals wrote anything

    def test_evidence_must_exist_in_our_data_for_that_pair(self):
        self.assertEqual(self.call(evidence_id="ddi:99999")["reason"], "missing_evidence")  # nonexistent record
        self.assertEqual(self.call(evidence_id="ddi:12")["reason"], "missing_evidence")  # a different pair's row
        self.assertEqual(self.call(drug_pair=["Warfarin", "NoSuchDrug"])["reason"], "missing_evidence")  # unknown drug
        self.assertEqual(self.state.review_queue, [])  # still nothing written
        self.assertEqual(self.call()["status"], "staged_in_simulation")  # the valid call finally succeeds


# Tests what happens when the write fails.
class TestErrorPath(AgentTestCase):
    def test_a_failed_write_reports_the_failure_and_changes_nothing(self):
        before = self.state.snapshot()  # state before the failing run
        # Force the queue write to raise a database error.
        with mock.patch.object(AgentState, "stage_note", side_effect=sqlite3.OperationalError("disk I/O error")):
            outcome = self.agent.run_guarded(CASES[0])  # run the otherwise valid case
        self.assertEqual(outcome["status"], "error")  # the failure is reported as an error
        self.assertIn("disk I/O error", outcome["tool_result"]["reason"])  # with the real reason
        self.assertFalse(outcome["action_occurred"])  # success is not claimed
        self.assertEqual(self.state.snapshot(), before)  # nothing changed
        self.assertNotIn("request-001", self.state.seen_request_ids)  # the id was not used up

    def test_retry_needs_the_error_fixed_and_approval_given_again(self):
        # First attempt fails because the write is broken.
        with mock.patch.object(AgentState, "stage_note", side_effect=sqlite3.OperationalError("disk I/O error")):
            self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "error")
        # The error is fixed, but the earlier approval does not carry over.
        self.assertEqual(self.agent.run_guarded(dict(CASES[0], approved=False))["status"], "approval_required")
        self.assertEqual(self.state.review_queue, [])  # still nothing staged
        self.assertEqual(self.agent.run_guarded(CASES[0])["status"], "staged_in_simulation")  # approved retry works
        self.assertEqual(len(self.state.review_queue), 1)  # one note queued

    def test_the_note_and_the_used_id_are_written_together_or_not_at_all(self):
        # 'already-used-004' is in the used-ID set, so the second statement fails after the first succeeded.
        note = {"evidence_id": "ddi:1", "drug_pair": ["Warfarin", "Ibuprofen"],
                "request_id": "already-used-004", "purpose": "clinician review only"}  # a note with a used id
        with self.assertRaises(sqlite3.IntegrityError):  # the write must fail
            self.state.stage_note(note)
        self.assertEqual(self.state.review_queue, [])  # the first insert was rolled back


# Tests that an ineligible source cannot be staged as documented.
class TestEvidenceRules(unittest.TestCase):
    def test_evidence_ineligible_pair_cannot_be_staged_as_documented(self):
        # A registry where the only source is archived, so its evidence is not eligible.
        reg = registry.build_registry({"approved_authority_classes": [],
                                       "sources": {"kaggle_ddi": {"status": "archived", "scope": "exact_pair"}}})
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder for a fake database
            # Build a one-row database (A + B, Major) and open it read-only.
            conn = ingest.open_readonly(support.build_db(Path(tmp), [ddi_row("A", "B", "Major")]), ingest.CORE_TABLES)
            try:
                state = AgentState.for_tests()  # fresh state
                self.addCleanup(state.close)  # close it afterwards
                agent = GuardedAgent(Checker(conn, registry=reg), state)  # agent using the strict registry
                outcome = agent.run_guarded(dict(CASES[0], drug_pair=("A", "B")))  # try to stage A + B
                self.assertEqual(outcome["status"], "missing_evidence")  # ineligible evidence counts as missing
                self.assertEqual(state.review_queue, [])  # nothing staged
            finally:
                conn.close()  # always close the fake database


# Tests that the state lives in its own file and that the interaction database is never changed.
class TestPersistence(unittest.TestCase):
    def test_queue_and_log_live_in_a_separate_file_and_survive_reopening(self):
        conn = support.open_real()  # real interaction database
        self.addCleanup(conn.close)  # close it afterwards
        checker = Checker(conn)  # checker over that database
        with tempfile.TemporaryDirectory() as tmp:  # temp folder for the state file
            path = Path(tmp) / "agent_state.db"  # the separate state file
            state = AgentState(path)  # create it
            GuardedAgent(checker, state).run_guarded(CASES[0])  # a staged request
            GuardedAgent(checker, state).run_guarded(CASES[1])  # a denied request
            state.close()  # close the file
            reopened = AgentState(path)  # open it again
            try:
                self.assertEqual([n["request_id"] for n in reopened.review_queue], ["request-001"])  # queue survived
                self.assertEqual(reopened.seen_request_ids, {"request-001"})  # not pre-seeded outside tests
                self.assertEqual(len(reopened.attempts("request-002")), 1)  # the denied attempt was logged
                self.assertEqual(reopened.attempts("request-002")[0]["gate"], "approval_required")  # with its gate
            finally:
                reopened.close()  # close the reopened file

    def test_state_is_never_stored_in_medications_db(self):
        with self.assertRaises(ValueError):  # trying to use the data file as state must be refused
            AgentState(Path("somewhere") / "medications.db")

    def test_the_interaction_database_is_never_written(self):
        path = support.real_db_path()  # the database file
        before = hashlib.sha256(path.read_bytes()).hexdigest()  # fingerprint it before
        conn = support.open_real()  # open read-only
        self.addCleanup(conn.close)  # close afterwards
        state = AgentState.for_tests()  # fresh state
        self.addCleanup(state.close)  # close afterwards
        agent = GuardedAgent(Checker(conn), state)  # agent over the database
        for case in CASES:  # run all six scenarios
            agent.run_guarded(case)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)  # the file is byte-identical

    def test_the_orchestrator_imports_no_language_model_code(self):
        tree = ast.parse(Path(agent_module.__file__).read_text(encoding="utf-8"))  # parse the agent's source
        imported = set()  # every module name it imports
        for node in ast.walk(tree):  # visit every syntax node
            if isinstance(node, ast.Import):  # plain "import x"
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):  # "from x import y"
                imported.add(node.module)  # the module
                imported |= {f"{node.module}.{a.name}" for a in node.names}  # and each name from it
        # None of the imports may mention a language-model, OpenAI or urllib module.
        self.assertEqual([name for name in imported if "llm" in name or "openai" in name or "urllib" in name], [])


# Tests the human-readable trace of an attempt.
class TestTrace(AgentTestCase):
    def test_trace_reports_the_attempt_gate_result_state_change_and_whether_it_happened(self):
        self.agent.run_guarded(CASES[0])  # one staged request
        self.agent.run_guarded(CASES[1])  # one denied request
        staged = format_trace("request-001", self.state.attempts("request-001"))  # trace text for the staged one
        self.assertIn("SIMULATED", staged)  # labelled as simulated
        self.assertIn("stage_interaction_review", staged)  # names the tool
        self.assertIn("gate result    : allowed", staged)  # the gate allowed it
        self.assertIn('"status": "staged_in_simulation"', staged)  # shows the tool result
        self.assertIn("review_queue 0 -> 1", staged)  # shows the state change
        self.assertIn("action happened: yes", staged)  # says the action happened

        denied = format_trace("request-002", self.state.attempts("request-002"))  # trace text for the denied one
        self.assertIn("gate result    : approval_required", denied)  # the gate that stopped it
        self.assertIn("not produced", denied)  # the tool never produced a result
        self.assertIn("review_queue 1 -> 1", denied)  # request-001 was already in the queue
        self.assertIn("action happened: no", denied)  # says the action did not happen
        self.assertIn("therefore the action did not occur", denied)  # and states it as a conclusion

    def test_unknown_request_id(self):
        self.assertIn("No attempts", format_trace("nope", self.state.attempts("nope")))  # says nothing was found

    def test_describe_attempt_matches_the_notebook_sentence(self):
        self.agent.run_guarded(CASES[2])  # the out-of-scope case
        sentence = describe_attempt(self.state.attempts("request-003")[0])  # the one-sentence description
        self.assertTrue(sentence.startswith("The agent attempted stage_interaction_review(args="))  # how it begins
        self.assertIn("the gate returned out_of_scope", sentence)  # names the gate result
        self.assertIn("therefore the action did not occur.", sentence)  # and the conclusion


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run the tests in this file


# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the guarded agent: the six notebook cases, the order of the six gates, the tool repeating its
# own checks, the error path, atomic queue writes, persistence in a separate state file, the guarantee that the
# interaction database is never written, and the trace output. It also checks the agent imports no language model code.
# ASSIGNMENT TOOL: guarded tool use with a simulated review queue (the agent, ported from the agent notebook).
# ---------------------------------------------------------------------------
