# Docstring describing the module: tests for the command-line interface.
"""Tests for the command-line interface."""

import contextlib  # redirects stdout/stderr so output can be captured
import hashlib  # fingerprints the database to prove it is untouched
import io  # in-memory text and byte streams
import json  # parses the CLI's --json output
import tempfile  # temporary folders for state files
import unittest  # the test framework
from pathlib import Path  # path objects
from unittest import mock  # patches for environment and embedding loading

from ddi_backend import cli, embeddings  # the CLI under test and the embeddings module (patched below)
from ddi_backend.database_context import configure_console_output  # makes console output safe on Windows
from tests import support  # shared database builders


# Runs the CLI with the given arguments and returns (exit code, captured stdout, captured stderr).
def run(argv):
    out, err = io.StringIO(), io.StringIO()  # buffers that stand in for the console
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):  # capture everything printed
        code = cli.main(argv)  # run the command line program
    return code, out.getvalue(), err.getvalue()  # hand back the exit code and the captured text


# Base class: gives each test a temporary folder, the real database and a state file path.
class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()  # a temporary folder
        self.addCleanup(self._tmp.cleanup)  # deleted when the test ends
        self.dir = Path(self._tmp.name)  # its path
        self.db = support.real_full_db_path()  # the real DDI + HPO database
        self.state = self.dir / "agent_state.db"  # where this test's state file will go

    def base(self):  # the shared command line options
        return ["--db", str(self.db), "--state", str(self.state)]  # point the CLI at the test database and state file


# Tests for the "check" command.
class TestCheck(CliTestCase):
    def test_major_first_and_the_disclaimer_is_shown(self):
        # Two current drugs (one with a dose) against a new drug.
        code, out, _ = run(["check", "--current", "Docusate", "--current", "Warfarin 5 mg", "--new", "Ibuprofen",
                            *self.base()])
        self.assertEqual(code, 0)  # success
        # The Major interaction text must appear before the "not documented" text.
        self.assertLess(out.index("documented Major interaction"), out.index("not documented in our data"))
        self.assertIn("not medical advice", out)  # the disclaimer is printed

    def test_json_output_has_one_result_per_pair(self):
        code, out, _ = run(["check", "--current", "Fluconazole", "--current", "Ibuprofen", "--new", "Warfarin",
                            "--json", *self.base()])  # ask for JSON output
        data = json.loads(out)  # the output must be valid JSON
        self.assertEqual(code, 0)  # success
        self.assertEqual([r["status"] for r in data["results"]], ["documented_interaction"] * 2)  # both pairs documented
        for key in ("status", "severity", "mechanism", "effect", "alternatives", "rules_fired", "source"):  # required fields
            self.assertIn(key, data["results"][0])  # each is present

    def test_include_current_pairs_adds_pairs(self):
        _, out, _ = run(["check", "--current", "Warfarin", "--current", "Fluconazole", "--new", "Docusate",
                         "--include-current-pairs", "--json", *self.base()])  # also check current drugs against each other
        self.assertEqual(len(json.loads(out)["results"]), 3)  # two new-drug pairs plus one current pair

    def test_unknown_drug_is_never_safe(self):
        _, out, _ = run(["check", "--current", "Cabergoline", "--new", "Rucaparib", *self.base()])  # neither is in the data
        self.assertIn("not a drug name in our data", out)  # said plainly
        self.assertIn("does not mean the combination is safe", out)  # and not called safe

    def test_explain_without_a_key_uses_the_template(self):
        with mock.patch.dict("os.environ", {}, clear=True):  # no OPENAI_API_KEY in the environment
            _, out, _ = run(["check", "--current", "Warfarin", "--new", "Fluconazole", "--explain", *self.base()])
        self.assertIn("(written by: template)", out)  # the template composer wrote the explanation
        self.assertIn("[drug interaction] ddi:12", out)  # the source record is printed beside the text

    def test_detail_levels_change_the_length(self):
        # Output length for brief, standard and detailed.
        sizes = [len(run(["check", "--current", "Warfarin", "--new", "Ibuprofen", "--detail", d, *self.base()])[1])
                 for d in ("brief", "standard", "detailed")]
        self.assertEqual(sizes, sorted(sizes))  # each level is at least as long as the one before
        self.assertLess(sizes[0], sizes[2])  # and detailed is strictly longer than brief


# Tests for the drug, avoid, ask, health and export commands.
class TestOtherCommands(CliTestCase):
    def test_drug_and_avoid(self):
        code, out, _ = run(["drug", "warfarin", *self.base()])  # single-drug view
        self.assertEqual(code, 0)  # success
        self.assertIn("not available in our data", out)  # monograph fields are marked unavailable
        code, out, _ = run(["avoid", "warfarin", *self.base()])  # avoid list
        self.assertEqual(code, 0)  # success
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))  # Major entry listed before Moderate

    def test_unknown_drug_gets_did_you_mean(self):
        code, out, _ = run(["drug", "Warfrin", *self.base()])  # a typo
        self.assertEqual(code, 1)  # failure exit code
        self.assertIn("Did you mean: Warfarin", out)  # with a suggestion

    def test_ask_the_adversarial_question(self):
        code, out, _ = run(["ask", "Which pain reliever should I not take with warfarin?", *self.base()])  # the "not" question
        self.assertEqual(code, 0)  # success
        self.assertIn("classes are not in our data", out)  # the class caveat is stated
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))  # ordered by severity, not by similarity

    def test_ask_ambiguous_abbreviation(self):
        _, out, _ = run(["ask", "What does MS interact with?", *self.base()])  # "MS" is ambiguous
        self.assertIn("abbreviation", out)  # the answer talks about the abbreviation instead of guessing

    def test_missing_database_is_reported_and_not_created(self):
        missing = self.dir / "nope.db"  # a database that does not exist
        code, _, err = run(["check", "--new", "X", "--db", str(missing), "--memory"])  # try to use it
        self.assertEqual(code, 1)  # failure
        self.assertIn("load_data load", err)  # the error tells the user how to build it
        self.assertFalse(missing.exists())  # and the file was not created

    def test_export_graph_writes_files_and_trains_nothing(self):
        code, out, _ = run(["export-graph", "--out", str(self.dir / "export"), *self.base()])  # export the graph
        self.assertEqual(code, 0)  # success
        self.assertTrue((self.dir / "export" / "nodes.csv").is_file())  # a nodes file was written
        self.assertIn("Nothing was trained", out)  # and it says no model was trained

    def test_health_reports_tiers(self):
        _, out, _ = run(["health", "--json", *self.base()])  # ask for tier status
        tiers = json.loads(out)["tiers"]  # the tiers section
        self.assertTrue(tiers["core"])  # core is available
        self.assertFalse(tiers["embeddings"])  # embeddings are off by default


# Tests for the review-queue commands and the trace command.
class TestReviewCommands(CliTestCase):
    def test_nothing_is_written_without_approval(self):
        code, out, _ = run(["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", *self.base()])  # no --approve
        self.assertEqual(code, 1)  # refused
        self.assertIn("approval_required", out)  # for lack of approval
        self.assertIn("SIMULATED", out)  # output is labelled simulated
        _, listing, _ = run(["review", "list", *self.base()])  # list the queue
        self.assertIn("0 note(s)", listing)  # still empty

    def test_approved_request_is_staged_listed_and_traceable(self):
        digest = hashlib.sha256(self.db.read_bytes()).hexdigest()  # fingerprint the database before
        code, out, _ = run(["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", "--approve",
                            *self.base()])  # an approved request
        self.assertEqual(code, 0)  # success
        self.assertIn("staged_in_simulation", out)  # staged
        self.assertTrue(self.state.is_file())  # the queue lives in its own file
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), digest)  # medications.db untouched
        _, listing, _ = run(["review", "list", *self.base()])  # list the queue
        self.assertIn("r1: ddi:1 Warfarin + Ibuprofen (clinician review only)", listing)  # the note appears
        code, trace, _ = run(["trace", "r1", *self.base()])  # trace the request
        self.assertEqual(code, 0)  # success
        # The trace must show each part of the attempt.
        for needle in ("attempted call", "gate result    : allowed", "tool result", "review_queue 0 -> 1",
                       "action happened: yes"):
            self.assertIn(needle, trace)  # each phrase appears

    def test_duplicate_request_id_is_refused_and_traced(self):
        args = ["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", "--approve", *self.base()]  # one request
        self.assertEqual(run(args)[0], 0)  # the first use succeeds
        code, out, _ = run(args)  # the same id again
        self.assertEqual(code, 1)  # refused
        self.assertIn("duplicate_request_id", out)  # for being a duplicate
        _, trace, _ = run(["trace", "r1", *self.base()])  # trace it
        self.assertIn("Attempt 2", trace)  # both attempts are recorded
        self.assertIn("action happened: no", trace)  # and the second did not happen

    def test_unknown_request_id_has_no_trace(self):
        code, out, _ = run(["trace", "never-seen", *self.base()])  # an id never used
        self.assertEqual(code, 1)  # failure
        self.assertIn("No attempts", out)  # says nothing was found


# Tests that a missing or broken embedding model never breaks the deterministic core.
class TestTierTwoNeverBreaksTierOne(CliTestCase):
    def test_embeddings_auto_falls_back_when_the_model_is_unavailable(self):
        with mock.patch.object(embeddings, "load_embedder", return_value=None):  # pretend the model cannot load
            code, out, _ = run(["check", "--current", "Warfarin", "--new", "Ibuprofen", "--embeddings", "auto",
                                *self.base()])
        self.assertEqual(code, 0)  # still succeeds
        self.assertIn("documented Major interaction", out)  # with the normal answer

    def test_embeddings_auto_survives_a_loader_that_raises(self):
        with mock.patch.object(embeddings, "load_embedder", side_effect=RuntimeError("boom")):  # loader crashes
            code, out, _ = run(["ask", "Do warfarin and ibuprofen interact?", "--embeddings", "auto", *self.base()])
        self.assertEqual(code, 0)  # the crash is absorbed


# Tests the Windows console fix for characters the console cannot encode.
class TestWindowsSafeOutput(unittest.TestCase):
    def test_console_helper_prevents_unicode_crashes(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")  # a strict Windows-style stream
        with self.assertRaises(UnicodeEncodeError):  # writing an arrow would normally crash
            stream.write("→")
        with mock.patch("sys.stdout", stream), mock.patch("sys.stderr", stream):  # make it the console
            configure_console_output()  # apply the fix
            stream.write("→ ok")  # now written with a backslash escape instead of raising
            stream.flush()  # push the bytes into the buffer
        self.assertIn(b"\\u2192", stream.buffer.getvalue())  # the arrow came out as an escape sequence


if __name__ == "__main__":  # only when run as a script
    unittest.main()  # run the tests in this file


# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the command-line interface end to end by calling cli.main and capturing its output: check
# (Major first, disclaimer, JSON, detail levels, explain), drug, avoid, ask, health, export-graph, the review
# stage/list commands, trace, graceful fallback when embeddings fail, and Windows-safe console output.
# ASSIGNMENT TOOL: supporting infrastructure (not one of the assignment tools) that drives all of them from a terminal:
# the knowledge graph and rules, the guarded agent and review queue, embeddings in their limited role, and the graph export.
# ---------------------------------------------------------------------------
