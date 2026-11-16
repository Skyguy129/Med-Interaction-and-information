"""Tests for the command-line interface."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ddi_backend import cli, embeddings
from ddi_backend.database_context import configure_console_output
from tests import support


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db = support.real_full_db_path()
        self.state = self.dir / "agent_state.db"

    def base(self):
        return ["--db", str(self.db), "--state", str(self.state)]


class TestCheck(CliTestCase):
    def test_major_first_and_the_disclaimer_is_shown(self):
        code, out, _ = run(["check", "--current", "Docusate", "--current", "Warfarin 5 mg", "--new", "Ibuprofen",
                            *self.base()])
        self.assertEqual(code, 0)
        self.assertLess(out.index("documented Major interaction"), out.index("not documented in our data"))
        self.assertIn("not medical advice", out)

    def test_json_output_has_one_result_per_pair(self):
        code, out, _ = run(["check", "--current", "Fluconazole", "--current", "Ibuprofen", "--new", "Warfarin",
                            "--json", *self.base()])
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual([r["status"] for r in data["results"]], ["documented_interaction"] * 2)
        for key in ("status", "severity", "mechanism", "effect", "alternatives", "rules_fired", "source"):
            self.assertIn(key, data["results"][0])

    def test_include_current_pairs_adds_pairs(self):
        _, out, _ = run(["check", "--current", "Warfarin", "--current", "Fluconazole", "--new", "Docusate",
                         "--include-current-pairs", "--json", *self.base()])
        self.assertEqual(len(json.loads(out)["results"]), 3)

    def test_unknown_drug_is_never_safe(self):
        _, out, _ = run(["check", "--current", "Cabergoline", "--new", "Rucaparib", *self.base()])
        self.assertIn("not a drug name in our data", out)
        self.assertIn("does not mean the combination is safe", out)

    def test_explain_without_a_key_uses_the_template(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, out, _ = run(["check", "--current", "Warfarin", "--new", "Fluconazole", "--explain", *self.base()])
        self.assertIn("(written by: template)", out)
        self.assertIn("[drug interaction] ddi:12", out)  # the source record is printed beside the text

    def test_detail_levels_change_the_length(self):
        sizes = [len(run(["check", "--current", "Warfarin", "--new", "Ibuprofen", "--detail", d, *self.base()])[1])
                 for d in ("brief", "standard", "detailed")]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[2])


class TestOtherCommands(CliTestCase):
    def test_drug_and_avoid(self):
        code, out, _ = run(["drug", "warfarin", *self.base()])
        self.assertEqual(code, 0)
        self.assertIn("not available in our data", out)
        code, out, _ = run(["avoid", "warfarin", *self.base()])
        self.assertEqual(code, 0)
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))

    def test_unknown_drug_gets_did_you_mean(self):
        code, out, _ = run(["drug", "Warfrin", *self.base()])
        self.assertEqual(code, 1)
        self.assertIn("Did you mean: Warfarin", out)

    def test_ask_the_adversarial_question(self):
        code, out, _ = run(["ask", "Which pain reliever should I not take with warfarin?", *self.base()])
        self.assertEqual(code, 0)
        self.assertIn("classes are not in our data", out)
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))

    def test_ask_ambiguous_abbreviation(self):
        _, out, _ = run(["ask", "What does MS interact with?", *self.base()])
        self.assertIn("abbreviation", out)

    def test_missing_database_is_reported_and_not_created(self):
        missing = self.dir / "nope.db"
        code, _, err = run(["check", "--new", "X", "--db", str(missing), "--memory"])
        self.assertEqual(code, 1)
        self.assertIn("load_data load", err)
        self.assertFalse(missing.exists())

    def test_export_graph_writes_files_and_trains_nothing(self):
        code, out, _ = run(["export-graph", "--out", str(self.dir / "export"), *self.base()])
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "export" / "nodes.csv").is_file())
        self.assertIn("Nothing was trained", out)

    def test_health_reports_tiers(self):
        _, out, _ = run(["health", "--json", *self.base()])
        tiers = json.loads(out)["tiers"]
        self.assertTrue(tiers["core"])
        self.assertFalse(tiers["embeddings"])


class TestReviewCommands(CliTestCase):
    def test_nothing_is_written_without_approval(self):
        code, out, _ = run(["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", *self.base()])
        self.assertEqual(code, 1)
        self.assertIn("approval_required", out)
        self.assertIn("SIMULATED", out)
        _, listing, _ = run(["review", "list", *self.base()])
        self.assertIn("0 note(s)", listing)

    def test_approved_request_is_staged_listed_and_traceable(self):
        digest = hashlib.sha256(self.db.read_bytes()).hexdigest()
        code, out, _ = run(["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", "--approve",
                            *self.base()])
        self.assertEqual(code, 0)
        self.assertIn("staged_in_simulation", out)
        self.assertTrue(self.state.is_file())  # the queue lives in its own file
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), digest)  # medications.db untouched
        _, listing, _ = run(["review", "list", *self.base()])
        self.assertIn("r1: ddi:1 Warfarin + Ibuprofen (clinician review only)", listing)
        code, trace, _ = run(["trace", "r1", *self.base()])
        self.assertEqual(code, 0)
        for needle in ("attempted call", "gate result    : allowed", "tool result", "review_queue 0 -> 1",
                       "action happened: yes"):
            self.assertIn(needle, trace)

    def test_duplicate_request_id_is_refused_and_traced(self):
        args = ["review", "stage", "--pair", "Warfarin", "Ibuprofen", "--request-id", "r1", "--approve", *self.base()]
        self.assertEqual(run(args)[0], 0)
        code, out, _ = run(args)
        self.assertEqual(code, 1)
        self.assertIn("duplicate_request_id", out)
        _, trace, _ = run(["trace", "r1", *self.base()])
        self.assertIn("Attempt 2", trace)
        self.assertIn("action happened: no", trace)

    def test_unknown_request_id_has_no_trace(self):
        code, out, _ = run(["trace", "never-seen", *self.base()])
        self.assertEqual(code, 1)
        self.assertIn("No attempts", out)


class TestTierTwoNeverBreaksTierOne(CliTestCase):
    def test_embeddings_auto_falls_back_when_the_model_is_unavailable(self):
        with mock.patch.object(embeddings, "load_embedder", return_value=None):
            code, out, _ = run(["check", "--current", "Warfarin", "--new", "Ibuprofen", "--embeddings", "auto",
                                *self.base()])
        self.assertEqual(code, 0)
        self.assertIn("documented Major interaction", out)

    def test_embeddings_auto_survives_a_loader_that_raises(self):
        with mock.patch.object(embeddings, "load_embedder", side_effect=RuntimeError("boom")):
            code, out, _ = run(["ask", "Do warfarin and ibuprofen interact?", "--embeddings", "auto", *self.base()])
        self.assertEqual(code, 0)


class TestWindowsSafeOutput(unittest.TestCase):
    def test_console_helper_prevents_unicode_crashes(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with self.assertRaises(UnicodeEncodeError):
            stream.write("→")
        with mock.patch("sys.stdout", stream), mock.patch("sys.stderr", stream):
            configure_console_output()
            stream.write("→ ok")  # now written with a backslash escape instead of raising
            stream.flush()
        self.assertIn(b"\\u2192", stream.buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
