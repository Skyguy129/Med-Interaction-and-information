"""Tests for full_system_test.py, the interactive terminal run of the whole system. No network."""

import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from ddi_backend import ingest, llm
from tests import support

spec = importlib.util.spec_from_file_location("full_system_test", Path(__file__).resolve().parent.parent / "full_system_test.py")
fst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fst)


def run(lines, extra=(), key=""):
    feed = iter(lines)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fst.main(["--db", str(support.real_full_db_path()), "--memory", *extra],
                        input_fn=lambda prompt="": next(feed), key_fn=lambda prompt="": key)
    return code, out.getvalue(), err.getvalue()


class TestWithoutAKey(unittest.TestCase):
    def test_deterministic_answer_shows_steps_response_and_source(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, out, _ = run(["Do warfarin and ibuprofen interact?", "/quit"])
        self.assertEqual(code, 0)
        for needle in ("[1] names found: Warfarin, Ibuprofen", "[3] lookup and rules fired", "[4] eligibility filter",
                       "template", "documented Major interaction", "[drug interaction] ddi:1", "not medical advice"):
            self.assertIn(needle, out)

    def test_negation_question_uses_the_avoid_list(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, out, _ = run(["Which pain reliever should I not take with warfarin?", "/quit"], ["--no-key-prompt"])
        self.assertIn("question type: avoid", out)
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))

    def test_abbreviation_gets_a_question(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, out, _ = run(["What does MS interact with?", "/quit"])
        self.assertIn("Which medication do you mean?", out)

    def test_medications_then_a_new_drug_name(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, out, _ = run(["/meds Warfarin 5 mg, Docusate", "/new Ibuprofen", "/quit"])
        self.assertLess(out.index("Warfarin + Ibuprofen"), out.index("Docusate + Ibuprofen"))

    def test_staging_needs_a_yes(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, out, _ = run(["/stage Warfarin Ibuprofen r1", "no", "/quit"])
        self.assertIn("approval_required", out)


class TestWithAModel(unittest.TestCase):
    def provider(self, reply):
        provider = llm.OpenAIProvider("test-key")
        provider.complete = lambda messages: reply if isinstance(reply, str) else json.dumps(reply)
        return provider

    def run_with(self, reply, lines):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
                mock.patch.object(llm, "provider_from_env", return_value=self.provider(reply)):
            return run(lines)

    def test_validated_model_text_is_used_and_the_source_is_shown(self):
        conn = support.open_real()
        record = next(r for r in ingest.load_interactions(conn) if r["id"] == "ddi:12")
        conn.close()
        reply = {"summary": f"Warfarin with Fluconazole is documented as a Major interaction. {record['effect']}.",
                 "claims": [{"record_id": "ddi:12", "field": f, "value": record[f]} for f in ("severity", "mechanism", "effect")],
                 "clarifying_question": None}
        _, out, _ = self.run_with(reply, ["Does warfarin interact with fluconazole?", "/quit"])
        self.assertIn("language model", out)
        self.assertIn("documented as a Major interaction", out)
        self.assertIn("[drug interaction] ddi:12", out)

    def test_an_invented_fact_falls_back_to_the_template(self):
        reply = {"summary": "Warfarin with Fluconazole causes hospitalization.", "claims": [], "clarifying_question": None}
        _, out, _ = self.run_with(reply, ["Does warfarin interact with fluconazole?", "/quit"])
        self.assertIn("failed validation", out)
        self.assertIn("Taking Warfarin with Fluconazole is a documented Major interaction", out)

    def test_list_answers_are_narrated_only_when_valid(self):
        good = {"summary": "Ibuprofen has a documented Major interaction with Warfarin.", "record_ids": ["ddi:1"]}
        _, out, _ = self.run_with(good, ["What should I not take with warfarin?", "/quit"])
        self.assertIn("Ibuprofen has a documented Major interaction with Warfarin.", out)
        self.assertIn("Structured result", out)
        bad = {"summary": "Take aspirin instead, it is safe.", "record_ids": ["ddi:1"]}
        _, out, _ = self.run_with(bad, ["What should I not take with warfarin?", "/quit"])
        self.assertIn("reply failed validation", out)
        self.assertIn("most severe first", out)

    def test_provider_errors_fall_back(self):
        provider = llm.OpenAIProvider("k")
        provider.complete = mock.Mock(side_effect=RuntimeError("HTTP 500"))
        with mock.patch.object(llm, "provider_from_env", return_value=provider):
            _, out, _ = run(["What should I not take with warfarin?", "/quit"])
        self.assertIn("language model unavailable", out)


if __name__ == "__main__":
    unittest.main()
