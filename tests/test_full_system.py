# Module docstring: this file tests the interactive terminal script full_system_test.py and uses no network.
"""Tests for full_system_test.py, the interactive terminal run of the whole system. No network."""

import contextlib  # redirect_stdout / redirect_stderr to capture what the script prints
import importlib.util  # loads full_system_test.py by file path, since it is not inside a package
import io  # in-memory text buffers for the captured output
import json  # turns scripted model replies into JSON text
import unittest  # standard-library test framework
from pathlib import Path  # path handling for locating the script
from unittest import mock  # patches environment variables and the language model provider

from ddi_backend import ingest, llm  # record loading, and the language model module being patched
from tests import support  # shared helpers: open the real database, find the real full database path

# Load the top-level script from the repository root as a module named full_system_test, without importing a package.
spec = importlib.util.spec_from_file_location("full_system_test", Path(__file__).resolve().parent.parent / "full_system_test.py")
fst = importlib.util.module_from_spec(spec)  # create an empty module object from that spec
spec.loader.exec_module(fst)  # run the script's top level so its functions (main etc.) exist on fst


def run(lines, extra=(), key=""):  # helper: run the script with scripted typed lines and return (exit code, stdout, stderr)
    feed = iter(lines)  # the lines the "user" will type, handed out one at a time
    out, err = io.StringIO(), io.StringIO()  # buffers for stdout and stderr
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):  # capture everything printed
        # Run main against the real database with an in-memory state file, plus any extra flags.
        code = fst.main(["--db", str(support.real_full_db_path()), "--memory", *extra],
                        # Fake the typed input (next scripted line) and the hidden key prompt (returns key).
                        input_fn=lambda prompt="": next(feed), key_fn=lambda prompt="": key)
    return code, out.getvalue(), err.getvalue()  # exit code and the two captured texts


class TestWithoutAKey(unittest.TestCase):  # tests the deterministic mode, with no API key
    def test_deterministic_answer_shows_steps_response_and_source(self):  # a normal pair question
        with mock.patch.dict("os.environ", {}, clear=True):  # empty environment, so no Anthropic key
            code, out, _ = run(["Do warfarin and ibuprofen interact?", "/quit"])  # ask, then quit
        self.assertEqual(code, 0)  # clean exit
        # Phrases that must appear: the numbered steps, the template composer, the answer, the source, and the disclaimer.
        for needle in ("[1] names found: Warfarin, Ibuprofen", "[3] lookup and rules fired", "[4] eligibility filter",
                       "template", "documented Major interaction", "[drug interaction] ddi:1", "not medical advice"):
            self.assertIn(needle, out)  # each one is in the output

    def test_negation_question_uses_the_avoid_list(self):  # "should I not take" is answered from the avoid list
        with mock.patch.dict("os.environ", {}, clear=True):  # no key
            # Ask the negation question; --no-key-prompt skips the key prompt.
            _, out, _ = run(["Which pain reliever should I not take with warfarin?", "/quit"], ["--no-key-prompt"])
        self.assertIn("question type: avoid", out)  # the question was classified as an avoid question
        self.assertLess(out.index("Amiodarone"), out.index("Acetaminophen"))  # ordered by severity: Major Amiodarone before Acetaminophen

    def test_abbreviation_gets_a_question(self):  # an abbreviation is never auto-resolved
        with mock.patch.dict("os.environ", {}, clear=True):  # no key
            _, out, _ = run(["What does MS interact with?", "/quit"])  # "MS" is ambiguous
        self.assertIn("Which medication do you mean?", out)  # the system asks for clarification

    def test_medications_then_a_new_drug_name(self):  # the /meds and /new commands
        with mock.patch.dict("os.environ", {}, clear=True):  # no key
            # Set current medications (one with a dose), then name a new drug to check against them.
            _, out, _ = run(["/meds Warfarin 5 mg, Docusate", "/new Ibuprofen", "/quit"])
        self.assertLess(out.index("Warfarin + Ibuprofen"), out.index("Docusate + Ibuprofen"))  # the Major pair is listed first

    def test_staging_needs_a_yes(self):  # the review-queue write needs approval
        with mock.patch.dict("os.environ", {}, clear=True):  # no key
            _, out, _ = run(["/stage Warfarin Ibuprofen r1", "no", "/quit"])  # try to stage, then answer "no"
        self.assertIn("approval_required", out)  # the gate stops it and names the gate


class TestWithAModel(unittest.TestCase):  # tests the language-model mode using a scripted stand-in provider
    def provider(self, reply):  # helper: an Anthropic provider whose network call is replaced with a canned reply
        provider = llm.AnthropicProvider("test-key")  # constructed with a dummy key, never contacted
        provider.complete = lambda messages: reply if isinstance(reply, str) else json.dumps(reply)  # strings pass through, dicts become JSON
        return provider

    def run_with(self, reply, lines):  # helper: run the script with the fake key and the scripted provider
        # Backslash continuation below, so this note sits above: set a fake key and make the provider lookup return our stand-in.
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
                mock.patch.object(llm, "provider_from_env", return_value=self.provider(reply)):
            return run(lines)  # run the scripted conversation

    def test_validated_model_text_is_used_and_the_source_is_shown(self):  # a valid model reply is accepted
        conn = support.open_real()  # open the real database
        record = next(r for r in ingest.load_interactions(conn) if r["id"] == "ddi:12")  # the warfarin + fluconazole record
        conn.close()  # done with the connection
        # A reply that restates the record faithfully: summary text plus one claim per checked field.
        reply = {"summary": f"Warfarin with Fluconazole is documented as a Major interaction. {record['effect']}.",
                 "claims": [{"record_id": "ddi:12", "field": f, "value": record[f]} for f in ("severity", "mechanism", "effect")],
                 "clarifying_question": None}  # no clarifying question needed
        _, out, _ = self.run_with(reply, ["Does warfarin interact with fluconazole?", "/quit"])  # ask the question
        self.assertIn("language model", out)  # output says the model wrote the text
        self.assertIn("documented as a Major interaction", out)  # the model's sentence is shown
        self.assertIn("[drug interaction] ddi:12", out)  # and the source record is shown beside it

    def test_an_invented_fact_falls_back_to_the_template(self):  # a reply with a new fact is rejected
        # The summary adds "hospitalization", which is not in the record, and gives no supporting claims.
        reply = {"summary": "Warfarin with Fluconazole causes hospitalization.", "claims": [], "clarifying_question": None}
        _, out, _ = self.run_with(reply, ["Does warfarin interact with fluconazole?", "/quit"])  # ask the question
        self.assertIn("failed validation", out)  # the validator rejected it
        self.assertIn("Taking Warfarin with Fluconazole is a documented Major interaction", out)  # template text is used instead

    def test_list_answers_are_narrated_only_when_valid(self):  # avoid-list narration is validated too
        good = {"summary": "Ibuprofen has a documented Major interaction with Warfarin.", "record_ids": ["ddi:1"]}  # faithful narration
        _, out, _ = self.run_with(good, ["What should I not take with warfarin?", "/quit"])  # ask the avoid question
        self.assertIn("Ibuprofen has a documented Major interaction with Warfarin.", out)  # valid narration is shown
        self.assertIn("Structured result", out)  # next to the structured result
        bad = {"summary": "Take aspirin instead, it is safe.", "record_ids": ["ddi:1"]}  # invented advice and a "safe" claim
        _, out, _ = self.run_with(bad, ["What should I not take with warfarin?", "/quit"])  # same question
        self.assertIn("reply failed validation", out)  # the reply is rejected
        self.assertIn("most severe first", out)  # and the deterministic list is shown instead

    def test_provider_errors_fall_back(self):  # a failing model call must not break the run
        provider = llm.AnthropicProvider("k")  # provider with a dummy key
        provider.complete = mock.Mock(side_effect=RuntimeError("HTTP 500"))  # every call raises
        with mock.patch.object(llm, "provider_from_env", return_value=provider):  # make the script use this provider
            _, out, _ = run(["What should I not take with warfarin?", "/quit"])  # ask a question
        self.assertIn("language model unavailable", out)  # it reports the problem and carries on


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests full_system_test.py, the interactive terminal run of the whole system, by feeding it scripted
# input and capturing its output with no network. It checks the printed steps, the avoid-list answer to a negation
# question, the clarifying question for an abbreviation, the /meds, /new and /stage commands (staging needs a yes),
# and, with a scripted stand-in model, that valid model text is used while invented facts and provider errors fall back
# to the template text.
# ASSIGNMENT TOOL: Exercises all the adopted tools together end to end: the knowledge graph with rule-based reasoning,
# eligibility filtering, the guarded agent with its simulated review queue, and the language model modes.
# ---------------------------------------------------------------------------
