# Docstring: says this file tests the template composer and the language model modes, with stub providers only.
"""Tests for the template composer and the language model modes. No network: every provider is a stub."""

import contextlib  # capture printed output in the selector test
import io  # BytesIO / StringIO for fake responses and captured output
import json  # build and read JSON replies and payloads
import os  # patch environment variables such as the API key
import unittest  # standard test framework
from pathlib import Path  # read the source files of llm.py and llm_chat.py
from unittest import mock  # replace the network, database and input calls with fakes

from ddi_backend import composer, llm, llm_chat  # the three modules under test
from ddi_backend.checker import Checker  # deterministic checker that produces the pair results to explain
# the explain-mode classes and functions under test
from ddi_backend.llm import AnthropicProvider, Explainer, Provider, Settings, provider_from_env, validate_explanation
from ddi_backend.retrieval import context_question  # builds a clarifying question about missing patient context
from tests import support  # helper that opens the real database


class StubProvider(Provider):  # a fake model provider that returns a canned reply or raises a canned error
    name = "stub"  # provider name shown in results

    def __init__(self, reply=None, error=None):  # reply: what to return; error: message to raise instead
        self.reply, self.error, self.calls = reply, error, []  # store them and start an empty call log

    def complete(self, messages):  # the method the explainer calls instead of a real API
        self.calls.append(messages)  # record that a call happened and what was sent
        if self.error:  # if an error was configured
            raise RuntimeError(self.error)  # raise it, simulating a provider failure
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)  # text as is, objects as JSON


class LlmTestCase(unittest.TestCase):  # base class: a real database, a checker and the list of drug names
    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.conn = support.open_real()  # open the real database
        cls.checker = Checker(cls.conn)  # deterministic checker over it
        cls.names = [r[0] for r in cls.conn.execute("SELECT name FROM drugs")]  # every known drug name

    @classmethod
    def tearDownClass(cls):  # runs once after the class
        cls.conn.close()  # close the shared connection

    def pair(self, current, new, **kwargs):  # helper: check one current drug against one new drug
        return self.checker.check([current], new, **kwargs)["results"][0]  # return the single pair result

    @staticmethod
    def good_reply(result, summary=None):  # helper: build a valid structured model reply for a result
        record = result["records"][0]  # the stored record the reply must be based on
        # one claim per checked field, copying the value from the record
        claims = [{"record_id": record["id"], "field": f, "value": record[f]} for f in ("severity", "mechanism", "effect")]
        return {"summary": summary or (  # use the given summary, or build one from the record
            f"{record['drug_a']} with {record['drug_b']} is documented as a {record['severity']} interaction. "
            f"Effect: {record['effect']}."),
            "claims": claims, "clarifying_question": None}  # claims list and no clarifying question


class TestSettings(unittest.TestCase):  # tests for the two user settings
    def test_values_are_checked(self):  # invalid settings are rejected
        self.assertEqual(Settings().context_level, "full_context")  # the default context level
        Settings("names_only", "brief")  # a valid combination is accepted
        with self.assertRaises(ValueError):  # an unknown context level is rejected
            Settings(context_level="everything")
        with self.assertRaises(ValueError):  # an unknown detail level is rejected
            Settings(detail_level="verbose")


class TestProvider(unittest.TestCase):  # tests for choosing and using the real provider
    def test_no_key_means_deterministic_only(self):  # no key, no provider
        self.assertIsNone(provider_from_env({}))  # no key at all gives None
        self.assertIsNone(provider_from_env({"ANTHROPIC_API_KEY": ""}))  # an empty key also gives None

    def test_key_and_model_come_from_the_environment(self):  # settings are read from environment variables
        provider = provider_from_env({"ANTHROPIC_API_KEY": "k", "ANTHROPIC_MODEL": "m"})  # both set
        self.assertEqual((provider.api_key, provider.model), ("k", "m"))  # both were used
        self.assertEqual(provider_from_env({"ANTHROPIC_API_KEY": "k"}).model, "claude-sonnet-4-5")  # default model

    def test_no_key_is_stored_in_code(self):  # no secret key is written in the source
        source = Path(llm.__file__).read_text(encoding="utf-8") + Path(llm_chat.__file__).read_text(encoding="utf-8")  # both files
        self.assertNotIn("sk-", source)  # no key-looking prefix appears

    def test_anthropic_provider_uses_urllib_and_sends_no_tools(self):  # the request carries no tools
        response = io.BytesIO(json.dumps({"content": [{"type": "text", "text": "{}"}]}).encode())  # fake response
        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response) as urlopen:  # fake the network call
            reply = AnthropicProvider("test-key", "test-model").complete([
                {"role": "system", "content": "rules"}, {"role": "user", "content": "x"}])  # one call
        self.assertEqual(reply, "{}")  # the reply text was extracted
        payload = json.loads(urlopen.call_args.args[0].data)  # the JSON body that was sent
        self.assertEqual(set(payload), {"model", "max_tokens", "system", "messages"})  # Anthropic request, no tools
        self.assertEqual(payload["model"], "test-model")  # the model name was sent
        self.assertEqual(payload["system"], "rules")  # system instructions use a separate field


class TestTemplateComposer(LlmTestCase):  # tests for the no-model text builder
    def test_standard_text_uses_every_stored_field_as_written(self):  # standard text carries all stored fields
        result = self.pair("Warfarin", "Amiodarone")  # a documented pair
        text = composer.compose_pair(result, "standard")  # build the standard-detail text
        record = result["records"][0]  # the stored record
        for needle in ("Major", record["mechanism"], record["effect"], "Reduce warfarin dose by 30-50%",
                       record["rationale"], record["reference"], "ddi:20"):  # things the text must contain
            self.assertIn(needle, text)
        self.assertIn("management action", text)  # the dose-reduction advice is labelled a management action

    def test_detail_levels_grow(self):  # more detail means more text
        result = self.pair("Warfarin", "Ibuprofen")  # a documented pair
        brief, standard, detailed = (composer.compose_pair(result, d) for d in composer.DETAIL_LEVELS)  # all three levels
        self.assertLess(len(brief), len(standard))  # brief is shorter than standard
        self.assertLess(len(standard), len(detailed))  # standard is shorter than detailed
        self.assertIn(result["records"][0]["rationale"], detailed)  # the rationale appears in detailed text
        self.assertNotIn(result["records"][0]["rationale"], brief)  # and not in brief text

    def test_alternative_warning_is_shown(self):  # a risky suggested alternative is flagged
        text = composer.compose_pair(self.pair("Warfarin", "Ibuprofen"), "standard")  # build the text
        self.assertIn("Warning: the suggested alternative Acetaminophen", text)  # acetaminophen itself interacts

    def test_conflict_is_stated_and_flagged(self):  # a repeated pair with two severities is explained
        text = composer.compose_pair(self.pair("Clopidogrel", "Omeprazole"), "standard")  # build the text
        self.assertIn("Major / Moderate", text)  # both severities are shown
        self.assertIn("pharmacist review", text)  # and review is recommended

    def test_doses_are_echoed_and_never_evaluated(self):  # entered doses are repeated but not judged
        result = self.checker.check([{"name": "Warfarin", "dose": "5 mg"}], "Ibuprofen")["results"][0]  # with a dose
        text = composer.compose_pair(result, "detailed")  # build the text
        self.assertIn("Warfarin 5 mg", text)  # the dose is echoed
        self.assertIn("does not evaluate or recommend", text)  # with a statement that it is not evaluated

    def test_absent_and_unknown_pairs_are_never_safe(self):  # "not found" is never "safe"
        for current, new, expected in (("Warfarin", "Docusate", "not documented in our data"),  # known, no row
                                       ("Cabergoline", "Rucaparib", "not a drug name in our data")):  # unknown drugs
            with self.subTest(pair=(current, new)):  # report each pair separately
                text = composer.compose_pair(self.pair(current, new), "standard")  # build the text
                self.assertIn(expected, text)  # the right "not found" wording is used
                self.assertIn("does not mean the combination is safe", text)  # with the safety caveat

    def test_whole_check_ends_with_the_disclaimer(self):  # the full text ends with the disclaimer
        out = self.checker.check(["Warfarin", "Docusate"], "Ibuprofen")  # one documented pair and one absent pair
        text = composer.compose_check(out)  # build the whole text
        self.assertTrue(text.index("Major") < text.index("not documented"))  # Major first
        self.assertIn("not medical advice", text)  # the disclaimer is present

    def test_avoid_text_lists_by_severity_and_says_classes_are_missing(self):  # text for the avoid list
        text = composer.compose_avoid(self.checker.avoid("Warfarin"))  # build the text for warfarin
        lines = [l for l in text.splitlines() if l.startswith("- ")]  # the list item lines
        self.assertIn("Major", lines[0])  # the first item is Major
        self.assertIn("classes are not in our data", text)  # drug classes are not covered


class TestExplainer(LlmTestCase):  # tests for the language-model explain mode and its safeguards
    def test_deterministic_only_mode_is_complete(self):  # no provider still gives a full answer
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        out = Explainer(None).explain_pair(result)  # explain with no provider
        self.assertEqual(out["generated_by"], "template")  # the text came from the template
        self.assertIn(result["records"][0]["effect"], out["text"])  # it contains the stored effect
        self.assertEqual([r["id"] for r in out["records"]], ["ddi:12"])  # text sits next to the source record
        self.assertIn("deterministic-only", out["fallback_reason"])  # the reason names this mode

    def test_valid_model_reply_is_used_and_shown_with_its_source(self):  # a valid reply is accepted
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        provider = StubProvider(self.good_reply(result))  # a stub returning a valid reply
        out = Explainer(provider, known_drug_names=self.names).explain_pair(result)  # explain with it
        self.assertEqual(out["generated_by"], "language_model")  # the model text was used
        self.assertIn("documented as a Major interaction", out["text"])  # the summary text is present
        self.assertIn("Source: ddi:12", out["text"])  # shown with its source
        self.assertEqual([r["id"] for r in out["records"]], ["ddi:12"])  # the source record is attached
        self.assertEqual(len(provider.calls), 1)  # exactly one model call was made

    def test_every_claim_points_to_a_record_id(self):  # claims must cite the record
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        out = Explainer(StubProvider(self.good_reply(result)), known_drug_names=self.names).explain_pair(result)  # explain
        self.assertTrue(all(c["record_id"] == "ddi:12" for c in out["claims"]))  # every claim cites ddi:12

    def test_bad_replies_fall_back_to_the_template(self):  # every kind of bad reply is rejected
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        good = self.good_reply(result)  # a valid reply to modify
        record = result["records"][0]  # the stored record
        bad = {  # label -> an invalid reply built by changing the valid one
            "not json": "The answer is Major.",  # plain text instead of JSON
            "extra key": dict(good, advice="stop taking it"),  # an unexpected key
            "unknown record": dict(good, claims=[{"record_id": "ddi:999", "field": "severity", "value": "Major"}]),  # fake ID
            # a claim whose value disagrees with the record
            "claim mismatch": dict(good, claims=[{"record_id": "ddi:12", "field": "severity", "value": "Minor"}] + good["claims"][1:]),
            "missing required claim": dict(good, claims=good["claims"][:1]),  # only one of the required claims
            "wrong severity": dict(good, summary="Warfarin with Fluconazole is a Minor interaction."),  # wrong severity
            "new drug": dict(good, summary="Warfarin with Fluconazole interacts and Aspirin also matters."),  # a new drug
            "new number": dict(good, summary="Warfarin with Fluconazole interacts; reduce the dose by 40%."),  # new number
            "new fact": dict(good, summary="Warfarin with Fluconazole causes hospitalization in elderly patients."),  # new fact
            "reassurance": dict(good, summary="Warfarin with Fluconazole is safe."),  # a safety claim
            "too long": dict(good, summary=("Warfarin with Fluconazole. " * 40)),  # over the length limit
            "wrong types": dict(good, claims="none"),  # claims is not a list
        }
        template = composer.compose_pair(result, "standard")  # the text that should be used instead
        for label, reply in bad.items():  # try each bad reply
            with self.subTest(label):  # report each separately
                out = Explainer(StubProvider(reply), known_drug_names=self.names).explain_pair(result)  # explain with it
                self.assertEqual(out["generated_by"], "template", out.get("fallback_reason"))  # fell back to the template
                self.assertEqual(out["text"], template)  # the text is exactly the template text
                self.assertIn("failed validation", out["fallback_reason"])  # the reason says validation failed
        self.assertTrue(record)  # sanity check that the record exists

    def test_validator_reports_the_new_fact(self):  # the validator names what was added
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        # a valid reply whose summary adds a new fact
        reply = json.dumps(dict(self.good_reply(result), summary="Warfarin with Fluconazole can lead to hospitalization."))
        parsed, problems = validate_explanation(reply, result, self.names)  # validate it
        self.assertIsNone(parsed)  # a rejected reply gives no parsed result
        self.assertTrue(any("hospitalization" in p for p in problems))  # a problem names the new word

    def test_provider_failure_falls_back(self):  # a provider error does not break the answer
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        out = Explainer(StubProvider(error="HTTP 500")).explain_pair(result)  # a provider that always fails
        self.assertEqual(out["generated_by"], "template")  # fell back to the template
        self.assertIn("language model unavailable", out["fallback_reason"])  # the reason says so

    def test_no_record_means_no_api_call(self):  # no evidence means the model is never called
        provider = StubProvider({})  # a stub that records its calls
        for current, new in (("Warfarin", "Docusate"), ("Cabergoline", "Rucaparib")):  # an absent pair and an unknown pair
            out = Explainer(provider).explain_pair(self.pair(current, new))  # explain it
            self.assertEqual(out["generated_by"], "template")  # the template was used
            self.assertIn("no language model call", out["fallback_reason"])  # the reason says no call was made
        self.assertEqual(provider.calls, [])  # the provider was never called
        # the absent-pair text says "not documented in our data"
        self.assertIn("not documented in our data", Explainer(provider).explain_pair(self.pair("Warfarin", "Docusate"))["text"])

    def test_model_sees_only_the_pair_and_its_records(self):  # what is sent to the model is limited
        # a result with a dose, plus another medication ("Zinc")
        result = self.checker.check([{"name": "Warfarin", "dose": "5 mg"}, "Zinc"], "Ibuprofen")["results"][0]
        for level, expect_doses in (("names_only", False), ("full_context", True)):  # the two context levels
            with self.subTest(level):  # report each level separately
                messages = llm.build_explain_messages(result, Settings(level, "standard"), ["Zinc"])  # build the prompt
                payload = json.loads(messages[1]["content"])  # the JSON the model receives
                self.assertEqual(set(payload) >= {"pair", "records", "detail_level"}, True)  # basic keys are always there
                self.assertEqual("doses_as_entered" in payload, expect_doses)  # doses only at full context
                self.assertEqual("other_medications" in payload, expect_doses)  # other medications only at full context
                self.assertEqual([r["id"] for r in payload["records"]], ["ddi:1"])  # only the pair's own record
        system = messages[0]["content"].casefold()  # the system instruction, lower-cased
        for phrase in ("no tools", "no internet", "only permitted source"):  # required rules in the instruction
            self.assertIn(phrase, system)

    def test_detail_level_reaches_the_model_and_the_validator(self):  # detail level limits the allowed length
        result = self.pair("Warfarin", "Fluconazole")  # a documented pair
        long_summary = "Warfarin with Fluconazole is documented as a Major interaction. " * 6  # ~390 characters
        reply = self.good_reply(result, long_summary)  # a valid reply with a long summary
        brief = Explainer(StubProvider(reply), Settings("full_context", "brief"), self.names).explain_pair(result)  # brief
        standard = Explainer(StubProvider(reply), Settings("full_context", "standard"), self.names).explain_pair(result)
        self.assertEqual(brief["generated_by"], "template")  # too long for brief, so it fell back
        self.assertEqual(standard["generated_by"], "language_model")  # fine for standard, so it was used

    def test_explain_check_covers_every_pair(self):  # explaining a whole check handles every pair
        out = self.checker.check(["Warfarin", "Docusate"], "Ibuprofen")  # one documented pair and one absent pair
        provider = StubProvider(self.good_reply(out["results"][0]))  # a stub with a valid reply for the first
        explained = Explainer(provider, known_drug_names=self.names).explain_check(out)  # explain every pair
        self.assertEqual([e["status"] for e in explained], ["documented_interaction", "no_documented_interaction"])  # both
        self.assertEqual(len(provider.calls), 1)  # only the documented pair used a model call


class TestPatientContext(unittest.TestCase):  # tests for the clarifying question about missing context
    def test_missing_context_gets_a_question_not_an_assumption(self):  # ask, do not assume
        question = context_question("Is warfarin with ibuprofen safe for children?")  # mentions children, no age given
        self.assertIn("children", question)  # the question names the context
        self.assertTrue(question.endswith("pharmacist or clinician."))  # and points to a professional
        self.assertIn("Can you tell me more", question)  # and invites more detail

    def test_no_question_when_the_context_was_given_or_not_mentioned(self):  # no needless questions
        self.assertIsNone(context_question("Is warfarin with ibuprofen safe for children?", {"children": "age 6"}))  # given
        self.assertIsNone(context_question("Do warfarin and ibuprofen interact?"))  # not mentioned


class TestSelectorModeStillWorks(unittest.TestCase):  # confirms the older selector mode is unchanged
    """The existing selector contract is unchanged: key set, types, candidate IDs, stored rows only."""

    RECORD = {"id": "ddi:1", "source": "drug interaction", "severity": "Major", "drug_a": "Warfarin", "drug_b": "Ibuprofen"}

    def test_contract(self):  # valid and invalid selector replies
        self.assertEqual(llm_chat.select_records('{"record_ids":["ddi:1"]}', [self.RECORD]), [self.RECORD])  # valid
        # four invalid replies: prose, unknown ID, extra key, wrong type
        for reply in ("prose", '{"record_ids":["ddi:2"]}', '{"record_ids":["ddi:1"],"x":1}', '{"record_ids":[1]}'):
            with self.subTest(reply=reply):  # report each separately
                with self.assertRaises(RuntimeError):  # each must be rejected
                    llm_chat.select_records(reply, [self.RECORD])

    def test_no_candidates_means_no_api_call(self):  # no candidates, no model call
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}), \
                mock.patch("ddi_backend.llm_chat.open_database", return_value=mock.Mock()), \
                mock.patch("ddi_backend.llm_chat.search_records", return_value=[]), \
                mock.patch("ddi_backend.llm_chat.get_reply") as get_reply, \
                mock.patch("builtins.input", side_effect=iter(["nothing", "/exit"])), \
                contextlib.redirect_stdout(io.StringIO()):
            # the five patches above set a fake key, a fake database, an empty search, a watched model call,
            # scripted input and hidden output
            llm_chat.main([])  # run the chat
        get_reply.assert_not_called()  # the model was never called


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the template composer (ddi_backend/composer.py) and the language model layer
# (ddi_backend/llm.py, llm_chat.py) using stub providers, so there is no network. It proves the deterministic text
# uses stored fields and never calls an absent pair safe, that a model reply is used only when it passes
# validation (record IDs, severity, drug names, no new facts), that any failure falls back to the template, that
# no record means no API call, that settings control what the model sees, and that selector mode is unchanged.
# ASSIGNMENT TOOL: the language model (explain mode and selector mode, with its safeguards), plus the
# deterministic template composer that is the complete fallback when there is no model.
# ---------------------------------------------------------------------------
