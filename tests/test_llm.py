"""Tests for the template composer and the language model modes. No network: every provider is a stub."""

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ddi_backend import composer, llm, llm_chat
from ddi_backend.checker import Checker
from ddi_backend.llm import Explainer, OpenAIProvider, Provider, Settings, provider_from_env, validate_explanation
from ddi_backend.retrieval import context_question
from tests import support


class StubProvider(Provider):
    name = "stub"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, messages):
        self.calls.append(messages)
        if self.error:
            raise RuntimeError(self.error)
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


class LlmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real()
        cls.checker = Checker(cls.conn)
        cls.names = [r[0] for r in cls.conn.execute("SELECT name FROM drugs")]

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def pair(self, current, new, **kwargs):
        return self.checker.check([current], new, **kwargs)["results"][0]

    @staticmethod
    def good_reply(result, summary=None):
        record = result["records"][0]
        claims = [{"record_id": record["id"], "field": f, "value": record[f]} for f in ("severity", "mechanism", "effect")]
        return {"summary": summary or (
            f"{record['drug_a']} with {record['drug_b']} is documented as a {record['severity']} interaction. "
            f"Effect: {record['effect']}."),
            "claims": claims, "clarifying_question": None}


class TestSettings(unittest.TestCase):
    def test_values_are_checked(self):
        self.assertEqual(Settings().context_level, "full_context")
        Settings("names_only", "brief")
        with self.assertRaises(ValueError):
            Settings(context_level="everything")
        with self.assertRaises(ValueError):
            Settings(detail_level="verbose")


class TestProvider(unittest.TestCase):
    def test_no_key_means_deterministic_only(self):
        self.assertIsNone(provider_from_env({}))
        self.assertIsNone(provider_from_env({"OPENAI_API_KEY": ""}))

    def test_key_and_model_come_from_the_environment(self):
        provider = provider_from_env({"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"})
        self.assertEqual((provider.api_key, provider.model), ("k", "m"))
        self.assertEqual(provider_from_env({"OPENAI_API_KEY": "k"}).model, "gpt-4o-mini")

    def test_no_key_is_stored_in_code(self):
        source = Path(llm.__file__).read_text(encoding="utf-8") + Path(llm_chat.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sk-", source)

    def test_openai_provider_uses_urllib_and_sends_no_tools(self):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response) as urlopen:
            reply = OpenAIProvider("test-key", "test-model").complete([{"role": "user", "content": "x"}])
        self.assertEqual(reply, "{}")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(set(payload), {"model", "messages", "response_format"})  # no tools, no browsing
        self.assertEqual(payload["model"], "test-model")


class TestTemplateComposer(LlmTestCase):
    def test_standard_text_uses_every_stored_field_as_written(self):
        result = self.pair("Warfarin", "Amiodarone")
        text = composer.compose_pair(result, "standard")
        record = result["records"][0]
        for needle in ("Major", record["mechanism"], record["effect"], "Reduce warfarin dose by 30-50%",
                       record["rationale"], record["reference"], "ddi:20"):
            self.assertIn(needle, text)
        self.assertIn("management action", text)

    def test_detail_levels_grow(self):
        result = self.pair("Warfarin", "Ibuprofen")
        brief, standard, detailed = (composer.compose_pair(result, d) for d in composer.DETAIL_LEVELS)
        self.assertLess(len(brief), len(standard))
        self.assertLess(len(standard), len(detailed))
        self.assertIn(result["records"][0]["rationale"], detailed)
        self.assertNotIn(result["records"][0]["rationale"], brief)

    def test_alternative_warning_is_shown(self):
        text = composer.compose_pair(self.pair("Warfarin", "Ibuprofen"), "standard")
        self.assertIn("Warning: the suggested alternative Acetaminophen", text)

    def test_conflict_is_stated_and_flagged(self):
        text = composer.compose_pair(self.pair("Clopidogrel", "Omeprazole"), "standard")
        self.assertIn("Major / Moderate", text)
        self.assertIn("pharmacist review", text)

    def test_doses_are_echoed_and_never_evaluated(self):
        result = self.checker.check([{"name": "Warfarin", "dose": "5 mg"}], "Ibuprofen")["results"][0]
        text = composer.compose_pair(result, "detailed")
        self.assertIn("Warfarin 5 mg", text)
        self.assertIn("does not evaluate or recommend", text)

    def test_absent_and_unknown_pairs_are_never_safe(self):
        for current, new, expected in (("Warfarin", "Docusate", "not documented in our data"),
                                       ("Cabergoline", "Rucaparib", "not a drug name in our data")):
            with self.subTest(pair=(current, new)):
                text = composer.compose_pair(self.pair(current, new), "standard")
                self.assertIn(expected, text)
                self.assertIn("does not mean the combination is safe", text)

    def test_whole_check_ends_with_the_disclaimer(self):
        out = self.checker.check(["Warfarin", "Docusate"], "Ibuprofen")
        text = composer.compose_check(out)
        self.assertTrue(text.index("Major") < text.index("not documented"))  # Major first
        self.assertIn("not medical advice", text)

    def test_avoid_text_lists_by_severity_and_says_classes_are_missing(self):
        text = composer.compose_avoid(self.checker.avoid("Warfarin"))
        lines = [l for l in text.splitlines() if l.startswith("- ")]
        self.assertIn("Major", lines[0])
        self.assertIn("classes are not in our data", text)


class TestExplainer(LlmTestCase):
    def test_deterministic_only_mode_is_complete(self):
        result = self.pair("Warfarin", "Fluconazole")
        out = Explainer(None).explain_pair(result)
        self.assertEqual(out["generated_by"], "template")
        self.assertIn(result["records"][0]["effect"], out["text"])
        self.assertEqual([r["id"] for r in out["records"]], ["ddi:12"])  # text sits next to the source record
        self.assertIn("deterministic-only", out["fallback_reason"])

    def test_valid_model_reply_is_used_and_shown_with_its_source(self):
        result = self.pair("Warfarin", "Fluconazole")
        provider = StubProvider(self.good_reply(result))
        out = Explainer(provider, known_drug_names=self.names).explain_pair(result)
        self.assertEqual(out["generated_by"], "language_model")
        self.assertIn("documented as a Major interaction", out["text"])
        self.assertIn("Source: ddi:12", out["text"])
        self.assertEqual([r["id"] for r in out["records"]], ["ddi:12"])
        self.assertEqual(len(provider.calls), 1)

    def test_every_claim_points_to_a_record_id(self):
        result = self.pair("Warfarin", "Fluconazole")
        out = Explainer(StubProvider(self.good_reply(result)), known_drug_names=self.names).explain_pair(result)
        self.assertTrue(all(c["record_id"] == "ddi:12" for c in out["claims"]))

    def test_bad_replies_fall_back_to_the_template(self):
        result = self.pair("Warfarin", "Fluconazole")
        good = self.good_reply(result)
        record = result["records"][0]
        bad = {
            "not json": "The answer is Major.",
            "extra key": dict(good, advice="stop taking it"),
            "unknown record": dict(good, claims=[{"record_id": "ddi:999", "field": "severity", "value": "Major"}]),
            "claim mismatch": dict(good, claims=[{"record_id": "ddi:12", "field": "severity", "value": "Minor"}] + good["claims"][1:]),
            "missing required claim": dict(good, claims=good["claims"][:1]),
            "wrong severity": dict(good, summary="Warfarin with Fluconazole is a Minor interaction."),
            "new drug": dict(good, summary="Warfarin with Fluconazole interacts and Aspirin also matters."),
            "new number": dict(good, summary="Warfarin with Fluconazole interacts; reduce the dose by 40%."),
            "new fact": dict(good, summary="Warfarin with Fluconazole causes hospitalization in elderly patients."),
            "reassurance": dict(good, summary="Warfarin with Fluconazole is safe."),
            "too long": dict(good, summary=("Warfarin with Fluconazole. " * 40)),
            "wrong types": dict(good, claims="none"),
        }
        template = composer.compose_pair(result, "standard")
        for label, reply in bad.items():
            with self.subTest(label):
                out = Explainer(StubProvider(reply), known_drug_names=self.names).explain_pair(result)
                self.assertEqual(out["generated_by"], "template", out.get("fallback_reason"))
                self.assertEqual(out["text"], template)
                self.assertIn("failed validation", out["fallback_reason"])
        self.assertTrue(record)

    def test_validator_reports_the_new_fact(self):
        result = self.pair("Warfarin", "Fluconazole")
        reply = json.dumps(dict(self.good_reply(result), summary="Warfarin with Fluconazole can lead to hospitalization."))
        parsed, problems = validate_explanation(reply, result, self.names)
        self.assertIsNone(parsed)
        self.assertTrue(any("hospitalization" in p for p in problems))

    def test_provider_failure_falls_back(self):
        result = self.pair("Warfarin", "Fluconazole")
        out = Explainer(StubProvider(error="HTTP 500")).explain_pair(result)
        self.assertEqual(out["generated_by"], "template")
        self.assertIn("language model unavailable", out["fallback_reason"])

    def test_no_record_means_no_api_call(self):
        provider = StubProvider({})
        for current, new in (("Warfarin", "Docusate"), ("Cabergoline", "Rucaparib")):
            out = Explainer(provider).explain_pair(self.pair(current, new))
            self.assertEqual(out["generated_by"], "template")
            self.assertIn("no language model call", out["fallback_reason"])
        self.assertEqual(provider.calls, [])
        self.assertIn("not documented in our data", Explainer(provider).explain_pair(self.pair("Warfarin", "Docusate"))["text"])

    def test_model_sees_only_the_pair_and_its_records(self):
        result = self.checker.check([{"name": "Warfarin", "dose": "5 mg"}, "Zinc"], "Ibuprofen")["results"][0]
        for level, expect_doses in (("names_only", False), ("full_context", True)):
            with self.subTest(level):
                messages = llm.build_explain_messages(result, Settings(level, "standard"), ["Zinc"])
                payload = json.loads(messages[1]["content"])
                self.assertEqual(set(payload) >= {"pair", "records", "detail_level"}, True)
                self.assertEqual("doses_as_entered" in payload, expect_doses)
                self.assertEqual("other_medications" in payload, expect_doses)
                self.assertEqual([r["id"] for r in payload["records"]], ["ddi:1"])
        system = messages[0]["content"].casefold()
        for phrase in ("no tools", "no internet", "only permitted source"):
            self.assertIn(phrase, system)

    def test_detail_level_reaches_the_model_and_the_validator(self):
        result = self.pair("Warfarin", "Fluconazole")
        long_summary = "Warfarin with Fluconazole is documented as a Major interaction. " * 6  # ~390 characters
        reply = self.good_reply(result, long_summary)
        brief = Explainer(StubProvider(reply), Settings("full_context", "brief"), self.names).explain_pair(result)
        standard = Explainer(StubProvider(reply), Settings("full_context", "standard"), self.names).explain_pair(result)
        self.assertEqual(brief["generated_by"], "template")
        self.assertEqual(standard["generated_by"], "language_model")

    def test_explain_check_covers_every_pair(self):
        out = self.checker.check(["Warfarin", "Docusate"], "Ibuprofen")
        provider = StubProvider(self.good_reply(out["results"][0]))
        explained = Explainer(provider, known_drug_names=self.names).explain_check(out)
        self.assertEqual([e["status"] for e in explained], ["documented_interaction", "no_documented_interaction"])
        self.assertEqual(len(provider.calls), 1)


class TestPatientContext(unittest.TestCase):
    def test_missing_context_gets_a_question_not_an_assumption(self):
        question = context_question("Is warfarin with ibuprofen safe for children?")
        self.assertIn("children", question)
        self.assertTrue(question.endswith("pharmacist or clinician."))
        self.assertIn("Can you tell me more", question)

    def test_no_question_when_the_context_was_given_or_not_mentioned(self):
        self.assertIsNone(context_question("Is warfarin with ibuprofen safe for children?", {"children": "age 6"}))
        self.assertIsNone(context_question("Do warfarin and ibuprofen interact?"))


class TestSelectorModeStillWorks(unittest.TestCase):
    """The existing selector contract is unchanged: key set, types, candidate IDs, stored rows only."""

    RECORD = {"id": "ddi:1", "source": "drug interaction", "severity": "Major", "drug_a": "Warfarin", "drug_b": "Ibuprofen"}

    def test_contract(self):
        self.assertEqual(llm_chat.select_records('{"record_ids":["ddi:1"]}', [self.RECORD]), [self.RECORD])
        for reply in ("prose", '{"record_ids":["ddi:2"]}', '{"record_ids":["ddi:1"],"x":1}', '{"record_ids":[1]}'):
            with self.subTest(reply=reply):
                with self.assertRaises(RuntimeError):
                    llm_chat.select_records(reply, [self.RECORD])

    def test_no_candidates_means_no_api_call(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
                mock.patch("ddi_backend.llm_chat.open_database", return_value=mock.Mock()), \
                mock.patch("ddi_backend.llm_chat.search_records", return_value=[]), \
                mock.patch("ddi_backend.llm_chat.get_reply") as get_reply, \
                mock.patch("builtins.input", side_effect=iter(["nothing", "/exit"])), \
                contextlib.redirect_stdout(io.StringIO()):
            llm_chat.main([])
        get_reply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
