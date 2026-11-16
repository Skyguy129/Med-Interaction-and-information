"""Tests for question handling: the Meeting 8 failure types, the RAG handout questions and
symptom mapping. Written before the old keyword search was replaced; the warfarin question
failed against it (see TestWarfarinQuestion). No embeddings are needed for anything here."""

import tempfile
import unittest
from pathlib import Path

from ddi_backend import database_context, ingest
from ddi_backend.common import NOT_SAFE_NOTE, severity_rank
from ddi_backend.retrieval import QuestionAnswerer, detect_negation, filter_by_population
from ddi_backend.checker import Checker
from tests import support
from tests.support import ddi_row

WARFARIN_QUESTION = "Which pain reliever should I not take with warfarin?"


class FullDbTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real_full()
        cls.checker = Checker(cls.conn)
        cls.qa = QuestionAnswerer(cls.checker, hpo_conn=cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def ask(self, question, **kwargs):
        return self.qa.ask(question, **kwargs)


class TestWarfarinQuestion(FullDbTestCase):
    """The adversarial Meeting 8 question. Embedding similarity ranked acetaminophen first
    because it ignored 'not'; the old LIKE search never reached warfarin at all."""

    def test_search_records_returns_warfarins_interactions_most_severe_first(self):
        records = database_context.search_records(self.conn, WARFARIN_QUESTION)
        self.assertTrue(records)
        self.assertTrue(all("Warfarin" in (r["drug_a"], r["drug_b"]) for r in records), [r["id"] for r in records])
        self.assertTrue(all(r["id"].startswith("ddi:") for r in records))  # no HPO distractors
        ranks = [severity_rank(r["severity"]) for r in records]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(records[0]["severity"], "Major")

    def test_acetaminophen_is_never_first_because_of_similarity(self):
        answer = self.ask(WARFARIN_QUESTION)
        self.assertEqual(answer["intent"], "avoid")
        names = [e["drug"] for e in answer["avoid"]["avoid"]]
        self.assertNotEqual(names[0], "Acetaminophen")
        self.assertEqual(answer["avoid"]["avoid"][0]["severity"], "Major")
        ranks = [severity_rank(e["severity"]) for e in answer["avoid"]["avoid"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertIn("classes are not in our data", " ".join(answer["notes"]))
        self.assertIn("avoid_list", answer["rules_fired"])
        self.assertIn("negation_detected", answer["rules_fired"])
        self.assertNotIn("similarity", str(answer["rules_fired"]))

    def test_class_wording_does_not_narrow_the_list(self):
        with_class = [e["drug"] for e in self.ask(WARFARIN_QUESTION)["avoid"]["avoid"]]
        without = [e["drug"] for e in self.ask("What should I not take with warfarin?")["avoid"]["avoid"]]
        self.assertEqual(with_class, without)

    def test_negation_phrasings(self):
        for text in ("What should I avoid with warfarin?", "Anything to steer clear of with warfarin? avoid it",
                     "Which drugs should I not combine with warfarin?",
                     "Take everything except what interacts with warfarin",
                     "Which drug is risky without a doctor with warfarin"):
            with self.subTest(text=text):
                self.assertTrue(detect_negation(text), text)
        self.assertFalse(detect_negation("What interacts with warfarin?"))
        self.assertFalse(detect_negation("Do warfarin and ibuprofen interact?"))

    def test_avoid_without_a_drug_asks_which_drug(self):
        answer = self.ask("What should I not take?")
        self.assertEqual(answer["status"], "needs_clarification")
        self.assertNotIn("avoid", answer)


class TestMeeting8FailureTypes(FullDbTestCase):
    def test_ambiguity_ms_gets_a_question_not_a_guess(self):
        answer = self.ask("What does MS interact with?")
        self.assertEqual(answer["status"], "needs_clarification")
        self.assertIn("MS", answer["clarifications"][0]["input"])
        self.assertIn("abbreviation", answer["clarifications"][0]["question"])
        self.assertNotIn("results", answer)
        self.assertNotIn("avoid", answer)

    def test_unknown_brand_name_asks_instead_of_guessing(self):
        answer = self.ask("Can I take Advil with warfarin?")
        self.assertEqual(answer["clarifications"][0]["input"], "Advil")
        self.assertEqual(answer["clarifications"][0]["status"], "unknown_drug")
        result = answer["results"][0]
        self.assertEqual(result["status"], "unknown_drug")
        self.assertNotIn("records", result)
        self.assertIn(NOT_SAFE_NOTE, result["notes"])

    def test_single_unresolved_name_gets_only_a_question(self):
        answer = self.ask("What does Advil interact with?")
        self.assertEqual(answer["status"], "needs_clarification")
        self.assertNotIn("results", answer)

    def test_typo_gets_did_you_mean_never_auto_resolved(self):
        answer = self.ask("Does Warfrin interact with Ibuprofen?")
        self.assertIn("Warfarin", answer["clarifications"][0]["did_you_mean"])
        self.assertEqual(answer["results"][0]["status"], "unknown_drug")
        self.assertNotIn("records", answer["results"][0])  # we did not quietly answer for Warfarin

    def test_context_population_is_not_covered_and_nothing_is_filtered(self):
        answer = self.ask("Is warfarin with ibuprofen safe for children?")
        self.assertEqual(answer["results"][0]["status"], "documented_interaction")
        text = " ".join(answer["notes"])
        self.assertIn("children", text)
        self.assertIn("does not cover", text)
        self.assertIn("population scope", text)
        self.assertEqual(len(answer["results"][0]["records"]), 1)  # no record was dropped for 'children'

    def test_population_filter_applies_only_where_a_record_carries_a_scope(self):
        records = [{"id": "a", "population": "pediatric"}, {"id": "b", "population": "adult"}, {"id": "c"}]
        kept, note = filter_by_population(records, "pediatric")
        self.assertEqual([r["id"] for r in kept], ["a"])
        self.assertIsNone(note)
        kept, note = filter_by_population([{"id": "x"}, {"id": "y"}], "pediatric")
        self.assertEqual([r["id"] for r in kept], ["x", "y"])
        self.assertIn("does not cover", note)

    def test_topics_outside_an_interaction_dataset_are_not_available(self):
        for text in ("What are the side effects of warfarin?", "What is warfarin used for?",
                     "What are the contraindications of ibuprofen?", "Does warfarin interact with food?",
                     "What are the ingredients in ibuprofen?"):
            with self.subTest(text=text):
                answer = self.ask(text)
                self.assertTrue(any("not available in our data" in n for n in answer["notes"]), answer["notes"])

    def test_dose_questions_are_never_answered(self):
        answer = self.ask("How much warfarin should I take with ibuprofen?")
        self.assertTrue(any("dose" in n and "never" in n for n in answer["notes"]))
        self.assertEqual(answer["results"][0]["status"], "documented_interaction")


class TestRagHandoutQuestions(FullDbTestCase):
    def first(self, question):
        answer = self.ask(question)
        return answer, answer["results"][0]

    def test_direct_question(self):
        _, result = self.first("Do warfarin and ibuprofen interact, and how severe is it?")
        self.assertEqual((result["status"], result["severity"]), ("documented_interaction", "Major"))
        self.assertEqual(result["source"]["record_ids"], ["ddi:1"])

    def test_paraphrase_retrieves_the_same_record(self):
        _, direct = self.first("Do warfarin and ibuprofen interact, and how severe is it?")
        _, reworded = self.first("Is it dangerous to take ibuprofen while on warfarin?")
        self.assertEqual(direct["source"]["record_ids"], reworded["source"]["record_ids"])

    def test_reworded_queries_for_a_repeated_pair_return_every_row(self):
        queries = ["Does taking clopidogrel with omeprazole reduce the antiplatelet effect?",
                   "clopidogrel omeprazole", "Omeprazole and Clopidogrel together?"]
        ids = [self.first(q)[1]["source"]["record_ids"] for q in queries]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])
        self.assertEqual(len(ids[0]), 2)

    def test_multi_part_question_returns_whole_record(self):
        _, result = self.first("What are the severity, clinical effect and mechanism of warfarin with fluconazole?")
        record = result["records"][0]
        self.assertEqual(record["id"], "ddi:12")
        self.assertEqual(result["severity"], "Major")
        for field in ("mechanism", "effect", "rationale", "reference", "safer_alternative"):
            self.assertTrue(record[field])
        self.assertEqual(result["effect"], record["effect"])

    def test_absent_pair_cabergoline_rucaparib_is_never_called_safe(self):
        answer, result = self.first("Do cabergoline and rucaparib interact?")
        self.assertEqual(result["status"], "unknown_drug")
        self.assertIn(NOT_SAFE_NOTE, result["notes"])
        self.assertNotIn("records", result)
        self.assertNotIn("severity", {k for k, v in result.items() if v})
        self.assertEqual(answer["status"], "answered")

    def test_two_known_drugs_with_no_row_is_no_documented_interaction(self):
        _, result = self.first("Do warfarin and docusate interact?")
        self.assertEqual(result["status"], "no_documented_interaction")
        self.assertIn(NOT_SAFE_NOTE, result["notes"])

    def test_single_drug_question_lists_its_interactions_most_severe_first(self):
        answer = self.ask("What interacts with warfarin?")
        self.assertEqual(answer["intent"], "drug")
        ranks = [severity_rank(r["severity"]) for r in answer["interactions"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertIn("ddi:1", [r["id"] for r in answer["interactions"]])

    def test_generic_question_with_no_drug_is_not_answered_from_guesses(self):
        answer = self.ask("What are interactions?")
        self.assertEqual(answer["status"], "no_match")
        self.assertNotIn("results", answer)


class TestSymptomMapping(FullDbTestCase):
    def test_symptom_word_matches_the_effect_text_with_hpo_only_helping(self):
        answer = self.ask("Which drugs cause bleeding?")
        self.assertEqual(answer["intent"], "symptom")
        self.assertTrue(answer["interactions"])
        self.assertTrue(all(i["id"].startswith("ddi:") for i in answer["interactions"]))
        for item in answer["interactions"]:
            self.assertIn(item["matched_on"], ("effect text", "mechanism text"))
        self.assertIn("matched on the effect text", " ".join(answer["notes"]))
        self.assertTrue(answer["phenotype_records"])
        self.assertTrue(all(p["id"].startswith("hpo:") for p in answer["phenotype_records"]))
        self.assertIn("vocabulary, not interaction evidence", answer["phenotype_note"])
        ranks = [severity_rank(i["severity"]) for i in answer["interactions"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_hpo_never_changes_what_a_pair_question_returns(self):
        with_hpo = self.qa.ask("Do warfarin and ibuprofen interact?")
        without = QuestionAnswerer(self.checker, hpo_conn=None).ask("Do warfarin and ibuprofen interact?")
        self.assertEqual(with_hpo["results"][0]["source"], without["results"][0]["source"])
        self.assertNotIn("phenotype_records", with_hpo)

    def test_symptom_mapping_can_be_switched_off(self):
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, symptom_mapping=False)
        answer = qa.ask("Which drugs cause bleeding?")
        self.assertNotIn("phenotype_records", answer)
        self.assertIn("Symptom mapping is off", " ".join(answer["notes"]))
        self.assertTrue(answer["interactions"])  # the literal word is still matched in the effect text

    def test_hpo_expansion_adds_synonyms_children_and_layperson_terms(self):
        from ddi_backend import hpo
        expansion = hpo.expand_symptom(self.conn, "dizziness")
        phrases = {p["phrase"] for p in expansion["phrases"]}
        self.assertIn("vertigo", phrases)            # label of the matched term
        self.assertIn("dizzy spell", phrases)         # layperson synonym
        self.assertIn("paroxysmal vertigo", phrases)  # child term from hpo_edges
        self.assertTrue(all(p["hp_id"].startswith("HP:") for p in expansion["phrases"]))

    def test_symptom_lists_and_plain_sentences_are_not_mistaken_for_drug_names(self):
        for text in ("Which drugs cause headache and nausea?", "my heart is racing after a new pill"):
            with self.subTest(text=text):
                answer = self.ask(text)
                self.assertNotIn("clarifications", answer)
                self.assertEqual(answer["intent"], "symptom")

    def test_symptom_drug_question_filters_that_drugs_interactions(self):
        answer = self.ask("Does warfarin cause bleeding?")
        self.assertEqual(answer["intent"], "drug")
        self.assertTrue(answer["interactions"])
        self.assertTrue(all("Warfarin" in (i["drug_a"], i["drug_b"]) for i in answer["interactions"]))

    def test_without_hpo_tables_symptom_mapping_switches_itself_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = ingest.open_readonly(
                support.build_db(Path(tmp), [ddi_row("A", "B", "Major", effect="Increased bleeding risk")]),
                ingest.CORE_TABLES)
            try:
                qa = QuestionAnswerer(Checker(conn), hpo_conn=conn)
                answer = qa.ask("Which drugs cause bleeding?")
                self.assertEqual([i["id"] for i in answer["interactions"]], ["ddi:1"])
                self.assertNotIn("phenotype_records", answer)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
