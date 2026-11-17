# Docstring: says this file tests question handling (Meeting 8 failure types, handout questions, symptom mapping).
"""Tests for question handling: the Meeting 8 failure types, the RAG handout questions and
symptom mapping. Written before the old keyword search was replaced; the warfarin question
failed against it (see TestWarfarinQuestion). No embeddings are needed for anything here."""

import tempfile  # temporary folder for a tiny database
import unittest  # standard test framework
from pathlib import Path  # path handling for that folder

from ddi_backend import database_context, ingest  # record search helper and read-only database access
from ddi_backend.common import NOT_SAFE_NOTE, severity_rank  # shared "not safe" note and severity ordering
from ddi_backend.retrieval import QuestionAnswerer, detect_negation, filter_by_population  # code under test
from ddi_backend.checker import Checker  # the deterministic checker the answerer is built on
from tests import support  # helpers that open the real database or build tiny ones
from tests.support import ddi_row  # builds one fake interaction row

WARFARIN_QUESTION = "Which pain reliever should I not take with warfarin?"  # the adversarial Meeting 8 question


class FullDbTestCase(unittest.TestCase):  # base class: a real database with both interaction and HPO tables
    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.conn = support.open_real_full()  # open the real database including HPO tables
        cls.checker = Checker(cls.conn)  # deterministic checker over it
        cls.qa = QuestionAnswerer(cls.checker, hpo_conn=cls.conn)  # question answerer with symptom mapping enabled

    @classmethod
    def tearDownClass(cls):  # runs once after the class
        cls.conn.close()  # close the shared connection

    def ask(self, question, **kwargs):  # helper: ask the shared answerer a question
        return self.qa.ask(question, **kwargs)  # pass through any options


class TestWarfarinQuestion(FullDbTestCase):  # tests around the "not take with warfarin" question
    """The adversarial Meeting 8 question. Embedding similarity ranked acetaminophen first
    because it ignored 'not'; the old LIKE search never reached warfarin at all."""

    def test_search_records_returns_warfarins_interactions_most_severe_first(self):  # record search works for it
        records = database_context.search_records(self.conn, WARFARIN_QUESTION)  # search with the whole question
        self.assertTrue(records)  # something was found
        # every record involves Warfarin (the IDs are shown if this fails)
        self.assertTrue(all("Warfarin" in (r["drug_a"], r["drug_b"]) for r in records), [r["id"] for r in records])
        self.assertTrue(all(r["id"].startswith("ddi:") for r in records))  # no HPO distractors
        ranks = [severity_rank(r["severity"]) for r in records]  # numeric severity order of the results
        self.assertEqual(ranks, sorted(ranks))  # already sorted, Major first
        self.assertEqual(records[0]["severity"], "Major")  # and the first record is Major

    def test_acetaminophen_is_never_first_because_of_similarity(self):  # answers come from the avoid list, not similarity
        answer = self.ask(WARFARIN_QUESTION)  # ask the question
        self.assertEqual(answer["intent"], "avoid")  # it was understood as an avoid-list question
        names = [e["drug"] for e in answer["avoid"]["avoid"]]  # drug names in the avoid list, in order
        self.assertNotEqual(names[0], "Acetaminophen")  # acetaminophen is not ranked first
        self.assertEqual(answer["avoid"]["avoid"][0]["severity"], "Major")  # the first entry is Major
        ranks = [severity_rank(e["severity"]) for e in answer["avoid"]["avoid"]]  # severity order of the list
        self.assertEqual(ranks, sorted(ranks))  # sorted most severe first
        self.assertIn("classes are not in our data", " ".join(answer["notes"]))  # notes say classes are not covered
        self.assertIn("avoid_list", answer["rules_fired"])  # the avoid-list rule is reported as fired
        self.assertIn("negation_detected", answer["rules_fired"])  # so is the negation rule
        self.assertNotIn("similarity", str(answer["rules_fired"]))  # no similarity step decided the answer

    def test_class_wording_does_not_narrow_the_list(self):  # "pain reliever" does not filter the result
        with_class = [e["drug"] for e in self.ask(WARFARIN_QUESTION)["avoid"]["avoid"]]  # list for the class wording
        without = [e["drug"] for e in self.ask("What should I not take with warfarin?")["avoid"]["avoid"]]  # no class
        self.assertEqual(with_class, without)  # identical lists

    def test_negation_phrasings(self):  # many ways of saying "not" are detected
        for text in ("What should I avoid with warfarin?", "Anything to steer clear of with warfarin? avoid it",
                     "Which drugs should I not combine with warfarin?",
                     "Take everything except what interacts with warfarin",
                     "Which drug is risky without a doctor with warfarin"):  # five negated phrasings
            with self.subTest(text=text):  # report each phrasing separately
                self.assertTrue(detect_negation(text), text)  # negation must be detected
        self.assertFalse(detect_negation("What interacts with warfarin?"))  # no negation here
        self.assertFalse(detect_negation("Do warfarin and ibuprofen interact?"))  # nor here

    def test_avoid_without_a_drug_asks_which_drug(self):  # an avoid question with no drug asks a follow-up
        answer = self.ask("What should I not take?")  # no drug named
        self.assertEqual(answer["status"], "needs_clarification")  # the system asks which drug
        self.assertNotIn("avoid", answer)  # and returns no avoid list


class TestMeeting8FailureTypes(FullDbTestCase):  # one test group per failure type found in Meeting 8
    def test_ambiguity_ms_gets_a_question_not_a_guess(self):  # an ambiguous abbreviation is clarified
        answer = self.ask("What does MS interact with?")  # "MS" could mean several things
        self.assertEqual(answer["status"], "needs_clarification")  # we ask instead of guessing
        self.assertIn("MS", answer["clarifications"][0]["input"])  # the clarification names the input
        self.assertIn("abbreviation", answer["clarifications"][0]["question"])  # and explains it is an abbreviation
        self.assertNotIn("results", answer)  # no pair results were produced
        self.assertNotIn("avoid", answer)  # and no avoid list

    def test_unknown_brand_name_asks_instead_of_guessing(self):  # brand names are unknown, not guessed
        answer = self.ask("Can I take Advil with warfarin?")  # Advil is not in our data
        self.assertEqual(answer["clarifications"][0]["input"], "Advil")  # the clarification is about Advil
        self.assertEqual(answer["clarifications"][0]["status"], "unknown_drug")  # status is unknown_drug
        result = answer["results"][0]  # the pair result
        self.assertEqual(result["status"], "unknown_drug")  # also unknown_drug
        self.assertNotIn("records", result)  # no records were invented
        self.assertIn(NOT_SAFE_NOTE, result["notes"])  # the "not found is not safe" note is attached

    def test_single_unresolved_name_gets_only_a_question(self):  # one unknown name gives only a question
        answer = self.ask("What does Advil interact with?")  # a one-drug question about an unknown drug
        self.assertEqual(answer["status"], "needs_clarification")  # we ask which drug is meant
        self.assertNotIn("results", answer)  # and produce no results

    def test_typo_gets_did_you_mean_never_auto_resolved(self):  # typos are suggested, not silently fixed
        answer = self.ask("Does Warfrin interact with Ibuprofen?")  # a typo of Warfarin
        self.assertIn("Warfarin", answer["clarifications"][0]["did_you_mean"])  # Warfarin is suggested
        self.assertEqual(answer["results"][0]["status"], "unknown_drug")  # the pair stays unknown
        self.assertNotIn("records", answer["results"][0])  # we did not quietly answer for Warfarin

    def test_context_population_is_not_covered_and_nothing_is_filtered(self):  # "children" is not in the data
        answer = self.ask("Is warfarin with ibuprofen safe for children?")  # population context in the question
        self.assertEqual(answer["results"][0]["status"], "documented_interaction")  # the interaction is still shown
        text = " ".join(answer["notes"])  # join the notes for searching
        self.assertIn("children", text)  # the notes name the population
        self.assertIn("does not cover", text)  # and say the data does not cover it
        self.assertIn("population scope", text)  # because no record carries a population scope
        self.assertEqual(len(answer["results"][0]["records"]), 1)  # no record was dropped for 'children'

    def test_population_filter_applies_only_where_a_record_carries_a_scope(self):  # the filter's exact rule
        records = [{"id": "a", "population": "pediatric"}, {"id": "b", "population": "adult"}, {"id": "c"}]  # sample
        kept, note = filter_by_population(records, "pediatric")  # filter for pediatric
        self.assertEqual([r["id"] for r in kept], ["a"])  # only the pediatric record is kept
        self.assertIsNone(note)  # no "not covered" note is needed
        kept, note = filter_by_population([{"id": "x"}, {"id": "y"}], "pediatric")  # records without any scope
        self.assertEqual([r["id"] for r in kept], ["x", "y"])  # none are dropped
        self.assertIn("does not cover", note)  # a note says the data does not cover it

    def test_topics_outside_an_interaction_dataset_are_not_available(self):  # drug-monograph topics are refused
        for text in ("What are the side effects of warfarin?", "What is warfarin used for?",
                     "What are the contraindications of ibuprofen?", "Does warfarin interact with food?",
                     "What are the ingredients in ibuprofen?"):  # topics our data does not hold
            with self.subTest(text=text):  # report each question separately
                answer = self.ask(text)  # ask it
                # a note must say the topic is not available in our data
                self.assertTrue(any("not available in our data" in n for n in answer["notes"]), answer["notes"])

    def test_dose_questions_are_never_answered(self):  # no dose advice is ever given
        answer = self.ask("How much warfarin should I take with ibuprofen?")  # a dose question
        self.assertTrue(any("dose" in n and "never" in n for n in answer["notes"]))  # a note says doses are never given
        self.assertEqual(answer["results"][0]["status"], "documented_interaction")  # the pair is still checked


class TestRagHandoutQuestions(FullDbTestCase):  # the questions from the retrieval handout
    def first(self, question):  # helper: ask and return the answer plus its first pair result
        answer = self.ask(question)  # ask the question
        return answer, answer["results"][0]  # the whole answer and the first result

    def test_direct_question(self):  # a plain pair question
        _, result = self.first("Do warfarin and ibuprofen interact, and how severe is it?")  # ask it
        self.assertEqual((result["status"], result["severity"]), ("documented_interaction", "Major"))  # documented, Major
        self.assertEqual(result["source"]["record_ids"], ["ddi:1"])  # sourced from ddi:1

    def test_paraphrase_retrieves_the_same_record(self):  # different wording, same record
        _, direct = self.first("Do warfarin and ibuprofen interact, and how severe is it?")  # direct wording
        _, reworded = self.first("Is it dangerous to take ibuprofen while on warfarin?")  # reworded
        self.assertEqual(direct["source"]["record_ids"], reworded["source"]["record_ids"])  # same record IDs

    def test_reworded_queries_for_a_repeated_pair_return_every_row(self):  # both rows of a repeated pair come back
        queries = ["Does taking clopidogrel with omeprazole reduce the antiplatelet effect?",
                   "clopidogrel omeprazole", "Omeprazole and Clopidogrel together?"]  # three phrasings
        ids = [self.first(q)[1]["source"]["record_ids"] for q in queries]  # record IDs for each phrasing
        self.assertEqual(ids[0], ids[1])  # first two phrasings agree
        self.assertEqual(ids[1], ids[2])  # last two phrasings agree
        self.assertEqual(len(ids[0]), 2)  # and both rows of the pair are returned

    def test_multi_part_question_returns_whole_record(self):  # one record carries every requested field
        _, result = self.first("What are the severity, clinical effect and mechanism of warfarin with fluconazole?")
        record = result["records"][0]  # the stored record
        self.assertEqual(record["id"], "ddi:12")  # it is ddi:12
        self.assertEqual(result["severity"], "Major")  # severity is Major
        for field in ("mechanism", "effect", "rationale", "reference", "safer_alternative"):  # every stored field
            self.assertTrue(record[field])  # is present and non-empty
        self.assertEqual(result["effect"], record["effect"])  # the headline effect equals the stored text

    def test_absent_pair_cabergoline_rucaparib_is_never_called_safe(self):  # unknown drugs are not "safe"
        answer, result = self.first("Do cabergoline and rucaparib interact?")  # neither drug is in our data
        self.assertEqual(result["status"], "unknown_drug")  # reported as unknown
        self.assertIn(NOT_SAFE_NOTE, result["notes"])  # with the "not safe" note
        self.assertNotIn("records", result)  # no records were invented
        self.assertNotIn("severity", {k for k, v in result.items() if v})  # no severity was filled in
        self.assertEqual(answer["status"], "answered")  # the question itself was answered

    def test_two_known_drugs_with_no_row_is_no_documented_interaction(self):  # known drugs, no row
        _, result = self.first("Do warfarin and docusate interact?")  # both exist but have no row together
        self.assertEqual(result["status"], "no_documented_interaction")  # correct status
        self.assertIn(NOT_SAFE_NOTE, result["notes"])  # with the "not safe" note

    def test_single_drug_question_lists_its_interactions_most_severe_first(self):  # one-drug lookup
        answer = self.ask("What interacts with warfarin?")  # list warfarin's interactions
        self.assertEqual(answer["intent"], "drug")  # understood as a drug lookup
        ranks = [severity_rank(r["severity"]) for r in answer["interactions"]]  # severity order of the list
        self.assertEqual(ranks, sorted(ranks))  # sorted most severe first
        self.assertIn("ddi:1", [r["id"] for r in answer["interactions"]])  # warfarin + ibuprofen is included

    def test_generic_question_with_no_drug_is_not_answered_from_guesses(self):  # no drug means no answer
        answer = self.ask("What are interactions?")  # a generic question
        self.assertEqual(answer["status"], "no_match")  # nothing matched
        self.assertNotIn("results", answer)  # no results were invented


class TestSymptomMapping(FullDbTestCase):  # tests for mapping symptom words to interaction text through HPO
    def test_symptom_word_matches_the_effect_text_with_hpo_only_helping(self):  # HPO helps search, not evidence
        answer = self.ask("Which drugs cause bleeding?")  # a symptom question
        self.assertEqual(answer["intent"], "symptom")  # understood as a symptom question
        self.assertTrue(answer["interactions"])  # some interactions matched
        self.assertTrue(all(i["id"].startswith("ddi:") for i in answer["interactions"]))  # all are interaction records
        for item in answer["interactions"]:  # check what each match was found in
            self.assertIn(item["matched_on"], ("effect text", "mechanism text"))  # matched in stored text only
        self.assertIn("matched on the effect text", " ".join(answer["notes"]))  # the notes say so
        self.assertTrue(answer["phenotype_records"])  # phenotype records are returned separately
        self.assertTrue(all(p["id"].startswith("hpo:") for p in answer["phenotype_records"]))  # with hpo: IDs
        self.assertIn("vocabulary, not interaction evidence", answer["phenotype_note"])  # labelled as vocabulary
        ranks = [severity_rank(i["severity"]) for i in answer["interactions"]]  # severity order of matches
        self.assertEqual(ranks, sorted(ranks))  # sorted most severe first

    def test_hpo_never_changes_what_a_pair_question_returns(self):  # HPO is irrelevant to pair lookups
        with_hpo = self.qa.ask("Do warfarin and ibuprofen interact?")  # answerer with HPO available
        without = QuestionAnswerer(self.checker, hpo_conn=None).ask("Do warfarin and ibuprofen interact?")  # without HPO
        self.assertEqual(with_hpo["results"][0]["source"], without["results"][0]["source"])  # same source either way
        self.assertNotIn("phenotype_records", with_hpo)  # no phenotype records on a pair question

    def test_symptom_mapping_can_be_switched_off(self):  # the --no-symptom-mapping option
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, symptom_mapping=False)  # mapping disabled
        answer = qa.ask("Which drugs cause bleeding?")  # ask a symptom question
        self.assertNotIn("phenotype_records", answer)  # no phenotype records
        self.assertIn("Symptom mapping is off", " ".join(answer["notes"]))  # a note says mapping is off
        self.assertTrue(answer["interactions"])  # the literal word is still matched in the effect text

    def test_hpo_expansion_adds_synonyms_children_and_layperson_terms(self):  # what the HPO expansion adds
        from ddi_backend import hpo  # imported here because only this test needs it
        expansion = hpo.expand_symptom(self.conn, "dizziness")  # expand a symptom word
        phrases = {p["phrase"] for p in expansion["phrases"]}  # the set of words to look for
        self.assertIn("vertigo", phrases)            # label of the matched term
        self.assertIn("dizzy spell", phrases)         # layperson synonym
        self.assertIn("paroxysmal vertigo", phrases)  # child term from hpo_edges
        self.assertTrue(all(p["hp_id"].startswith("HP:") for p in expansion["phrases"]))  # every phrase has an HP ID

    def test_symptom_lists_and_plain_sentences_are_not_mistaken_for_drug_names(self):  # no false drug-name guesses
        for text in ("Which drugs cause headache and nausea?", "my heart is racing after a new pill"):  # two symptom texts
            with self.subTest(text=text):  # report each separately
                answer = self.ask(text)  # ask it
                self.assertNotIn("clarifications", answer)  # no drug-name clarification was triggered
                self.assertEqual(answer["intent"], "symptom")  # understood as a symptom question

    def test_symptom_drug_question_filters_that_drugs_interactions(self):  # symptom plus a drug name
        answer = self.ask("Does warfarin cause bleeding?")  # drug and symptom together
        self.assertEqual(answer["intent"], "drug")  # treated as a drug lookup
        self.assertTrue(answer["interactions"])  # some interactions matched
        # every match involves Warfarin
        self.assertTrue(all("Warfarin" in (i["drug_a"], i["drug_b"]) for i in answer["interactions"]))

    def test_without_hpo_tables_symptom_mapping_switches_itself_off(self):  # a database lacking HPO still works
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder for a tiny database
            conn = ingest.open_readonly(  # open a database that has only the core tables
                support.build_db(Path(tmp), [ddi_row("A", "B", "Major", effect="Increased bleeding risk")]),  # one row
                ingest.CORE_TABLES)  # require only the core tables
            try:  # make sure the connection is closed afterwards
                qa = QuestionAnswerer(Checker(conn), hpo_conn=conn)  # HPO connection that has no HPO tables
                answer = qa.ask("Which drugs cause bleeding?")  # ask a symptom question
                self.assertEqual([i["id"] for i in answer["interactions"]], ["ddi:1"])  # matched by literal word
                self.assertNotIn("phenotype_records", answer)  # no phenotype records without HPO tables
            finally:
                conn.close()  # always close the connection


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests free-text question handling (ddi_backend/retrieval.py, database_context.py and hpo.py)
# against the real database. It covers the adversarial warfarin question answered from the avoid list, negation
# detection, clarification for abbreviations, brand names and typos, population and dose limits, topics the data
# cannot answer, the handout questions including repeated pairs, and HPO-assisted symptom mapping that only
# searches the effect and mechanism text.
# ASSIGNMENT TOOL: embeddings in a limited role (question handling and candidate finding that never decides
# severity), together with the knowledge graph + rule-based reasoning that answers from the graph's avoid list;
# it also records why plain RAG was rejected and replaced by exact record lookup.
# ---------------------------------------------------------------------------
