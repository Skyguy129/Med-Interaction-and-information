# Docstring: says this file tests name resolution that is exact, case-insensitive and never guesses.
"""Tests for name resolution: exact, case-insensitive, never guessing."""

import tempfile  # temporary folder for the alias file test
import unittest  # standard test framework
from pathlib import Path  # build the path of the temporary alias file

from ddi_backend import resolver  # the module under test (used for load_aliases)
# status constants and the classes/functions under test
from ddi_backend.resolver import AMBIGUOUS_NAME, RESOLVED, UNKNOWN_DRUG, Resolver, normalize_name
from tests import support  # helper that opens the real database


class TestNormalize(unittest.TestCase):  # tests for normalize_name, which splits a drug name from a dose
    def test_strips_case_whitespace_and_dose(self):  # name and dose are separated, extra spaces removed
        self.assertEqual(normalize_name("  Warfarin   5 mg "), ("Warfarin", "5 mg"))  # spaces trimmed, dose split off
        self.assertEqual(normalize_name("ibuprofen 200mg"), ("ibuprofen", "200mg"))  # dose with no space
        self.assertEqual(normalize_name("Insulin 10 units"), ("Insulin", "10 units"))  # a units-style dose
        self.assertEqual(normalize_name("Warfarin"), ("Warfarin", None))  # no dose gives None

    def test_empty_input(self):  # nothing in, nothing out
        self.assertEqual(normalize_name(None), ("", None))  # None becomes an empty name
        self.assertEqual(normalize_name("   "), ("", None))  # only spaces becomes an empty name


class TestResolver(unittest.TestCase):  # tests for resolving text to a real drug in the database
    @classmethod
    def setUpClass(cls):  # runs once for the class
        conn = support.open_real()  # open the real medications database read-only
        cls.resolver = Resolver.from_connection(conn)  # build a resolver from its drug list
        conn.close()  # the resolver holds the names in memory, so the connection can close

    def test_exact_match_ignores_case_and_dose(self):  # case and dose do not stop an exact match
        for text in ("warfarin", "WARFARIN", " Warfarin 5 mg "):  # three spellings of the same drug
            with self.subTest(text=text):  # report each spelling separately if it fails
                res = self.resolver.resolve(text)  # resolve the text
                # resolved, canonical name, and found by exact match
                self.assertEqual((res.status, res.name, res.via), (RESOLVED, "Warfarin", "exact"))
        self.assertEqual(self.resolver.resolve("Warfarin 5 mg").dose, "5 mg")  # the entered dose is echoed back

    def test_brand_name_is_unknown_not_guessed(self):  # brand names are not in the data
        res = self.resolver.resolve("Advil")  # a brand name for ibuprofen
        self.assertEqual(res.status, UNKNOWN_DRUG)  # reported as unknown
        self.assertIsNone(res.drug_id)  # no drug ID was assigned

    def test_typo_is_never_auto_resolved(self):  # a misspelling only gets suggestions
        res = self.resolver.resolve("Warfrin")  # a typo of Warfarin
        self.assertEqual(res.status, UNKNOWN_DRUG)  # still unknown
        self.assertIsNone(res.name)  # no name was silently chosen
        self.assertIn("Warfarin", res.candidates)  # Warfarin is offered as a suggestion
        self.assertIn("Did you mean", res.question)  # the user is asked a did-you-mean question

    def test_short_abbreviation_asks_what_is_meant(self):  # a short abbreviation is ambiguous
        res = self.resolver.resolve("MS")  # could mean several things
        self.assertEqual(res.status, AMBIGUOUS_NAME)  # flagged ambiguous
        self.assertIn("abbreviation", res.question)  # the question explains it is an abbreviation

    def test_partial_name_matching_several_drugs_is_ambiguous(self):  # a prefix matching many drugs is ambiguous
        res = self.resolver.resolve("Valpro")  # prefix of two valproate drugs
        self.assertEqual(res.status, AMBIGUOUS_NAME)  # flagged ambiguous
        self.assertEqual(set(res.candidates), {"Valproic Acid", "Sodium Valproate"})  # both candidates are listed

    def test_drugs_absent_from_the_data_are_unknown(self):  # real drugs missing from our data are unknown
        for name in ("Cabergoline", "Rucaparib"):  # two drugs that are not in the database
            self.assertEqual(self.resolver.resolve(name).status, UNKNOWN_DRUG)  # each is unknown

    def test_blank_input_is_unknown(self):  # blank text cannot resolve
        self.assertEqual(self.resolver.resolve("  ").status, UNKNOWN_DRUG)  # blank is unknown


class TestAliases(unittest.TestCase):  # tests for the optional alias file
    def test_default_alias_file_is_empty(self):  # the committed data/aliases.json has no aliases
        self.assertEqual(resolver.load_aliases(), {})  # loads as an empty dict

    def test_alias_is_used_only_when_listed(self):  # aliases work only if supplied
        drugs = [(1, "Ibuprofen")]  # a one-drug database as (id, name)
        self.assertEqual(Resolver(drugs).resolve("Advil").status, UNKNOWN_DRUG)  # with no alias, Advil is unknown
        res = Resolver(drugs, aliases={"Advil": "Ibuprofen"}).resolve("advil 200 mg")  # with an alias it resolves
        # resolved to the real drug, and marked as found through an alias
        self.assertEqual((res.status, res.name, res.via), (RESOLVED, "Ibuprofen", "alias"))

    def test_alias_pointing_at_missing_drug_does_not_resolve(self):  # an alias to a non-existent drug is ignored
        res = Resolver([(1, "Ibuprofen")], aliases={"advil": "Nonexistentium"}).resolve("Advil")  # bad target
        self.assertEqual(res.status, UNKNOWN_DRUG)  # stays unknown

    def test_alias_file_must_be_an_object(self):  # the alias file must hold a JSON object
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder, removed afterwards
            path = Path(tmp) / "aliases.json"  # where the bad alias file goes
            path.write_text("[1, 2]", encoding="utf-8")  # write a JSON list instead of an object
            with self.assertRaises(ValueError):  # loading it must raise
                resolver.load_aliases(path)


class TestSuggester(unittest.TestCase):  # tests for the optional suggestion hook (such as embeddings)
    def test_suggester_can_only_add_real_names(self):  # suggestions are filtered to names in the data
        res = Resolver([(1, "Ibuprofen"), (2, "Warfarin")],  # a two-drug database
                       suggester=lambda text: ["Warfarin", "Made Up"]).resolve("blood thinner")  # suggester returns one fake
        self.assertEqual(res.status, UNKNOWN_DRUG)  # the suggester never auto-resolves
        self.assertEqual(res.candidates, ["Warfarin"])  # only the real drug name survives


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests drug-name resolution (ddi_backend/resolver.py): dose stripping, exact case-insensitive
# matching, unknown brand names and typos that only get did-you-mean suggestions, ambiguous abbreviations and
# prefixes, the optional alias file, and a suggester that can only add real drug names. It protects the rule
# that a name is never guessed.
# ASSIGNMENT TOOL: knowledge graph + rule-based reasoning (resolving the name is the first deterministic step
# before the graph lookup); the suggester hook is where embeddings play their limited role.
# ---------------------------------------------------------------------------
