"""Tests for name resolution: exact, case-insensitive, never guessing."""

import tempfile
import unittest
from pathlib import Path

from ddi_backend import resolver
from ddi_backend.resolver import AMBIGUOUS_NAME, RESOLVED, UNKNOWN_DRUG, Resolver, normalize_name
from tests import support


class TestNormalize(unittest.TestCase):
    def test_strips_case_whitespace_and_dose(self):
        self.assertEqual(normalize_name("  Warfarin   5 mg "), ("Warfarin", "5 mg"))
        self.assertEqual(normalize_name("ibuprofen 200mg"), ("ibuprofen", "200mg"))
        self.assertEqual(normalize_name("Insulin 10 units"), ("Insulin", "10 units"))
        self.assertEqual(normalize_name("Warfarin"), ("Warfarin", None))

    def test_empty_input(self):
        self.assertEqual(normalize_name(None), ("", None))
        self.assertEqual(normalize_name("   "), ("", None))


class TestResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conn = support.open_real()
        cls.resolver = Resolver.from_connection(conn)
        conn.close()

    def test_exact_match_ignores_case_and_dose(self):
        for text in ("warfarin", "WARFARIN", " Warfarin 5 mg "):
            with self.subTest(text=text):
                res = self.resolver.resolve(text)
                self.assertEqual((res.status, res.name, res.via), (RESOLVED, "Warfarin", "exact"))
        self.assertEqual(self.resolver.resolve("Warfarin 5 mg").dose, "5 mg")

    def test_brand_name_is_unknown_not_guessed(self):
        res = self.resolver.resolve("Advil")
        self.assertEqual(res.status, UNKNOWN_DRUG)
        self.assertIsNone(res.drug_id)

    def test_typo_is_never_auto_resolved(self):
        res = self.resolver.resolve("Warfrin")
        self.assertEqual(res.status, UNKNOWN_DRUG)
        self.assertIsNone(res.name)
        self.assertIn("Warfarin", res.candidates)
        self.assertIn("Did you mean", res.question)

    def test_short_abbreviation_asks_what_is_meant(self):
        res = self.resolver.resolve("MS")
        self.assertEqual(res.status, AMBIGUOUS_NAME)
        self.assertIn("abbreviation", res.question)

    def test_partial_name_matching_several_drugs_is_ambiguous(self):
        res = self.resolver.resolve("Valpro")
        self.assertEqual(res.status, AMBIGUOUS_NAME)
        self.assertEqual(set(res.candidates), {"Valproic Acid", "Sodium Valproate"})

    def test_drugs_absent_from_the_data_are_unknown(self):
        for name in ("Cabergoline", "Rucaparib"):
            self.assertEqual(self.resolver.resolve(name).status, UNKNOWN_DRUG)

    def test_blank_input_is_unknown(self):
        self.assertEqual(self.resolver.resolve("  ").status, UNKNOWN_DRUG)


class TestAliases(unittest.TestCase):
    def test_default_alias_file_is_empty(self):
        self.assertEqual(resolver.load_aliases(), {})

    def test_alias_is_used_only_when_listed(self):
        drugs = [(1, "Ibuprofen")]
        self.assertEqual(Resolver(drugs).resolve("Advil").status, UNKNOWN_DRUG)
        res = Resolver(drugs, aliases={"Advil": "Ibuprofen"}).resolve("advil 200 mg")
        self.assertEqual((res.status, res.name, res.via), (RESOLVED, "Ibuprofen", "alias"))

    def test_alias_pointing_at_missing_drug_does_not_resolve(self):
        res = Resolver([(1, "Ibuprofen")], aliases={"advil": "Nonexistentium"}).resolve("Advil")
        self.assertEqual(res.status, UNKNOWN_DRUG)

    def test_alias_file_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aliases.json"
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolver.load_aliases(path)


class TestSuggester(unittest.TestCase):
    def test_suggester_can_only_add_real_names(self):
        res = Resolver([(1, "Ibuprofen"), (2, "Warfarin")],
                       suggester=lambda text: ["Warfarin", "Made Up"]).resolve("blood thinner")
        self.assertEqual(res.status, UNKNOWN_DRUG)
        self.assertEqual(res.candidates, ["Warfarin"])


if __name__ == "__main__":
    unittest.main()
