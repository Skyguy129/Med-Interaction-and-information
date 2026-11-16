"""Tests for the source registry and the committed seed file."""

import unittest
from pathlib import Path

from ddi_backend import load_data, registry


class TestBuildRegistry(unittest.TestCase):
    def test_seed_is_loaded_without_filling_gaps(self):
        reg = registry.build_registry({
            "approved_authority_classes": ["course_designated_dataset"],
            "sources": {"s": {"status": "current", "scope": "exact_pair"}},
        })
        source = reg.get("s")
        self.assertEqual(source.status, "current")
        self.assertIsNone(source.authority_class)  # never invented
        self.assertIsNone(source.effective_date)
        self.assertIsNone(source.owner)
        self.assertIsNone(reg.get("not_registered"))

    def test_unknown_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            registry.build_registry({"sources": {"s": {"scope": "galaxy"}}})
        with self.assertRaisesRegex(ValueError, "status"):
            registry.build_registry({"sources": {"s": {"status": "fresh"}}})

    def test_scope_none_is_allowed(self):
        self.assertEqual(registry.build_registry({"sources": {"s": {"scope": "none"}}}).get("s").scope, "none")

    def test_missing_seed_file_registers_nothing(self):
        reg = registry.load_registry(Path("no/such/seed.json"))
        self.assertEqual(reg.sources, {})
        self.assertFalse(reg.is_approved("course_designated_dataset"))


class TestCommittedSeed(unittest.TestCase):
    def setUp(self):
        self.reg = registry.load_registry()

    def test_kaggle_ddi_is_the_approved_interaction_source(self):
        s = self.reg.get("kaggle_ddi")
        self.assertEqual((s.status, s.authority_class, s.scope), ("current", "course_designated_dataset", "exact_pair"))
        self.assertIsNone(s.effective_date)  # unknown: not guessed
        self.assertTrue(self.reg.is_approved(s.authority_class))

    def test_hpo_is_a_vocabulary_not_evidence(self):
        s = self.reg.get("hpo")
        self.assertEqual((s.authority_class, s.scope), ("vocabulary", "none"))
        self.assertFalse(self.reg.is_approved("vocabulary"))

    @unittest.skipUnless(load_data.DEFAULT_HPO.exists(), "hp.json not present")
    def test_hpo_release_date_matches_hp_json(self):
        graph = load_data.read_json(load_data.DEFAULT_HPO)["graphs"][0]
        self.assertIn(f"/{self.reg.get('hpo').effective_date}/", graph["meta"]["version"])


if __name__ == "__main__":
    unittest.main()
