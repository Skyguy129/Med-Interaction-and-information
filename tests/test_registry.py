# Docstring: says this file tests the source registry and the committed seed file.
"""Tests for the source registry and the committed seed file."""

import unittest  # standard test framework
from pathlib import Path  # used to point at a seed file path that does not exist

from ddi_backend import load_data, registry  # registry is under test; load_data supplies the hp.json path


class TestBuildRegistry(unittest.TestCase):  # tests for building a registry from a dict
    def test_seed_is_loaded_without_filling_gaps(self):  # missing metadata must stay missing
        # build a registry from an in-memory seed with one source that has only status and scope
        reg = registry.build_registry({
            "approved_authority_classes": ["course_designated_dataset"],  # the list of approved classes
            "sources": {"s": {"status": "current", "scope": "exact_pair"}},  # source "s" with partial metadata
        })
        source = reg.get("s")  # look the source up by its key
        self.assertEqual(source.status, "current")  # the given status was kept
        self.assertIsNone(source.authority_class)  # never invented
        self.assertIsNone(source.effective_date)  # a missing date stays None
        self.assertIsNone(source.owner)  # a missing owner stays None
        self.assertIsNone(reg.get("not_registered"))  # an unregistered key gives None, not an error or a guess

    def test_unknown_values_are_rejected(self):  # bad vocabulary values are errors
        with self.assertRaisesRegex(ValueError, "scope"):  # an invalid scope must raise and mention "scope"
            registry.build_registry({"sources": {"s": {"scope": "galaxy"}}})
        with self.assertRaisesRegex(ValueError, "status"):  # an invalid status must raise and mention "status"
            registry.build_registry({"sources": {"s": {"status": "fresh"}}})

    def test_scope_none_is_allowed(self):  # "none" is a legal scope (used for vocabularies)
        self.assertEqual(registry.build_registry({"sources": {"s": {"scope": "none"}}}).get("s").scope, "none")

    def test_missing_seed_file_registers_nothing(self):  # no file means an empty registry
        reg = registry.load_registry(Path("no/such/seed.json"))  # load from a path that does not exist
        self.assertEqual(reg.sources, {})  # no sources were registered
        self.assertFalse(reg.is_approved("course_designated_dataset"))  # and no authority class is approved


class TestCommittedSeed(unittest.TestCase):  # tests against the real data/sources_seed.json
    def setUp(self):  # runs before every test in this class
        self.reg = registry.load_registry()  # load the default committed seed file

    def test_kaggle_ddi_is_the_approved_interaction_source(self):  # the interaction dataset is registered correctly
        s = self.reg.get("kaggle_ddi")  # fetch the interaction source
        # status, authority class and scope must match the team's designation
        self.assertEqual((s.status, s.authority_class, s.scope), ("current", "course_designated_dataset", "exact_pair"))
        self.assertIsNone(s.effective_date)  # unknown: not guessed
        self.assertTrue(self.reg.is_approved(s.authority_class))  # its authority class is on the approved list

    def test_hpo_is_a_vocabulary_not_evidence(self):  # the phenotype source must not count as evidence
        s = self.reg.get("hpo")  # fetch the HPO source
        self.assertEqual((s.authority_class, s.scope), ("vocabulary", "none"))  # it is a vocabulary with no scope
        self.assertFalse(self.reg.is_approved("vocabulary"))  # vocabulary is not an approved evidence class

    # skip this test when the big hp.json file is not on disk
    @unittest.skipUnless(load_data.DEFAULT_HPO.exists(), "hp.json not present")
    def test_hpo_release_date_matches_hp_json(self):  # the seed date must match the real file's version
        graph = load_data.read_json(load_data.DEFAULT_HPO)["graphs"][0]  # first graph of the HPO file
        # the version URL inside hp.json must contain the registry's effective date
        self.assertIn(f"/{self.reg.get('hpo').effective_date}/", graph["meta"]["version"])


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the source registry (ddi_backend/registry.py): that metadata is loaded from the seed
# without inventing missing values, that unknown statuses or scopes are rejected, that a missing seed file gives an
# empty registry, and that the committed seed marks the interaction dataset as approved and HPO as a vocabulary.
# ASSIGNMENT TOOL: metadata-aware eligibility filtering (the registry is the source metadata that the filter reads).
# ---------------------------------------------------------------------------
