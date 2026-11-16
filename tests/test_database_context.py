"""Tests for retrieval from only the loaded SQLite DDI/HPO tables."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import ddi_backend.database_context as database_context
import ddi_backend.load_data as load_data


class TestDatabaseContext(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "medications.db"
        self.conn = load_data.connect(self.db_path)
        self.addCleanup(self.conn.close)

        ddi_path = self.root / "ddi.json"
        ddi_path.write_text(
            json.dumps(
                {
                    "drug_interactions": {
                        "major": [
                            {
                                "drug_a": "Warfarin",
                                "drug_b": "Ibuprofen",
                                "severity": "Major",
                                "mechanism": "test mechanism",
                                "effect": "Increased bleeding risk",
                                "Safer_alternative": "Acetaminophen",
                                "rationale": "test rationale",
                                "reference": "test reference",
                            }
                        ],
                        "moderate": [],
                        "minor": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        hpo_path = self.root / "hpo.json"
        hpo_path.write_text(
            json.dumps(
                {
                    "graphs": [
                        {
                            "nodes": [
                                {
                                    "id": "http://purl.obolibrary.org/obo/HP_0000001",
                                    "lbl": "Bleeding",
                                    "type": "CLASS",
                                    "meta": {
                                        "definition": {"val": "Bleeding description"},
                                        "synonyms": [
                                            {
                                                "pred": "hasExactSynonym",
                                                "val": "Hemorrhage",
                                            }
                                        ],
                                    },
                                }
                            ],
                            "edges": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        load_data.load_ddi(self.conn, ddi_path)
        load_data.load_hpo(self.conn, hpo_path)

    def test_opens_database_created_by_loader(self):
        conn = database_context.open_database(self.db_path)
        try:
            self.assertEqual(
                set(load_data.table_names(conn)), database_context.REQUIRED_TABLES
            )
        finally:
            conn.close()

    def test_retrieves_only_matching_interaction_and_phenotype_rows(self):
        interactions = database_context.search_records(self.conn, "What interacts with WARFARIN?")
        phenotypes = database_context.search_records(self.conn, "hemorrhage symptom")

        self.assertEqual([record["id"] for record in interactions], ["ddi:1"])
        self.assertEqual(interactions[0]["effect"], "Increased bleeding risk")
        self.assertEqual([record["id"] for record in phenotypes], ["hpo:HP:0000001"])
        self.assertEqual(phenotypes[0]["label"], "Bleeding")

    def test_generic_prompt_has_no_matches(self):
        self.assertEqual(
            database_context.search_records(self.conn, "What are interactions?"),
            [],
        )

    def test_limit_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            database_context.search_records(self.conn, "warfarin", limit=0)

    def test_missing_database_is_not_created_implicitly(self):
        missing_path = self.root / "missing.db"

        with self.assertRaisesRegex(RuntimeError, "python -m ddi_backend.load_data load"):
            database_context.open_database(missing_path)

        self.assertFalse(missing_path.exists())

    def test_partial_database_is_rejected(self):
        partial_path = self.root / "partial.db"
        conn = sqlite3.connect(partial_path)
        conn.execute("CREATE TABLE drugs (drug_id INTEGER)")
        conn.close()

        with self.assertRaisesRegex(RuntimeError, "missing required tables"):
            database_context.open_database(partial_path)

    def test_format_uses_only_present_database_values(self):
        formatted = database_context.format_record(
            {
                "id": "ddi:1",
                "source": "drug interaction",
                "drug_a": "Warfarin",
                "drug_b": "Ibuprofen",
                "effect": None,
            }
        )

        self.assertIn("Warfarin", formatted)
        self.assertIn("Ibuprofen", formatted)
        self.assertNotIn("Effect:", formatted)
        self.assertNotIn("diagnosis", formatted)


if __name__ == "__main__":
    unittest.main()
