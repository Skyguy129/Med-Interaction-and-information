# Docstring describing the module: tests for retrieval from only the loaded SQLite DDI/HPO tables.
"""Tests for retrieval from only the loaded SQLite DDI/HPO tables."""

import json  # writes the fake DDI and HPO input files
import sqlite3  # makes a deliberately incomplete database
import tempfile  # temporary folder for the databases
import unittest  # the test framework
from pathlib import Path  # path objects

import ddi_backend.database_context as database_context  # the retrieval module under test
import ddi_backend.load_data as load_data  # the loader used to build the test database


# Tests for the database_context module using a tiny hand-made database.
class TestDatabaseContext(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()  # a temporary folder
        self.addCleanup(self.temp_dir.cleanup)  # deleted when the test ends
        self.root = Path(self.temp_dir.name)  # its path
        self.db_path = self.root / "medications.db"  # where the test database goes
        self.conn = load_data.connect(self.db_path)  # create it with the real schema
        self.addCleanup(self.conn.close)  # close it when the test ends

        ddi_path = self.root / "ddi.json"  # fake interaction file
        ddi_path.write_text(  # write one interaction to it
            json.dumps(
                {
                    "drug_interactions": {  # the same nesting the real file uses
                        "major": [  # severity group "major" with one row
                            {
                                "drug_a": "Warfarin",  # first drug
                                "drug_b": "Ibuprofen",  # second drug
                                "severity": "Major",  # how severe
                                "mechanism": "test mechanism",  # filler mechanism
                                "effect": "Increased bleeding risk",  # the effect text the symptom search will match
                                "Safer_alternative": "Acetaminophen",  # suggested alternative
                                "rationale": "test rationale",  # filler reason
                                "reference": "test reference",  # filler source
                            }
                        ],
                        "moderate": [],  # no Moderate rows
                        "minor": [],  # no Minor rows
                    }
                }
            ),
            encoding="utf-8",  # file encoding
        )
        hpo_path = self.root / "hpo.json"  # fake phenotype vocabulary file
        hpo_path.write_text(  # write one HPO term to it
            json.dumps(
                {
                    "graphs": [  # HPO files hold a list of graphs
                        {
                            "nodes": [  # the terms
                                {
                                    "id": "http://purl.obolibrary.org/obo/HP_0000001",  # the term's web id
                                    "lbl": "Bleeding",  # its label
                                    "type": "CLASS",  # it is a class (a real term)
                                    "meta": {  # extra information
                                        "definition": {"val": "Bleeding description"},  # its definition
                                        "synonyms": [  # its synonyms
                                            {
                                                "pred": "hasExactSynonym",  # synonym kind
                                                "val": "Hemorrhage",  # the synonym itself
                                            }
                                        ],
                                    },
                                }
                            ],
                            "edges": [],  # no parent/child links
                        }
                    ]
                }
            ),
            encoding="utf-8",  # file encoding
        )
        load_data.load_ddi(self.conn, ddi_path)  # load the interaction into the database
        load_data.load_hpo(self.conn, hpo_path)  # load the HPO term into the database

    def test_opens_database_created_by_loader(self):
        conn = database_context.open_database(self.db_path)  # open the loader's database
        try:
            self.assertEqual(
                set(load_data.table_names(conn)), database_context.REQUIRED_TABLES  # it has exactly the required tables
            )
        finally:
            conn.close()  # always close it

    def test_retrieves_only_matching_interaction_and_phenotype_rows(self):
        interactions = database_context.search_records(self.conn, "What interacts with WARFARIN?")  # drug question
        phenotypes = database_context.search_hpo_records(self.conn, "hemorrhage symptom")  # symptom search (via synonym)

        self.assertEqual([record["id"] for record in interactions], ["ddi:1"])  # the one interaction matches
        self.assertEqual(interactions[0]["effect"], "Increased bleeding risk")  # with its stored effect text
        self.assertEqual([record["id"] for record in phenotypes], ["hpo:HP:0000001"])  # the HPO term matches
        self.assertEqual(phenotypes[0]["label"], "Bleeding")  # with its label

    def test_symptom_words_find_interactions_through_their_effect_text_not_hpo_rows(self):
        # HPO is a vocabulary: it helps find the effect text, and its terms are never returned as evidence.
        records = database_context.search_records(self.conn, "hemorrhage symptom")  # a symptom-only query
        self.assertEqual([record["id"] for record in records], ["ddi:1"])  # returns the interaction, not the HPO row

    def test_generic_prompt_has_no_matches(self):
        self.assertEqual(
            database_context.search_records(self.conn, "What are interactions?"),  # only generic words, no drug name
            [],  # so nothing is returned
        )

    def test_limit_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "limit"):  # a zero limit must be rejected with a message about the limit
            database_context.search_records(self.conn, "warfarin", limit=0)

    def test_missing_database_is_not_created_implicitly(self):
        missing_path = self.root / "missing.db"  # a path with no file

        with self.assertRaisesRegex(RuntimeError, "python -m ddi_backend.load_data load"):  # error explains how to build it
            database_context.open_database(missing_path)

        self.assertFalse(missing_path.exists())  # and opening did not create the file

    def test_partial_database_is_rejected(self):
        partial_path = self.root / "partial.db"  # a new file
        conn = sqlite3.connect(partial_path)  # create it
        conn.execute("CREATE TABLE drugs (drug_id INTEGER)")  # with only one of the required tables
        conn.close()  # save it

        with self.assertRaisesRegex(RuntimeError, "missing required tables"):  # opening must fail clearly
            database_context.open_database(partial_path)

    def test_format_uses_only_present_database_values(self):
        formatted = database_context.format_record(  # turn a record into display text
            {
                "id": "ddi:1",  # record id
                "source": "drug interaction",  # kind of record
                "drug_a": "Warfarin",  # first drug
                "drug_b": "Ibuprofen",  # second drug
                "effect": None,  # missing effect, so it must not be printed
            }
        )

        self.assertIn("Warfarin", formatted)  # first drug appears
        self.assertIn("Ibuprofen", formatted)  # second drug appears
        self.assertNotIn("Effect:", formatted)  # no empty effect line
        self.assertNotIn("diagnosis", formatted)  # nothing invented


if __name__ == "__main__":  # only when run as a script
    unittest.main()  # run the tests in this file


# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests database_context.py, the original read-only retrieval helper over the loaded SQLite tables: it
# opens only a complete loader-built database, finds interaction rows by drug name or symptom words in the effect text,
# returns HPO phenotype rows separately, rejects a zero limit, a missing file and a partial database, and formats a
# record using only values that are present.
# ASSIGNMENT TOOL: the database/retrieval layer that feeds the knowledge graph core (exact lookup over stored records,
# with HPO used only as a vocabulary to expand symptom words); it deliberately is not the rejected retrieve-and-generate (RAG) approach.
# ---------------------------------------------------------------------------
