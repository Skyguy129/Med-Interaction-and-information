# Module docstring: this file tests read-only database access, the record shape and alternative classification.
"""Tests for read-only access, record shape and alternative classification."""

import sqlite3  # needed for the OperationalError raised when a write is attempted
import tempfile  # throwaway folders for small test databases
import unittest  # standard-library test framework
from pathlib import Path  # path objects for the temporary folder

from ddi_backend import ingest  # the module under test
from tests import support  # shared helpers: build a small database, open the real one


class TestOpenReadonly(unittest.TestCase):  # tests that the application's database connection cannot write
    def setUp(self):  # runs before each test in this class
        self._tmp = tempfile.TemporaryDirectory()  # a fresh temporary folder
        self.addCleanup(self._tmp.cleanup)  # delete it after each test
        self.dir = Path(self._tmp.name)  # remember the folder path
        # Build a tiny database containing a single Warfarin + Ibuprofen Major row.
        self.db = support.build_db(self.dir, [support.ddi_row("Warfarin", "Ibuprofen", "Major")])

    def test_cannot_write(self):  # the read-only connection rejects every write
        conn = ingest.open_readonly(self.db, ingest.CORE_TABLES)  # open with only the core tables required
        self.addCleanup(conn.close)  # close afterwards
        with self.assertRaises(sqlite3.OperationalError):  # a delete must fail
            conn.execute("DELETE FROM interactions")
        with self.assertRaises(sqlite3.OperationalError):  # an insert must fail
            conn.execute("INSERT INTO drugs (name) VALUES ('X')")

    def test_missing_database_is_not_created(self):  # opening a missing file must not create it
        missing = self.dir / "missing.db"  # a path that does not exist
        with self.assertRaisesRegex(RuntimeError, "ddi_backend.load_data load"):  # error tells the user how to build it
            ingest.open_readonly(missing)
        self.assertFalse(missing.exists())  # and no file appeared

    def test_missing_tables_named(self):  # a partial database is rejected with the missing table named
        # Built from DDI only, so the default (full) requirement fails but the core one passes.
        with self.assertRaisesRegex(RuntimeError, "hpo_edges"):  # the HPO table is absent, and the error says so
            ingest.open_readonly(self.db)
        ingest.open_readonly(self.db, ingest.CORE_TABLES).close()  # asking for core tables only succeeds


class TestRecords(unittest.TestCase):  # tests the shape of interaction records read from the real database
    def test_record_keeps_loader_id_and_fields(self):  # IDs and fields match what the loader wrote
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        records = {r["id"]: r for r in ingest.load_interactions(conn)}  # index all records by their ddi: ID
        self.assertEqual(len(records), 180)  # the dataset has 180 interactions
        r = records["ddi:1"]  # the first record
        self.assertEqual((r["severity"], r["drug_a"], r["drug_b"]), ("Major", "Warfarin", "Ibuprofen"))  # known values
        self.assertEqual(r["source_key"], "kaggle_ddi")  # tagged with its source key
        self.assertEqual(r["role"], "direct")  # and marked as direct evidence
        self.assertEqual(r["safer_alternative"], "Acetaminophen (Paracetamol)")  # the alternative text is kept as written

    def test_records_sorted_major_first(self):  # the most serious records come first
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        # Turn each severity into a rank number (Major lowest) in the order the records come back.
        ranks = [{"Major": 0, "Moderate": 1, "Minor": 2}[r["severity"]]
                 for r in ingest.load_interactions(conn)]
        self.assertEqual(ranks, sorted(ranks))  # the ranks must already be in ascending order


class TestParseAlternative(unittest.TestCase):  # tests how safer-alternative text is split into drugs vs actions
    def classify(self, text, known=()):  # helper: parse alternative text, optionally with a list of known drug names
        return ingest.parse_alternative(text, known)

    def test_dose_instruction_is_a_management_action(self):  # dose advice is an action, not a drug
        got = self.classify("Reduce warfarin dose by 30-50%", ["Warfarin"])  # text mentions a known drug
        self.assertEqual(got["type"], ingest.MANAGEMENT_ACTION)  # still classed as an action
        self.assertEqual(got["drug_names"], [])  # no alternative drug is extracted
        self.assertEqual(got["text"], "Reduce warfarin dose by 30-50%")  # echoed, never altered

    def test_single_and_multiple_drugs(self):  # drug alternatives are split into names
        self.assertEqual(self.classify("Sertraline or Citalopram")["drug_names"], ["Sertraline", "Citalopram"])  # two drugs joined by "or"
        self.assertEqual(self.classify("Sertraline")["type"], ingest.ALTERNATIVE_DRUG)  # one name is an alternative drug

    def test_known_lowercase_drug_after_or(self):  # a lowercase name is recognised when it is a known drug
        got = self.classify("Famotidine or ranitidine", ["Ranitidine", "Famotidine"])  # known list supplies the names
        self.assertEqual(got["drug_names"], ["Famotidine", "ranitidine"])  # names come back in text order, text casing

    def test_class_with_example_names_the_example(self):  # "class (e.g., Drug)" keeps only the example drug
        got = self.classify("Alternative antibiotic (e.g., Ceftazidime)")  # a class with one example
        self.assertEqual((got["type"], got["drug_names"]), (ingest.ALTERNATIVE_DRUG, ["Ceftazidime"]))  # example is the drug
        self.assertEqual(self.classify("ARB (e.g., Losartan) or CCB")["drug_names"], ["Losartan"])  # class names without examples are dropped

    def test_non_specific_text_is_not_a_drug(self):  # vague advice must not be treated as a drug name
        # A set of phrases that look like advice or a bare class rather than a named drug, plus the empty string.
        for text in ("Alternative antibiotic", "Other SSRIs if appropriate", "H2 blockers if appropriate",
                     "Monitor INR closely or choose different antibiotic", "Non-hormonal IUD or higher-dose OC",
                     "None needed", "Use lowest effective dose", ""):
            with self.subTest(text=text):  # report each phrase separately
                got = self.classify(text)  # classify it
                self.assertEqual((got["type"], got["drug_names"]), (ingest.MANAGEMENT_ACTION, []))  # an action with no drugs

    def test_qualifiers_are_not_part_of_the_name(self):  # bracketed qualifiers are stripped from names
        self.assertEqual(self.classify("Domperidone (if available)")["drug_names"], ["Domperidone"])  # "(if available)" dropped
        self.assertEqual(self.classify("Amiodarone (with dose adjustment)")["drug_names"], ["Amiodarone"])  # dose note dropped
        self.assertEqual(self.classify("Loop Diuretic (Furosemide)")["drug_names"], ["Furosemide"])  # the bracketed drug is kept


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests ddi_backend/ingest.py, the layer between the database and the rest of the system. It checks that
# the application's connection is truly read-only and fails clearly when the database or tables are missing, that
# interaction records keep their loader IDs and fields and come back Major first, and that safer-alternative text is
# sorted into "alternative drug" versus "management action" without ever altering the stored text.
# ASSIGNMENT TOOL: Supporting infrastructure (not one of the assignment tools): it feeds the knowledge graph and
# rule-based checker with records and underpins the safer-alternative handling in the core engine.
# ---------------------------------------------------------------------------
