"""Tests for read-only access, record shape and alternative classification."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ddi_backend import ingest
from tests import support


class TestOpenReadonly(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db = support.build_db(self.dir, [support.ddi_row("Warfarin", "Ibuprofen", "Major")])

    def test_cannot_write(self):
        conn = ingest.open_readonly(self.db, ingest.CORE_TABLES)
        self.addCleanup(conn.close)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM interactions")
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO drugs (name) VALUES ('X')")

    def test_missing_database_is_not_created(self):
        missing = self.dir / "missing.db"
        with self.assertRaisesRegex(RuntimeError, "ddi_backend.load_data load"):
            ingest.open_readonly(missing)
        self.assertFalse(missing.exists())

    def test_missing_tables_named(self):
        # Built from DDI only, so the default (full) requirement fails but the core one passes.
        with self.assertRaisesRegex(RuntimeError, "hpo_edges"):
            ingest.open_readonly(self.db)
        ingest.open_readonly(self.db, ingest.CORE_TABLES).close()


class TestRecords(unittest.TestCase):
    def test_record_keeps_loader_id_and_fields(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        records = {r["id"]: r for r in ingest.load_interactions(conn)}
        self.assertEqual(len(records), 180)
        r = records["ddi:1"]
        self.assertEqual((r["severity"], r["drug_a"], r["drug_b"]), ("Major", "Warfarin", "Ibuprofen"))
        self.assertEqual(r["source_key"], "kaggle_ddi")
        self.assertEqual(r["role"], "direct")
        self.assertEqual(r["safer_alternative"], "Acetaminophen (Paracetamol)")

    def test_records_sorted_major_first(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        ranks = [{"Major": 0, "Moderate": 1, "Minor": 2}[r["severity"]]
                 for r in ingest.load_interactions(conn)]
        self.assertEqual(ranks, sorted(ranks))


class TestParseAlternative(unittest.TestCase):
    def classify(self, text, known=()):
        return ingest.parse_alternative(text, known)

    def test_dose_instruction_is_a_management_action(self):
        got = self.classify("Reduce warfarin dose by 30-50%", ["Warfarin"])
        self.assertEqual(got["type"], ingest.MANAGEMENT_ACTION)
        self.assertEqual(got["drug_names"], [])
        self.assertEqual(got["text"], "Reduce warfarin dose by 30-50%")  # echoed, never altered

    def test_single_and_multiple_drugs(self):
        self.assertEqual(self.classify("Sertraline or Citalopram")["drug_names"], ["Sertraline", "Citalopram"])
        self.assertEqual(self.classify("Sertraline")["type"], ingest.ALTERNATIVE_DRUG)

    def test_known_lowercase_drug_after_or(self):
        got = self.classify("Famotidine or ranitidine", ["Ranitidine", "Famotidine"])
        self.assertEqual(got["drug_names"], ["Famotidine", "ranitidine"])

    def test_class_with_example_names_the_example(self):
        got = self.classify("Alternative antibiotic (e.g., Ceftazidime)")
        self.assertEqual((got["type"], got["drug_names"]), (ingest.ALTERNATIVE_DRUG, ["Ceftazidime"]))
        self.assertEqual(self.classify("ARB (e.g., Losartan) or CCB")["drug_names"], ["Losartan"])

    def test_non_specific_text_is_not_a_drug(self):
        for text in ("Alternative antibiotic", "Other SSRIs if appropriate", "H2 blockers if appropriate",
                     "Monitor INR closely or choose different antibiotic", "Non-hormonal IUD or higher-dose OC",
                     "None needed", "Use lowest effective dose", ""):
            with self.subTest(text=text):
                got = self.classify(text)
                self.assertEqual((got["type"], got["drug_names"]), (ingest.MANAGEMENT_ACTION, []))

    def test_qualifiers_are_not_part_of_the_name(self):
        self.assertEqual(self.classify("Domperidone (if available)")["drug_names"], ["Domperidone"])
        self.assertEqual(self.classify("Amiodarone (with dose adjustment)")["drug_names"], ["Amiodarone"])
        self.assertEqual(self.classify("Loop Diuretic (Furosemide)")["drug_names"], ["Furosemide"])


if __name__ == "__main__":
    unittest.main()
