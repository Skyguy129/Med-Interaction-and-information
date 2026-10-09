"""Tests for load_data.py. Standard library only.

Run from this folder:

    python -m unittest -v

Most tests use tiny fixtures written to a temp folder. The last two classes
run against the real DDI_Database.json / hp.json if they sit next to this file.
"""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import load_data

IRI = "http://purl.obolibrary.org/obo/"


def ddi_row(a, b, severity, **overrides):
    row = {
        "drug_a": a, "drug_b": b, "severity": severity,
        "mechanism": "mech", "effect": "effect text",
        "Safer_alternative": "alt", "rationale": "why", "reference": "ref",
    }
    row.update(overrides)
    return row


DDI_FIXTURE = {"drug_interactions": {
    "major": [
        ddi_row("Warfarin", "Ibuprofen", "Major", Safer_alternative="Acetaminophen"),
        ddi_row("Clopidogrel", "Omeprazole", "Major"),
    ],
    # Same pair as above: reversed order, different case, conflicting severity.
    "moderate": [ddi_row("omeprazole", "Clopidogrel", "Moderate")],
    "minor": [],
}}

HPO_FIXTURE = {"graphs": [{
    "id": "http://purl.obolibrary.org/obo/hp.json",
    "nodes": [
        {"id": IRI + "HP_0000001", "lbl": "All", "type": "CLASS",
         "meta": {"comments": ["Root of all terms."]}},
        {"id": IRI + "HP_0000002", "lbl": "Abnormality of body height", "type": "CLASS",
         "meta": {
             "definition": {"val": "Deviation from the norm of height."},
             "synonyms": [{"pred": "hasExactSynonym", "val": "Height abnormality",
                           "synonymType": "http://purl.obolibrary.org/obo/hp#layperson"}],
         }},
        {"id": IRI + "HP_0000057", "lbl": "obsolete Clitoromegaly", "type": "CLASS",
         "meta": {"deprecated": True}},
        # A relation node, not a phenotype term: must be ignored.
        {"id": IRI + "BFO_0000051", "lbl": "has part", "type": "PROPERTY"},
    ],
    "edges": [
        {"sub": IRI + "HP_0000002", "pred": "is_a", "obj": IRI + "HP_0000001"},
        {"sub": IRI + "HP_0000002", "pred": "is_a", "obj": IRI + "HP_9999999"},  # parent missing
    ],
}]}


def quiet_main(argv):
    """Run the CLI without cluttering test output; returns (exit_code, printed_text)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = load_data.main(argv)
    return code, out.getvalue() + err.getvalue()


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db = self.dir / "test.db"
        self.conn = load_data.connect(self.db)
        self.addCleanup(self.conn.close)

    def write_json(self, name, data):
        path = self.dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestDDILoader(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.write_json("ddi.json", DDI_FIXTURE)

    def test_counts(self):
        n_inter, n_drugs = load_data.load_ddi(self.conn, self.path)
        self.assertEqual(n_inter, 3)
        self.assertEqual(n_drugs, 4)
        self.assertEqual(self.count("interactions"), 3)
        self.assertEqual(self.count("drugs"), 4)

    def test_drug_names_are_case_insensitive(self):
        load_data.load_ddi(self.conn, self.path)
        # 'Omeprazole' and 'omeprazole' must be the same drug.
        n = self.conn.execute("SELECT COUNT(*) FROM drugs WHERE name = 'OMEPRAZOLE'").fetchone()[0]
        self.assertEqual(n, 1)

    def test_fields_are_stored(self):
        load_data.load_ddi(self.conn, self.path)
        rows = load_data.find_interactions(self.conn, "warfarin")
        self.assertEqual(rows, [("Major", "Warfarin", "Ibuprofen", "effect text", "Acetaminophen")])

    def test_repeated_pairs_are_kept_and_reported(self):
        load_data.load_ddi(self.conn, self.path)
        repeated = load_data.find_repeated_pairs(self.conn)
        self.assertEqual(len(repeated), 1)
        a, b, severities = repeated[0]
        self.assertEqual({a, b}, {"Clopidogrel", "Omeprazole"})
        self.assertEqual(set(severities.split(" / ")), {"Major", "Moderate"})

    def test_reload_does_not_duplicate(self):
        load_data.load_ddi(self.conn, self.path)
        load_data.load_ddi(self.conn, self.path)
        self.assertEqual(self.count("interactions"), 3)
        self.assertEqual(self.count("drugs"), 4)

    def test_no_orphan_foreign_keys(self):
        load_data.load_ddi(self.conn, self.path)
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_wrong_structure_raises(self):
        bad = self.write_json("bad.json", {"something_else": []})
        with self.assertRaises(ValueError):
            load_data.load_ddi(self.conn, bad)

    def test_missing_field_raises_and_saves_nothing(self):
        broken = {"drug_interactions": {"major": [{"drug_a": "X", "drug_b": "Y"}]}}  # no severity
        path = self.write_json("broken.json", broken)
        with self.assertRaises(ValueError):
            load_data.load_ddi(self.conn, path)
        self.assertEqual(self.count("interactions"), 0)


class TestHPOLoader(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.write_json("hp.json", HPO_FIXTURE)
        self.counts = load_data.load_hpo(self.conn, self.path)

    def test_only_hp_terms_are_loaded(self):
        self.assertEqual(self.counts["terms"], 3)  # relation node 'has part' is skipped
        self.assertEqual(self.count("hpo_terms"), 3)

    def test_iri_converted_to_hp_id(self):
        row = self.conn.execute(
            "SELECT label, definition FROM hpo_terms WHERE hp_id = 'HP:0000002'").fetchone()
        self.assertEqual(row, ("Abnormality of body height", "Deviation from the norm of height."))

    def test_obsolete_flag_and_comment(self):
        self.assertEqual(
            self.conn.execute("SELECT is_obsolete FROM hpo_terms WHERE hp_id = 'HP:0000057'").fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute("SELECT is_obsolete FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT comment FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()[0],
            "Root of all terms.")

    def test_synonym_scope_and_type(self):
        row = self.conn.execute("SELECT hp_id, synonym, scope, synonym_type FROM hpo_synonyms").fetchone()
        self.assertEqual(row, ("HP:0000002", "Height abnormality", "exact", "layperson"))

    def test_edges_skip_unknown_parents(self):
        self.assertEqual(self.counts["edges"], 1)
        self.assertEqual(self.counts["skipped_edges"], 1)
        self.assertEqual(self.conn.execute("SELECT child_id, parent_id FROM hpo_edges").fetchall(),
                         [("HP:0000002", "HP:0000001")])

    def test_search_finds_label_and_synonym_but_not_obsolete(self):
        self.assertEqual(load_data.search_hpo(self.conn, "body height"),
                         [("HP:0000002", "Abnormality of body height")])
        self.assertEqual(load_data.search_hpo(self.conn, "Height abnormality"),
                         [("HP:0000002", "Abnormality of body height")])  # matched via synonym
        self.assertEqual(load_data.search_hpo(self.conn, "Clitoromegaly"), [])  # obsolete

    def test_no_orphan_foreign_keys(self):
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])


class TestCommandLine(TempDirTestCase):
    def test_load_stats_drug_hpo_end_to_end(self):
        ddi = self.write_json("ddi.json", DDI_FIXTURE)
        hpo = self.write_json("hp.json", HPO_FIXTURE)
        base = ["--db", str(self.db)]

        code, out = quiet_main(["load", *base, "--ddi", str(ddi), "--hpo", str(hpo)])
        self.assertEqual(code, 0, out)
        self.assertIn("loaded 3 interactions across 4 drugs", out)

        code, out = quiet_main(["stats", *base])
        self.assertEqual(code, 0, out)
        self.assertIn("pairs listed more than once: 1", out)

        code, out = quiet_main(["drug", "WARFARIN", *base])  # case-insensitive
        self.assertEqual(code, 0, out)
        self.assertIn("Warfarin + Ibuprofen", out)

        code, out = quiet_main(["hpo", "height", *base])
        self.assertEqual(code, 0, out)
        self.assertIn("HP:0000002", out)

    def test_unknown_drug_returns_nonzero(self):
        ddi = self.write_json("ddi.json", DDI_FIXTURE)
        quiet_main(["load", "--db", str(self.db), "--ddi", str(ddi), "--only", "ddi"])
        code, out = quiet_main(["drug", "nonexistentium", "--db", str(self.db)])
        self.assertEqual(code, 1)
        self.assertIn("No interactions found", out)

    def test_missing_json_file_gives_friendly_error(self):
        code, out = quiet_main(["load", "--db", str(self.db), "--only", "ddi",
                                "--ddi", str(self.dir / "nope.json")])
        self.assertEqual(code, 1)
        self.assertIn("Error:", out)

    def test_query_before_load_gives_hint(self):
        code, out = quiet_main(["stats", "--db", str(self.dir / "never_created.db")])
        self.assertEqual(code, 1)
        self.assertIn("load", out)


@unittest.skipUnless(load_data.DEFAULT_DDI.exists(), "DDI_Database.json not next to this file")
class TestRealDDIFile(TempDirTestCase):
    def test_loads_cleanly(self):
        n_inter, n_drugs = load_data.load_ddi(self.conn, load_data.DEFAULT_DDI)
        self.assertGreater(n_inter, 0)
        self.assertGreater(n_drugs, 0)
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        severities = {r[0] for r in self.conn.execute("SELECT DISTINCT severity FROM interactions")}
        self.assertLessEqual(severities, {"Major", "Moderate", "Minor"})


@unittest.skipUnless(load_data.DEFAULT_HPO.exists(), "hp.json not next to this file")
class TestRealHPOFile(TempDirTestCase):
    def test_loads_cleanly(self):
        counts = load_data.load_hpo(self.conn, load_data.DEFAULT_HPO)
        self.assertGreater(counts["terms"], 1000)
        self.assertGreater(counts["edges"], 1000)
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        root = self.conn.execute("SELECT label FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()
        self.assertIsNotNone(root)


if __name__ == "__main__":
    unittest.main()
