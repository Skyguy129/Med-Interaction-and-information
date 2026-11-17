# Docstring: says this file tests load_data.py, uses only the standard library, and explains how to run it.
"""Tests for load_data.py. Standard library only.

Run from this folder:

    python -m unittest -v

Most tests use tiny fixtures written to a temp folder. The last two classes
run against the real DDI_Database.json / hp.json if they sit next to this file.
"""

import contextlib  # redirect_stdout / redirect_stderr to capture CLI output
import io  # StringIO buffers for captured output
import json  # write the JSON fixtures to disk
import sqlite3  # imported for database types (not used directly in this file)
import tempfile  # temporary folder for the test database and fixture files
import unittest  # standard test framework
from pathlib import Path  # path handling for the temp files

import ddi_backend.load_data as load_data  # the module under test

IRI = "http://purl.obolibrary.org/obo/"  # prefix that HPO term IDs carry in the JSON file


def ddi_row(a, b, severity, **overrides):  # builds one fake interaction row for the DDI fixture
    row = {  # default values for every field the loader requires
        "drug_a": a, "drug_b": b, "severity": severity,  # the pair and its severity
        "mechanism": "mech", "effect": "effect text",  # placeholder mechanism and effect text
        "Safer_alternative": "alt", "rationale": "why", "reference": "ref",  # placeholder remaining fields
    }
    row.update(overrides)  # let the caller replace any field
    return row  # the finished row


# A tiny fake DDI file: severities are grouped under major / moderate / minor.
DDI_FIXTURE = {"drug_interactions": {
    "major": [  # rows listed as major
        ddi_row("Warfarin", "Ibuprofen", "Major", Safer_alternative="Acetaminophen"),  # a normal documented pair
        ddi_row("Clopidogrel", "Omeprazole", "Major"),  # the first half of a repeated pair
    ],
    # Same pair as above: reversed order, different case, conflicting severity.
    "moderate": [ddi_row("omeprazole", "Clopidogrel", "Moderate")],
    "minor": [],  # no minor rows
}}

# A tiny fake HPO file in the same graph layout as the real hp.json.
HPO_FIXTURE = {"graphs": [{
    "id": "http://purl.obolibrary.org/obo/hp.json",  # graph ID
    "nodes": [  # the terms
        {"id": IRI + "HP_0000001", "lbl": "All", "type": "CLASS",  # root term
         "meta": {"comments": ["Root of all terms."]}},  # it carries a comment
        {"id": IRI + "HP_0000002", "lbl": "Abnormality of body height", "type": "CLASS",  # a normal term
         "meta": {
             "definition": {"val": "Deviation from the norm of height."},  # its definition text
             "synonyms": [{"pred": "hasExactSynonym", "val": "Height abnormality",  # an exact synonym
                           "synonymType": "http://purl.obolibrary.org/obo/hp#layperson"}],  # of the layperson type
         }},
        {"id": IRI + "HP_0000057", "lbl": "obsolete Clitoromegaly", "type": "CLASS",  # an obsolete term
         "meta": {"deprecated": True}},  # flagged as deprecated
        # A relation node, not a phenotype term: must be ignored.
        {"id": IRI + "BFO_0000051", "lbl": "has part", "type": "PROPERTY"},
    ],
    "edges": [  # parent-child links
        {"sub": IRI + "HP_0000002", "pred": "is_a", "obj": IRI + "HP_0000001"},  # body height is a child of All
        {"sub": IRI + "HP_0000002", "pred": "is_a", "obj": IRI + "HP_9999999"},  # parent missing
    ],
}]}


def quiet_main(argv):  # runs the CLI while capturing everything it prints
    """Run the CLI without cluttering test output; returns (exit_code, printed_text)."""
    out, err = io.StringIO(), io.StringIO()  # buffers for standard output and standard error
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):  # capture both streams
        code = load_data.main(argv)  # run the loader's command line with these arguments
    return code, out.getvalue() + err.getvalue()  # exit code plus all text printed


class TempDirTestCase(unittest.TestCase):  # base class giving each test a fresh temp folder and database
    def setUp(self):  # runs before every test
        self._tmp = tempfile.TemporaryDirectory()  # create the temp folder
        self.addCleanup(self._tmp.cleanup)  # delete it when the test ends
        self.dir = Path(self._tmp.name)  # folder path
        self.db = self.dir / "test.db"  # where the test database lives
        self.conn = load_data.connect(self.db)  # open (and create) the database with its schema
        self.addCleanup(self.conn.close)  # close it when the test ends

    def write_json(self, name, data):  # helper: write a JSON file into the temp folder
        path = self.dir / name  # target path
        path.write_text(json.dumps(data), encoding="utf-8")  # write the data as JSON
        return path  # return the path for the test to use

    def count(self, table):  # helper: number of rows in a table
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # run COUNT(*) and return the number


class TestDDILoader(TempDirTestCase):  # tests for loading interactions
    def setUp(self):  # runs before every test in this class
        super().setUp()  # create the temp folder and database
        self.path = self.write_json("ddi.json", DDI_FIXTURE)  # write the fake DDI file

    def test_counts(self):  # row and drug counts are right
        n_inter, n_drugs = load_data.load_ddi(self.conn, self.path)  # load the fixture
        self.assertEqual(n_inter, 3)  # three interaction rows reported
        self.assertEqual(n_drugs, 4)  # four distinct drugs reported
        self.assertEqual(self.count("interactions"), 3)  # three rows are really in the table
        self.assertEqual(self.count("drugs"), 4)  # four rows are really in the drugs table

    def test_drug_names_are_case_insensitive(self):  # "Omeprazole" and "omeprazole" are one drug
        load_data.load_ddi(self.conn, self.path)  # load the fixture
        # 'Omeprazole' and 'omeprazole' must be the same drug.
        n = self.conn.execute("SELECT COUNT(*) FROM drugs WHERE name = 'OMEPRAZOLE'").fetchone()[0]  # look it up in caps
        self.assertEqual(n, 1)  # exactly one drug row matches

    def test_fields_are_stored(self):  # stored columns come back as loaded
        load_data.load_ddi(self.conn, self.path)  # load the fixture
        rows = load_data.find_interactions(self.conn, "warfarin")  # look up warfarin's interactions
        # severity, both drugs, effect and safer alternative as stored
        self.assertEqual(rows, [("Major", "Warfarin", "Ibuprofen", "effect text", "Acetaminophen")])

    def test_repeated_pairs_are_kept_and_reported(self):  # a pair listed twice is kept and flagged
        load_data.load_ddi(self.conn, self.path)  # load the fixture
        repeated = load_data.find_repeated_pairs(self.conn)  # ask which pairs appear more than once
        self.assertEqual(len(repeated), 1)  # exactly one repeated pair
        a, b, severities = repeated[0]  # its two drugs and the joined severities
        self.assertEqual({a, b}, {"Clopidogrel", "Omeprazole"})  # it is the clopidogrel/omeprazole pair
        self.assertEqual(set(severities.split(" / ")), {"Major", "Moderate"})  # both severities are listed

    def test_reload_does_not_duplicate(self):  # loading twice leaves the same data
        load_data.load_ddi(self.conn, self.path)  # first load
        load_data.load_ddi(self.conn, self.path)  # second load of the same file
        self.assertEqual(self.count("interactions"), 3)  # still three interactions
        self.assertEqual(self.count("drugs"), 4)  # still four drugs

    def test_no_orphan_foreign_keys(self):  # every link points at a real row
        load_data.load_ddi(self.conn, self.path)  # load the fixture
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])  # SQLite reports no violations

    def test_wrong_structure_raises(self):  # a file without the expected keys is rejected
        bad = self.write_json("bad.json", {"something_else": []})  # a file with the wrong layout
        with self.assertRaises(ValueError):  # loading must fail
            load_data.load_ddi(self.conn, bad)

    def test_missing_field_raises_and_saves_nothing(self):  # a bad row rolls back the whole load
        broken = {"drug_interactions": {"major": [{"drug_a": "X", "drug_b": "Y"}]}}  # no severity
        path = self.write_json("broken.json", broken)  # write the broken file
        with self.assertRaises(ValueError):  # loading must fail
            load_data.load_ddi(self.conn, path)
        self.assertEqual(self.count("interactions"), 0)  # and nothing was saved


class TestHPOLoader(TempDirTestCase):  # tests for loading phenotype terms
    def setUp(self):  # runs before every test in this class
        super().setUp()  # create the temp folder and database
        self.path = self.write_json("hp.json", HPO_FIXTURE)  # write the fake HPO file
        self.counts = load_data.load_hpo(self.conn, self.path)  # load it and keep the reported counts

    def test_only_hp_terms_are_loaded(self):  # non-phenotype nodes are skipped
        self.assertEqual(self.counts["terms"], 3)  # relation node 'has part' is skipped
        self.assertEqual(self.count("hpo_terms"), 3)  # three terms are in the table

    def test_iri_converted_to_hp_id(self):  # long IRIs become short HP:<number> IDs
        row = self.conn.execute(  # fetch one term by its short ID
            "SELECT label, definition FROM hpo_terms WHERE hp_id = 'HP:0000002'").fetchone()
        self.assertEqual(row, ("Abnormality of body height", "Deviation from the norm of height."))  # label and text

    def test_obsolete_flag_and_comment(self):  # obsolete flag and comment are stored
        self.assertEqual(  # the deprecated term is flagged obsolete
            self.conn.execute("SELECT is_obsolete FROM hpo_terms WHERE hp_id = 'HP:0000057'").fetchone()[0], 1)
        self.assertEqual(  # a normal term is not obsolete
            self.conn.execute("SELECT is_obsolete FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()[0], 0)
        self.assertEqual(  # the root term's comment is stored
            self.conn.execute("SELECT comment FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()[0],
            "Root of all terms.")

    def test_synonym_scope_and_type(self):  # synonym details are parsed
        row = self.conn.execute("SELECT hp_id, synonym, scope, synonym_type FROM hpo_synonyms").fetchone()  # one synonym
        self.assertEqual(row, ("HP:0000002", "Height abnormality", "exact", "layperson"))  # scope and type simplified

    def test_edges_skip_unknown_parents(self):  # links to missing terms are dropped and counted
        self.assertEqual(self.counts["edges"], 1)  # one edge was kept
        self.assertEqual(self.counts["skipped_edges"], 1)  # one was skipped
        self.assertEqual(self.conn.execute("SELECT child_id, parent_id FROM hpo_edges").fetchall(),  # the kept edge
                         [("HP:0000002", "HP:0000001")])

    def test_search_finds_label_and_synonym_but_not_obsolete(self):  # search rules for phenotype terms
        self.assertEqual(load_data.search_hpo(self.conn, "body height"),  # match on the label
                         [("HP:0000002", "Abnormality of body height")])
        self.assertEqual(load_data.search_hpo(self.conn, "Height abnormality"),  # match on the synonym
                         [("HP:0000002", "Abnormality of body height")])  # matched via synonym
        self.assertEqual(load_data.search_hpo(self.conn, "Clitoromegaly"), [])  # obsolete

    def test_no_orphan_foreign_keys(self):  # every link points at a real row
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])  # SQLite reports no violations


class TestCommandLine(TempDirTestCase):  # tests for the load / stats / drug / hpo commands
    def test_load_stats_drug_hpo_end_to_end(self):  # run the four commands one after another
        ddi = self.write_json("ddi.json", DDI_FIXTURE)  # fake DDI file
        hpo = self.write_json("hp.json", HPO_FIXTURE)  # fake HPO file
        base = ["--db", str(self.db)]  # the shared database argument

        code, out = quiet_main(["load", *base, "--ddi", str(ddi), "--hpo", str(hpo)])  # load both files
        self.assertEqual(code, 0, out)  # success
        self.assertIn("loaded 3 interactions across 4 drugs", out)  # summary line is printed

        code, out = quiet_main(["stats", *base])  # show statistics
        self.assertEqual(code, 0, out)  # success
        self.assertIn("pairs listed more than once: 1", out)  # the repeated pair is reported

        code, out = quiet_main(["drug", "WARFARIN", *base])  # case-insensitive
        self.assertEqual(code, 0, out)  # success
        self.assertIn("Warfarin + Ibuprofen", out)  # the interaction is printed

        code, out = quiet_main(["hpo", "height", *base])  # search phenotype terms
        self.assertEqual(code, 0, out)  # success
        self.assertIn("HP:0000002", out)  # the matching term is printed

    def test_unknown_drug_returns_nonzero(self):  # no match gives exit code 1
        ddi = self.write_json("ddi.json", DDI_FIXTURE)  # fake DDI file
        quiet_main(["load", "--db", str(self.db), "--ddi", str(ddi), "--only", "ddi"])  # load only interactions
        code, out = quiet_main(["drug", "nonexistentium", "--db", str(self.db)])  # look up a drug that is not there
        self.assertEqual(code, 1)  # failure exit code
        self.assertIn("No interactions found", out)  # friendly message

    def test_missing_json_file_gives_friendly_error(self):  # a missing input file is reported, not a crash
        code, out = quiet_main(["load", "--db", str(self.db), "--only", "ddi",
                                "--ddi", str(self.dir / "nope.json")])  # point at a file that does not exist
        self.assertEqual(code, 1)  # failure exit code
        self.assertIn("Error:", out)  # message starts with "Error:"

    def test_query_before_load_gives_hint(self):  # querying an empty database suggests loading first
        code, out = quiet_main(["stats", "--db", str(self.dir / "never_created.db")])  # a database that was never built
        self.assertEqual(code, 1)  # failure exit code
        self.assertIn("load", out)  # message mentions the load command


# run this class only if the real interaction file is present
@unittest.skipUnless(load_data.DEFAULT_DDI.exists(), "DDI_Database.json not next to this file")
class TestRealDDIFile(TempDirTestCase):  # sanity checks on the real data file
    def test_loads_cleanly(self):  # the real file loads without errors
        n_inter, n_drugs = load_data.load_ddi(self.conn, load_data.DEFAULT_DDI)  # load the real file
        self.assertGreater(n_inter, 0)  # some interactions were loaded
        self.assertGreater(n_drugs, 0)  # some drugs were loaded
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])  # no broken links
        severities = {r[0] for r in self.conn.execute("SELECT DISTINCT severity FROM interactions")}  # severities used
        self.assertLessEqual(severities, {"Major", "Moderate", "Minor"})  # only the three allowed values


# run this class only if the real HPO file is present
@unittest.skipUnless(load_data.DEFAULT_HPO.exists(), "hp.json not next to this file")
class TestRealHPOFile(TempDirTestCase):  # sanity checks on the real HPO file
    def test_loads_cleanly(self):  # the real file loads without errors
        counts = load_data.load_hpo(self.conn, load_data.DEFAULT_HPO)  # load the real file
        self.assertGreater(counts["terms"], 1000)  # many terms loaded
        self.assertGreater(counts["edges"], 1000)  # many parent-child links loaded
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])  # no broken links
        root = self.conn.execute("SELECT label FROM hpo_terms WHERE hp_id = 'HP:0000001'").fetchone()  # the root term
        self.assertIsNotNone(root)  # the root exists


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the data loader (ddi_backend/load_data.py) that builds medications.db from the interaction
# JSON and the HPO JSON. It covers counts, case-insensitive drug names, transactional loading with rollback,
# idempotent reloads, foreign-key integrity, repeated pairs being kept and reported, HPO ID conversion, obsolete
# terms, synonyms, the command line, and sanity runs on the real data files.
# ASSIGNMENT TOOL: knowledge graph + rule-based reasoning (the loader builds the tables that become the graph's
# nodes and edges); it also loads HPO, the vocabulary used for symptom mapping.
# ---------------------------------------------------------------------------
