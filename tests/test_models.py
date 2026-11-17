# Docstring: says this file tests the deferred neural model stubs, the offline graph export and the pair-level split.
"""Tests for the deferred neural model stubs, the offline export and the pair-level split."""

import ast  # parse source files to check what they import
import contextlib  # capture the script's printed output
import io  # StringIO buffer for that output
import tempfile  # temporary folders for exported files
import unittest  # standard test framework
from pathlib import Path  # path handling for exported and source files

from ddi_backend import models  # the module under test
from tests import support  # helpers: open the real database, build tiny databases
from tests.support import ddi_row  # builds one fake interaction row


class TestPrediction(unittest.TestCase):  # tests for the Prediction value object
    def test_every_prediction_is_labelled_predicted_not_documented(self):  # predictions can never pass as documented
        # build a valid prediction: pair A/B, probability 0.7, severity probabilities, and an uncertainty of 0.4
        p = models.Prediction("A", "B", 0.7, {"Major": 0.5, "Moderate": 0.3, "Minor": 0.2}, 0.4)
        self.assertEqual(p.label, "predicted, not documented")  # the label is fixed

    def test_invalid_predictions_are_rejected(self):  # bad inputs raise errors
        with self.assertRaises(ValueError):  # a probability above 1 is invalid
            models.Prediction("A", "B", 1.5)
        with self.assertRaises(ValueError):  # severity probabilities that sum to more than 1 are invalid
            models.Prediction("A", "B", 0.5, {"Major": 0.9, "Moderate": 0.9, "Minor": 0.9})
        with self.assertRaises(ValueError):  # the label cannot be overridden to "documented"
            models.Prediction("A", "B", 0.5, label="documented")

    def test_the_interface_is_abstract_and_the_placeholder_trains_nothing(self):  # only an interface exists
        with self.assertRaises(TypeError):  # an abstract class cannot be created
            models.InteractionPredictor()
        with self.assertRaisesRegex(NotImplementedError, "deferred"):  # the placeholder refuses to predict
            models.DeferredPredictor().predict("A", "B")


class TestExportAndSplit(unittest.TestCase):  # tests for the offline graph export and the train/test split
    def test_export_keeps_every_row_including_repeated_pairs(self):  # nothing is lost on export
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close it when the test ends
        with tempfile.TemporaryDirectory() as tmp:  # temporary output folder
            nodes, edges = models.export_graph(conn, tmp)  # write the node and edge files
            node_lines = nodes.read_text(encoding="utf-8").splitlines()  # read the node file lines
            edge_lines = edges.read_text(encoding="utf-8").splitlines()  # read the edge file lines
        self.assertEqual(len(node_lines) - 1, 210)  # 210 drugs (minus the header line)
        self.assertEqual(len(edge_lines) - 1, 180)  # 180 interaction rows (minus the header line)
        self.assertEqual(edge_lines[0], "interaction_id,source_drug_id,target_drug_id,severity")  # edge header columns

    def test_test_pairs_never_appear_in_training(self):  # no leakage between train and test
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close it when the test ends
        split = models.split_pairs(conn, test_fraction=0.25, seed=3)  # split with a fixed seed
        train = {(low, high) for low, high, _ in split["train"]}  # set of training drug pairs
        test = {(low, high) for low, high, _ in split["test"]}  # set of test drug pairs
        self.assertTrue(train and test)  # both sides have pairs
        self.assertEqual(train & test, set())  # and they do not overlap
        # count every interaction row on both sides
        total_rows = sum(len(s) for *_, s in split["train"]) + sum(len(s) for *_, s in split["test"])
        self.assertEqual(total_rows, 180)  # every row lands on exactly one side
        self.assertEqual(len(train) + len(test), 172)  # 180 rows, 8 repeated pairs

    def test_a_repeated_pair_stays_together_whichever_order_it_is_written_in(self):  # A+B and B+A are one pair
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder for a tiny database
            # build a database with A/B and B/A (same pair, different severities) plus ten unrelated minor pairs
            db = support.build_db(tmp, [ddi_row("A", "B", "Major"), ddi_row("B", "A", "Moderate")]
                                  + [ddi_row(f"X{i}", f"Y{i}", "Minor") for i in range(10)])
            from ddi_backend import ingest  # imported here because only this test needs it
            conn = ingest.open_readonly(db, ingest.CORE_TABLES)  # open the tiny database read-only
            try:  # make sure the connection is closed afterwards
                for seed in range(8):  # try several random seeds
                    split = models.split_pairs(conn, 0.3, seed)  # split with this seed
                    # find the sides holding the repeated pair (the one with two severities)
                    sides = [name for name, rows in split.items() for low, high, sev in rows if len(sev) == 2]
                    self.assertEqual(len(sides), 1)  # the pair appears on exactly one side
            finally:
                conn.close()  # always close the connection

    def test_split_is_deterministic_for_a_seed_and_validates_the_fraction(self):  # reproducible and validated
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close it when the test ends
        self.assertEqual(models.split_pairs(conn, 0.2, 1), models.split_pairs(conn, 0.2, 1))  # same seed, same split
        self.assertNotEqual(models.split_pairs(conn, 0.2, 1), models.split_pairs(conn, 0.2, 2))  # different seed differs
        with self.assertRaises(ValueError):  # a fraction above 1 is invalid
            models.split_pairs(conn, 1.5)

    def test_the_script_runs(self):  # the command line entry point works end to end
        with tempfile.TemporaryDirectory() as tmp:  # temporary output folder
            with contextlib.redirect_stdout(io.StringIO()):  # hide the script's printed output
                code = models.main(["--db", str(support.real_db_path()), "--out", tmp])  # run on the real database
            self.assertEqual(code, 0)  # exit code 0 means success
            self.assertTrue((Path(tmp) / "train_pairs.csv").is_file())  # training pairs file was written
            self.assertTrue((Path(tmp) / "test_pairs.csv").is_file())  # test pairs file was written


class TestNotOnTheUserPath(unittest.TestCase):  # proves the unused neural stub is isolated
    def test_nothing_on_the_user_path_imports_the_neural_model_stub(self):  # no user-path module imports models
        package = Path(models.__file__).parent  # the ddi_backend folder
        # the modules a user request can reach
        for name in ("api", "checker", "retrieval", "service", "agent", "llm", "composer", "rules", "eligibility"):
            tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))  # parse the module's source
            for node in ast.walk(tree):  # visit every syntax node
                imported = []  # names imported by this node
                if isinstance(node, ast.Import):  # "import x" statements
                    imported = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):  # "from x import y" statements
                    imported = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                # no import may end in "models"
                self.assertFalse([i for i in imported if i and i.endswith("models")], f"{name} imports models")

    def test_the_cli_imports_it_only_inside_the_offline_export_command(self):  # cli imports it lazily at most
        source = (Path(models.__file__).parent / "cli.py").read_text(encoding="utf-8")  # read cli.py
        tree = ast.parse(source)  # parse it
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]  # module-level imports only
        self.assertFalse([n for n in top_level if "models" in ast.dump(n)])  # none of them mention models


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the deferred neural-model module (ddi_backend/models.py): predictions are always labelled
# "predicted, not documented", the predictor is only an interface with a placeholder that trains nothing, the graph
# export keeps every row, and the train/test split is pair-level so no pair (including repeated pairs) leaks
# between sides. It also checks that no module on the user path imports this stub.
# ASSIGNMENT TOOL: the deferred neural network / graph neural network (Meetings 10 and 11): interface, offline
# graph export and pair-level split only; nothing is trained.
# ---------------------------------------------------------------------------
