"""Tests for the deferred neural model stubs, the offline export and the pair-level split."""

import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ddi_backend import models
from tests import support
from tests.support import ddi_row


class TestPrediction(unittest.TestCase):
    def test_every_prediction_is_labelled_predicted_not_documented(self):
        p = models.Prediction("A", "B", 0.7, {"Major": 0.5, "Moderate": 0.3, "Minor": 0.2}, 0.4)
        self.assertEqual(p.label, "predicted, not documented")

    def test_invalid_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            models.Prediction("A", "B", 1.5)
        with self.assertRaises(ValueError):
            models.Prediction("A", "B", 0.5, {"Major": 0.9, "Moderate": 0.9, "Minor": 0.9})
        with self.assertRaises(ValueError):
            models.Prediction("A", "B", 0.5, label="documented")

    def test_the_interface_is_abstract_and_the_placeholder_trains_nothing(self):
        with self.assertRaises(TypeError):
            models.InteractionPredictor()
        with self.assertRaisesRegex(NotImplementedError, "deferred"):
            models.DeferredPredictor().predict("A", "B")


class TestExportAndSplit(unittest.TestCase):
    def test_export_keeps_every_row_including_repeated_pairs(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        with tempfile.TemporaryDirectory() as tmp:
            nodes, edges = models.export_graph(conn, tmp)
            node_lines = nodes.read_text(encoding="utf-8").splitlines()
            edge_lines = edges.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(node_lines) - 1, 210)
        self.assertEqual(len(edge_lines) - 1, 180)
        self.assertEqual(edge_lines[0], "interaction_id,source_drug_id,target_drug_id,severity")

    def test_test_pairs_never_appear_in_training(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        split = models.split_pairs(conn, test_fraction=0.25, seed=3)
        train = {(low, high) for low, high, _ in split["train"]}
        test = {(low, high) for low, high, _ in split["test"]}
        self.assertTrue(train and test)
        self.assertEqual(train & test, set())
        total_rows = sum(len(s) for *_, s in split["train"]) + sum(len(s) for *_, s in split["test"])
        self.assertEqual(total_rows, 180)  # every row lands on exactly one side
        self.assertEqual(len(train) + len(test), 172)  # 180 rows, 8 repeated pairs

    def test_a_repeated_pair_stays_together_whichever_order_it_is_written_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = support.build_db(tmp, [ddi_row("A", "B", "Major"), ddi_row("B", "A", "Moderate")]
                                  + [ddi_row(f"X{i}", f"Y{i}", "Minor") for i in range(10)])
            from ddi_backend import ingest
            conn = ingest.open_readonly(db, ingest.CORE_TABLES)
            try:
                for seed in range(8):
                    split = models.split_pairs(conn, 0.3, seed)
                    sides = [name for name, rows in split.items() for low, high, sev in rows if len(sev) == 2]
                    self.assertEqual(len(sides), 1)
            finally:
                conn.close()

    def test_split_is_deterministic_for_a_seed_and_validates_the_fraction(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        self.assertEqual(models.split_pairs(conn, 0.2, 1), models.split_pairs(conn, 0.2, 1))
        self.assertNotEqual(models.split_pairs(conn, 0.2, 1), models.split_pairs(conn, 0.2, 2))
        with self.assertRaises(ValueError):
            models.split_pairs(conn, 1.5)

    def test_the_script_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = models.main(["--db", str(support.real_db_path()), "--out", tmp])
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "train_pairs.csv").is_file())
            self.assertTrue((Path(tmp) / "test_pairs.csv").is_file())


class TestNotOnTheUserPath(unittest.TestCase):
    def test_nothing_on_the_user_path_imports_the_neural_model_stub(self):
        package = Path(models.__file__).parent
        for name in ("api", "checker", "retrieval", "service", "agent", "llm", "composer", "rules", "eligibility"):
            tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                self.assertFalse([i for i in imported if i and i.endswith("models")], f"{name} imports models")

    def test_the_cli_imports_it_only_inside_the_offline_export_command(self):
        source = (Path(models.__file__).parent / "cli.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse([n for n in top_level if "models" in ast.dump(n)])


if __name__ == "__main__":
    unittest.main()
