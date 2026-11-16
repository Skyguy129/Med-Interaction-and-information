"""The deferred neural model: an interface, an offline export and a pair-level split. Nothing is trained.

The neural network and graph neural network from Meetings 10 and 11 are deferred. This
module only defines what a predictor would look like, exports the graph so a future
prototype can start from files, and builds a pair-level train/test split so that a test
pair never appears in training. It is not connected to the user path. If a prototype is
ever built, start with a simple scikit-learn baseline (as in Meeting 10) before any graph
model, and label every prediction "predicted, not documented".
"""

import argparse
import csv
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ddi_backend import ingest, load_data
from ddi_backend.common import SEVERITIES

PREDICTED_LABEL = "predicted, not documented"


@dataclass(frozen=True)
class Prediction:
    """What a predictor returns. It is never documented evidence."""
    drug_a: str
    drug_b: str
    interaction_probability: float
    severity_distribution: dict = field(default_factory=dict)  # {"Major": p, "Moderate": p, "Minor": p}
    confidence: float = 0.0
    label: str = PREDICTED_LABEL

    def __post_init__(self):
        if not 0.0 <= self.interaction_probability <= 1.0:
            raise ValueError("interaction_probability must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.severity_distribution:
            if set(self.severity_distribution) != set(SEVERITIES):
                raise ValueError(f"severity_distribution needs exactly the keys {SEVERITIES}")
            if abs(sum(self.severity_distribution.values()) - 1.0) > 1e-6:
                raise ValueError("severity_distribution must sum to 1")
        if self.label != PREDICTED_LABEL:
            raise ValueError(f"a prediction must be labelled {PREDICTED_LABEL!r}")


class InteractionPredictor(ABC):
    """Interface for a future model: an interaction probability, a severity distribution and a confidence."""

    @abstractmethod
    def predict(self, drug_a, drug_b):
        """Return a Prediction for the pair."""


class DeferredPredictor(InteractionPredictor):
    """The placeholder: the neural model is deferred, so asking it for anything is an error."""

    def predict(self, drug_a, drug_b):
        raise NotImplementedError("The neural model is deferred; nothing is trained or deployed.")


# --------------------------------------------------------------------------
# Offline export
# --------------------------------------------------------------------------
def export_graph(conn, out_dir):
    """Write nodes.csv and edges.csv for the interaction graph. Returns the two paths.

    One edge row per interaction row, so a repeated pair stays visible with every severity.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes, edges = out / "nodes.csv", out / "edges.csv"
    with open(nodes, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_id", "name"])
        writer.writerows(conn.execute("SELECT drug_id, name FROM drugs ORDER BY drug_id"))
    with open(edges, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["interaction_id", "source_drug_id", "target_drug_id", "severity"])
        writer.writerows(conn.execute(
            "SELECT interaction_id, MIN(drug_a_id, drug_b_id), MAX(drug_a_id, drug_b_id), severity "
            "FROM interactions ORDER BY interaction_id"))
    return nodes, edges


def split_pairs(conn, test_fraction=0.2, seed=0):
    """Pair-level split. Every row of a pair goes to the same side, so no test pair is ever in training.

    Returns {"train": [(low_id, high_id, [severities])], "test": [...]} with unordered pairs.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    pairs = {}
    for low, high, severity in conn.execute(
            "SELECT MIN(drug_a_id, drug_b_id), MAX(drug_a_id, drug_b_id), severity FROM interactions "
            "ORDER BY interaction_id"):
        pairs.setdefault((low, high), []).append(severity)
    keys = sorted(pairs)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    test, train = keys[:n_test], keys[n_test:]
    as_rows = lambda ks: [(low, high, pairs[(low, high)]) for low, high in sorted(ks)]  # noqa: E731
    return {"train": as_rows(train), "test": as_rows(test)}


def write_split(conn, out_dir, test_fraction=0.2, seed=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = split_pairs(conn, test_fraction, seed)
    paths = {}
    for name, rows in split.items():
        path = out / f"{name}_pairs.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_drug_id", "target_drug_id", "severities"])
            writer.writerows((low, high, "/".join(sev)) for low, high, sev in rows)
        paths[name] = path
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the interaction graph and a pair-level split (offline; trains nothing).")
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=load_data.ROOT / "data" / "graph_export")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        conn = ingest.open_readonly(args.db, ingest.CORE_TABLES)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        nodes, edges = export_graph(conn, args.out)
        paths = write_split(conn, args.out, args.test_fraction, args.seed)
    finally:
        conn.close()
    print(f"Wrote {nodes}, {edges}, {paths['train']}, {paths['test']}")
    print("Nothing was trained. Any future prediction must be labelled " + repr(PREDICTED_LABEL) + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
