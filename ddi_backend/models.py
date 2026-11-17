# Module docstring: this module is only an interface, an export and a split for the deferred neural model.
"""The deferred neural model: an interface, an offline export and a pair-level split. Nothing is trained.

The neural network and graph neural network from Meetings 10 and 11 are deferred. This
module only defines what a predictor would look like, exports the graph so a future
prototype can start from files, and builds a pair-level train/test split so that a test
pair never appears in training. It is not connected to the user path. If a prototype is
ever built, start with a simple scikit-learn baseline (as in Meeting 10) before any graph
model, and label every prediction "predicted, not documented".
"""

import argparse  # command-line options for main()
import csv  # writes the node, edge and split CSV files
import random  # seeded shuffle for the train/test split
import sys  # stderr and exit code
from abc import ABC, abstractmethod  # lets InteractionPredictor be an abstract interface
from dataclasses import dataclass, field  # Prediction is an immutable record
from pathlib import Path  # output folder paths

from ddi_backend import ingest, load_data  # read-only database access and the default database path
from ddi_backend.common import SEVERITIES  # the allowed severity names (Major, Moderate, Minor)

# The only label a prediction may carry, so it can never be mistaken for documented evidence.
PREDICTED_LABEL = "predicted, not documented"


@dataclass(frozen=True)  # frozen: a Prediction cannot be edited after it is made
class Prediction:
    # Docstring: a prediction is never documented evidence.
    """What a predictor returns. It is never documented evidence."""
    drug_a: str  # first drug name
    drug_b: str  # second drug name
    interaction_probability: float  # model's chance that the pair interacts, 0 to 1
    severity_distribution: dict = field(default_factory=dict)  # {"Major": p, "Moderate": p, "Minor": p}
    confidence: float = 0.0  # model confidence, 0 to 1
    label: str = PREDICTED_LABEL  # must stay the predicted-not-documented label

    def __post_init__(self):  # runs right after creation to validate the values
        if not 0.0 <= self.interaction_probability <= 1.0:  # probability out of range
            raise ValueError("interaction_probability must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:  # confidence out of range
            raise ValueError("confidence must be between 0 and 1")
        if self.severity_distribution:  # only checked when a distribution is given
            if set(self.severity_distribution) != set(SEVERITIES):  # keys must be exactly the three severities
                raise ValueError(f"severity_distribution needs exactly the keys {SEVERITIES}")
            if abs(sum(self.severity_distribution.values()) - 1.0) > 1e-6:  # must be a probability distribution
                raise ValueError("severity_distribution must sum to 1")
        if self.label != PREDICTED_LABEL:  # the label cannot be changed to look like evidence
            raise ValueError(f"a prediction must be labelled {PREDICTED_LABEL!r}")


class InteractionPredictor(ABC):
    # Docstring: the contract a future model would follow.
    """Interface for a future model: an interaction probability, a severity distribution and a confidence."""

    @abstractmethod  # subclasses must implement predict
    def predict(self, drug_a, drug_b):
        # Docstring: returns a Prediction for the pair.
        """Return a Prediction for the pair."""


class DeferredPredictor(InteractionPredictor):
    # Docstring: placeholder because the neural model is deferred.
    """The placeholder: the neural model is deferred, so asking it for anything is an error."""

    def predict(self, drug_a, drug_b):
        # Always refuse, so nothing can accidentally use an untrained model.
        raise NotImplementedError("The neural model is deferred; nothing is trained or deployed.")


# --------------------------------------------------------------------------
# Offline export
# --------------------------------------------------------------------------
def export_graph(conn, out_dir):
    # Docstring: explains the two output files and that repeated pairs keep every row.
    """Write nodes.csv and edges.csv for the interaction graph. Returns the two paths.

    One edge row per interaction row, so a repeated pair stays visible with every severity.
    """
    out = Path(out_dir)  # make sure we have a Path
    out.mkdir(parents=True, exist_ok=True)  # create the output folder if needed
    nodes, edges = out / "nodes.csv", out / "edges.csv"  # the two file paths
    with open(nodes, "w", newline="", encoding="utf-8") as f:  # newline="" stops blank rows on Windows
        writer = csv.writer(f)  # CSV writer for the nodes file
        writer.writerow(["drug_id", "name"])  # header row
        writer.writerows(conn.execute("SELECT drug_id, name FROM drugs ORDER BY drug_id"))  # one node per drug
    with open(edges, "w", newline="", encoding="utf-8") as f:  # edges file
        writer = csv.writer(f)  # CSV writer for the edges file
        writer.writerow(["interaction_id", "source_drug_id", "target_drug_id", "severity"])  # header row
        # One edge per interaction row; MIN/MAX put the lower drug id first so edges are unordered pairs.
        writer.writerows(conn.execute(
            "SELECT interaction_id, MIN(drug_a_id, drug_b_id), MAX(drug_a_id, drug_b_id), severity "
            "FROM interactions ORDER BY interaction_id"))
    return nodes, edges  # let the caller report the paths


def split_pairs(conn, test_fraction=0.2, seed=0):
    # Docstring: the split is by pair so a test pair is never in training.
    """Pair-level split. Every row of a pair goes to the same side, so no test pair is ever in training.

    Returns {"train": [(low_id, high_id, [severities])], "test": [...]} with unordered pairs.
    """
    if not 0.0 < test_fraction < 1.0:  # a split needs both sides non-empty
        raise ValueError("test_fraction must be between 0 and 1")
    pairs = {}  # (low_id, high_id) -> list of severities for that pair
    # Read every interaction row with the lower drug id first, in a fixed order.
    for low, high, severity in conn.execute(
            "SELECT MIN(drug_a_id, drug_b_id), MAX(drug_a_id, drug_b_id), severity FROM interactions "
            "ORDER BY interaction_id"):
        pairs.setdefault((low, high), []).append(severity)  # group repeated pairs together
    keys = sorted(pairs)  # sorted first so the shuffle is repeatable for a given seed
    random.Random(seed).shuffle(keys)  # seeded shuffle, independent of global random state
    n_test = max(1, round(len(keys) * test_fraction))  # at least one test pair
    test, train = keys[:n_test], keys[n_test:]  # first n_test shuffled pairs are test, the rest train
    # Turn a list of pair keys into sorted (low, high, severities) rows.
    as_rows = lambda ks: [(low, high, pairs[(low, high)]) for low, high in sorted(ks)]  # noqa: E731
    return {"train": as_rows(train), "test": as_rows(test)}  # the two sides


def write_split(conn, out_dir, test_fraction=0.2, seed=0):
    out = Path(out_dir)  # make sure we have a Path
    out.mkdir(parents=True, exist_ok=True)  # create the output folder if needed
    split = split_pairs(conn, test_fraction, seed)  # compute the train/test split
    paths = {}  # name ("train"/"test") -> file path
    for name, rows in split.items():  # one file per side
        path = out / f"{name}_pairs.csv"  # train_pairs.csv or test_pairs.csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)  # CSV writer
            writer.writerow(["source_drug_id", "target_drug_id", "severities"])  # header row
            # Join repeated-pair severities with "/" so the pair stays on one row.
            writer.writerows((low, high, "/".join(sev)) for low, high, sev in rows)
        paths[name] = path  # remember the path for the caller
    return paths  # {"train": path, "test": path}


def main(argv=None):
    # Build the command-line parser.
    parser = argparse.ArgumentParser(description="Export the interaction graph and a pair-level split (offline; trains nothing).")
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)  # database to read
    parser.add_argument("--out", type=Path, default=load_data.ROOT / "data" / "graph_export")  # output folder
    parser.add_argument("--test-fraction", type=float, default=0.2)  # share of pairs held out for testing
    parser.add_argument("--seed", type=int, default=0)  # shuffle seed
    args = parser.parse_args(argv)  # read the options
    try:
        conn = ingest.open_readonly(args.db, ingest.CORE_TABLES)  # open the database read-only
    except RuntimeError as error:  # missing or incomplete database
        print(f"Error: {error}", file=sys.stderr)
        return 1  # failure exit code
    try:
        nodes, edges = export_graph(conn, args.out)  # write nodes.csv and edges.csv
        paths = write_split(conn, args.out, args.test_fraction, args.seed)  # write train and test CSVs
    finally:
        conn.close()  # always release the database
    print(f"Wrote {nodes}, {edges}, {paths['train']}, {paths['test']}")  # list the four files written
    print("Nothing was trained. Any future prediction must be labelled " + repr(PREDICTED_LABEL) + ".")  # reminder
    return 0  # success exit code


# Only run when executed as a script.
if __name__ == "__main__":
    sys.exit(main())  # run main and pass its return code to the shell

# ---------------------------------------------------------------------------
# FILE SUMMARY: Holds everything that exists for the deferred neural model: a validated Prediction record that must
# carry the "predicted, not documented" label, an abstract InteractionPredictor interface, a DeferredPredictor that
# refuses to run, and offline helpers that export nodes/edges CSVs and make a pair-level train/test split. Nothing is
# trained and nothing here is on the user path.
# ASSIGNMENT TOOL: Deferred neural network / graph neural network (Meetings 10 and 11).
# ---------------------------------------------------------------------------
