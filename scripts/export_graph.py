# Module docstring: says this script exports the graph and a train/test split offline and trains nothing, plus usage.
"""Offline export of the interaction graph and a pair-level train/test split. Trains nothing.

    python scripts/export_graph.py --out data/graph_export
"""

import sys  # used for sys.path and the process exit code
from pathlib import Path  # builds the repository-root path

# Put the repository root on the import path so `ddi_backend` can be imported when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the real command-line entry point from the models module (import is after the path edit, hence noqa).
from ddi_backend.models import main  # noqa: E402

# Only run when executed as a script, not when imported.
if __name__ == "__main__":
    sys.exit(main())  # run the export and pass its return code (0 ok, 1 error) to the shell

# ---------------------------------------------------------------------------
# FILE SUMMARY: A thin launcher script. It makes the repository importable and calls ddi_backend.models.main(), which
# writes nodes.csv, edges.csv and the pair-level train/test CSVs for the interaction graph. It trains nothing.
# ASSIGNMENT TOOL: Deferred neural network / graph neural network (Meetings 10 and 11): this is the offline graph
# export that a future prototype would start from.
# ---------------------------------------------------------------------------
