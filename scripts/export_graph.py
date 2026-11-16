"""Offline export of the interaction graph and a pair-level train/test split. Trains nothing.

    python scripts/export_graph.py --out data/graph_export
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi_backend.models import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
