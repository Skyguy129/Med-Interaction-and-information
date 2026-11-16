"""Source registry: stable keys with manually maintained metadata.

Sources are identified by key (``kaggle_ddi``, ``hpo``), never by the S1-S5
labels used in different class experiments. Metadata comes from the seed file
and is never filled in by code: a field that is missing stays ``None``/"unknown".
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ddi_backend import load_data

DEFAULT_SEED = load_data.DATA_DIR / "sources_seed.json"

STATUSES = ("current", "archived", "unknown")
SCOPES = ("exact_pair", "single_drug", "class", "none")
ROLE_DIRECT = "direct"    # names the exact pair
ROLE_CONTEXT = "context"  # only supports it


@dataclass(frozen=True)
class Source:
    key: str
    status: str = "unknown"
    authority_class: str = None
    scope: str = "none"
    effective_date: str = None
    owner: str = None
    label: str = ""


@dataclass(frozen=True)
class Registry:
    sources: dict
    approved_authority_classes: frozenset

    def get(self, key):
        """The Source for a key, or None when the key was never registered."""
        return self.sources.get(key)

    def is_approved(self, authority_class):
        return authority_class in self.approved_authority_classes


def build_registry(seed):
    """Validate a seed dict and return a Registry. Raises ValueError on bad values."""
    sources = {}
    for key, meta in seed.get("sources", {}).items():
        status = meta.get("status", "unknown")
        scope = meta.get("scope", "none")
        if status not in STATUSES:
            raise ValueError(f"source {key!r}: status must be one of {STATUSES}, got {status!r}")
        if scope not in SCOPES:
            raise ValueError(f"source {key!r}: scope must be one of {SCOPES}, got {scope!r}")
        sources[key] = Source(
            key=key,
            status=status,
            authority_class=meta.get("authority_class"),
            scope=scope,
            effective_date=meta.get("effective_date"),
            owner=meta.get("owner"),
            label=meta.get("label", ""),
        )
    approved = frozenset(seed.get("approved_authority_classes", []))
    return Registry(sources=sources, approved_authority_classes=approved)


def load_registry(path=None):
    """Load the registry from the seed file. A missing seed means no source is registered."""
    path = Path(path) if path is not None else DEFAULT_SEED
    if not path.is_file():
        return Registry(sources={}, approved_authority_classes=frozenset())
    return build_registry(load_data.read_json(path))
