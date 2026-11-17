# Module docstring: explains the registry holds source metadata by stable key and never invents missing values
"""Source registry: stable keys with manually maintained metadata.

Sources are identified by key (``kaggle_ddi``, ``hpo``), never by the S1-S5
labels used in different class experiments. Metadata comes from the seed file
and is never filled in by code: a field that is missing stays ``None``/"unknown".
"""

import json  # (imported for JSON handling; the actual file reading goes through load_data.read_json)
from dataclasses import dataclass  # lets Source and Registry be plain immutable data holders
from pathlib import Path  # used to treat the seed file location as a path object

from ddi_backend import load_data  # supplies DATA_DIR and the shared JSON reader

DEFAULT_SEED = load_data.DATA_DIR / "sources_seed.json"  # where the team's source metadata lives

STATUSES = ("current", "archived", "unknown")  # the only allowed source statuses
SCOPES = ("exact_pair", "single_drug", "class", "none")  # the only allowed source scopes
ROLE_DIRECT = "direct"    # names the exact pair
ROLE_CONTEXT = "context"  # only supports it


@dataclass(frozen=True)  # immutable record: a source's metadata cannot change after it is built
class Source:
    key: str  # stable identifier such as "kaggle_ddi"
    status: str = "unknown"  # current / archived / unknown; only "current" can count as evidence
    authority_class: str = None  # kind of authority (e.g. course_designated_dataset); None when not supplied
    scope: str = "none"  # what the source covers; "none" means it is never interaction evidence
    effective_date: str = None  # date the source applies from; left None when unknown (never guessed)
    owner: str = None  # who maintains the source; None until the team fills it in
    label: str = ""  # optional human-readable name


@dataclass(frozen=True)  # immutable bundle of all sources plus the approved authority list
class Registry:
    sources: dict  # key -> Source
    approved_authority_classes: frozenset  # authority classes the project accepts as evidence

    def get(self, key):  # look a source up by key
        """The Source for a key, or None when the key was never registered."""
        return self.sources.get(key)  # None for unregistered keys, so the filter can abstain

    def is_approved(self, authority_class):  # check an authority class against the approved list
        return authority_class in self.approved_authority_classes  # True only for listed classes


def build_registry(seed):  # turn the raw seed dict into a validated Registry
    """Validate a seed dict and return a Registry. Raises ValueError on bad values."""
    sources = {}  # collects the Source objects by key
    for key, meta in seed.get("sources", {}).items():  # go through each source entry in the seed
        status = meta.get("status", "unknown")  # missing status stays "unknown"
        scope = meta.get("scope", "none")  # missing scope stays "none"
        if status not in STATUSES:  # reject statuses we do not recognise
            raise ValueError(f"source {key!r}: status must be one of {STATUSES}, got {status!r}")
        if scope not in SCOPES:  # reject scopes we do not recognise
            raise ValueError(f"source {key!r}: scope must be one of {SCOPES}, got {scope!r}")
        sources[key] = Source(  # build the immutable Source for this key
            key=key,  # keep the stable key
            status=status,  # validated status
            authority_class=meta.get("authority_class"),  # None if absent: never filled in by code
            scope=scope,  # validated scope
            effective_date=meta.get("effective_date"),  # None if absent
            owner=meta.get("owner"),  # None if absent
            label=meta.get("label", ""),  # empty label if absent
        )
    approved = frozenset(seed.get("approved_authority_classes", []))  # the approved list, empty if none given
    return Registry(sources=sources, approved_authority_classes=approved)  # hand back the finished registry


def load_registry(path=None):  # read the seed file from disk and build a Registry
    """Load the registry from the seed file. A missing seed means no source is registered."""
    path = Path(path) if path is not None else DEFAULT_SEED  # use the default seed when no path is given
    if not path.is_file():  # no seed file at all
        return Registry(sources={}, approved_authority_classes=frozenset())  # empty registry: nothing is eligible
    return build_registry(load_data.read_json(path))  # parse the JSON and validate it

# ---------------------------------------------------------------------------
# FILE SUMMARY: Reads data/sources_seed.json and turns it into an immutable registry of sources (kaggle_ddi, hpo) with
# status, authority class, scope, date and owner, plus the list of approved authority classes. It validates values and
# never fills in missing metadata, so unknowns stay unknown for the eligibility filter to act on.
# ASSIGNMENT TOOL: metadata-aware eligibility filtering (it is the source-metadata half; eligibility.py is the filter).
# ---------------------------------------------------------------------------
