"""Eligibility filtering: which records count as evidence.

A record is evidence only if its source is current, has an approved authority
class and a scope other than none, and the record itself covers the exact pair
(interaction records) or the drug (monograph information). Every record gets an ``eligible`` flag and a reason
string. Missing metadata is never filled in: the filter abstains, excludes the
record, and the result goes to human review.
"""

from dataclasses import dataclass

from ddi_backend.registry import ROLE_CONTEXT, ROLE_DIRECT

INTERACTION_RECORD = "interaction"
MONOGRAPH_RECORD = "monograph"


@dataclass(frozen=True)
class Evidence:
    record: dict
    source_key: str
    role: str            # direct = names the exact pair; context = only supports it
    eligible: bool
    reason: str
    metadata: dict       # the source's full metadata as registered; None values stay None

    def to_dict(self, include_record=False):
        out = {
            "record_id": self.record.get("id"),
            "source_key": self.source_key,
            "role": self.role,
            "eligible": self.eligible,
            "reason": self.reason,
            "metadata": self.metadata,
        }
        if include_record:
            out["record"] = self.record
        return out


def _role(record, pair_names):
    """direct when the record names exactly the requested pair, otherwise context."""
    if pair_names is None:
        return ROLE_DIRECT if record.get("role", ROLE_DIRECT) == ROLE_DIRECT else ROLE_CONTEXT
    named = {str(record.get("drug_a", "")).casefold(), str(record.get("drug_b", "")).casefold()}
    return ROLE_DIRECT if named == {name.casefold() for name in pair_names} else ROLE_CONTEXT


def _names_any(record, pair_names):
    named = {str(record.get("drug_a", "")).casefold(), str(record.get("drug_b", "")).casefold(),
             str(record.get("drug", "")).casefold()}
    return bool(named & {name.casefold() for name in pair_names})


class EligibilityFilter:
    def __init__(self, registry):
        self.registry = registry

    def evaluate(self, record, pair_names=None, record_type=INTERACTION_RECORD, drug_name=None):
        """Return the Evidence verdict for one record.

        ``pair_names`` is the requested pair (interaction records); ``drug_name`` is the
        requested drug (monograph records).
        """
        key = record.get("source_key")
        role = _role(record, pair_names) if record_type == INTERACTION_RECORD else ROLE_CONTEXT
        source = self.registry.get(key) if key else None
        metadata = (
            {"status": source.status, "authority_class": source.authority_class, "scope": source.scope,
             "effective_date": source.effective_date, "owner": source.owner}
            if source else {}
        )

        def verdict(eligible, reason):
            return Evidence(record, key, role, eligible, reason, metadata)

        if not key:
            return verdict(False, "record has no source key; its source metadata is missing")
        if source is None:
            return verdict(False, f"source '{key}' is not in the registry; its metadata is missing")
        if source.status != "current":
            return verdict(False, f"source '{key}' status is '{source.status}', not current")
        if not source.authority_class:
            return verdict(False, f"source '{key}' has no authority class (missing authority)")
        if not self.registry.is_approved(source.authority_class):
            return verdict(False, f"authority class '{source.authority_class}' is not approved as evidence")
        if source.scope == "none":
            return verdict(False, f"source '{key}' has scope 'none': a vocabulary, not interaction evidence")

        if record_type == INTERACTION_RECORD:
            if role == ROLE_DIRECT:
                return verdict(True, "current source, approved authority, names the exact pair")
            if pair_names is not None and not _names_any(record, pair_names):
                return verdict(False, "record does not concern either drug in the requested pair")
            return verdict(True, "current source, approved authority; context only, it does not name the exact pair")

        # Monograph information must cover the drug itself.
        if drug_name is not None and str(record.get("drug", "")).casefold() != drug_name.casefold():
            return verdict(False, "record is about a different drug")
        return verdict(True, "current source, approved authority, covers the drug")

    def partition(self, records, pair_names=None):
        """Split records into (eligible, excluded) Evidence lists, preserving order."""
        eligible, excluded = [], []
        for record in records:
            evidence = self.evaluate(record, pair_names)
            (eligible if evidence.eligible else excluded).append(evidence)
        return eligible, excluded
