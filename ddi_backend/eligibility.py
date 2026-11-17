# Module docstring: explains which records count as evidence and that missing metadata means exclusion
"""Eligibility filtering: which records count as evidence.

A record is evidence only if its source is current, has an approved authority
class and a scope other than none, and the record itself covers the exact pair
(interaction records) or the drug (monograph information). Every record gets an ``eligible`` flag and a reason
string. Missing metadata is never filled in: the filter abstains, excludes the
record, and the result goes to human review.
"""

from dataclasses import dataclass  # makes Evidence a simple immutable data holder

from ddi_backend.registry import ROLE_CONTEXT, ROLE_DIRECT  # the two roles a record can play

INTERACTION_RECORD = "interaction"  # record type: a drug-pair interaction row
MONOGRAPH_RECORD = "monograph"  # record type: single-drug information (not present in our data)


@dataclass(frozen=True)  # immutable verdict object
class Evidence:
    record: dict  # the record that was judged
    source_key: str  # which registered source it came from
    role: str            # direct = names the exact pair; context = only supports it
    eligible: bool  # True if the record may be used as evidence
    reason: str  # plain-language explanation of the verdict
    metadata: dict       # the source's full metadata as registered; None values stay None

    def to_dict(self, include_record=False):  # JSON-friendly form of the verdict
        out = {  # the always-included fields
            "record_id": self.record.get("id"),  # e.g. "ddi:12"
            "source_key": self.source_key,  # source key
            "role": self.role,  # direct or context
            "eligible": self.eligible,  # the verdict flag
            "reason": self.reason,  # why
            "metadata": self.metadata,  # source metadata snapshot
        }
        if include_record:  # only attach the full record when asked
            out["record"] = self.record  # the original record
        return out  # hand back the dict


def _role(record, pair_names):  # decide whether a record names the requested pair exactly
    """direct when the record names exactly the requested pair, otherwise context."""
    if pair_names is None:  # no requested pair to compare against
        return ROLE_DIRECT if record.get("role", ROLE_DIRECT) == ROLE_DIRECT else ROLE_CONTEXT  # trust the record's own role
    named = {str(record.get("drug_a", "")).casefold(), str(record.get("drug_b", "")).casefold()}  # the record's two drugs, lower-cased
    return ROLE_DIRECT if named == {name.casefold() for name in pair_names} else ROLE_CONTEXT  # exact match of both drugs = direct


def _names_any(record, pair_names):  # does the record mention at least one drug from the pair?
    named = {str(record.get("drug_a", "")).casefold(), str(record.get("drug_b", "")).casefold(),
             str(record.get("drug", "")).casefold()}  # every drug name the record carries, lower-cased
    return bool(named & {name.casefold() for name in pair_names})  # True when the two sets overlap


class EligibilityFilter:  # applies the registry's rules to records
    def __init__(self, registry):  # remember which registry to consult
        self.registry = registry  # the source metadata

    def evaluate(self, record, pair_names=None, record_type=INTERACTION_RECORD, drug_name=None):  # judge one record
        """Return the Evidence verdict for one record.

        ``pair_names`` is the requested pair (interaction records); ``drug_name`` is the
        requested drug (monograph records).
        """
        key = record.get("source_key")  # which source the record claims
        role = _role(record, pair_names) if record_type == INTERACTION_RECORD else ROLE_CONTEXT  # monographs are context
        source = self.registry.get(key) if key else None  # look the source up, or None if no key / not registered
        metadata = (  # snapshot of the source's metadata for the verdict
            {"status": source.status, "authority_class": source.authority_class, "scope": source.scope,
             "effective_date": source.effective_date, "owner": source.owner}  # copied as registered, None stays None
            if source else {}  # no source means no metadata
        )

        def verdict(eligible, reason):  # small helper that builds the Evidence result
            return Evidence(record, key, role, eligible, reason, metadata)  # pack everything into one object

        if not key:  # the record names no source at all
            return verdict(False, "record has no source key; its source metadata is missing")  # abstain and exclude
        if source is None:  # the source key is unknown to the registry
            return verdict(False, f"source '{key}' is not in the registry; its metadata is missing")  # exclude
        if source.status != "current":  # archived or unknown sources cannot be evidence
            return verdict(False, f"source '{key}' status is '{source.status}', not current")  # exclude
        if not source.authority_class:  # no authority recorded
            return verdict(False, f"source '{key}' has no authority class (missing authority)")  # exclude
        if not self.registry.is_approved(source.authority_class):  # authority exists but is not on the approved list
            return verdict(False, f"authority class '{source.authority_class}' is not approved as evidence")  # exclude
        if source.scope == "none":  # vocabulary-type sources (like HPO) never count as interaction evidence
            return verdict(False, f"source '{key}' has scope 'none': a vocabulary, not interaction evidence")  # exclude

        if record_type == INTERACTION_RECORD:  # source passed; now judge the record itself
            if role == ROLE_DIRECT:  # it names exactly the requested pair
                return verdict(True, "current source, approved authority, names the exact pair")  # best case: eligible
            if pair_names is not None and not _names_any(record, pair_names):  # unrelated to both requested drugs
                return verdict(False, "record does not concern either drug in the requested pair")  # exclude
            return verdict(True, "current source, approved authority; context only, it does not name the exact pair")  # context evidence

        # Monograph information must cover the drug itself.
        if drug_name is not None and str(record.get("drug", "")).casefold() != drug_name.casefold():  # wrong drug
            return verdict(False, "record is about a different drug")  # exclude
        return verdict(True, "current source, approved authority, covers the drug")  # monograph is eligible

    def partition(self, records, pair_names=None):  # sort many records into eligible and excluded
        """Split records into (eligible, excluded) Evidence lists, preserving order."""
        eligible, excluded = [], []  # two result lists
        for record in records:  # go through the records in order
            evidence = self.evaluate(record, pair_names)  # judge each one
            (eligible if evidence.eligible else excluded).append(evidence)  # file it in the matching list
        return eligible, excluded  # hand back both lists

# ---------------------------------------------------------------------------
# FILE SUMMARY: Judges every record before it may be used as evidence. A record passes only if its source is registered,
# current, has an approved authority class and a real scope, and the record names the requested pair; it returns an
# Evidence object with an eligible flag and a plain-language reason. Missing metadata leads to exclusion, never a guess.
# ASSIGNMENT TOOL: metadata-aware eligibility filtering (adopted); excluded evidence becomes the evidence_ineligible
# status rather than "no interaction".
# ---------------------------------------------------------------------------
