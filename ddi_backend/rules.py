"""Named, tested rules. Every response lists which of them fired."""

from ddi_backend.common import severity_rank

RULE_EXACT_NAME = "exact_name_resolution"
RULE_UNRESOLVED_NAME = "unresolved_name"
RULE_PAIR_LOOKUP = "pair_lookup"
RULE_SEVERITY_PRECEDENCE = "severity_precedence"
RULE_REPEATED_PAIR = "repeated_pair_conflict"
RULE_DUPLICATE_COLLAPSED = "duplicate_record_collapsed"
RULE_ALTERNATIVE_CROSS_CHECK = "alternative_cross_check"
RULE_ABSENCE_NOT_SAFETY = "absence_is_not_safety"
RULE_ELIGIBILITY = "eligibility_filter"
RULE_EVIDENCE_INELIGIBLE = "evidence_ineligible"
RULE_AVOID_LIST = "avoid_list"

_CONTENT_FIELDS = ("severity", "mechanism", "effect", "safer_alternative", "rationale", "reference")


def collapse_duplicates(records):
    """Drop rows whose content is identical to an earlier row for the same pair.

    Returns (kept, dropped_ids). A duplicated row must change nothing: not the
    headline, not the conflict flag, not the avoid list.
    """
    seen, kept, dropped = set(), [], []
    for record in sorted(records, key=lambda r: r["id"]):
        key = tuple((record.get(f) or "").strip().casefold() for f in _CONTENT_FIELDS)
        if key in seen:
            dropped.append(record["id"])
        else:
            seen.add(key)
            kept.append(record)
    return kept, dropped


def severity_precedence(records):
    """Most severe first. The first record is the headline; severity is reported exactly as stored."""
    ordered = sorted(records, key=lambda r: (severity_rank(r["severity"]), r["id"]))
    return ordered[0] if ordered else None, ordered


def repeated_pair_conflict(records):
    """A pair listed more than once with different severities is a conflict to review."""
    severities = []
    for record in sorted(records, key=lambda r: severity_rank(r["severity"])):
        if record["severity"] not in severities:
            severities.append(record["severity"])
    return {
        "repeated": len(records) > 1,
        "severity_conflict": len(severities) > 1,
        "severities": severities,
    }


def alternative_cross_check(record, graph, pair_ids, current_ids):
    """Check the drugs named as a safer alternative against the graph.

    An alternative drug is checked against both drugs of the pair and against
    everything the user takes. A documented interaction is shown with a warning
    and flagged for review. A drug that is not in our data is reported as such;
    that is not a statement that it is safe. Management actions name no drug,
    so there is nothing to check (their text is shown as written).
    """
    alternative = record["alternative"]
    findings = []
    if alternative["type"] != "alternative_drug":
        return findings
    others = {i for i in set(pair_ids) | set(current_ids)}
    for name in alternative["drug_names"]:
        alt_id = graph.drug_id(name)
        finding = {"alternative": name, "in_data": alt_id is not None, "interactions": [], "warning": None}
        if alt_id is None:
            finding["note"] = f"'{name}' is not in our data, so it could not be checked; that does not mean it is safe."
            findings.append(finding)
            continue
        for other in sorted(others - {alt_id}, key=lambda i: (graph.name(i) or "").casefold()):
            for hit in graph.pair(alt_id, other):
                finding["interactions"].append({
                    "with": graph.name(other),
                    "severity": hit["severity"],
                    "record_id": hit["id"],
                    "effect": hit["effect"],
                    "scope": "current_medication" if other in current_ids else "pair",
                })
        if finding["interactions"]:
            finding["interactions"].sort(key=lambda h: (severity_rank(h["severity"]), h["record_id"]))
            worst = finding["interactions"][0]
            finding["warning"] = (
                f"Warning: the suggested alternative {name} has its own documented "
                f"{worst['severity']} interaction with {worst['with']} ({worst['record_id']}). "
                "Flagged for review."
            )
        findings.append(finding)
    return findings


def avoid_list(graph, drug_id, minimum="Moderate"):
    """Every drug with a Major or Moderate edge to ``drug_id``, most severe first.

    Drug classes are not in our data, so the list cannot be narrowed to a class.
    """
    cutoff = severity_rank(minimum)
    entries = []
    for other in graph.neighbors(drug_id):
        records, _ = collapse_duplicates(graph.pair(drug_id, other))
        headline, ordered = severity_precedence(records)
        if severity_rank(headline["severity"]) > cutoff:
            continue
        conflict = repeated_pair_conflict(records)
        entries.append({
            "drug": graph.name(other),
            "severity": headline["severity"],
            "severities": conflict["severities"],
            "severity_conflict": conflict["severity_conflict"],
            "record_ids": [r["id"] for r in ordered],
            "effect": headline["effect"],
        })
    entries.sort(key=lambda e: (severity_rank(e["severity"]), e["drug"].casefold()))
    return entries
