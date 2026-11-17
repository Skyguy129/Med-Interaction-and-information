# Module docstring: this file holds the named rules, and every response reports which rules fired
"""Named, tested rules. Every response lists which of them fired."""

from ddi_backend.common import severity_rank  # maps Major/Moderate/Minor to a sortable number (0 is most severe)

# Rule names: short strings put into each response's "rules fired" list
RULE_EXACT_NAME = "exact_name_resolution"  # a drug name matched exactly (ignoring case)
RULE_UNRESOLVED_NAME = "unresolved_name"  # a name could not be matched to a drug
RULE_PAIR_LOOKUP = "pair_lookup"  # the pair was looked up directly in the data
RULE_SEVERITY_PRECEDENCE = "severity_precedence"  # the most severe row becomes the headline
RULE_REPEATED_PAIR = "repeated_pair_conflict"  # one pair appears in several rows with different severities
RULE_DUPLICATE_COLLAPSED = "duplicate_record_collapsed"  # an identical repeated row was dropped
RULE_ALTERNATIVE_CROSS_CHECK = "alternative_cross_check"  # a suggested safer drug was checked against the graph
RULE_ABSENCE_NOT_SAFETY = "absence_is_not_safety"  # no row never means "safe"
RULE_ELIGIBILITY = "eligibility_filter"  # evidence went through the eligibility filter
RULE_EVIDENCE_INELIGIBLE = "evidence_ineligible"  # the only evidence failed the eligibility filter
RULE_AVOID_LIST = "avoid_list"  # an avoid list was built from the graph

# Fields compared to decide whether two rows are duplicates of each other
_CONTENT_FIELDS = ("severity", "mechanism", "effect", "safer_alternative", "rationale", "reference")


def collapse_duplicates(records):  # remove repeated rows that say exactly the same thing
    """Drop rows whose content is identical to an earlier row for the same pair.

    Returns (kept, dropped_ids). A duplicated row must change nothing: not the
    headline, not the conflict flag, not the avoid list.
    """
    seen, kept, dropped = set(), [], []  # content keys already seen, rows to keep, IDs of rows dropped
    for record in sorted(records, key=lambda r: r["id"]):  # go in ID order so the earliest row is the one kept
        key = tuple((record.get(f) or "").strip().casefold() for f in _CONTENT_FIELDS)  # normalised content fingerprint
        if key in seen:  # same content as an earlier row
            dropped.append(record["id"])  # note the dropped row's ID
        else:  # first time this content appears
            seen.add(key)  # remember the fingerprint
            kept.append(record)  # keep the row
    return kept, dropped  # both lists, so callers can report what was collapsed


def severity_precedence(records):  # pick the headline row by severity
    """Most severe first. The first record is the headline; severity is reported exactly as stored."""
    ordered = sorted(records, key=lambda r: (severity_rank(r["severity"]), r["id"]))  # most severe first, ties by ID
    return ordered[0] if ordered else None, ordered  # (headline or None if empty, the full ordered list)


def repeated_pair_conflict(records):  # detect a pair that appears more than once
    """A pair listed more than once with different severities is a conflict to review."""
    severities = []  # distinct severities, most severe first
    for record in sorted(records, key=lambda r: severity_rank(r["severity"])):  # walk rows from most to least severe
        if record["severity"] not in severities:  # only add each severity once
            severities.append(record["severity"])
    return {  # summary the checker puts in the response
        "repeated": len(records) > 1,  # more than one row for the pair
        "severity_conflict": len(severities) > 1,  # rows disagree on severity, so route to review
        "severities": severities,  # the distinct severities found
    }


def alternative_cross_check(record, graph, pair_ids, current_ids):  # test the suggested alternative against the graph
    """Check the drugs named as a safer alternative against the graph.

    An alternative drug is checked against both drugs of the pair and against
    everything the user takes. A documented interaction is shown with a warning
    and flagged for review. A drug that is not in our data is reported as such;
    that is not a statement that it is safe. Management actions name no drug,
    so there is nothing to check (their text is shown as written).
    """
    alternative = record["alternative"]  # the typed alternative (drug names or a management action)
    findings = []  # one finding per alternative drug
    if alternative["type"] != "alternative_drug":  # a management action names no drug
        return findings  # nothing to check, return the empty list
    others = {i for i in set(pair_ids) | set(current_ids)}  # the pair's drugs plus everything the user takes
    for name in alternative["drug_names"]:  # check each named alternative drug
        alt_id = graph.drug_id(name)  # find it in the graph by exact name
        finding = {"alternative": name, "in_data": alt_id is not None, "interactions": [], "warning": None}  # blank result
        if alt_id is None:  # the alternative drug is not in our data
            finding["note"] = f"'{name}' is not in our data, so it could not be checked; that does not mean it is safe."
            findings.append(finding)  # report it as unchecked
            continue  # move to the next alternative
        for other in sorted(others - {alt_id}, key=lambda i: (graph.name(i) or "").casefold()):  # each other drug, by name
            for hit in graph.pair(alt_id, other):  # every documented row between the alternative and that drug
                finding["interactions"].append({  # record the hit
                    "with": graph.name(other),  # the other drug's name
                    "severity": hit["severity"],  # severity as stored
                    "record_id": hit["id"],  # the ddi: ID so it can be cited
                    "effect": hit["effect"],  # the stored clinical effect
                    "scope": "current_medication" if other in current_ids else "pair",  # which group the other drug is from
                })
        if finding["interactions"]:  # the alternative has its own documented interaction
            finding["interactions"].sort(key=lambda h: (severity_rank(h["severity"]), h["record_id"]))  # worst first
            worst = finding["interactions"][0]  # the most severe hit leads the warning
            finding["warning"] = (  # build the warning text
                f"Warning: the suggested alternative {name} has its own documented "
                f"{worst['severity']} interaction with {worst['with']} ({worst['record_id']}). "
                "Flagged for review."
            )
        findings.append(finding)  # keep the finding, with or without a warning
    return findings  # all findings for this record


def avoid_list(graph, drug_id, minimum="Moderate"):  # build the "drugs to avoid with X" list
    """Every drug with a Major or Moderate edge to ``drug_id``, most severe first.

    Drug classes are not in our data, so the list cannot be narrowed to a class.
    """
    cutoff = severity_rank(minimum)  # numeric cut-off: anything ranked above this number is too mild to list
    entries = []  # one entry per interacting drug
    for other in graph.neighbors(drug_id):  # every drug that shares a row with this one
        records, _ = collapse_duplicates(graph.pair(drug_id, other))  # drop identical repeated rows first
        headline, ordered = severity_precedence(records)  # most severe row is the headline
        if severity_rank(headline["severity"]) > cutoff:  # milder than the minimum (for example Minor)
            continue  # skip it
        conflict = repeated_pair_conflict(records)  # note whether the rows disagree on severity
        entries.append({  # add the entry for this drug
            "drug": graph.name(other),  # the interacting drug's name
            "severity": headline["severity"],  # headline severity
            "severities": conflict["severities"],  # all distinct severities for this pair
            "severity_conflict": conflict["severity_conflict"],  # True when the rows disagree
            "record_ids": [r["id"] for r in ordered],  # all supporting ddi: IDs, most severe first
            "effect": headline["effect"],  # the headline row's effect text
        })
    entries.sort(key=lambda e: (severity_rank(e["severity"]), e["drug"].casefold()))  # Major first, then alphabetical
    return entries  # the finished avoid list


# ---------------------------------------------------------------------------
# FILE SUMMARY: Defines the names of the rules the system reports, and the rule functions themselves: collapsing
# identical repeated rows, picking the most severe row as the headline, flagging repeated pairs with conflicting
# severities, cross-checking suggested safer alternatives, and building avoid lists. Every response lists which of
# these rules fired.
# ASSIGNMENT TOOL: Knowledge graph with rule-based reasoning (the core engine); this is the rule-based reasoning half
# (severity precedence, repeated-pair conflict detection, alternative cross-check).
# ---------------------------------------------------------------------------
