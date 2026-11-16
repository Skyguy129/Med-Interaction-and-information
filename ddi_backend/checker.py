"""The deterministic checking flow: resolve names, look up pairs, apply the rules.

No embedding, no language model and no HPO is involved here. Only the drugs and
interactions tables are read.
"""

from itertools import combinations

from ddi_backend import ingest, rules
from ddi_backend.common import DISCLAIMER, NOT_AVAILABLE, NOT_SAFE_NOTE, severity_rank
from ddi_backend.graph import Graph
from ddi_backend.resolver import AMBIGUOUS_NAME, RESOLVED, UNKNOWN_DRUG, Resolver

DOCUMENTED = "documented_interaction"
NO_DOCUMENTED = "no_documented_interaction"
EVIDENCE_INELIGIBLE = "evidence_ineligible"

# Display order: documented (Major first), then evidence that needs review, then
# names we could not resolve, and last the pairs with nothing documented.
_STATUS_ORDER = {DOCUMENTED: 0, EVIDENCE_INELIGIBLE: 1, AMBIGUOUS_NAME: 2, UNKNOWN_DRUG: 2, NO_DOCUMENTED: 3}


def as_medication(item):
    """Accept "Warfarin 5 mg" or {"name": ..., "dose": ..., "details": ...}."""
    if isinstance(item, str):
        return {"name": item}
    if isinstance(item, dict) and isinstance(item.get("name"), str):
        return dict(item)
    raise ValueError("each medication must be a name or an object with a 'name'")


class Checker:
    def __init__(self, conn, aliases=None, suggester=None, evidence_filter=None):
        self.conn = conn
        self.graph = Graph.from_connection(conn)
        self.resolver = Resolver.from_connection(conn, aliases=aliases, suggester=suggester)
        self._known = frozenset(ingest.drug_names(conn))
        self.evidence_filter = evidence_filter  # set in phase 3; None means every row is evidence

    # ------------------------------------------------------------------
    # Medications
    # ------------------------------------------------------------------
    def _resolve_medication(self, item):
        med = as_medication(item)
        resolution = self.resolver.resolve(med["name"])
        entry = {"input": med["name"], "resolution": resolution}
        entry["dose"] = med.get("dose") or resolution.dose  # echoed, never recommended
        if med.get("details"):
            entry["details"] = med["details"]
        return entry

    # ------------------------------------------------------------------
    # Pairs
    # ------------------------------------------------------------------
    def check_pair(self, first, second, current_ids=()):
        """Check two resolved-or-not medication entries (from _resolve_medication)."""
        ra, rb = first["resolution"], second["resolution"]
        fired = [rules.RULE_EXACT_NAME] if (ra.resolved or rb.resolved) else []
        unresolved = [r for r in (ra, rb) if not r.resolved]
        pair_names = [ra.name or ra.query, rb.name or rb.query]
        if unresolved:
            fired.append(rules.RULE_UNRESOLVED_NAME)
            worst = next((r for r in unresolved if r.status == UNKNOWN_DRUG), unresolved[0])
            return {
                "pair": pair_names,
                "status": worst.status,
                "severity": None,
                "unresolved": [r.to_dict() for r in unresolved],
                "message": " ".join(r.question for r in unresolved if r.question),
                "notes": [NOT_SAFE_NOTE],
                "rules_fired": fired,
                "needs_review": False,
            }

        fired.append(rules.RULE_PAIR_LOOKUP)
        rows = ingest.pair_records(self.conn, ra.drug_id, rb.drug_id, self._known)
        result = {"pair": pair_names, "doses": {pair_names[0]: first.get("dose"), pair_names[1]: second.get("dose")}}
        if not rows:
            fired.append(rules.RULE_ABSENCE_NOT_SAFETY)
            result.update(status=NO_DOCUMENTED, severity=None, records=[], alternatives=[],
                          notes=[NOT_SAFE_NOTE], rules_fired=fired, needs_review=False)
            return result

        rows, dropped = rules.collapse_duplicates(rows)
        if dropped:
            fired.append(rules.RULE_DUPLICATE_COLLAPSED)
            result["duplicate_record_ids"] = dropped
        if self.evidence_filter is not None:
            return self.evidence_filter.apply(self, result, rows, fired, ra, rb, current_ids)
        return self.documented_result(result, rows, fired, ra, rb, current_ids)

    def documented_result(self, result, rows, fired, ra, rb, current_ids, notes=()):
        """Build a documented_interaction result from the eligible rows of a pair."""
        headline, ordered = rules.severity_precedence(rows)
        conflict = rules.repeated_pair_conflict(rows)
        fired.append(rules.RULE_SEVERITY_PRECEDENCE)
        reasons = []
        if conflict["severity_conflict"]:
            fired.append(rules.RULE_REPEATED_PAIR)
            reasons.append("severity_conflict: " + " / ".join(conflict["severities"]))

        pair_ids = (ra.drug_id, rb.drug_id)
        alternatives = []
        cross_checked = False
        for record in ordered:
            alt = record["alternative"]
            entry = {
                "record_id": record["id"],
                "type": alt["type"],
                "text": alt["text"],  # exactly as written; never generated or calculated
                "rationale": record["rationale"],
                "source": {"source_key": record["source_key"], "reference": record["reference"]},
                "cross_check": rules.alternative_cross_check(record, self.graph, pair_ids, current_ids),
            }
            if alt["type"] == ingest.ALTERNATIVE_DRUG:
                cross_checked = True
            flagged = [f for f in entry["cross_check"] if f["interactions"]]
            if flagged:
                reason = "alternative_has_documented_interaction: " + ", ".join(f["alternative"] for f in flagged)
                if reason not in reasons:
                    reasons.append(reason)
            alternatives.append(entry)
        if cross_checked:
            fired.append(rules.RULE_ALTERNATIVE_CROSS_CHECK)

        result.update(
            status=DOCUMENTED,
            severity=headline["severity"],
            severity_conflict=conflict["severity_conflict"],
            severities=conflict["severities"],
            mechanism=headline["mechanism"],
            effect=headline["effect"],
            records=ordered,
            alternatives=alternatives,
            source={
                "source_key": headline["source_key"],
                "record_ids": [r["id"] for r in ordered],
                "citations": [{"id": r["id"], "reference": r["reference"]} for r in ordered],
            },
            notes=list(notes),
            rules_fired=fired,
            needs_review=bool(reasons),
            review_reasons=reasons,
        )
        return result

    # ------------------------------------------------------------------
    # The main flow
    # ------------------------------------------------------------------
    def check(self, current, new, include_current_pairs=False):
        """Check ``new`` against each current medication; optionally the current ones against each other."""
        current_entries = [self._resolve_medication(item) for item in current]
        new_entry = self._resolve_medication(new)

        # Collapse a medication listed twice; a drug is never checked against itself.
        notes, seen, unique_current = [], set(), []
        new_id = new_entry["resolution"].drug_id
        for entry in current_entries:
            res = entry["resolution"]
            key = res.drug_id if res.resolved else ("?", res.query.casefold())
            if key in seen:
                notes.append(f"'{res.name or res.query}' was listed more than once; it is checked once.")
                continue
            seen.add(key)
            if res.resolved and res.drug_id == new_id:
                notes.append(f"'{res.name}' is both a current and the new medication; it is not checked against itself.")
                continue
            unique_current.append(entry)

        current_ids = {e["resolution"].drug_id for e in unique_current if e["resolution"].resolved}
        if new_id is not None:
            current_ids.add(new_id)
        results = [dict(self.check_pair(entry, new_entry, current_ids), kind="new_vs_current")
                   for entry in unique_current]
        if include_current_pairs:
            for a, b in combinations(unique_current, 2):
                results.append(dict(self.check_pair(a, b, current_ids), kind="current_vs_current"))
        results.sort(key=lambda r: (_STATUS_ORDER[r["status"]], severity_rank(r.get("severity"))))

        fired = []
        for r in results:
            for rule in r["rules_fired"]:
                if rule not in fired:
                    fired.append(rule)
        counts = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {
            "disclaimer": DISCLAIMER,
            "inputs": {
                "current": [self._echo(e) for e in current_entries],
                "new": self._echo(new_entry),
            },
            "results": results,
            "summary": {"pairs_checked": len(results), "by_status": counts,
                        "major_documented": sum(1 for r in results if r.get("severity") == "Major")},
            "rules_fired": fired,
            "notes": notes,
        }

    @staticmethod
    def _echo(entry):
        out = {"input": entry["input"], "status": entry["resolution"].status,
               "name": entry["resolution"].name, "dose": entry.get("dose")}
        if entry.get("details"):
            out["details"] = entry["details"]
        if entry["resolution"].candidates:
            out["did_you_mean"] = entry["resolution"].candidates
        return out

    # ------------------------------------------------------------------
    # Drug and avoid-list views
    # ------------------------------------------------------------------
    def drug(self, name):
        """The drug's name and the interactions it appears in; monograph fields are not in our data."""
        res = self.resolver.resolve(name)
        if not res.resolved:
            return {"status": res.status, "query": res.query, "did_you_mean": res.candidates,
                    "message": res.question, "disclaimer": DISCLAIMER}
        record = ingest.drug_record(self.conn, res.drug_id, self._known)
        return {
            "status": RESOLVED,
            "drug": {"drug_id": record["drug_id"], "name": record["name"]},
            "interactions": record["interactions"],
            "not_available": {
                field: NOT_AVAILABLE
                for field in ("uses", "side_effects", "warnings", "contraindications", "ingredients",
                              "brand_names", "food_interactions")
            },
            "disclaimer": DISCLAIMER,
        }

    def avoid(self, name):
        """Every drug with a Major or Moderate edge to ``name``, most severe first."""
        res = self.resolver.resolve(name)
        if not res.resolved:
            return {"status": res.status, "query": res.query, "did_you_mean": res.candidates,
                    "message": res.question, "disclaimer": DISCLAIMER}
        return {
            "status": RESOLVED,
            "drug": res.name,
            "avoid": rules.avoid_list(self.graph, res.drug_id),
            "rules_fired": [rules.RULE_EXACT_NAME, rules.RULE_AVOID_LIST],
            "notes": [
                "Drug classes are not in our data, so this list cannot be narrowed to a class "
                "such as pain relievers; it is the full list of documented Major and Moderate interactions.",
                NOT_SAFE_NOTE,
            ],
            "disclaimer": DISCLAIMER,
        }

