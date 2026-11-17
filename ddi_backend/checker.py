# Module docstring: describes the main deterministic flow (resolve names, look up pairs, apply rules)
"""The deterministic checking flow: resolve names, look up pairs, apply the rules.

No embedding, no language model and no HPO is involved here. Only the drugs and
interactions tables are read.
"""

from itertools import combinations  # every unordered pair from a list, for current-vs-current checks

from ddi_backend import ingest, rules  # database queries and the named rules
from ddi_backend.common import DISCLAIMER, NOT_AVAILABLE, NOT_SAFE_NOTE, severity_rank  # shared texts and severity ordering
from ddi_backend.eligibility import EligibilityFilter  # decides whether a record counts as evidence
from ddi_backend.graph import Graph  # the in-memory knowledge graph
from ddi_backend.registry import ROLE_DIRECT, load_registry  # source registry and the "direct evidence" role name
from ddi_backend.resolver import AMBIGUOUS_NAME, RESOLVED, UNKNOWN_DRUG, Resolver  # name resolution and its status names

# The three pair statuses the checker itself creates (resolver statuses cover the other two)
DOCUMENTED = "documented_interaction"  # at least one eligible row exists for the pair
NO_DOCUMENTED = "no_documented_interaction"  # no row at all; this never means "safe"
EVIDENCE_INELIGIBLE = "evidence_ineligible"  # rows exist but none passed the eligibility filter

# Display order: documented (Major first), then evidence that needs review, then
# names we could not resolve, and last the pairs with nothing documented.
_STATUS_ORDER = {DOCUMENTED: 0, EVIDENCE_INELIGIBLE: 1, AMBIGUOUS_NAME: 2, UNKNOWN_DRUG: 2, NO_DOCUMENTED: 3}  # lower sorts first


def as_medication(item):  # normalise one input medication into a dict
    """Accept "Warfarin 5 mg" or {"name": ..., "dose": ..., "details": ...}."""
    if isinstance(item, str):  # plain text name (the dose may be inside it)
        return {"name": item}  # wrap it in the dict form
    if isinstance(item, dict) and isinstance(item.get("name"), str):  # already an object with a text name
        return dict(item)  # copy so the caller's dict is not changed
    raise ValueError("each medication must be a name or an object with a 'name'")  # anything else is invalid input


class Checker:  # the deterministic main flow
    def __init__(self, conn, aliases=None, suggester=None, registry=None, eligibility=True,
                 extra_evidence=None):  # optional parts: alias file, did-you-mean helper, registry, extra evidence
        """``eligibility`` True builds the filter from the source registry (the seed file unless
        ``registry`` is given); None or False turns it off. ``extra_evidence(names) -> [records]``
        adds records from other sources for a pair; the database never supplies them."""
        self.conn = conn  # read-only database connection
        self.resolver = Resolver.from_connection(conn, aliases=aliases, suggester=suggester)  # exact-name resolver
        self._known = frozenset(ingest.drug_names(conn))  # every drug name in the data, for fast membership checks
        if eligibility is True:  # build the default filter
            eligibility = EligibilityFilter(registry if registry is not None else load_registry())  # use the given or default registry
        self.eligibility = eligibility or None  # False and None both mean "filter off"
        self.extra_evidence = extra_evidence  # optional callback that supplies records from other sources
        # The graph holds only eligible evidence, so avoid lists and the alternative
        # cross-check never rest on a record that failed the filter.
        records, self.excluded_evidence = self.eligible_records(ingest.load_interactions(conn))  # split all rows by eligibility
        self.graph = Graph(records, ingest.drug_names(conn))  # graph built from eligible rows only

    def eligible_records(self, records):  # split records into kept and excluded
        """Split interaction records into (eligible records, excluded Evidence list)."""
        if self.eligibility is None:  # filter is off
            return list(records), []  # everything is kept, nothing excluded
        kept, excluded = [], []  # records that pass, Evidence objects that fail
        for record in records:  # judge each row on its own
            evidence = self.eligibility.evaluate(record, (record["drug_a"], record["drug_b"]))  # apply the filter to this row
            if evidence.eligible and evidence.role == ROLE_DIRECT:  # must be eligible AND direct evidence for the pair
                kept.append(record)  # keep the row
            else:
                excluded.append(evidence)  # keep the Evidence object so the reason can be shown
        return kept, excluded  # both lists

    # ------------------------------------------------------------------
    # Medications
    # ------------------------------------------------------------------
    def _resolve_medication(self, item):  # turn one input into a resolved entry
        med = as_medication(item)  # normalise the input form
        resolution = self.resolver.resolve(med["name"])  # exact, case-insensitive name resolution
        entry = {"input": med["name"], "resolution": resolution}  # keep the original text and the result
        entry["dose"] = med.get("dose") or resolution.dose  # echoed, never recommended
        if med.get("details"):  # optional free details from the caller
            entry["details"] = med["details"]  # carried through unchanged
        return entry

    # ------------------------------------------------------------------
    # Pairs
    # ------------------------------------------------------------------
    def check_pair(self, first, second, current_ids=()):  # the core: one pair in, one result dict out
        """Check two resolved-or-not medication entries (from _resolve_medication)."""
        ra, rb = first["resolution"], second["resolution"]  # the two name resolutions
        fired = [rules.RULE_EXACT_NAME] if (ra.resolved or rb.resolved) else []  # rules fired list starts here
        unresolved = [r for r in (ra, rb) if not r.resolved]  # names that did not resolve to a drug
        pair_names = [ra.name or ra.query, rb.name or rb.query]  # canonical name if resolved, else what was typed
        if unresolved:  # at least one name is unknown or ambiguous
            fired.append(rules.RULE_UNRESOLVED_NAME)  # record that rule
            worst = next((r for r in unresolved if r.status == UNKNOWN_DRUG), unresolved[0])  # prefer reporting unknown_drug
            return {  # a result without a severity: nothing can be looked up
                "pair": pair_names,  # the pair as named
                "status": worst.status,  # unknown_drug or ambiguous_name
                "severity": None,  # no severity without a lookup
                "unresolved": [r.to_dict() for r in unresolved],  # details about each unresolved name
                "message": " ".join(r.question for r in unresolved if r.question),  # clarifying question(s) for the user
                "notes": [NOT_SAFE_NOTE],  # reminder that not found never means safe
                "rules_fired": fired,
                "needs_review": False,
            }

        fired.append(rules.RULE_PAIR_LOOKUP)  # both names resolved, so look the pair up
        rows = ingest.pair_records(self.conn, ra.drug_id, rb.drug_id, self._known)  # direct query, both orders of the pair
        if self.extra_evidence is not None:  # optional records from other sources
            rows = rows + list(self.extra_evidence(pair_names))  # add them to the database rows
        result = {"pair": pair_names, "doses": {pair_names[0]: first.get("dose"), pair_names[1]: second.get("dose")}}  # echo doses
        if not rows:  # no row at all for this pair
            fired.append(rules.RULE_ABSENCE_NOT_SAFETY)  # absence is not safety
            result.update(status=NO_DOCUMENTED, severity=None, records=[], alternatives=[],
                          notes=[NOT_SAFE_NOTE], rules_fired=fired, needs_review=False)  # fill in the "nothing documented" result
            return result

        rows, dropped = rules.collapse_duplicates(rows)  # drop rows identical to an earlier one
        if dropped:  # something was collapsed
            fired.append(rules.RULE_DUPLICATE_COLLAPSED)  # record the rule
            result["duplicate_record_ids"] = dropped  # list which IDs were dropped
        if self.eligibility is None:  # filter off: every row counts
            return self.documented_result(result, rows, fired, ra, rb, current_ids)

        fired.append(rules.RULE_ELIGIBILITY)  # the filter is applied from here
        eligible, excluded = self.eligibility.partition(rows, pair_names)  # split the rows into eligible and excluded Evidence
        result["evidence"] = [e.to_dict() for e in eligible + excluded]  # show every row's eligibility verdict
        direct = [e.record for e in eligible if e.role == ROLE_DIRECT]  # eligible rows that speak directly to the pair
        context = [e.record for e in eligible if e.role != ROLE_DIRECT]  # eligible rows that are only background
        excluded_reasons = [f"excluded_record {e.record['id']}: {e.reason}" for e in excluded]  # why each row was excluded
        if direct:  # there is usable direct evidence
            result = self.documented_result(result, direct, fired, ra, rb, current_ids)  # build the documented answer
            if context:  # also attach any background evidence
                result["context_evidence"] = context
            if excluded_reasons:  # the rest of the evidence is dropped and sent to a human
                result["excluded_evidence"] = [e.to_dict() for e in excluded]  # show what was dropped
                result["needs_review"] = True  # flag for human review
                result["review_reasons"] = result["review_reasons"] + excluded_reasons  # add the exclusion reasons
            return result

        # No eligible direct evidence. If something was excluded that must never read as "no interaction".
        if excluded:  # rows exist but all failed the filter
            fired.append(rules.RULE_EVIDENCE_INELIGIBLE)  # record the rule
            result.update(
                status=EVIDENCE_INELIGIBLE,  # a distinct status, not "no interaction"
                severity=None,
                records=[],
                alternatives=[],
                excluded_evidence=[e.to_dict() for e in excluded],  # why the evidence was rejected
                message=("Our data has evidence for this pair, but none of it is eligible: "
                         + "; ".join(e.reason for e in excluded)
                         + ". This is not a statement that there is no interaction."),  # plain statement that this is not "safe"
                notes=[NOT_SAFE_NOTE],
                rules_fired=fired,
                needs_review=True,  # a human should look at it
                review_reasons=excluded_reasons,
            )
            if context:  # keep any background evidence
                result["context_evidence"] = context
            return result
        fired.append(rules.RULE_ABSENCE_NOT_SAFETY)  # nothing direct and nothing excluded
        result.update(status=NO_DOCUMENTED, severity=None, records=[], alternatives=[],
                      context_evidence=context, notes=[NOT_SAFE_NOTE], rules_fired=fired, needs_review=False)  # nothing documented
        return result

    def documented_result(self, result, rows, fired, ra, rb, current_ids, notes=()):  # build the answer when evidence exists
        """Build a documented_interaction result from the eligible rows of a pair."""
        headline, ordered = rules.severity_precedence(rows)  # most severe row is the headline
        conflict = rules.repeated_pair_conflict(rows)  # do repeated rows disagree on severity
        fired.append(rules.RULE_SEVERITY_PRECEDENCE)  # record the rule
        reasons = []  # reasons this pair needs human review
        if conflict["severity_conflict"]:  # rows disagree
            fired.append(rules.RULE_REPEATED_PAIR)  # record the rule
            reasons.append("severity_conflict: " + " / ".join(conflict["severities"]))  # explain, e.g. "Major / Moderate"

        pair_ids = (ra.drug_id, rb.drug_id)  # IDs of the two drugs in the pair
        alternatives = []  # one entry per stored safer alternative
        cross_checked = False  # becomes True if any alternative drug gets checked
        for record in ordered:  # every row, most severe first
            alt = record["alternative"]  # typed alternative: a drug or a management action
            entry = {
                "record_id": record["id"],  # which row it came from
                "type": alt["type"],  # alternative_drug or management_action
                "text": alt["text"],  # exactly as written; never generated or calculated
                "rationale": record["rationale"],  # the row's stored reason
                "source": {"source_key": record["source_key"], "reference": record["reference"]},  # where it came from
                "cross_check": rules.alternative_cross_check(record, self.graph, pair_ids, current_ids),  # graph check of the alternative
            }
            if alt["type"] == ingest.ALTERNATIVE_DRUG:  # a real drug was named
                cross_checked = True  # so the cross-check rule applies
            flagged = [f for f in entry["cross_check"] if f["interactions"]]  # alternatives with their own interactions
            if flagged:  # at least one alternative is itself a problem
                reason = "alternative_has_documented_interaction: " + ", ".join(f["alternative"] for f in flagged)  # explain
                if reason not in reasons:  # no repeated reasons
                    reasons.append(reason)
            alternatives.append(entry)  # keep the entry
        if cross_checked:  # at least one alternative drug was checked
            fired.append(rules.RULE_ALTERNATIVE_CROSS_CHECK)  # record the rule

        result.update(  # fill in the documented result
            status=DOCUMENTED,
            severity=headline["severity"],  # exactly as stored
            severity_conflict=conflict["severity_conflict"],  # True when severities differ
            severities=conflict["severities"],  # all distinct severities
            mechanism=headline["mechanism"],  # the headline row's mechanism
            effect=headline["effect"],  # the headline row's clinical effect
            records=ordered,  # every row, most severe first
            alternatives=alternatives,  # the typed alternatives with their cross-checks
            source={
                "source_key": headline["source_key"],  # which source the headline came from
                "record_ids": [r["id"] for r in ordered],  # all ddi: IDs
                "citations": [{"id": r["id"], "reference": r["reference"]} for r in ordered],  # each ID with its reference text
            },
            notes=list(notes),  # extra notes passed by the caller
            rules_fired=fired,  # which rules fired
            needs_review=bool(reasons),  # True if any review reason was found
            review_reasons=reasons,
        )
        return result

    # ------------------------------------------------------------------
    # The main flow
    # ------------------------------------------------------------------
    def check(self, current, new, include_current_pairs=False):  # the public entry point
        """Check ``new`` against each current medication; optionally the current ones against each other."""
        current_entries = [self._resolve_medication(item) for item in current]  # resolve every current medication
        new_entry = self._resolve_medication(new)  # resolve the new one

        # Collapse a medication listed twice; a drug is never checked against itself.
        notes, seen, unique_current = [], set(), []  # user-facing notes, keys already seen, medications to check
        new_id = new_entry["resolution"].drug_id  # the new drug's ID (None if unresolved)
        for entry in current_entries:  # look at each current medication
            res = entry["resolution"]
            key = res.drug_id if res.resolved else ("?", res.query.casefold())  # ID if resolved, else the typed text
            if key in seen:  # already listed
                notes.append(f"'{res.name or res.query}' was listed more than once; it is checked once.")
                continue  # skip the duplicate
            seen.add(key)  # remember it
            if res.resolved and res.drug_id == new_id:  # same drug as the new one
                notes.append(f"'{res.name}' is both a current and the new medication; it is not checked against itself.")
                continue  # skip it
            unique_current.append(entry)  # keep this one for checking

        current_ids = {e["resolution"].drug_id for e in unique_current if e["resolution"].resolved}  # IDs of everything taken
        if new_id is not None:  # include the new drug in the set too
            current_ids.add(new_id)
        results = [dict(self.check_pair(entry, new_entry, current_ids), kind="new_vs_current")
                   for entry in unique_current]  # one result per current medication against the new one
        if include_current_pairs:  # optionally also check the current ones against each other
            for a, b in combinations(unique_current, 2):  # every unordered pair
                results.append(dict(self.check_pair(a, b, current_ids), kind="current_vs_current"))
        results.sort(key=lambda r: (_STATUS_ORDER[r["status"]], severity_rank(r.get("severity"))))  # documented and Major first

        fired = []  # all rules fired, in order of first use, without repeats
        for r in results:
            for rule in r["rules_fired"]:
                if rule not in fired:
                    fired.append(rule)
        counts = {}  # how many results had each status
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {  # the full response
            "disclaimer": DISCLAIMER,  # not medical advice
            "inputs": {
                "current": [self._echo(e) for e in current_entries],  # what was understood for each current medication
                "new": self._echo(new_entry),  # and for the new one
            },
            "results": results,  # one result per pair
            "summary": {"pairs_checked": len(results), "by_status": counts,
                        "major_documented": sum(1 for r in results if r.get("severity") == "Major")},  # counts, with Major called out
            "rules_fired": fired,
            "notes": notes,
        }

    @staticmethod  # needs no instance
    def _echo(entry):  # summarise how an input was understood
        out = {"input": entry["input"], "status": entry["resolution"].status,
               "name": entry["resolution"].name, "dose": entry.get("dose")}  # typed text, status, resolved name, echoed dose
        if entry.get("details"):  # carry details through if given
            out["details"] = entry["details"]
        if entry["resolution"].candidates:  # suggestions for unresolved names
            out["did_you_mean"] = entry["resolution"].candidates
        return out

    # ------------------------------------------------------------------
    # Drug and avoid-list views
    # ------------------------------------------------------------------
    def drug(self, name):  # single-drug view
        """The drug's name and the interactions it appears in; monograph fields are not in our data."""
        res = self.resolver.resolve(name)  # resolve the name
        if not res.resolved:  # unknown or ambiguous
            return {"status": res.status, "query": res.query, "did_you_mean": res.candidates,
                    "message": res.question, "disclaimer": DISCLAIMER}  # answer with suggestions, nothing else
        record = ingest.drug_record(self.conn, res.drug_id, self._known)  # the drug and all its interaction rows
        interactions, excluded = self.eligible_records(record["interactions"])  # filter them by eligibility
        return {
            "status": RESOLVED,
            "drug": {"drug_id": record["drug_id"], "name": record["name"]},
            "interactions": interactions,  # eligible rows only
            "excluded_evidence": [e.to_dict() for e in excluded],  # what the filter removed and why
            "not_available": {  # fields the data does not hold
                field: NOT_AVAILABLE  # each is answered with the standard "not available in our data" text
                for field in ("uses", "side_effects", "warnings", "contraindications", "ingredients",
                              "brand_names", "food_interactions")
            },
            "disclaimer": DISCLAIMER,
        }

    def avoid(self, name):  # avoid-list view
        """Every drug with a Major or Moderate edge to ``name``, most severe first."""
        res = self.resolver.resolve(name)  # resolve the name
        if not res.resolved:  # unknown or ambiguous
            return {"status": res.status, "query": res.query, "did_you_mean": res.candidates,
                    "message": res.question, "disclaimer": DISCLAIMER}
        return {
            "status": RESOLVED,
            "drug": res.name,
            "avoid": rules.avoid_list(self.graph, res.drug_id),  # list built from the graph by the avoid_list rule
            "rules_fired": [rules.RULE_EXACT_NAME, rules.RULE_AVOID_LIST],
            "notes": [
                "Drug classes are not in our data, so this list cannot be narrowed to a class "
                "such as pain relievers; it is the full list of documented Major and Moderate interactions.",
                NOT_SAFE_NOTE,  # absence from the list is not safety
            ],
            "disclaimer": DISCLAIMER,
        }


# ---------------------------------------------------------------------------
# FILE SUMMARY: The deterministic main flow. It resolves medication names, looks pairs up directly in the data,
# applies the eligibility filter and the named rules (severity precedence, repeated-pair conflict, duplicate
# collapsing, alternative cross-check), and returns one structured result per pair with status, severity, mechanism,
# effect, alternatives, source and the rules that fired. It also provides the single-drug and avoid-list views. No
# embedding, language model or HPO is involved.
# ASSIGNMENT TOOL: Knowledge graph with rule-based reasoning (the core engine and source of every factual answer),
# using metadata-aware eligibility filtering.
# ---------------------------------------------------------------------------
