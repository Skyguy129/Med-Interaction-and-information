# Module docstring: the composer is the complete no-model mode and also the fallback for the language model.
"""The template composer: plain-language responses built only from stored fields.

This is the complete deterministic mode. It writes the response from a record's
severity, mechanism, effect, rationale, safer alternative and reference exactly as
stored, plus the status wording the rules require. It is also the fallback when a
language model's output fails validation or the provider is unavailable.
Nothing is generated, calculated or recommended; doses are echoed as typed.
"""

from ddi_backend.common import DISCLAIMER, NOT_SAFE_NOTE  # shared fixed wording: the not-medical-advice line and the "not found is not safe" note

DETAIL_LEVELS = ("brief", "standard", "detailed")  # the three response sizes users can pick
NOT_DOCUMENTED = "not documented in our data"  # standard phrase used instead of "no interaction"


def _sentence(text):  # make sure a piece of stored text ends like a sentence
    text = (text or "").strip()  # treat None as empty and trim spaces
    return text if text.endswith((".", "!", "?")) else text + "."  # add a full stop only when missing


def _alternative_lines(alternatives, detail):  # turn safer alternatives into output lines
    lines = []  # collected output lines
    for alt in alternatives:  # each alternative the checker found
        kind = "suggested drug" if alt["type"] == "alternative_drug" else "management action"  # label which type it is
        line = f"Safer alternative ({kind}, as written in {alt['record_id']}): {alt['text']}"  # text exactly as stored, with its record ID
        if alt.get("rationale"):  # a stored reason exists
            line += f" Reason given: {_sentence(alt['rationale'])}"  # append it
        lines.append(line)  # keep the line
        for finding in alt.get("cross_check", []):  # results of checking the alternative against the user's drugs
            if finding.get("warning"):  # the alternative has its own documented interaction
                lines.append(finding["warning"])  # always show the warning
            elif not finding.get("in_data") and detail != "brief":  # a drug we know nothing about, and not in brief mode
                lines.append(finding["note"])  # say it is not in our data
    return lines  # hand the lines back


def compose_documented(result, detail="standard"):  # text for a pair that has a documented interaction
    a, b = result["pair"]  # the two drug names
    records = result["records"]  # every interaction row for the pair
    headline = records[0]  # the row that headlines (highest severity)
    lines = []  # output lines
    if detail == "brief":  # shortest form
        lines.append(f"{a} and {b}: documented {result['severity']} interaction ({headline['id']}). "  # one-line headline with the record ID
                     + _sentence(headline.get("effect") or ""))  # followed by the stored effect
        if result.get("severity_conflict"):  # the pair appears with different severities
            lines.append("Our data lists this pair with more than one severity; the highest is shown and it is "  # first half of the notice
                         "flagged for review.")  # second half of the notice
        lines.append(f"Source: {headline['id']}, {headline.get('reference')}")  # cite the record and its Reference field
        return "\n".join(lines)  # brief text is complete here

    lines.append(f"Taking {a} with {b} is a documented {result['severity']} interaction in our data.")  # standard/detailed opening sentence
    if result.get("severity_conflict"):  # repeated pair with different severities
        lines.append(f"Our data lists this pair {len(records)} times with different severities "  # how many rows
                     f"({' / '.join(result['severities'])}). The highest, {result['severity']}, is shown, and the "  # which severities, and the headline
                     "pair is flagged for pharmacist review.")  # routed to review
    if detail == "standard":  # middle size
        if headline.get("mechanism"):  # only print fields that exist
            lines.append(f"Mechanism: {headline['mechanism']}")  # stored mechanism text
        if headline.get("effect"):  # only print fields that exist
            lines.append(f"Clinical effect: {headline['effect']}")  # stored effect text
        alternatives = result.get("alternatives", [])  # safer alternatives found for the pair
        # The first alternative belongs to the headline record; show them all when the pair has a conflict.
        lines += _alternative_lines(alternatives if result.get("severity_conflict") else alternatives[:1], detail)
    else:  # detailed: show everything stored
        for record in records:  # every row, not just the headline
            lines.append(f"Record {record['id']} ({record['severity']}): {record['drug_a']} + {record['drug_b']}")  # row header
            for label, key in (("Mechanism", "mechanism"), ("Clinical effect", "effect"),  # the stored fields to show, with display labels
                               ("Safer alternative", "safer_alternative"), ("Rationale", "rationale"),  # more stored fields
                               ("Reference", "reference")):  # and the source text
                if record.get(key):  # skip empty fields
                    lines.append(f"  {label}: {record[key]}")  # indented stored value
        lines += _alternative_lines(result.get("alternatives", []), detail)  # all alternatives with their cross-checks
        doses = {name: dose for name, dose in (result.get("doses") or {}).items() if dose}  # only doses the user actually typed
        if doses:  # something to echo
            lines.append("Doses as you entered them: " + ", ".join(f"{n} {d}" for n, d in doses.items())  # echo exactly as typed
                         + ". This system stores and echoes doses; it does not evaluate or recommend them.")  # make clear no dose advice
    if result.get("needs_review"):  # the checker flagged the pair
        lines.append("Flagged for review: " + "; ".join(result.get("review_reasons", [])) + ".")  # list the reasons
    lines.append("Source: " + "; ".join(f"{c['id']} ({c['reference']})" for c in result["source"]["citations"]))  # cite every record used
    return "\n".join(lines)  # one line per piece of text


def compose_pair(result, detail="standard"):  # text for one pair, whatever its status
    """Plain-language text for one pair result, whatever its status."""
    status = result["status"]  # which of the five statuses this is
    a, b = result["pair"]  # the two drug names
    if status == "documented_interaction":  # found and eligible
        return compose_documented(result, detail)  # use the full composer above
    if status == "no_documented_interaction":  # both drugs known, no row
        return f"An interaction between {a} and {b} is {NOT_DOCUMENTED}. " + NOT_SAFE_NOTE  # never say "safe"
    if status == "evidence_ineligible":  # a row exists but failed the eligibility filter
        return result["message"] + " It is flagged for review. " + NOT_SAFE_NOTE  # use the filter's message and route to review
    # ambiguous_name / unknown_drug
    return (result.get("message") or f"We could not match {a} or {b} to a drug name in our data.") \
        + " " + NOT_SAFE_NOTE  # name problems also never imply safety


def compose_check(check_output, detail="standard"):  # text for a whole /check response
    """The full text for a /check style response: one block per pair, Major first."""
    blocks = [compose_pair(r, detail) for r in check_output["results"]]  # one block per pair (already sorted Major first)
    footer = [DISCLAIMER]  # the not-medical-advice line goes last
    return "\n\n".join(blocks + footer)  # blank line between blocks


def compose_avoid(avoid_output, detail="standard"):  # text for an "avoid list" answer
    if avoid_output.get("status") != "resolved":  # the drug name was not matched
        return avoid_output.get("message", "") + " " + NOT_SAFE_NOTE  # show the name problem plus the note
    entries = avoid_output["avoid"]  # the drugs to avoid, most severe first
    drug = avoid_output["drug"]  # the resolved drug name
    if not entries:  # nothing Major or Moderate found
        return f"A Major or Moderate interaction with {drug} is {NOT_DOCUMENTED}. " + NOT_SAFE_NOTE  # "not documented", never "safe"
    lines = [f"Drugs with a documented Major or Moderate interaction with {drug}, most severe first:"]  # heading line
    for entry in entries:  # one bullet per drug
        extra = ""  # optional conflict note
        if entry["severity_conflict"]:  # repeated pair with different severities
            extra = f" (listed with more than one severity: {' / '.join(entry['severities'])}; flagged for review)"  # explain the conflict
        line = f"- {entry['drug']}: {entry['severity']}{extra} [{', '.join(entry['record_ids'])}]"  # drug, severity and record IDs
        if detail != "brief" and entry.get("effect"):  # longer modes add the effect
            line += f" {_sentence(entry['effect'])}"  # append the stored effect
        lines.append(line)  # keep the bullet
    lines += list(avoid_output.get("notes", []))  # trailing notes such as "classes are not in our data"
    return "\n".join(lines)  # one bullet per line


# ---------------------------------------------------------------------------
# FILE SUMMARY: Builds the plain-language response text for pair checks and avoid lists purely from stored record
# fields (severity, mechanism, effect, rationale, safer alternative, reference) plus fixed status wording. It is the
# complete answer mode when no language model key exists and the fallback when a model's output fails validation.
# ASSIGNMENT TOOL: Part of the language model tool's design (the deterministic template mode and fallback beside
# llm.py), reading results produced by the knowledge graph and rule-based reasoning in checker.py.
# ---------------------------------------------------------------------------
