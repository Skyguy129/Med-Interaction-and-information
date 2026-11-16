"""The template composer: plain-language responses built only from stored fields.

This is the complete deterministic mode. It writes the response from a record's
severity, mechanism, effect, rationale, safer alternative and reference exactly as
stored, plus the status wording the rules require. It is also the fallback when a
language model's output fails validation or the provider is unavailable.
Nothing is generated, calculated or recommended; doses are echoed as typed.
"""

from ddi_backend.common import DISCLAIMER, NOT_SAFE_NOTE

DETAIL_LEVELS = ("brief", "standard", "detailed")
NOT_DOCUMENTED = "not documented in our data"


def _sentence(text):
    text = (text or "").strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _alternative_lines(alternatives, detail):
    lines = []
    for alt in alternatives:
        kind = "suggested drug" if alt["type"] == "alternative_drug" else "management action"
        line = f"Safer alternative ({kind}, as written in {alt['record_id']}): {alt['text']}"
        if alt.get("rationale"):
            line += f" Reason given: {_sentence(alt['rationale'])}"
        lines.append(line)
        for finding in alt.get("cross_check", []):
            if finding.get("warning"):
                lines.append(finding["warning"])
            elif not finding.get("in_data") and detail != "brief":
                lines.append(finding["note"])
    return lines


def compose_documented(result, detail="standard"):
    a, b = result["pair"]
    records = result["records"]
    headline = records[0]
    lines = []
    if detail == "brief":
        lines.append(f"{a} and {b}: documented {result['severity']} interaction ({headline['id']}). "
                     + _sentence(headline.get("effect") or ""))
        if result.get("severity_conflict"):
            lines.append("Our data lists this pair with more than one severity; the highest is shown and it is "
                         "flagged for review.")
        lines.append(f"Source: {headline['id']}, {headline.get('reference')}")
        return "\n".join(lines)

    lines.append(f"Taking {a} with {b} is a documented {result['severity']} interaction in our data.")
    if result.get("severity_conflict"):
        lines.append(f"Our data lists this pair {len(records)} times with different severities "
                     f"({' / '.join(result['severities'])}). The highest, {result['severity']}, is shown, and the "
                     "pair is flagged for pharmacist review.")
    if detail == "standard":
        if headline.get("mechanism"):
            lines.append(f"Mechanism: {headline['mechanism']}")
        if headline.get("effect"):
            lines.append(f"Clinical effect: {headline['effect']}")
        alternatives = result.get("alternatives", [])
        # The first alternative belongs to the headline record; show them all when the pair has a conflict.
        lines += _alternative_lines(alternatives if result.get("severity_conflict") else alternatives[:1], detail)
    else:
        for record in records:
            lines.append(f"Record {record['id']} ({record['severity']}): {record['drug_a']} + {record['drug_b']}")
            for label, key in (("Mechanism", "mechanism"), ("Clinical effect", "effect"),
                               ("Safer alternative", "safer_alternative"), ("Rationale", "rationale"),
                               ("Reference", "reference")):
                if record.get(key):
                    lines.append(f"  {label}: {record[key]}")
        lines += _alternative_lines(result.get("alternatives", []), detail)
        doses = {name: dose for name, dose in (result.get("doses") or {}).items() if dose}
        if doses:
            lines.append("Doses as you entered them: " + ", ".join(f"{n} {d}" for n, d in doses.items())
                         + ". This system stores and echoes doses; it does not evaluate or recommend them.")
    if result.get("needs_review"):
        lines.append("Flagged for review: " + "; ".join(result.get("review_reasons", [])) + ".")
    lines.append("Source: " + "; ".join(f"{c['id']} ({c['reference']})" for c in result["source"]["citations"]))
    return "\n".join(lines)


def compose_pair(result, detail="standard"):
    """Plain-language text for one pair result, whatever its status."""
    status = result["status"]
    a, b = result["pair"]
    if status == "documented_interaction":
        return compose_documented(result, detail)
    if status == "no_documented_interaction":
        return f"An interaction between {a} and {b} is {NOT_DOCUMENTED}. " + NOT_SAFE_NOTE
    if status == "evidence_ineligible":
        return result["message"] + " It is flagged for review. " + NOT_SAFE_NOTE
    # ambiguous_name / unknown_drug
    return (result.get("message") or f"We could not match {a} or {b} to a drug name in our data.") \
        + " " + NOT_SAFE_NOTE


def compose_check(check_output, detail="standard"):
    """The full text for a /check style response: one block per pair, Major first."""
    blocks = [compose_pair(r, detail) for r in check_output["results"]]
    footer = [DISCLAIMER]
    return "\n\n".join(blocks + footer)


def compose_avoid(avoid_output, detail="standard"):
    if avoid_output.get("status") != "resolved":
        return avoid_output.get("message", "") + " " + NOT_SAFE_NOTE
    entries = avoid_output["avoid"]
    drug = avoid_output["drug"]
    if not entries:
        return f"A Major or Moderate interaction with {drug} is {NOT_DOCUMENTED}. " + NOT_SAFE_NOTE
    lines = [f"Drugs with a documented Major or Moderate interaction with {drug}, most severe first:"]
    for entry in entries:
        extra = ""
        if entry["severity_conflict"]:
            extra = f" (listed with more than one severity: {' / '.join(entry['severities'])}; flagged for review)"
        line = f"- {entry['drug']}: {entry['severity']}{extra} [{', '.join(entry['record_ids'])}]"
        if detail != "brief" and entry.get("effect"):
            line += f" {_sentence(entry['effect'])}"
        lines.append(line)
    lines += list(avoid_output.get("notes", []))
    return "\n".join(lines)
