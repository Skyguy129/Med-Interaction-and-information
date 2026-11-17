# Module docstring: says this file holds wording and severity ordering that every layer reuses
"""Constants shared by every layer: wording that must appear on responses, and severity order."""

# Disclaimer attached to every response so nothing reads as medical advice
DISCLAIMER = (
    "Class prototype, not medical advice. It reports only what the project's "  # states it is a class prototype
    "interaction data documents; talk to a pharmacist or clinician before "  # limits claims to the documented data
    "starting, stopping or changing any medication."  # tells the user to consult a professional
)
# Label put on anything simulated (the review queue) so it is never mistaken for a real action
SIMULATED_NOTE = "SIMULATED: nothing here reaches a real pharmacist, patient or medical record."
# Wording for pairs with no row: "not found" must never be presented as "safe"
NOT_SAFE_NOTE = (
    "No documented interaction in our data does not mean the combination is safe. "  # the key rule 1 sentence
    "Our data covers a limited set of drugs and pairs."  # explains why absence proves nothing
)
# Standard phrase for facts the data does not hold (uses, side effects, doses, etc.)
NOT_AVAILABLE = "not available in our data"

# The only three severities in the data, most serious first
SEVERITIES = ("Major", "Moderate", "Minor")
# Maps each severity to its position, so Major = 0, Moderate = 1, Minor = 2
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}  # lower = more severe


def severity_rank(severity):  # sort-key helper used wherever results are ordered Major first
    # Docstring: explains the ordering and that unknown severities go to the end
    """Sort key where Major < Moderate < Minor; unknown values sort last."""
    return SEVERITY_RANK.get(severity, len(SEVERITIES))  # unknown severity gets a rank after Minor

# ---------------------------------------------------------------------------
# FILE SUMMARY: Holds the fixed wording (disclaimer, "simulated" label, "not safe" note, "not available" phrase) and the
# severity order that every other module imports. Keeping them in one place guarantees every response carries the same
# safety language and sorts Major findings first.
# ASSIGNMENT TOOL: supporting infrastructure (not one of the assignment tools): it supplies the severity-precedence
# ordering used by the knowledge graph and rules, and the safety wording used by the composer and the agent.
# ---------------------------------------------------------------------------
