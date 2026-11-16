"""Constants shared by every layer: wording that must appear on responses, and severity order."""

DISCLAIMER = (
    "Class prototype, not medical advice. It reports only what the project's "
    "interaction data documents; talk to a pharmacist or clinician before "
    "starting, stopping or changing any medication."
)
SIMULATED_NOTE = "SIMULATED: nothing here reaches a real pharmacist, patient or medical record."
NOT_SAFE_NOTE = (
    "No documented interaction in our data does not mean the combination is safe. "
    "Our data covers a limited set of drugs and pairs."
)
NOT_AVAILABLE = "not available in our data"

SEVERITIES = ("Major", "Moderate", "Minor")
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}  # lower = more severe


def severity_rank(severity):
    """Sort key where Major < Moderate < Minor; unknown values sort last."""
    return SEVERITY_RANK.get(severity, len(SEVERITIES))
