"""Read-only access to medications.db and the shape of an interaction record.

``load_data.py`` builds the database; everything else opens it through
``open_readonly`` so the application can never alter the drugs or interactions
tables. Record IDs keep the loader's form: ``ddi:<interaction_id>``.
"""

import re
import sqlite3
from pathlib import Path

from ddi_backend import load_data
from ddi_backend.common import severity_rank

SOURCE_KEY = "kaggle_ddi"

CORE_TABLES = {"drugs", "interactions"}  # all the deterministic core needs
REQUIRED_TABLES = CORE_TABLES | {"hpo_terms", "hpo_synonyms", "hpo_edges"}

INTERACTION_SQL = """
    SELECT i.interaction_id, i.severity, a.name, b.name, a.drug_id, b.drug_id,
           i.mechanism, i.effect, i.safer_alternative, i.rationale, i.reference
    FROM interactions i
    JOIN drugs a ON a.drug_id = i.drug_a_id
    JOIN drugs b ON b.drug_id = i.drug_b_id
"""


def open_readonly(db_path=None, required=REQUIRED_TABLES):
    """Open the database read-only and check that the required tables exist.

    Never creates the file: a missing database raises RuntimeError with the
    command that builds it.
    """
    path = Path(db_path) if db_path is not None else load_data.DEFAULT_DB
    load_cmd = f'python -m ddi_backend.load_data load --db "{path}"'
    if not path.is_file():
        raise RuntimeError(
            f"Database not found: {path}. Load both source files first with {load_cmd}."
        )
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        missing = set(required) - set(load_data.table_names(conn))
        if missing:
            raise RuntimeError(
                f"Database is missing required tables: {', '.join(sorted(missing))}. "
                f"Load both source files with {load_cmd}."
            )
    except (RuntimeError, sqlite3.Error):
        conn.close()
        raise
    return conn


# --------------------------------------------------------------------------
# Safer alternatives
# --------------------------------------------------------------------------
# The Safer alternative field is sometimes a drug ("Sertraline or Citalopram")
# and sometimes an instruction ("Reduce warfarin dose by 30-50%"). We classify
# it so the cross-check knows which one it is dealing with. The text itself is
# always shown exactly as written; nothing is generated or calculated.
ALTERNATIVE_DRUG = "alternative_drug"
MANAGEMENT_ACTION = "management_action"

_ACTION_WORDS = {
    "adjust", "alternative", "avoid", "consistent", "monitor", "none", "other",
    "reduce", "slow", "space", "start", "use", "withhold",
}
_EXAMPLE_PARENS = re.compile(r"\(e\.g\.,?\s*([^)]*)\)", re.IGNORECASE)
_PARENS = re.compile(r"\(([^)]*)\)")
_SPLITTERS = re.compile(r"\s+or\s+|\s+instead of\s+|\s+and\s+|[,;/]", re.IGNORECASE)
_TRAILING_QUALIFIER = re.compile(r"\s+(?:if|with|before|after)\b.*$", re.IGNORECASE)
_CAPITALISED_NAME = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?")


def parse_alternative(text, known_drug_names=frozenset()):
    """Classify a Safer alternative string.

    Returns {"text", "type", "drug_names"}. ``type`` is ``alternative_drug`` when
    the text names at least one specific drug and ``management_action``
    otherwise (instructions, monitoring advice, or a non-specific class such as
    "Alternative antibiotic"). ``drug_names`` lists the specific drugs named, as
    written, for the cross-check; it is empty for management actions.
    """
    text = (text or "").strip()
    first_word = re.split(r"[\s(]", text, maxsplit=1)[0].casefold() if text else ""
    if not text:
        return {"text": text, "type": MANAGEMENT_ACTION, "drug_names": []}
    leads_with_advice = first_word in _ACTION_WORDS

    known = {name.casefold() for name in known_drug_names}
    names = []

    def add(fragment, anything_goes=False):
        fragment = _TRAILING_QUALIFIER.sub("", fragment.strip(" .")).strip()
        words = fragment.split()
        if not words or words[0].casefold() in _ACTION_WORDS or fragment.isupper():
            return  # advice, or an abbreviation for a class (ARB, CCB)
        specific = (
            fragment.casefold() in known
            or _CAPITALISED_NAME.fullmatch(fragment)
            or (anything_goes and len(words) == 1 and fragment.isalpha())
        )
        if specific and fragment not in names:
            names.append(fragment)

    # "LMWH (e.g., enoxaparin)": anything after e.g. is an example drug.
    for example in _EXAMPLE_PARENS.findall(text):
        for part in _SPLITTERS.split(example):
            add(part, anything_goes=True)
    if leads_with_advice:
        # "Alternative antibiotic (e.g., Ceftazidime)": only the named example is a drug.
        kind = ALTERNATIVE_DRUG if names else MANAGEMENT_ACTION
        return {"text": text, "type": kind, "drug_names": names}
    rest = _EXAMPLE_PARENS.sub("", text)
    # "Loop Diuretic (Furosemide)": a lone capitalised word in parentheses is the drug.
    parenthetical = [p for p in _PARENS.findall(rest) if re.fullmatch(r"[A-Z][a-z]+", p.strip())]
    rest = _PARENS.sub(" ", rest)
    for part in _SPLITTERS.split(rest):
        if parenthetical and len(part.split()) > 1:
            continue  # the parenthetical drug names the class, e.g. "Loop Diuretic"
        add(part)
    for inner in parenthetical:
        add(inner.strip())
    return {
        "text": text,
        "type": ALTERNATIVE_DRUG if names else MANAGEMENT_ACTION,
        "drug_names": names,
    }


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def drug_names(conn):
    return [row[0] for row in conn.execute("SELECT name FROM drugs ORDER BY name")]


def interaction_record(row, known_drug_names=frozenset()):
    """Turn one INTERACTION_SQL row into the record dict used across the app."""
    (interaction_id, severity, name_a, name_b, id_a, id_b,
     mechanism, effect, alternative, rationale, reference) = row
    return {
        "id": f"ddi:{interaction_id}",
        "source": "drug interaction",
        "source_key": SOURCE_KEY,
        "role": "direct",  # one interaction source, so every record names its exact pair
        "severity": severity,
        "drug_a": name_a,
        "drug_b": name_b,
        "drug_a_id": id_a,
        "drug_b_id": id_b,
        "mechanism": mechanism,
        "effect": effect,
        "safer_alternative": alternative,
        "alternative": parse_alternative(alternative, known_drug_names),
        "rationale": rationale,
        "reference": reference,
    }


def load_interactions(conn):
    """Every interaction row as a record, most severe first, then by drug names."""
    known = frozenset(drug_names(conn))
    records = [interaction_record(row, known) for row in conn.execute(INTERACTION_SQL)]
    records.sort(key=lambda r: (severity_rank(r["severity"]), r["drug_a"].casefold(),
                                r["drug_b"].casefold(), r["id"]))
    return records
