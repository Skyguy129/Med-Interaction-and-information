# Module docstring: explains this file is the read-only door to medications.db and defines the record shape
"""Read-only access to medications.db and the shape of an interaction record.

``load_data.py`` builds the database; everything else opens it through
``open_readonly`` so the application can never alter the drugs or interactions
tables. Record IDs keep the loader's form: ``ddi:<interaction_id>``.
"""

import re  # regular expressions for parsing safer-alternative text
import sqlite3  # the database driver
from pathlib import Path  # path handling for the database file

from ddi_backend import load_data  # supplies DEFAULT_DB and table_names
from ddi_backend.common import severity_rank  # sort key: Major first

SOURCE_KEY = "kaggle_ddi"  # registry key stamped on every interaction record

CORE_TABLES = {"drugs", "interactions"}  # all the deterministic core needs
REQUIRED_TABLES = CORE_TABLES | {"hpo_terms", "hpo_synonyms", "hpo_edges"}  # the full set including the HPO tables

# SQL that joins each interaction to both drug names; later queries append a WHERE or ORDER BY to it
INTERACTION_SQL = """
    SELECT i.interaction_id, i.severity, a.name, b.name, a.drug_id, b.drug_id,
           i.mechanism, i.effect, i.safer_alternative, i.rationale, i.reference
    FROM interactions i
    JOIN drugs a ON a.drug_id = i.drug_a_id
    JOIN drugs b ON b.drug_id = i.drug_b_id
"""


def open_readonly(db_path=None, required=REQUIRED_TABLES, check_same_thread=True):  # the only way the app opens the database
    """Open the database read-only and check that the required tables exist.

    Never creates the file: a missing database raises RuntimeError with the
    command that builds it.
    """
    path = Path(db_path) if db_path is not None else load_data.DEFAULT_DB  # default to medications.db in the repo root
    load_cmd = f'python -m ddi_backend.load_data load --db "{path}"'  # the command to quote in error messages
    if not path.is_file():  # refuse to create a new empty database
        raise RuntimeError(
            f"Database not found: {path}. Load both source files first with {load_cmd}."  # tell the user how to build it
        )
    # open through a mode=ro URI so SQLite itself blocks writes
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=check_same_thread)
    try:  # make sure the connection is closed if the checks fail
        conn.execute("PRAGMA query_only = ON")  # second layer of protection against writes
        missing = set(required) - set(load_data.table_names(conn))  # which required tables are absent
        if missing:  # a partial database is rejected
            raise RuntimeError(
                f"Database is missing required tables: {', '.join(sorted(missing))}. "  # list them
                f"Load both source files with {load_cmd}."  # and how to fix it
            )
    except (RuntimeError, sqlite3.Error):  # any failure while checking
        conn.close()  # do not leak the connection
        raise  # pass the original error on
    return conn  # a safe read-only connection


# --------------------------------------------------------------------------
# Safer alternatives
# --------------------------------------------------------------------------
# The Safer alternative field is sometimes a drug ("Sertraline or Citalopram")
# and sometimes an instruction ("Reduce warfarin dose by 30-50%"). We classify
# it so the cross-check knows which one it is dealing with. The text itself is
# always shown exactly as written; nothing is generated or calculated.
ALTERNATIVE_DRUG = "alternative_drug"  # type label: the text names a specific drug
MANAGEMENT_ACTION = "management_action"  # type label: the text is an instruction

_ACTION_WORDS = {  # leading words that mark advice rather than a drug name
    "adjust", "alternative", "avoid", "consistent", "monitor", "none", "other",
    "reduce", "slow", "space", "start", "use", "withhold",
}
_EXAMPLE_PARENS = re.compile(r"\(e\.g\.,?\s*([^)]*)\)", re.IGNORECASE)  # "(e.g., enoxaparin)" examples
_PARENS = re.compile(r"\(([^)]*)\)")  # any parenthesised text
_SPLITTERS = re.compile(r"\s+or\s+|\s+instead of\s+|\s+and\s+|[,;/]", re.IGNORECASE)  # separators between several drugs
_TRAILING_QUALIFIER = re.compile(r"\s+(?:if|with|before|after)\b.*$", re.IGNORECASE)  # "... if needed" tails to drop
_CAPITALISED_NAME = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?")  # looks like a one- or two-word drug name


def parse_alternative(text, known_drug_names=frozenset()):  # classify a Safer alternative string
    """Classify a Safer alternative string.

    Returns {"text", "type", "drug_names"}. ``type`` is ``alternative_drug`` when
    the text names at least one specific drug and ``management_action``
    otherwise (instructions, monitoring advice, or a non-specific class such as
    "Alternative antibiotic"). ``drug_names`` lists the specific drugs named, as
    written, for the cross-check; it is empty for management actions.
    """
    text = (text or "").strip()  # treat None as empty
    first_word = re.split(r"[\s(]", text, maxsplit=1)[0].casefold() if text else ""  # first word, lower-cased
    if not text:  # nothing to classify
        return {"text": text, "type": MANAGEMENT_ACTION, "drug_names": []}  # empty text is not a drug
    leads_with_advice = first_word in _ACTION_WORDS  # does it start like an instruction?

    known = {name.casefold() for name in known_drug_names}  # real drug names from the database, lower-cased
    names = []  # drug names found so far

    def add(fragment, anything_goes=False):  # test one fragment and keep it if it is a drug name
        fragment = _TRAILING_QUALIFIER.sub("", fragment.strip(" .")).strip()  # trim punctuation and trailing qualifiers
        words = fragment.split()  # words in the fragment
        if not words or words[0].casefold() in _ACTION_WORDS or fragment.isupper():  # empty, advice, or class abbreviation
            return  # advice, or an abbreviation for a class (ARB, CCB)
        specific = (  # is this fragment a specific drug?
            fragment.casefold() in known  # it is a drug in our database
            or _CAPITALISED_NAME.fullmatch(fragment)  # or it looks like a drug name
            or (anything_goes and len(words) == 1 and fragment.isalpha())  # or a lone word inside an e.g. example
        )
        if specific and fragment not in names:  # avoid duplicates
            names.append(fragment)  # keep the drug name as written

    # "LMWH (e.g., enoxaparin)": anything after e.g. is an example drug.
    for example in _EXAMPLE_PARENS.findall(text):  # every "e.g." parenthesis
        for part in _SPLITTERS.split(example):  # split a list of examples
            add(part, anything_goes=True)  # examples are trusted as drug names
    if leads_with_advice:  # an instruction such as "Alternative antibiotic (e.g., ...)"
        # "Alternative antibiotic (e.g., Ceftazidime)": only the named example is a drug.
        kind = ALTERNATIVE_DRUG if names else MANAGEMENT_ACTION  # a drug type only if an example drug was found
        return {"text": text, "type": kind, "drug_names": names}  # finished classification
    rest = _EXAMPLE_PARENS.sub("", text)  # text with the e.g. parts removed
    # "Loop Diuretic (Furosemide)": a lone capitalised word in parentheses is the drug.
    parenthetical = [p for p in _PARENS.findall(rest) if re.fullmatch(r"[A-Z][a-z]+", p.strip())]  # lone-word parentheses
    rest = _PARENS.sub(" ", rest)  # remove all parentheses from the main text
    for part in _SPLITTERS.split(rest):  # split the main text into separate drugs
        if parenthetical and len(part.split()) > 1:  # a multi-word part is the class name, not the drug
            continue  # the parenthetical drug names the class, e.g. "Loop Diuretic"
        add(part)  # test the part as a drug name
    for inner in parenthetical:  # then the parenthetical drugs
        add(inner.strip())  # test each
    return {  # final classification
        "text": text,  # original text, shown exactly as written
        "type": ALTERNATIVE_DRUG if names else MANAGEMENT_ACTION,  # drug if any name was found
        "drug_names": names,  # the names, for the cross-check
    }


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def drug_names(conn):  # every drug name in the database
    return [row[0] for row in conn.execute("SELECT name FROM drugs ORDER BY name")]  # alphabetical list


def interaction_record(row, known_drug_names=frozenset()):  # convert one database row to the app-wide record dict
    """Turn one INTERACTION_SQL row into the record dict used across the app."""
    (interaction_id, severity, name_a, name_b, id_a, id_b,
     mechanism, effect, alternative, rationale, reference) = row  # unpack the columns in query order
    return {
        "id": f"ddi:{interaction_id}",  # stable record id used for citations
        "source": "drug interaction",  # human-readable source type
        "source_key": SOURCE_KEY,  # key the eligibility filter looks up
        "role": "direct",  # one interaction source, so every record names its exact pair
        "severity": severity,  # Major / Moderate / Minor
        "drug_a": name_a,  # first drug name
        "drug_b": name_b,  # second drug name
        "drug_a_id": id_a,  # first drug id
        "drug_b_id": id_b,  # second drug id
        "mechanism": mechanism,  # how the interaction happens
        "effect": effect,  # what it does clinically
        "safer_alternative": alternative,  # the raw text as stored
        "alternative": parse_alternative(alternative, known_drug_names),  # the same text, typed and parsed
        "rationale": rationale,  # why the alternative is suggested
        "reference": reference,  # the record's own citation
    }


def load_interactions(conn):  # every interaction as a record
    """Every interaction row as a record, most severe first, then by drug names."""
    known = frozenset(drug_names(conn))  # known drug names for alternative parsing
    records = [interaction_record(row, known) for row in conn.execute(INTERACTION_SQL)]  # convert each row
    records.sort(key=lambda r: (severity_rank(r["severity"]), r["drug_a"].casefold(),
                                r["drug_b"].casefold(), r["id"]))  # Major first, then names, then id
    return records  # the sorted list


def pair_records(conn, id_a, id_b, known_drug_names=frozenset()):  # all rows for one unordered pair
    """Every row for the unordered pair, found by a direct query on the two drug IDs.

    Repeated pairs are returned in full (the loader keeps them on purpose).
    """
    rows = conn.execute(  # direct query, so a Major row is never missed by a search step
        INTERACTION_SQL
        + """ WHERE (i.drug_a_id = ? AND i.drug_b_id = ?) OR (i.drug_a_id = ? AND i.drug_b_id = ?)
              ORDER BY i.interaction_id""",  # match the pair in either order
        (id_a, id_b, id_b, id_a),  # values for the four placeholders
    ).fetchall()  # read all matching rows
    return [interaction_record(row, known_drug_names) for row in rows]  # convert to records


def drug_record(conn, drug_id, known_drug_names=frozenset()):  # a drug and the interactions it appears in
    """The drug's name and the interactions it appears in. No monograph data exists."""
    row = conn.execute("SELECT drug_id, name FROM drugs WHERE drug_id = ?", (drug_id,)).fetchone()  # find the drug
    if row is None:  # no such drug
        return None  # caller reports unknown
    rows = conn.execute(  # every interaction where this drug is either side
        INTERACTION_SQL + " WHERE i.drug_a_id = ? OR i.drug_b_id = ? ORDER BY i.interaction_id",
        (drug_id, drug_id),  # same id for both placeholders
    ).fetchall()  # read them
    interactions = [interaction_record(r, known_drug_names) for r in rows]  # convert to records
    interactions.sort(key=lambda r: (severity_rank(r["severity"]), r["id"]))  # most severe first
    return {"drug_id": row[0], "name": row[1], "interactions": interactions}  # package the result

# ---------------------------------------------------------------------------
# FILE SUMMARY: The only door through which the application reads medications.db. It opens the database read-only,
# checks the tables exist, turns interaction rows into ddi:<id> record dictionaries, finds every row for a pair by
# direct query, and classifies a "safer alternative" as an alternative drug or a management action.
# ASSIGNMENT TOOL: knowledge graph with rule-based reasoning (it supplies the nodes and edges, i.e. drugs and interaction
# records, that graph.py and rules.py reason over).
# ---------------------------------------------------------------------------
