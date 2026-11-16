"""Search only the DDI and HPO records loaded by load_data.py."""

import re
import sys

from ddi_backend import ingest

REQUIRED_TABLES = ingest.REQUIRED_TABLES
QUERY_WORDS = re.compile(r"[a-z0-9]+", re.IGNORECASE)
IGNORED_TERMS = {
    "about",
    "and",
    "are",
    "can",
    "condition",
    "conditions",
    "database",
    "describe",
    "does",
    "drug",
    "drugs",
    "effect",
    "effects",
    "explain",
    "find",
    "for",
    "give",
    "hpo",
    "involve",
    "involves",
    "involving",
    "information",
    "interaction",
    "interactions",
    "is",
    "medical",
    "me",
    "of",
    "please",
    "related",
    "show",
    "side",
    "symptom",
    "symptoms",
    "tell",
    "the",
    "these",
    "this",
    "what",
    "phenotype",
    "phenotypes",
    "with",
}


def open_database(db_path=None):
    """Open and validate the database populated by ``load_data.py load``.

    The connection is read-only, so nothing in the application can alter
    medications.db.
    """
    return ingest.open_readonly(db_path, REQUIRED_TABLES)


def configure_console_output():
    """Prevent database Unicode characters from crashing legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def query_terms(query):
    """Extract useful search words, excluding generic question wording."""
    terms = []
    seen = set()
    for match in QUERY_WORDS.findall(query.casefold()):
        if len(match) < 3 or match in IGNORED_TERMS or match in seen:
            continue
        seen.add(match)
        terms.append(match)
    return terms


def _like_pattern(term):
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_records(conn, query, limit=12):
    """Return database records matching words in a query, with stable IDs.

    Both data families are searched in the SQLite database created by
    ``load_data.py``. No source text, external files, or model memory is used
    to construct returned records.
    """
    if limit < 1:
        raise ValueError("limit must be greater than zero")

    records = {}
    for term in query_terms(query):
        pattern = _like_pattern(term)
        ddi_rows = conn.execute(
            """SELECT i.interaction_id, i.severity, a.name, b.name,
                      i.mechanism, i.effect, i.safer_alternative,
                      i.rationale, i.reference
               FROM interactions i
               JOIN drugs a ON a.drug_id = i.drug_a_id
               JOIN drugs b ON b.drug_id = i.drug_b_id
               WHERE a.name LIKE ? ESCAPE '\\'
                  OR b.name LIKE ? ESCAPE '\\'
                  OR COALESCE(i.effect, '') LIKE ? ESCAPE '\\'
                  OR COALESCE(i.mechanism, '') LIKE ? ESCAPE '\\'
               ORDER BY CASE i.severity
                            WHEN 'Major' THEN 1
                            WHEN 'Moderate' THEN 2
                            ELSE 3
                        END, a.name, b.name
               LIMIT ?""",
            (pattern, pattern, pattern, pattern, limit),
        ).fetchall()
        for row in ddi_rows:
            record_id, severity, drug_a, drug_b, mechanism, effect, alternative, rationale, reference = row
            record = {
                "id": f"ddi:{record_id}",
                "source": "drug interaction",
                "severity": severity,
                "drug_a": drug_a,
                "drug_b": drug_b,
                "mechanism": mechanism,
                "effect": effect,
                "safer_alternative": alternative,
                "rationale": rationale,
                "reference": reference,
            }
            records[record["id"]] = record

        hpo_rows = conn.execute(
            """SELECT DISTINCT t.hp_id, t.label, t.definition, t.comment
               FROM hpo_terms t
               LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
               WHERE t.is_obsolete = 0
                 AND (t.label LIKE ? ESCAPE '\\'
                      OR COALESCE(s.synonym, '') LIKE ? ESCAPE '\\')
               ORDER BY t.label
               LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        for hp_id, label, definition, comment in hpo_rows:
            record = {
                "id": f"hpo:{hp_id}",
                "source": "Human Phenotype Ontology",
                "hp_id": hp_id,
                "label": label,
                "definition": definition,
                "comment": comment,
            }
            records[record["id"]] = record

    return list(records.values())[:limit]


def format_record(record):
    """Render one database row using only its stored values."""
    lines = [f"[{record['source']}] {record['id']}"]
    field_labels = {
        "severity": "Severity",
        "drug_a": "Drug A",
        "drug_b": "Drug B",
        "mechanism": "Mechanism",
        "effect": "Effect",
        "safer_alternative": "Safer alternative",
        "rationale": "Rationale",
        "reference": "Reference",
        "hp_id": "HPO ID",
        "label": "Label",
        "definition": "Definition",
        "comment": "Comment",
    }
    for field, label in field_labels.items():
        value = record.get(field)
        if value is not None and str(value).strip():
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)
