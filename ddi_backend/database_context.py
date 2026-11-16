"""Search only the DDI and HPO records loaded by load_data.py.

``search_records`` finds interaction records for a question with the same rules as
the question answerer: exact drug names first, negation and avoid lists, symptom
words through HPO, most severe first. It no longer mixes HPO terms into the
results; HPO is a vocabulary, so phenotype records come from ``search_hpo_records``.
The old word-by-word LIKE search let generic words ("not", "take", "which", "pain")
use up the result slots before the drug name contributed, so it was replaced.
"""

import re
import sys

from ddi_backend import ingest

REQUIRED_TABLES = ingest.REQUIRED_TABLES
QUERY_WORDS = re.compile(r"[a-z0-9]+", re.IGNORECASE)


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


def search_records(conn, query, limit=12):
    """Interaction records for a question, most severe first, with stable ``ddi:`` IDs.

    Only the drugs and interactions tables are read to build the answer. A question with
    no drug the data knows (and no symptom word) returns nothing.
    """
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    from ddi_backend.retrieval import candidate_records  # retrieval builds on this module's siblings

    return candidate_records(conn, query)[:limit]


def search_hpo_records(conn, query, limit=12):
    """Phenotype records for symptom words in the query. A vocabulary, never interaction evidence."""
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    from ddi_backend import hpo
    from ddi_backend.retrieval import STOPWORDS

    records, seen = [], set()
    for word in QUERY_WORDS.findall(query.casefold()):
        if word in STOPWORDS or len(word) < 4:
            continue
        for term in hpo.expand_symptom(conn, word)["terms"]:
            if term["id"] not in seen:
                seen.add(term["id"])
                records.append({
                    "id": term["id"], "source": "Human Phenotype Ontology", "hp_id": term["hp_id"],
                    "label": term["label"], "definition": term["definition"], "comment": term["comment"],
                })
    return records[:limit]


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
