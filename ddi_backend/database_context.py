# Module docstring: this file is the search entry point over the loaded DDI and HPO records
"""Search only the DDI and HPO records loaded by load_data.py.

``search_records`` finds interaction records for a question with the same rules as
the question answerer: exact drug names first, negation and avoid lists, symptom
words through HPO, most severe first. It no longer mixes HPO terms into the
results; HPO is a vocabulary, so phenotype records come from ``search_hpo_records``.
The old word-by-word LIKE search let generic words ("not", "take", "which", "pain")
use up the result slots before the drug name contributed, so it was replaced.
"""

import re  # regular expressions to split a query into words
import sys  # access to the stdout and stderr streams

from ddi_backend import ingest  # read-only database helpers

REQUIRED_TABLES = ingest.REQUIRED_TABLES  # the tables a complete database must contain
QUERY_WORDS = re.compile(r"[a-z0-9]+", re.IGNORECASE)  # pattern that pulls letters and digits out of a question


def open_database(db_path=None):  # open medications.db safely
    """Open and validate the database populated by ``load_data.py load``.

    The connection is read-only, so nothing in the application can alter
    medications.db.
    """
    return ingest.open_readonly(db_path, REQUIRED_TABLES)  # read-only open that also rejects a partial database


def configure_console_output():  # make printing safe on old Windows consoles
    """Prevent database Unicode characters from crashing legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):  # both normal and error output
        reconfigure = getattr(stream, "reconfigure", None)  # not every stream type supports reconfigure
        if reconfigure is not None:  # only act when it exists
            reconfigure(errors="backslashreplace")  # unprintable characters print as escapes instead of crashing


def search_records(conn, query, limit=12):  # question -> interaction records
    """Interaction records for a question, most severe first, with stable ``ddi:`` IDs.

    Only the drugs and interactions tables are read to build the answer. A question with
    no drug the data knows (and no symptom word) returns nothing.
    """
    if limit < 1:  # a zero or negative limit makes no sense
        raise ValueError("limit must be greater than zero")  # reject it loudly
    from ddi_backend.retrieval import candidate_records  # retrieval builds on this module's siblings

    return candidate_records(conn, query)[:limit]  # shared candidate search, cut down to the limit


def search_hpo_records(conn, query, limit=12):  # question -> phenotype (symptom) records
    """Phenotype records for symptom words in the query. A vocabulary, never interaction evidence."""
    if limit < 1:  # same limit check as above
        raise ValueError("limit must be greater than zero")
    from ddi_backend import hpo  # symptom word expansion through HPO terms
    from ddi_backend.retrieval import STOPWORDS  # common words to ignore

    records, seen = [], set()  # results so far and HPO term IDs already added
    for word in QUERY_WORDS.findall(query.casefold()):  # each lower-case word in the question
        if word in STOPWORDS or len(word) < 4:  # skip filler and very short words
            continue
        for term in hpo.expand_symptom(conn, word)["terms"]:  # HPO terms matching this word
            if term["id"] not in seen:  # avoid the same term twice
                seen.add(term["id"])  # remember it
                records.append({  # add a phenotype record
                    "id": term["id"], "source": "Human Phenotype Ontology", "hp_id": term["hp_id"],
                    "label": term["label"], "definition": term["definition"], "comment": term["comment"],
                })
    return records[:limit]  # stop at the limit


def format_record(record):  # turn one stored row into printable text
    """Render one database row using only its stored values."""
    lines = [f"[{record['source']}] {record['id']}"]  # header line with the source and the stable ID
    field_labels = {  # display label for each field, in print order
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
    for field, label in field_labels.items():  # go through the fields in order
        value = record.get(field)  # this record may not have every field
        if value is not None and str(value).strip():  # print only values that are present and non-blank
            lines.append(f"  {label}: {value}")  # indented "Label: value" line
    return "\n".join(lines)  # join into one block of text


# ---------------------------------------------------------------------------
# FILE SUMMARY: A small search layer over the loaded DDI and HPO records. It opens the database read-only, finds
# interaction records for a question through the retrieval code (exact drug names first, most severe first),
# returns phenotype records separately because HPO is only a vocabulary, and formats a stored row using only values
# that are actually present.
# ASSIGNMENT TOOL: Supporting infrastructure for the retrieval side of the system (the question answering that uses the
# knowledge graph and rules, with embeddings only in a limited role). It does not use RAG: one interaction row is one
# whole record, and nothing is split into pieces.
# ---------------------------------------------------------------------------
