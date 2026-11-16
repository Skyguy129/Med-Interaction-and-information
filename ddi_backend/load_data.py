#!/usr/bin/env python3
"""Load the DDI and HPO JSON files into a local SQLite database.

Run from the repository root:

    python -m ddi_backend.load_data load              # build medications.db from both JSON files
    python -m ddi_backend.load_data stats             # row counts + data-quality report
    python -m ddi_backend.load_data drug warfarin     # interactions that involve a drug
    python -m ddi_backend.load_data hpo bleeding      # search HPO terms and synonyms

Only the Python standard library is used (sqlite3, json, argparse).
Running `load` again rebuilds the tables, so it is safe to repeat.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Paths are relative to the repository root, so it works no matter where the
# terminal is opened. Override with --db / --ddi / --hpo.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DB = ROOT / "medications.db"


def first_existing(path_candidates, fallback):
    """Return the first existing path, or the fallback if none exist."""
    for path in path_candidates:
        if path.exists():
            return path
    return fallback


DATA_DIR = ROOT / "data"
DEFAULT_DDI = first_existing(
    [DATA_DIR / "DDI_Database.json", DATA_DIR / "DDI Database.json",
     ROOT / "DDI_Database.json", ROOT / "DDI Database.json"],
    DATA_DIR / "DDI_Database.json",
)
DEFAULT_HPO = first_existing(
    [DATA_DIR / "hp.json", DATA_DIR / "HP.json", ROOT / "hp.json", ROOT / "HP.json"],
    DATA_DIR / "hp.json",
)

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
# Drugs are nodes; interactions are edges between two drugs (see the dossier).
# There is deliberately NO unique constraint on a drug pair: the source data
# lists some pairs more than once with different severities, and we want to
# keep every row and report the conflicts rather than silently drop one.
DDI_SCHEMA = """
DROP TABLE IF EXISTS interactions;
DROP TABLE IF EXISTS drugs;

CREATE TABLE drugs (
    drug_id INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE interactions (
    interaction_id    INTEGER PRIMARY KEY,
    drug_a_id         INTEGER NOT NULL REFERENCES drugs(drug_id),
    drug_b_id         INTEGER NOT NULL REFERENCES drugs(drug_id),
    severity          TEXT NOT NULL CHECK (severity IN ('Major', 'Moderate', 'Minor')),
    mechanism         TEXT,
    effect            TEXT,
    safer_alternative TEXT,
    rationale         TEXT,
    reference         TEXT
);

CREATE INDEX idx_interactions_a ON interactions(drug_a_id);
CREATE INDEX idx_interactions_b ON interactions(drug_b_id);
"""

# Human Phenotype Ontology: terms, their synonyms, and is_a (child -> parent) links.
HPO_SCHEMA = """
DROP TABLE IF EXISTS hpo_edges;
DROP TABLE IF EXISTS hpo_synonyms;
DROP TABLE IF EXISTS hpo_terms;

CREATE TABLE hpo_terms (
    hp_id       TEXT PRIMARY KEY,              -- e.g. 'HP:0000002'
    label       TEXT NOT NULL,
    definition  TEXT,
    comment     TEXT,
    is_obsolete INTEGER NOT NULL DEFAULT 0     -- 1 = deprecated term
);

CREATE TABLE hpo_synonyms (
    synonym_id   INTEGER PRIMARY KEY,
    hp_id        TEXT NOT NULL REFERENCES hpo_terms(hp_id),
    synonym      TEXT NOT NULL,
    scope        TEXT,                         -- exact / related / broad / narrow
    synonym_type TEXT                          -- layperson, abbreviation, uk_spelling, ...
);

CREATE TABLE hpo_edges (
    child_id  TEXT NOT NULL REFERENCES hpo_terms(hp_id),
    parent_id TEXT NOT NULL REFERENCES hpo_terms(hp_id),
    PRIMARY KEY (child_id, parent_id)
);

CREATE INDEX idx_hpo_synonyms_hp ON hpo_synonyms(hp_id);
CREATE INDEX idx_hpo_edges_parent ON hpo_edges(parent_id);
"""

SYNONYM_SCOPE = {
    "hasExactSynonym": "exact",
    "hasRelatedSynonym": "related",
    "hasBroadSynonym": "broad",
    "hasNarrowSynonym": "narrow",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hp_id_from_iri(iri):
    """'http://purl.obolibrary.org/obo/HP_0000002' -> 'HP:0000002'"""
    return iri.rsplit("/", 1)[-1].replace("_", ":", 1)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_ddi(conn, path):
    """Load DDI_Database.json. Returns (interaction_count, drug_count)."""
    path = Path(path)
    data = read_json(path)
    # Expected shape: {"drug_interactions": {"major": [...], "moderate": [...], "minor": [...]}}
    buckets = data.get("drug_interactions") if isinstance(data, dict) else None
    if not isinstance(buckets, dict):
        raise ValueError(f"{path.name}: expected a top-level 'drug_interactions' object")

    conn.executescript(DDI_SCHEMA)
    drug_ids = {}  # lower-cased name -> drug_id, so 'Omeprazole' and 'omeprazole' are one drug

    def get_drug_id(name):
        name = name.strip()
        key = name.lower()
        if key not in drug_ids:
            cur = conn.execute("INSERT INTO drugs (name) VALUES (?)", (name,))
            drug_ids[key] = cur.lastrowid
        return drug_ids[key]

    count = 0
    try:
        with conn:  # one transaction: all rows are saved, or none
            for rows in buckets.values():
                for r in rows:
                    conn.execute(
                        """INSERT INTO interactions
                           (drug_a_id, drug_b_id, severity, mechanism, effect,
                            safer_alternative, rationale, reference)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            get_drug_id(r["drug_a"]),
                            get_drug_id(r["drug_b"]),
                            r["severity"].strip().capitalize(),
                            r.get("mechanism"),
                            r.get("effect"),
                            r.get("Safer_alternative"),
                            r.get("rationale"),
                            r.get("reference"),
                        ),
                    )
                    count += 1
    except KeyError as e:
        raise ValueError(f"{path.name}: an interaction row is missing the field {e}") from None
    return count, len(drug_ids)


def load_hpo(conn, path):
    """Load hp.json (OBO-graphs format). Returns a dict of row counts."""
    path = Path(path)
    data = read_json(path)
    try:
        graph = data["graphs"][0]
        nodes, edges = graph["nodes"], graph["edges"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"{path.name}: expected an OBO-graphs file with graphs[0].nodes and .edges")

    terms, synonyms = [], []
    for node in nodes:
        # Keep HP terms only; this skips relation nodes such as 'has part'.
        if node.get("type") != "CLASS" or "/HP_" not in node["id"]:
            continue
        hp_id = hp_id_from_iri(node["id"])
        meta = node.get("meta", {})
        terms.append((
            hp_id,
            node.get("lbl", ""),
            (meta.get("definition") or {}).get("val"),
            "\n".join(meta.get("comments", [])) or None,
            1 if meta.get("deprecated") else 0,
        ))
        for syn in meta.get("synonyms", []):
            synonyms.append((
                hp_id,
                syn["val"],
                SYNONYM_SCOPE.get(syn.get("pred"), syn.get("pred")),
                (syn.get("synonymType") or "").rsplit("#", 1)[-1] or None,
            ))

    term_ids = {t[0] for t in terms}
    edge_rows, skipped_edges = [], 0
    for e in edges:
        if e.get("pred") != "is_a":
            continue
        child, parent = hp_id_from_iri(e["sub"]), hp_id_from_iri(e["obj"])
        if child in term_ids and parent in term_ids:
            edge_rows.append((child, parent))
        else:
            skipped_edges += 1

    conn.executescript(HPO_SCHEMA)
    with conn:
        conn.executemany("INSERT INTO hpo_terms VALUES (?, ?, ?, ?, ?)", terms)
        conn.executemany(
            "INSERT INTO hpo_synonyms (hp_id, synonym, scope, synonym_type) VALUES (?, ?, ?, ?)",
            synonyms,
        )
        conn.executemany("INSERT OR IGNORE INTO hpo_edges VALUES (?, ?)", edge_rows)

    n_edges = conn.execute("SELECT COUNT(*) FROM hpo_edges").fetchone()[0]
    return {"terms": len(terms), "synonyms": len(synonyms), "edges": n_edges, "skipped_edges": skipped_edges}


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def find_repeated_pairs(conn):
    """Drug pairs that appear in more than one interaction row (either order).

    Returns [(drug_1, drug_2, 'Major / Moderate'), ...].
    """
    return conn.execute(
        """SELECT d1.name, d2.name, p.severities
           FROM (SELECT MIN(drug_a_id, drug_b_id) AS lo,
                        MAX(drug_a_id, drug_b_id) AS hi,
                        COUNT(*) AS n,
                        GROUP_CONCAT(severity, ' / ') AS severities
                 FROM interactions GROUP BY lo, hi HAVING n > 1) AS p
           JOIN drugs d1 ON d1.drug_id = p.lo
           JOIN drugs d2 ON d2.drug_id = p.hi
           ORDER BY d1.name, d2.name"""
    ).fetchall()


def find_interactions(conn, name):
    """Interactions where either drug name contains `name` (case-insensitive)."""
    like = f"%{name}%"
    return conn.execute(
        """SELECT i.severity, a.name, b.name, i.effect, i.safer_alternative
           FROM interactions i
           JOIN drugs a ON a.drug_id = i.drug_a_id
           JOIN drugs b ON b.drug_id = i.drug_b_id
           WHERE a.name LIKE ? OR b.name LIKE ?
           ORDER BY CASE i.severity WHEN 'Major' THEN 1 WHEN 'Moderate' THEN 2 ELSE 3 END,
                    a.name, b.name""",
        (like, like),
    ).fetchall()


def search_hpo(conn, text, limit=15):
    """Non-obsolete HPO terms whose label or a synonym contains `text`."""
    like = f"%{text}%"
    return conn.execute(
        """SELECT DISTINCT t.hp_id, t.label
           FROM hpo_terms t LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (t.label LIKE ? OR s.synonym LIKE ?)
           ORDER BY t.label LIMIT ?""",
        (like, like, limit),
    ).fetchall()


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------
def table_names(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def cmd_load(args):
    conn = connect(args.db)
    try:
        if args.only in (None, "ddi"):
            n_inter, n_drugs = load_ddi(conn, args.ddi)
            print(f"DDI: loaded {n_inter} interactions across {n_drugs} drugs  ({Path(args.ddi).name})")
        if args.only in (None, "hpo"):
            c = load_hpo(conn, args.hpo)
            print(f"HPO: loaded {c['terms']:,} terms, {c['synonyms']:,} synonyms, "
                  f"{c['edges']:,} is_a edges  ({Path(args.hpo).name})")
            if c["skipped_edges"]:
                print(f"     note: skipped {c['skipped_edges']} edges that point outside the HP terms")
        print(f"Database: {args.db}")
        return 0
    finally:
        conn.close()


def cmd_stats(args):
    conn = connect(args.db)
    try:
        tables = table_names(conn)
        print("Row counts")
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<14}{n:>9,}")

        if "interactions" in tables:
            print("\nInteractions by severity")
            for sev, n in conn.execute(
                "SELECT severity, COUNT(*) FROM interactions GROUP BY severity ORDER BY COUNT(*) DESC"
            ):
                print(f"  {sev:<10}{n:>5}")
            repeated = find_repeated_pairs(conn)
            print(f"\nData check: drug pairs listed more than once: {len(repeated)}")
            for a, b, severities in repeated:
                print(f"  {a} + {b}: {severities}")

        if "hpo_terms" in tables:
            obsolete = conn.execute("SELECT COUNT(*) FROM hpo_terms WHERE is_obsolete = 1").fetchone()[0]
            print(f"\nHPO: {obsolete} of the terms are marked obsolete")
        return 0
    finally:
        conn.close()


def cmd_drug(args):
    conn = connect(args.db)
    try:
        rows = find_interactions(conn, args.name)
        if not rows:
            print(f"No interactions found for '{args.name}'.")
            return 1
        print(f"{len(rows)} interaction(s) matching '{args.name}':")
        for severity, a, b, effect, alternative in rows:
            print(f"\n  [{severity}] {a} + {b}")
            print(f"      effect: {effect}")
            print(f"      safer alternative: {alternative}")
        return 0
    finally:
        conn.close()


def cmd_hpo(args):
    conn = connect(args.db)
    try:
        rows = search_hpo(conn, args.text, args.limit)
        if not rows:
            print(f"No HPO terms found for '{args.text}'.")
            return 1
        for hp_id, label in rows:
            print(f"{hp_id}  {label}")
        return 0
    finally:
        conn.close()


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="SQLite file (default: medications.db in the repository root)")

    parser = argparse.ArgumentParser(description="Load DDI + HPO JSON files into SQLite.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("load", parents=[common], help="read the JSON files and (re)build the database")
    p.add_argument("--ddi", type=Path, default=DEFAULT_DDI, help="path to DDI_Database.json")
    p.add_argument("--hpo", type=Path, default=DEFAULT_HPO, help="path to hp.json")
    p.add_argument("--only", choices=["ddi", "hpo"], help="load just one of the two files")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("stats", parents=[common], help="row counts and data-quality checks")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("drug", parents=[common], help="show interactions for a drug name")
    p.add_argument("name")
    p.set_defaults(func=cmd_drug)

    p = sub.add_parser("hpo", parents=[common], help="search HPO terms and synonyms")
    p.add_argument("text")
    p.add_argument("-n", "--limit", type=int, default=15)
    p.set_defaults(func=cmd_hpo)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command != "load" and not Path(args.db).exists():
        print(f"{args.db} does not exist yet. Run:  python -m ddi_backend.load_data load")
        return 1
    try:
        return args.func(args)
    except BrokenPipeError:
        # Output was piped into something like `head` that stopped reading early.
        # That is not an error; silence Python's shutdown flush and exit normally.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (OSError, ValueError) as e:  # missing file, bad JSON, unexpected structure
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
