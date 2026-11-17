#!/usr/bin/env python3
# Module docstring: explains what the loader does and shows the four commands it offers
"""Load the DDI and HPO JSON files into a local SQLite database.

Run from the repository root:

    python -m ddi_backend.load_data load              # build medications.db from both JSON files
    python -m ddi_backend.load_data stats             # row counts + data-quality report
    python -m ddi_backend.load_data drug warfarin     # interactions that involve a drug
    python -m ddi_backend.load_data hpo bleeding      # search HPO terms and synonyms

Only the Python standard library is used (sqlite3, json, argparse).
Running `load` again rebuilds the tables, so it is safe to repeat.
"""

import argparse  # parses the command-line arguments
import json  # reads the two JSON source files
import os  # used to silence output if a pipe closes early
import sqlite3  # the database engine
import sys  # stdout/stderr and exit codes
from pathlib import Path  # path handling

# Paths are relative to the repository root, so it works no matter where the
# terminal is opened. Override with --db / --ddi / --hpo.
HERE = Path(__file__).resolve().parent  # the ddi_backend folder
ROOT = HERE.parent  # the repository root
DEFAULT_DB = ROOT / "medications.db"  # where the generated database goes (not tracked in git)


def first_existing(path_candidates, fallback):  # pick the first candidate path that exists
    """Return the first existing path, or the fallback if none exist."""
    for path in path_candidates:  # try each candidate in order
        if path.exists():  # found one
            return path  # use it
    return fallback  # none exist, so use the fallback


DATA_DIR = ROOT / "data"  # folder holding the source data files
DEFAULT_DDI = first_existing(  # find the interaction file, accepting either spelling of its name
    [DATA_DIR / "DDI_Database.json", DATA_DIR / "DDI Database.json",
     ROOT / "DDI_Database.json", ROOT / "DDI Database.json"],  # candidates in data/ then the repo root
    DATA_DIR / "DDI_Database.json",  # fallback when none exists
)
DEFAULT_HPO = first_existing(  # find the HPO ontology file
    [DATA_DIR / "hp.json", DATA_DIR / "HP.json", ROOT / "hp.json", ROOT / "HP.json"],  # candidates
    DATA_DIR / "hp.json",  # fallback when none exists
)

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
# Drugs are nodes; interactions are edges between two drugs (see the dossier).
# There is deliberately NO unique constraint on a drug pair: the source data
# lists some pairs more than once with different severities, and we want to
# keep every row and report the conflicts rather than silently drop one.
# SQL below: drops old tables, then creates drugs (unique names, case-insensitive), interactions
# (two foreign keys to drugs, severity limited to three values) and an index on each drug column
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
# SQL below: drops old HPO tables and creates the terms, synonyms and parent/child edge tables with indexes
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

SYNONYM_SCOPE = {  # translates OBO synonym predicates into short scope words
    "hasExactSynonym": "exact",  # same meaning
    "hasRelatedSynonym": "related",  # loosely related
    "hasBroadSynonym": "broad",  # wider meaning
    "hasNarrowSynonym": "narrow",  # narrower meaning
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def connect(db_path):  # open a read-write connection (used only by the loader and its commands)
    conn = sqlite3.connect(str(db_path))  # open or create the file
    conn.execute("PRAGMA foreign_keys = ON")  # make SQLite enforce the foreign keys
    return conn  # ready to use


def read_json(path):  # shared JSON reader used across the package
    with open(path, encoding="utf-8") as f:  # open as UTF-8 text
        return json.load(f)  # parse and return the data


def hp_id_from_iri(iri):  # convert an ontology web address to a short HP id
    """'http://purl.obolibrary.org/obo/HP_0000002' -> 'HP:0000002'"""
    return iri.rsplit("/", 1)[-1].replace("_", ":", 1)  # keep the last path part and swap the first underscore for a colon


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_ddi(conn, path):  # load the drug-interaction JSON into the drugs and interactions tables
    """Load DDI_Database.json. Returns (interaction_count, drug_count)."""
    path = Path(path)  # make sure it is a Path
    data = read_json(path)  # parse the file
    # Expected shape: {"drug_interactions": {"major": [...], "moderate": [...], "minor": [...]}}
    buckets = data.get("drug_interactions") if isinstance(data, dict) else None  # the severity groups
    if not isinstance(buckets, dict):  # wrong file shape
        raise ValueError(f"{path.name}: expected a top-level 'drug_interactions' object")

    conn.executescript(DDI_SCHEMA)  # drop and recreate the two tables
    drug_ids = {}  # lower-cased name -> drug_id, so 'Omeprazole' and 'omeprazole' are one drug

    def get_drug_id(name):  # return the id for a drug, inserting it the first time it is seen
        name = name.strip()  # trim stray spaces
        key = name.lower()  # case-insensitive key
        if key not in drug_ids:  # first time we see this drug
            cur = conn.execute("INSERT INTO drugs (name) VALUES (?)", (name,))  # add it
            drug_ids[key] = cur.lastrowid  # remember its new id
        return drug_ids[key]  # the id

    count = 0  # interactions loaded so far
    try:  # a missing field is turned into a clear error below
        with conn:  # one transaction: all rows are saved, or none
            for rows in buckets.values():  # each severity group
                for r in rows:  # each interaction row
                    # SQL below inserts one interaction row with the eight stored columns
                    conn.execute(
                        """INSERT INTO interactions
                           (drug_a_id, drug_b_id, severity, mechanism, effect,
                            safer_alternative, rationale, reference)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            get_drug_id(r["drug_a"]),  # first drug id (created if new)
                            get_drug_id(r["drug_b"]),  # second drug id (created if new)
                            r["severity"].strip().capitalize(),  # normalise to Major / Moderate / Minor
                            r.get("mechanism"),  # how it happens
                            r.get("effect"),  # what it does
                            r.get("Safer_alternative"),  # note the capital S in the source file
                            r.get("rationale"),  # why
                            r.get("reference"),  # citation
                        ),
                    )
                    count += 1  # tally this row
    except KeyError as e:  # a required field (drug_a, drug_b, severity) was absent
        raise ValueError(f"{path.name}: an interaction row is missing the field {e}") from None
    return count, len(drug_ids)  # row count and number of distinct drugs


def load_hpo(conn, path):  # load the HPO ontology into the three hpo_ tables
    """Load hp.json (OBO-graphs format). Returns a dict of row counts."""
    path = Path(path)  # make sure it is a Path
    data = read_json(path)  # parse the (large) file
    try:  # check the OBO-graphs structure
        graph = data["graphs"][0]  # the single graph in the file
        nodes, edges = graph["nodes"], graph["edges"]  # its terms and links
    except (KeyError, IndexError, TypeError):  # structure is not as expected
        raise ValueError(f"{path.name}: expected an OBO-graphs file with graphs[0].nodes and .edges")

    terms, synonyms = [], []  # rows to insert
    for node in nodes:  # each ontology node
        # Keep HP terms only; this skips relation nodes such as 'has part'.
        if node.get("type") != "CLASS" or "/HP_" not in node["id"]:  # not a phenotype term
            continue  # skip it
        hp_id = hp_id_from_iri(node["id"])  # short id such as HP:0000002
        meta = node.get("meta", {})  # extra info (definition, synonyms, ...)
        terms.append((  # one hpo_terms row
            hp_id,  # id
            node.get("lbl", ""),  # label (name)
            (meta.get("definition") or {}).get("val"),  # definition text, or None
            "\n".join(meta.get("comments", [])) or None,  # comments joined, or None
            1 if meta.get("deprecated") else 0,  # obsolete flag
        ))
        for syn in meta.get("synonyms", []):  # each synonym of the term
            synonyms.append((  # one hpo_synonyms row
                hp_id,  # which term
                syn["val"],  # synonym text
                SYNONYM_SCOPE.get(syn.get("pred"), syn.get("pred")),  # scope word
                (syn.get("synonymType") or "").rsplit("#", 1)[-1] or None,  # type such as layperson
            ))

    term_ids = {t[0] for t in terms}  # the ids that exist
    edge_rows, skipped_edges = [], 0  # edges to insert and a count of those we drop
    for e in edges:  # each link in the ontology
        if e.get("pred") != "is_a":  # only parent/child links
            continue  # skip other relations
        child, parent = hp_id_from_iri(e["sub"]), hp_id_from_iri(e["obj"])  # convert both ends to short ids
        if child in term_ids and parent in term_ids:  # both ends are real HP terms
            edge_rows.append((child, parent))  # keep it
        else:  # points outside the HP terms
            skipped_edges += 1  # count it as skipped

    conn.executescript(HPO_SCHEMA)  # drop and recreate the three tables
    with conn:  # one transaction for all inserts
        conn.executemany("INSERT INTO hpo_terms VALUES (?, ?, ?, ?, ?)", terms)  # insert the terms
        conn.executemany(  # insert the synonyms
            "INSERT INTO hpo_synonyms (hp_id, synonym, scope, synonym_type) VALUES (?, ?, ?, ?)",
            synonyms,  # the collected rows
        )
        conn.executemany("INSERT OR IGNORE INTO hpo_edges VALUES (?, ?)", edge_rows)  # insert edges, ignoring duplicates

    n_edges = conn.execute("SELECT COUNT(*) FROM hpo_edges").fetchone()[0]  # edges actually stored
    return {"terms": len(terms), "synonyms": len(synonyms), "edges": n_edges, "skipped_edges": skipped_edges}  # summary counts


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def find_repeated_pairs(conn):  # data-quality check for pairs listed more than once
    """Drug pairs that appear in more than one interaction row (either order).

    Returns [(drug_1, drug_2, 'Major / Moderate'), ...].
    """
    # SQL below: group rows by the unordered pair (lowest id, highest id), keep groups with more than one row,
    # then join back to drug names and list the severities found
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
    ).fetchall()  # return every repeated pair


def find_interactions(conn, name):  # simple substring search used by the loader's own "drug" command
    """Interactions where either drug name contains `name` (case-insensitive)."""
    like = f"%{name}%"  # SQL wildcard pattern around the text
    # SQL below: rows where either drug matches, ordered Major, Moderate, Minor, then by names
    return conn.execute(
        """SELECT i.severity, a.name, b.name, i.effect, i.safer_alternative
           FROM interactions i
           JOIN drugs a ON a.drug_id = i.drug_a_id
           JOIN drugs b ON b.drug_id = i.drug_b_id
           WHERE a.name LIKE ? OR b.name LIKE ?
           ORDER BY CASE i.severity WHEN 'Major' THEN 1 WHEN 'Moderate' THEN 2 ELSE 3 END,
                    a.name, b.name""",
        (like, like),  # the pattern fills both placeholders
    ).fetchall()  # all matching rows


def search_hpo(conn, text, limit=15):  # search HPO terms by label or synonym
    """Non-obsolete HPO terms whose label or a synonym contains `text`."""
    like = f"%{text}%"  # SQL wildcard pattern
    # SQL below: distinct terms whose label or synonym matches, skipping obsolete ones, capped by the limit
    return conn.execute(
        """SELECT DISTINCT t.hp_id, t.label
           FROM hpo_terms t LEFT JOIN hpo_synonyms s ON s.hp_id = t.hp_id
           WHERE t.is_obsolete = 0 AND (t.label LIKE ? OR s.synonym LIKE ?)
           ORDER BY t.label LIMIT ?""",
        (like, like, limit),  # pattern twice, then the row limit
    ).fetchall()  # matching terms


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------
def table_names(conn):  # list the user tables in a database
    return [r[0] for r in conn.execute(  # take the name column of each row
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]  # skip internal tables


def cmd_load(args):  # the "load" command: build the database
    conn = connect(args.db)  # open (or create) the database file
    try:  # always close the connection
        if args.only in (None, "ddi"):  # load interactions unless --only hpo
            n_inter, n_drugs = load_ddi(conn, args.ddi)  # do the load
            print(f"DDI: loaded {n_inter} interactions across {n_drugs} drugs  ({Path(args.ddi).name})")  # report counts
        if args.only in (None, "hpo"):  # load the ontology unless --only ddi
            c = load_hpo(conn, args.hpo)  # do the load
            print(f"HPO: loaded {c['terms']:,} terms, {c['synonyms']:,} synonyms, "
                  f"{c['edges']:,} is_a edges  ({Path(args.hpo).name})")  # report counts
            if c["skipped_edges"]:  # some edges were dropped
                print(f"     note: skipped {c['skipped_edges']} edges that point outside the HP terms")
        print(f"Database: {args.db}")  # tell the user where the file is
        return 0  # success
    finally:
        conn.close()  # close the connection


def cmd_stats(args):  # the "stats" command: counts and data-quality report
    conn = connect(args.db)  # open the database
    try:  # always close the connection
        tables = table_names(conn)  # which tables exist
        print("Row counts")  # heading
        for t in tables:  # each table
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # count its rows
            print(f"  {t:<14}{n:>9,}")  # print aligned

        if "interactions" in tables:  # only when interactions were loaded
            print("\nInteractions by severity")  # heading
            for sev, n in conn.execute(  # count per severity
                "SELECT severity, COUNT(*) FROM interactions GROUP BY severity ORDER BY COUNT(*) DESC"
            ):
                print(f"  {sev:<10}{n:>5}")  # print aligned
            repeated = find_repeated_pairs(conn)  # pairs listed more than once
            print(f"\nData check: drug pairs listed more than once: {len(repeated)}")  # how many
            for a, b, severities in repeated:  # list each
                print(f"  {a} + {b}: {severities}")  # pair and its severities

        if "hpo_terms" in tables:  # only when HPO was loaded
            obsolete = conn.execute("SELECT COUNT(*) FROM hpo_terms WHERE is_obsolete = 1").fetchone()[0]  # count deprecated terms
            print(f"\nHPO: {obsolete} of the terms are marked obsolete")  # report it
        return 0  # success
    finally:
        conn.close()  # close the connection


def cmd_drug(args):  # the "drug" command: list interactions for a name
    conn = connect(args.db)  # open the database
    try:  # always close the connection
        rows = find_interactions(conn, args.name)  # substring search
        if not rows:  # nothing found
            print(f"No interactions found for '{args.name}'.")  # say so (not that it is safe)
            return 1  # failure code
        print(f"{len(rows)} interaction(s) matching '{args.name}':")  # how many
        for severity, a, b, effect, alternative in rows:  # each row
            print(f"\n  [{severity}] {a} + {b}")  # severity and pair
            print(f"      effect: {effect}")  # effect text
            print(f"      safer alternative: {alternative}")  # alternative text
        return 0  # success
    finally:
        conn.close()  # close the connection


def cmd_hpo(args):  # the "hpo" command: search HPO terms
    conn = connect(args.db)  # open the database
    try:  # always close the connection
        rows = search_hpo(conn, args.text, args.limit)  # run the search
        if not rows:  # nothing found
            print(f"No HPO terms found for '{args.text}'.")  # say so
            return 1  # failure code
        for hp_id, label in rows:  # each match
            print(f"{hp_id}  {label}")  # id and label
        return 0  # success
    finally:
        conn.close()  # close the connection


def build_parser():  # define the command-line options
    common = argparse.ArgumentParser(add_help=False)  # shared options, not a command on its own
    common.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="SQLite file (default: medications.db in the repository root)")  # database path option

    parser = argparse.ArgumentParser(description="Load DDI + HPO JSON files into SQLite.")  # top-level parser
    sub = parser.add_subparsers(dest="command", required=True)  # a command word is required

    p = sub.add_parser("load", parents=[common], help="read the JSON files and (re)build the database")  # "load" command
    p.add_argument("--ddi", type=Path, default=DEFAULT_DDI, help="path to DDI_Database.json")  # interaction file option
    p.add_argument("--hpo", type=Path, default=DEFAULT_HPO, help="path to hp.json")  # ontology file option
    p.add_argument("--only", choices=["ddi", "hpo"], help="load just one of the two files")  # partial load option
    p.set_defaults(func=cmd_load)  # run cmd_load

    p = sub.add_parser("stats", parents=[common], help="row counts and data-quality checks")  # "stats" command
    p.set_defaults(func=cmd_stats)  # run cmd_stats

    p = sub.add_parser("drug", parents=[common], help="show interactions for a drug name")  # "drug" command
    p.add_argument("name")  # the drug name to search
    p.set_defaults(func=cmd_drug)  # run cmd_drug

    p = sub.add_parser("hpo", parents=[common], help="search HPO terms and synonyms")  # "hpo" command
    p.add_argument("text")  # the search text
    p.add_argument("-n", "--limit", type=int, default=15)  # how many results to show
    p.set_defaults(func=cmd_hpo)  # run cmd_hpo
    return parser  # the finished parser


def main(argv=None):  # entry point for this module's own commands
    args = build_parser().parse_args(argv)  # parse the arguments
    if args.command != "load" and not Path(args.db).exists():  # other commands need an existing database
        print(f"{args.db} does not exist yet. Run:  python -m ddi_backend.load_data load")  # tell the user how to build it
        return 1  # failure code
    try:  # run the chosen command and handle errors
        return args.func(args)  # call cmd_load / cmd_stats / cmd_drug / cmd_hpo
    except BrokenPipeError:  # output was cut off by the reader
        # Output was piped into something like `head` that stopped reading early.
        # That is not an error; silence Python's shutdown flush and exit normally.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # point stdout at the null device
        return 0  # treat as success
    except (OSError, ValueError) as e:  # missing file, bad JSON, unexpected structure
        print(f"Error: {e}", file=sys.stderr)  # report on stderr
        return 1  # failure code


if __name__ == "__main__":  # only when run directly as a script or module
    sys.exit(main())  # run and exit with the returned code

# ---------------------------------------------------------------------------
# FILE SUMMARY: The existing loader that builds medications.db from DDI_Database.json and hp.json in one safe,
# repeatable transaction. It defines the schema (drugs as nodes, interactions as edges with no unique pair constraint so
# repeated pairs are kept), the HPO tables, shared helpers (connect, read_json, table_names) and four commands: load,
# stats, drug and hpo.
# ASSIGNMENT TOOL: knowledge graph with rule-based reasoning (it creates the data the graph is built from); the HPO
# tables it builds also feed the symptom mapping used by retrieval.
# ---------------------------------------------------------------------------
