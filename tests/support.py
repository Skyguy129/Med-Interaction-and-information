# Docstring describing the module: shared helpers that build throwaway databases for tests.
"""Helpers for building throwaway databases in tests."""

import atexit  # lets us register cleanup of temp folders when the test run ends
import json  # used to write the fake DDI json file
import shutil  # used to delete temp directories
import tempfile  # makes temporary directories for the generated databases
from pathlib import Path  # path objects for building file locations

from ddi_backend import ingest, load_data  # the read-only opener and the real loader we build databases with


# Builds one fake interaction row (the shape of an entry in DDI_Database.json).
def ddi_row(a, b, severity, **overrides):
    # Start from a complete row with filler text so tests only specify what they care about.
    row = {
        "drug_a": a, "drug_b": b, "severity": severity,  # the two drugs and how severe the interaction is
        "mechanism": "mech", "effect": "effect text",  # filler mechanism and clinical effect
        "Safer_alternative": "alt", "rationale": "why", "reference": "ref",  # filler alternative, reason, source
    }
    row.update(overrides)  # let the caller replace any field (e.g. mechanism=...)
    return row  # hand the finished row back


# Builds a small database file from a list of rows.
def build_db(directory, rows):
    # Docstring below explains the contract; it is a string so notes sit above it.
    """Write rows (a list of ddi_row dicts) into <directory>/test.db and return its path."""
    directory = Path(directory)  # accept str or Path
    ddi_path = directory / "ddi.json"  # where the fake DDI json goes
    # Write the rows in the same nested layout the real file uses.
    ddi_path.write_text(json.dumps({"drug_interactions": {"all": rows}}), encoding="utf-8")
    db_path = directory / "test.db"  # where the sqlite database will live
    conn = load_data.connect(db_path)  # create/open the database with the real schema
    try:
        load_data.load_ddi(conn, ddi_path)  # load the fake rows through the real loader
    finally:
        conn.close()  # always close the connection, even if loading failed
    return db_path  # return the location so the test can open it


# Cache so the real database is built only once per test run.
_real = {}


# Returns the path of a database built from the real DDI file.
def real_db_path():
    # The docstring below says this is built once per run and holds interactions only.
    """A database built once per test run from the real DDI file (interactions only)."""
    if "path" not in _real:  # only build if we have not already
        tmp = tempfile.mkdtemp()  # fresh temp folder
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # delete it when Python exits
        db_path = Path(tmp) / "real.db"  # database file name inside the temp folder
        conn = load_data.connect(db_path)  # create the empty database
        try:
            load_data.load_ddi(conn, load_data.DEFAULT_DDI)  # load the real interaction data
        finally:
            conn.close()  # close the writer connection
        _real["path"] = db_path  # remember it for later calls
    return _real["path"]  # return the cached path


# Opens the real interaction database read-only, the way the application does.
def open_real():
    return ingest.open_readonly(real_db_path(), ingest.CORE_TABLES)  # read-only; requires only the core tables


# Returns the path of a database that also has the HPO phenotype data.
def real_full_db_path():
    # The docstring below says this one has both the DDI and HPO data.
    """A database built once per test run from the real DDI and HPO files."""
    if "full" not in _real:  # build only once
        tmp = tempfile.mkdtemp()  # fresh temp folder
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # clean up at exit
        db_path = Path(tmp) / "full.db"  # database file name
        conn = load_data.connect(db_path)  # create the empty database
        try:
            load_data.load_ddi(conn, load_data.DEFAULT_DDI)  # load interactions
            load_data.load_hpo(conn, load_data.DEFAULT_HPO)  # load the symptom vocabulary
        finally:
            conn.close()  # close the writer connection
        _real["full"] = db_path  # cache the path
    return _real["full"]  # return the cached path


# Opens the full (DDI + HPO) database read-only.
def open_real_full():
    return ingest.open_readonly(real_full_db_path())  # read-only; default table requirements


# ---------------------------------------------------------------------------
# FILE SUMMARY: Shared test fixtures. It builds small fake interaction databases (ddi_row, build_db) and a real
# database built once per run from the shipped data (real_db_path, open_real, real_full_db_path, open_real_full),
# so every other test file can use the actual loader and the read-only opener without touching medications.db.
# ASSIGNMENT TOOL: supporting infrastructure (not one of the assignment tools); it supports tests of the knowledge
# graph and rule-based core by feeding it the same loader and read-only database access the application uses.
# ---------------------------------------------------------------------------
