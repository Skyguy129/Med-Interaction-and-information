"""Helpers for building throwaway databases in tests."""

import atexit
import json
import shutil
import tempfile
from pathlib import Path

from ddi_backend import ingest, load_data


def ddi_row(a, b, severity, **overrides):
    row = {
        "drug_a": a, "drug_b": b, "severity": severity,
        "mechanism": "mech", "effect": "effect text",
        "Safer_alternative": "alt", "rationale": "why", "reference": "ref",
    }
    row.update(overrides)
    return row


def build_db(directory, rows):
    """Write rows (a list of ddi_row dicts) into <directory>/test.db and return its path."""
    directory = Path(directory)
    ddi_path = directory / "ddi.json"
    ddi_path.write_text(json.dumps({"drug_interactions": {"all": rows}}), encoding="utf-8")
    db_path = directory / "test.db"
    conn = load_data.connect(db_path)
    try:
        load_data.load_ddi(conn, ddi_path)
    finally:
        conn.close()
    return db_path


_real = {}


def real_db_path():
    """A database built once per test run from the real DDI file (interactions only)."""
    if "path" not in _real:
        tmp = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        db_path = Path(tmp) / "real.db"
        conn = load_data.connect(db_path)
        try:
            load_data.load_ddi(conn, load_data.DEFAULT_DDI)
        finally:
            conn.close()
        _real["path"] = db_path
    return _real["path"]


def open_real():
    return ingest.open_readonly(real_db_path(), ingest.CORE_TABLES)


def real_full_db_path():
    """A database built once per test run from the real DDI and HPO files."""
    if "full" not in _real:
        tmp = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        db_path = Path(tmp) / "full.db"
        conn = load_data.connect(db_path)
        try:
            load_data.load_ddi(conn, load_data.DEFAULT_DDI)
            load_data.load_hpo(conn, load_data.DEFAULT_HPO)
        finally:
            conn.close()
        _real["full"] = db_path
    return _real["full"]


def open_real_full():
    return ingest.open_readonly(real_full_db_path())
