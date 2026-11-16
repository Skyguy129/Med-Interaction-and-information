"""Mutable state for the guarded agent, kept apart from medications.db.

The review queue, the request log (every attempt, including blocked ones) and
the set of used request IDs live in a small SQLite file such as agent_state.db.
With no path the state is in memory, which is what the tests use. This module
never opens medications.db and refuses to be pointed at it.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ddi_backend import load_data

DEFAULT_STATE_DB = load_data.ROOT / "agent_state.db"
PURPOSE = "clinician review only"
PRE_SEEDED_REQUEST_ID = "already-used-004"  # the notebook's duplicate case

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    queue_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL,
    drug_pair   TEXT NOT NULL,                 -- JSON list of two names
    request_id  TEXT NOT NULL UNIQUE,
    purpose     TEXT NOT NULL,
    simulated   INTEGER NOT NULL DEFAULT 1,
    staged_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seen_request_ids (request_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS request_log (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    entry      TEXT NOT NULL,                  -- JSON: the full attempt, gate, tool result, state change
    logged_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_log_request ON request_log(request_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentState:
    def __init__(self, path=None, seed_request_ids=()):
        if path is not None and Path(path).name.casefold() == load_data.DEFAULT_DB.name.casefold():
            raise ValueError("agent state must not be stored in medications.db")
        self.path = None if path is None else Path(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(":memory:" if path is None else str(path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.executemany("INSERT OR IGNORE INTO seen_request_ids VALUES (?)",
                                   [(rid,) for rid in seed_request_ids])

    @classmethod
    def for_tests(cls):
        """In-memory state with the notebook's already-used request ID pre-seeded."""
        return cls(None, seed_request_ids=(PRE_SEEDED_REQUEST_ID,))

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @property
    def review_queue(self):
        """Queue notes: evidence_id, drug_pair, request_id and the purpose."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT evidence_id, drug_pair, request_id, purpose FROM review_queue ORDER BY queue_id").fetchall()
        return [{"evidence_id": e, "drug_pair": json.loads(p), "request_id": r, "purpose": u}
                for e, p, r, u in rows]

    def queue_rows(self):
        """The same notes with the simulated flag and timestamp, for listings."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT evidence_id, drug_pair, request_id, purpose, simulated, staged_at "
                "FROM review_queue ORDER BY queue_id").fetchall()
        return [{"evidence_id": e, "drug_pair": json.loads(p), "request_id": r, "purpose": u,
                 "simulated": bool(s), "staged_at": t} for e, p, r, u, s, t in rows]

    @property
    def seen_request_ids(self):
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT request_id FROM seen_request_ids")}

    @property
    def tool_trace(self):
        """Entries for attempts where the tool actually ran (the notebook's tool_trace)."""
        with self._lock:
            rows = self._conn.execute("SELECT entry FROM request_log ORDER BY log_id").fetchall()
        entries = [json.loads(r[0]) for r in rows]
        return [e["tool_trace_entry"] for e in entries if e.get("tool_trace_entry")]

    def snapshot(self):
        return {"review_queue": self.review_queue, "seen_request_ids": sorted(self.seen_request_ids)}

    # ------------------------------------------------------------------
    # Writes (only the review queue, the used-ID set and the log)
    # ------------------------------------------------------------------
    def stage_note(self, note):
        """Add one note and mark its request ID used, in one transaction: both happen or neither."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO review_queue (evidence_id, drug_pair, request_id, purpose, simulated, staged_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (note["evidence_id"], json.dumps(list(note["drug_pair"])), note["request_id"],
                 note["purpose"], _now()),
            )
            self._conn.execute("INSERT INTO seen_request_ids VALUES (?)", (note["request_id"],))

    def log_attempt(self, request_id, entry):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO request_log (request_id, entry, logged_at) VALUES (?, ?, ?)",
                (request_id, json.dumps(entry, sort_keys=True, default=list), _now()),
            )

    def attempts(self, request_id):
        """Every logged attempt for one request ID, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry, logged_at FROM request_log WHERE request_id = ? ORDER BY log_id",
                (request_id,)).fetchall()
        return [dict(json.loads(entry), logged_at=at) for entry, at in rows]
