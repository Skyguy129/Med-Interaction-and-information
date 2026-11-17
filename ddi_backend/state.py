# Module docstring: this file stores the guarded agent's changing state apart from the read-only medications.db
"""Mutable state for the guarded agent, kept apart from medications.db.

The review queue, the request log (every attempt, including blocked ones) and
the set of used request IDs live in a small SQLite file such as agent_state.db.
With no path the state is in memory, which is what the tests use. This module
never opens medications.db and refuses to be pointed at it.
"""

import json  # stores lists and dicts as text in SQLite
import sqlite3  # the small database the state lives in
import threading  # lock for safe use from several HTTP threads
from datetime import datetime, timezone  # timestamps for queue and log rows
from pathlib import Path  # path handling

from ddi_backend import load_data  # for the project root path and the name of the main database

DEFAULT_STATE_DB = load_data.ROOT / "agent_state.db"  # default state file, beside medications.db
PURPOSE = "clinician review only"  # the fixed purpose written on every queue note
PRE_SEEDED_REQUEST_ID = "already-used-004"  # the notebook's duplicate case

# SQL that creates the three tables; written as one script so it runs in a single call
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


def _now():  # current time as text
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # UTC, ISO format, whole seconds


class AgentState:  # wraps the SQLite file that holds the queue, used IDs and request log
    def __init__(self, path=None, seed_request_ids=()):  # path None means in-memory
        if path is not None and Path(path).name.casefold() == load_data.DEFAULT_DB.name.casefold():  # pointed at medications.db?
            raise ValueError("agent state must not be stored in medications.db")  # refuse, the main database stays read-only
        self.path = None if path is None else Path(path)  # keep the path (or None) for reference
        self._lock = threading.RLock()  # guards every database call
        self._conn = sqlite3.connect(":memory:" if path is None else str(path), check_same_thread=False)  # open or create the file
        with self._lock, self._conn:  # one locked transaction for setup
            self._conn.executescript(_SCHEMA)  # create the tables if they do not exist yet
            self._conn.executemany("INSERT OR IGNORE INTO seen_request_ids VALUES (?)",
                                   [(rid,) for rid in seed_request_ids])  # pre-mark any seed request IDs as used

    @classmethod  # alternate constructor for tests
    def for_tests(cls):
        """In-memory state with the notebook's already-used request ID pre-seeded."""
        return cls(None, seed_request_ids=(PRE_SEEDED_REQUEST_ID,))  # memory only, with the duplicate-case ID already used

    def close(self):  # release the database connection
        self._conn.close()

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @property  # read as an attribute, not a call
    def review_queue(self):
        """Queue notes: evidence_id, drug_pair, request_id and the purpose."""
        with self._lock:  # safe against concurrent writes
            rows = self._conn.execute(
                "SELECT evidence_id, drug_pair, request_id, purpose FROM review_queue ORDER BY queue_id").fetchall()  # oldest first
        return [{"evidence_id": e, "drug_pair": json.loads(p), "request_id": r, "purpose": u}
                for e, p, r, u in rows]  # turn each row into a note, decoding the stored JSON pair

    def queue_rows(self):  # richer listing of the queue
        """The same notes with the simulated flag and timestamp, for listings."""
        with self._lock:  # lock while reading
            rows = self._conn.execute(
                "SELECT evidence_id, drug_pair, request_id, purpose, simulated, staged_at "
                "FROM review_queue ORDER BY queue_id").fetchall()  # all columns, oldest first
        return [{"evidence_id": e, "drug_pair": json.loads(p), "request_id": r, "purpose": u,
                 "simulated": bool(s), "staged_at": t} for e, p, r, u, s, t in rows]  # add the simulated flag and time

    @property
    def seen_request_ids(self):  # every request ID that has already been used
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT request_id FROM seen_request_ids")}  # a set of IDs

    @property
    def tool_trace(self):  # the log entries where the tool really ran
        """Entries for attempts where the tool actually ran (the notebook's tool_trace)."""
        with self._lock:
            rows = self._conn.execute("SELECT entry FROM request_log ORDER BY log_id").fetchall()  # all logged attempts in order
        entries = [json.loads(r[0]) for r in rows]  # decode each stored JSON entry
        return [e["tool_trace_entry"] for e in entries if e.get("tool_trace_entry")]  # keep only attempts that reached the tool

    def snapshot(self):  # compact view of the state, used by tests and traces
        return {"review_queue": self.review_queue, "seen_request_ids": sorted(self.seen_request_ids)}  # queue plus used IDs

    # ------------------------------------------------------------------
    # Writes (only the review queue, the used-ID set and the log)
    # ------------------------------------------------------------------
    def stage_note(self, note):  # the one real write: put a note on the simulated queue
        """Add one note and mark its request ID used, in one transaction: both happen or neither."""
        with self._lock, self._conn:  # the connection context commits on success and rolls back on error
            self._conn.execute(
                "INSERT INTO review_queue (evidence_id, drug_pair, request_id, purpose, simulated, staged_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",  # simulated is always 1
                (note["evidence_id"], json.dumps(list(note["drug_pair"])), note["request_id"],
                 note["purpose"], _now()),  # values for the placeholders, with the pair stored as JSON
            )
            self._conn.execute("INSERT INTO seen_request_ids VALUES (?)", (note["request_id"],))  # mark the request ID used

    def log_attempt(self, request_id, entry):  # record every attempt, including blocked ones
        with self._lock, self._conn:  # transaction
            self._conn.execute(
                "INSERT INTO request_log (request_id, entry, logged_at) VALUES (?, ?, ?)",
                (request_id, json.dumps(entry, sort_keys=True, default=list), _now()),  # entry as stable JSON, sets become lists
            )

    def attempts(self, request_id):  # history of one request, used by the trace command
        """Every logged attempt for one request ID, oldest first."""
        with self._lock:  # lock while reading
            rows = self._conn.execute(
                "SELECT entry, logged_at FROM request_log WHERE request_id = ? ORDER BY log_id",
                (request_id,)).fetchall()  # only this request's rows
        return [dict(json.loads(entry), logged_at=at) for entry, at in rows]  # decode each entry and add its timestamp


# ---------------------------------------------------------------------------
# FILE SUMMARY: Owns the small SQLite state file (agent_state.db, or memory in tests) that holds the simulated
# pharmacist review queue, the set of request IDs already used, and a log of every attempt including blocked ones.
# Writing a note and marking its request ID used happen in one transaction, and the module refuses to be pointed at
# medications.db so the main database stays read-only.
# ASSIGNMENT TOOL: Guarded tool use with a simulated review queue (the agent). This is the agent's memory: the queue
# the one write goes into, the duplicate-ID record, and the trace log.
# ---------------------------------------------------------------------------
