# Module docstring: this file wires the whole backend together once for the CLI and the API
"""Builds the whole backend once, for the command line and the HTTP API to share.

Tier one (the deterministic core and the guarded agent) needs only medications.db.
Tier two (embeddings) and tier three (a language model) switch on only when their
inputs exist, and their absence or failure never breaks tier one.
"""

import os  # reads the DDI_EMBEDDINGS environment variable
import threading  # provides the lock that makes the shared App safe across HTTP threads
from pathlib import Path  # path handling for the state file

from ddi_backend import ingest, llm, state  # database access, language model code, agent state storage
from ddi_backend.agent import GuardedAgent  # the guarded agent that stages review requests
from ddi_backend.checker import Checker  # the deterministic interaction checker
from ddi_backend.common import DISCLAIMER  # the not-medical-advice text shown in health output
from ddi_backend.retrieval import QuestionAnswerer  # answers free-text questions

EMBEDDING_MODES = ("off", "auto")  # the two accepted settings for the embedding tier


class App:  # one object holding every part of the backend
    def __init__(self, db_path=None, state_path=None, symptom_mapping=True, embeddings="off",
                 settings=None, provider="env", in_memory_state=False, check_same_thread=True):  # all options have safe defaults
        self.lock = threading.RLock()  # re-entrant lock callers use to serialise access to the shared connections
        self.conn = ingest.open_readonly(db_path, ingest.CORE_TABLES, check_same_thread=check_same_thread)  # read-only DB
        # Check whether the HPO tables are present, so symptom mapping is only offered when they are
        self.has_hpo = ingest.REQUIRED_TABLES <= set(row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"))
        self.state_path = Path(state_path) if state_path else state.DEFAULT_STATE_DB  # where the agent state file lives
        self.state = state.AgentState(None if in_memory_state else self.state_path)  # in memory for tests, file otherwise

        self.embedder = None  # stays None unless tier two loads successfully
        self.interaction_index = self.hpo_index = None  # embedding indexes, built only if there is an embedder
        suggester = None  # did-you-mean helper that uses embeddings, optional
        if embeddings == "auto":  # tier two was requested
            self._start_embeddings()  # try to load the model; failure just leaves embedder as None
            if self.embedder is not None:  # the model loaded
                from ddi_backend import embeddings as emb  # imported late so tier one never needs numpy
                suggester = emb.build_name_suggester(ingest.drug_names(self.conn), self.embedder,
                                                     None if in_memory_state else self.state_path)  # cache beside the state file
        self.checker = Checker(self.conn, suggester=suggester)  # the deterministic core, with the suggester if one exists
        if self.embedder is not None:  # build the record and symptom indexes only when embeddings are on
            from ddi_backend import embeddings as emb  # late import, same reason as above
            from ddi_backend import hpo  # HPO document source for the symptom index
            cache = None if in_memory_state else self.state_path  # where computed vectors are cached
            self.interaction_index = emb.build_interaction_index(self.checker.graph.records, self.embedder, cache)  # one record, one document
            if self.has_hpo:  # HPO keeps its own index, separate from the interaction records
                self.hpo_index = emb.build_hpo_index(hpo.documents(self.conn), self.embedder, cache)
        self.qa = QuestionAnswerer(self.checker, hpo_conn=self.conn if self.has_hpo else None,
                                   symptom_mapping=symptom_mapping, embeddings=self.interaction_index,
                                   hpo_index=self.hpo_index)  # free-text question answering using the pieces above
        self.agent = GuardedAgent(self.checker, self.state)  # guarded agent over the checker and the state file
        self.settings = settings or llm.Settings()  # detail and context settings, default if none given
        self.provider = llm.provider_from_env() if provider == "env" else provider  # language model provider (None with no key)
        self.explainer = llm.Explainer(self.provider, self.settings, ingest.drug_names(self.conn))  # explain-mode writer with a validator

    def _start_embeddings(self):  # try to load the embedding model
        try:  # any failure must be contained here
            from ddi_backend import embeddings as emb  # late import so numpy is only needed when asked for
            self.embedder = emb.load_embedder()  # load the pinned sentence-embedding model
        except Exception:  # tier two must never break tier one
            self.embedder = None  # fall back to exact names plus difflib suggestions

    def health(self):  # report which tiers are available, for /health and the CLI
        return {
            "status": "ok",  # the app is up
            "tiers": {
                "core": True,  # tier one is always on
                "embeddings": self.embedder is not None,  # tier two loaded
                "symptom_mapping": self.has_hpo and self.qa.symptom_mapping,  # HPO tables present and the feature enabled
                "language_model": self.provider is not None,  # tier three has a provider (a key was found)
            },
            "detail_level": self.settings.detail_level,  # current detail setting (brief, standard, detailed)
            "context_level": self.settings.context_level,  # current context setting (names_only or full_context)
            "disclaimer": DISCLAIMER,  # not-medical-advice text
        }

    def close(self):  # release both database connections
        self.state.close()  # close agent_state.db
        self.conn.close()  # close medications.db


def default_embedding_mode():  # the embedding mode used when the user did not pass one
    return os.environ.get("DDI_EMBEDDINGS", "off")  # off unless DDI_EMBEDDINGS says otherwise


# ---------------------------------------------------------------------------
# FILE SUMMARY: Builds the entire backend in one place (database connection, agent state, checker, question answerer,
# guarded agent, language model explainer) so the command line and the HTTP API share identical setup. It also works
# out which tiers are available and reports them, and makes sure the optional embedding and language model tiers can
# fail without breaking the deterministic core.
# ASSIGNMENT TOOL: Supporting infrastructure (not one of the assignment tools). It assembles the knowledge graph core,
# the guarded agent, the embeddings tier and the language model tier into one object.
# ---------------------------------------------------------------------------
