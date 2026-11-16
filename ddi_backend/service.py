"""Builds the whole backend once, for the command line and the HTTP API to share.

Tier one (the deterministic core and the guarded agent) needs only medications.db.
Tier two (embeddings) and tier three (a language model) switch on only when their
inputs exist, and their absence or failure never breaks tier one.
"""

import os
import threading
from pathlib import Path

from ddi_backend import ingest, llm, state
from ddi_backend.agent import GuardedAgent
from ddi_backend.checker import Checker
from ddi_backend.common import DISCLAIMER
from ddi_backend.retrieval import QuestionAnswerer

EMBEDDING_MODES = ("off", "auto")


class App:
    def __init__(self, db_path=None, state_path=None, symptom_mapping=True, embeddings="off",
                 settings=None, provider="env", in_memory_state=False, check_same_thread=True):
        self.lock = threading.RLock()
        self.conn = ingest.open_readonly(db_path, ingest.CORE_TABLES, check_same_thread=check_same_thread)
        self.has_hpo = ingest.REQUIRED_TABLES <= set(row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"))
        self.state_path = Path(state_path) if state_path else state.DEFAULT_STATE_DB
        self.state = state.AgentState(None if in_memory_state else self.state_path)

        self.embedder = None
        self.interaction_index = self.hpo_index = None
        suggester = None
        if embeddings == "auto":
            self._start_embeddings()
            if self.embedder is not None:
                from ddi_backend import embeddings as emb
                suggester = emb.build_name_suggester(ingest.drug_names(self.conn), self.embedder,
                                                     None if in_memory_state else self.state_path)
        self.checker = Checker(self.conn, suggester=suggester)
        if self.embedder is not None:
            from ddi_backend import embeddings as emb
            from ddi_backend import hpo
            cache = None if in_memory_state else self.state_path
            self.interaction_index = emb.build_interaction_index(self.checker.graph.records, self.embedder, cache)
            if self.has_hpo:  # HPO keeps its own index, separate from the interaction records
                self.hpo_index = emb.build_hpo_index(hpo.documents(self.conn), self.embedder, cache)
        self.qa = QuestionAnswerer(self.checker, hpo_conn=self.conn if self.has_hpo else None,
                                   symptom_mapping=symptom_mapping, embeddings=self.interaction_index,
                                   hpo_index=self.hpo_index)
        self.agent = GuardedAgent(self.checker, self.state)
        self.settings = settings or llm.Settings()
        self.provider = llm.provider_from_env() if provider == "env" else provider
        self.explainer = llm.Explainer(self.provider, self.settings, ingest.drug_names(self.conn))

    def _start_embeddings(self):
        try:
            from ddi_backend import embeddings as emb
            self.embedder = emb.load_embedder()
        except Exception:  # tier two must never break tier one
            self.embedder = None

    def health(self):
        return {
            "status": "ok",
            "tiers": {
                "core": True,
                "embeddings": self.embedder is not None,
                "symptom_mapping": self.has_hpo and self.qa.symptom_mapping,
                "language_model": self.provider is not None,
            },
            "detail_level": self.settings.detail_level,
            "context_level": self.settings.context_level,
            "disclaimer": DISCLAIMER,
        }

    def close(self):
        self.state.close()
        self.conn.close()


def default_embedding_mode():
    return os.environ.get("DDI_EMBEDDINGS", "off")
