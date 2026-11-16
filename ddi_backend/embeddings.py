"""Embeddings in a limited role (tier two). Nothing here is needed for tier one.

Embeddings only produce candidate record IDs: candidates for free-text questions,
"did you mean" names, and symptom-word candidates for HPO. A similarity score never
decides whether an interaction exists or how severe it is, and it is never shown as
confidence, so the public methods return records or names, not scores. There is no
cut-off score anywhere: candidates are always the top few, and every candidate is still
passed through the eligibility filter by the caller.

One interaction row is one whole document, never split up. The HPO terms have their own
index. Vectors are computed once, cached on disk beside the state file, and the
cache is keyed on the model, its pinned revision and a hash of the texts, so a
change to any of them rebuilds it.

If numpy or sentence-transformers is missing, or the model cannot be downloaded or
loaded, ``load_embedder`` returns None and the system falls back to exact name
resolution plus difflib "did you mean" suggestions.
"""

import hashlib
import json
import os
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # the model the class notebooks used
# Pinned model version (a Hugging Face commit). Override with DDI_EMBEDDING_REVISION; None means latest.
MODEL_REVISION = os.environ.get("DDI_EMBEDDING_REVISION", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")


class EmbeddingUnavailable(RuntimeError):
    """Raised when numpy or the sentence-transformers model cannot be used."""


def _numpy():
    try:
        import numpy
    except ImportError as error:
        raise EmbeddingUnavailable("numpy is not installed") from error
    return numpy


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 through sentence-transformers. Loaded lazily; failures become EmbeddingUnavailable."""

    def __init__(self, model_name=MODEL_NAME, revision=MODEL_REVISION):
        self.model_name, self.revision = model_name, revision
        try:
            from sentence_transformers import SentenceTransformer
            kwargs = {"revision": revision} if revision else {}
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as error:  # missing package, no network, bad revision, corrupt cache ...
            raise EmbeddingUnavailable(f"could not load {model_name}: {error}") from error
        import sentence_transformers
        self.version = f"sentence-transformers {sentence_transformers.__version__}"

    @property
    def identity(self):
        return f"{self.model_name}@{self.revision or 'latest'} ({self.version})"

    def encode(self, texts):
        numpy = _numpy()
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return numpy.asarray(vectors, dtype="float32")


def load_embedder():
    """The real embedder, or None when tier two cannot run. Never raises."""
    try:
        _numpy()
        return SentenceTransformerEmbedder()
    except EmbeddingUnavailable:
        return None


def interaction_text(record):
    """One interaction record as one document: its fields, labelled, as the RAG notebook laid them out."""
    fields = [("Severity", record.get("severity")), ("Drug A", record.get("drug_a")),
              ("Drug B", record.get("drug_b")), ("Mechanism", record.get("mechanism")),
              ("Effect", record.get("effect")), ("Safer alternative", record.get("safer_alternative")),
              ("Rationale", record.get("rationale"))]
    return "\n".join(f"{label}: {value}" for label, value in fields if value is not None and str(value).strip())


def cache_path(state_path, kind):
    """Cache file beside the state file: agent_state.interactions.embeddings.npz."""
    state_path = Path(state_path)
    return state_path.with_name(f"{state_path.stem}.{kind}.embeddings.npz")


class EmbeddingIndex:
    """Cosine nearest neighbours over a list of (id, text) documents."""

    def __init__(self, ids, texts, embedder, payloads=None, cache_file=None, kind="index"):
        numpy = _numpy()
        # A duplicated record changes nothing: identical documents collapse to the first one.
        seen, keep = set(), []
        for position, text in enumerate(texts):
            key = " ".join(text.casefold().split())
            if key not in seen:
                seen.add(key)
                keep.append(position)
        self.ids = [ids[p] for p in keep]
        self.texts = [texts[p] for p in keep]
        self.payloads = [payloads[p] for p in keep] if payloads is not None else [None] * len(keep)
        self.embedder = embedder
        self.kind = kind
        self.cache_file = Path(cache_file) if cache_file else None
        self.from_cache = False
        self.matrix = self._load_or_build(numpy)

    @property
    def _signature(self):
        digest = hashlib.sha256("\x00".join(self.ids + self.texts).encode("utf-8")).hexdigest()
        identity = getattr(self.embedder, "identity", type(self.embedder).__name__)
        return json.dumps({"model": identity, "docs": digest, "count": len(self.ids)}, sort_keys=True)

    def _load_or_build(self, numpy):
        if self.cache_file and self.cache_file.is_file():
            try:
                with numpy.load(self.cache_file, allow_pickle=False) as data:
                    if str(data["signature"]) == self._signature:
                        self.from_cache = True
                        return data["matrix"]
            except Exception:
                pass  # unreadable or stale cache: rebuild it
        matrix = self.embedder.encode(self.texts) if self.texts else numpy.zeros((0, 1), dtype="float32")
        if self.cache_file:
            try:
                numpy.savez_compressed(self.cache_file, matrix=matrix, signature=self._signature)
            except OSError:
                pass  # caching is an optimisation only
        return matrix

    def _nearest_positions(self, query, k):
        numpy = _numpy()
        if not len(self.ids):
            return []
        vector = self.embedder.encode([query])[0]
        scores = self.matrix @ vector
        # Stable, deterministic order: score descending, then id.
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return order[:k]

    def nearest_ids(self, query, k=5):
        """Candidate IDs, most similar first. No scores are exposed."""
        return [self.ids[i] for i in self._nearest_positions(query, k)]

    def candidates(self, query, k=5):
        """Candidate payloads (e.g. interaction records) for the nearest documents."""
        return [self.payloads[i] for i in self._nearest_positions(query, k)]


def build_interaction_index(records, embedder, state_path=None):
    """Index of interaction records, cached beside the state file when one is given."""
    texts = [interaction_text(r) for r in records]
    cache = cache_path(state_path, "interactions") if state_path else None
    return EmbeddingIndex([r["id"] for r in records], texts, embedder, payloads=list(records),
                          cache_file=cache, kind="interactions")


def build_hpo_index(terms, embedder, state_path=None):
    """Separate index of HPO terms: [(hp_id, text)] where text is the label plus synonyms."""
    cache = cache_path(state_path, "hpo") if state_path else None
    return EmbeddingIndex([t[0] for t in terms], [t[1] for t in terms], embedder, cache_file=cache, kind="hpo")


def build_name_suggester(drug_names, embedder, state_path=None):
    """A suggester(text) -> [names] for the resolver: nearest drug names, top-k, never auto-chosen."""
    cache = cache_path(state_path, "drugs") if state_path else None
    index = EmbeddingIndex(list(drug_names), list(drug_names), embedder, cache_file=cache, kind="drugs")
    return lambda text: index.nearest_ids(text, k=3)
