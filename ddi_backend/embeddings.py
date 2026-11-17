# Module docstring: says embeddings are tier two and only ever propose candidates, never decide facts.
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

import hashlib  # SHA-256 fingerprint of the documents, used to detect a stale cache
import json  # builds the cache signature string
import os  # reads the optional DDI_EMBEDDING_REVISION environment variable
from pathlib import Path  # builds cache file paths

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # the model the class notebooks used
# Pinned model version (a Hugging Face commit). Override with DDI_EMBEDDING_REVISION; None means latest.
MODEL_REVISION = os.environ.get("DDI_EMBEDDING_REVISION", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")  # default is the pinned commit


class EmbeddingUnavailable(RuntimeError):  # error type meaning "tier two cannot run"
    """Raised when numpy or the sentence-transformers model cannot be used."""


def _numpy():  # import numpy only when needed so tier one never requires it
    try:  # numpy may be missing
        import numpy  # imported lazily on purpose
    except ImportError as error:  # not installed
        raise EmbeddingUnavailable("numpy is not installed") from error  # turn it into our own error
    return numpy  # hand the module to the caller


class SentenceTransformerEmbedder:  # the real embedding model wrapper
    """all-MiniLM-L6-v2 through sentence-transformers. Loaded lazily; failures become EmbeddingUnavailable."""

    def __init__(self, model_name=MODEL_NAME, revision=MODEL_REVISION):  # load the model at creation
        self.model_name, self.revision = model_name, revision  # remember which model and version
        try:  # loading can fail in many ways
            from sentence_transformers import SentenceTransformer  # imported lazily so tier one stays light
            kwargs = {"revision": revision} if revision else {}  # pin the revision when one is set
            self._model = SentenceTransformer(model_name, **kwargs)  # download or load the model
        except Exception as error:  # missing package, no network, bad revision, corrupt cache ...
            raise EmbeddingUnavailable(f"could not load {model_name}: {error}") from error  # report as unavailable
        import sentence_transformers  # needed only for its version number
        self.version = f"sentence-transformers {sentence_transformers.__version__}"  # recorded in the cache identity

    @property
    def identity(self):  # a string that names exactly which model produced the vectors
        return f"{self.model_name}@{self.revision or 'latest'} ({self.version})"  # model, revision and library version

    def encode(self, texts):  # turn texts into vectors
        numpy = _numpy()  # get numpy (or raise EmbeddingUnavailable)
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)  # unit-length vectors so dot product = cosine
        return numpy.asarray(vectors, dtype="float32")  # compact numeric array


def load_embedder():  # the single place that decides whether tier two is on
    """The real embedder, or None when tier two cannot run. Never raises."""
    try:  # attempt to start tier two
        _numpy()  # check numpy first because it is cheap
        return SentenceTransformerEmbedder()  # then load the model
    except EmbeddingUnavailable:  # anything missing
        return None  # callers fall back to exact names and difflib suggestions


def interaction_text(record):  # build the document text for one interaction row
    """One interaction record as one document: its fields, labelled, as the RAG notebook laid them out."""
    fields = [("Severity", record.get("severity")), ("Drug A", record.get("drug_a")),  # labelled fields, in a fixed order
              ("Drug B", record.get("drug_b")), ("Mechanism", record.get("mechanism")),  # second drug and mechanism
              ("Effect", record.get("effect")), ("Safer alternative", record.get("safer_alternative")),  # effect and alternative
              ("Rationale", record.get("rationale"))]  # reason for the alternative
    return "\n".join(f"{label}: {value}" for label, value in fields if value is not None and str(value).strip())  # one line per non-empty field


def cache_path(state_path, kind):  # where a cached vector file lives
    """Cache file beside the state file: agent_state.interactions.embeddings.npz."""
    state_path = Path(state_path)  # accept strings or Paths
    return state_path.with_name(f"{state_path.stem}.{kind}.embeddings.npz")  # same folder, name includes the index kind


class EmbeddingIndex:  # a searchable set of vectors
    """Cosine nearest neighbours over a list of (id, text) documents."""

    def __init__(self, ids, texts, embedder, payloads=None, cache_file=None, kind="index"):  # build (or load) the index
        numpy = _numpy()  # numpy is required here
        # A duplicated record changes nothing: identical documents collapse to the first one.
        seen, keep = set(), []  # normalised texts seen so far, and positions to keep
        for position, text in enumerate(texts):  # go through every document
            key = " ".join(text.casefold().split())  # normalise case and spacing
            if key not in seen:  # first time this text appears
                seen.add(key)  # remember it
                keep.append(position)  # keep this position
        self.ids = [ids[p] for p in keep]  # IDs of the kept documents
        self.texts = [texts[p] for p in keep]  # texts of the kept documents
        self.payloads = [payloads[p] for p in keep] if payloads is not None else [None] * len(keep)  # the original records, if given
        self.embedder = embedder  # object that turns text into vectors
        self.kind = kind  # which index this is (interactions, hpo, drugs)
        self.cache_file = Path(cache_file) if cache_file else None  # optional on-disk cache
        self.from_cache = False  # set to True when the cache was reused
        self.matrix = self._load_or_build(numpy)  # the vectors, from cache or freshly computed

    @property
    def _signature(self):  # fingerprint of everything that determines the vectors
        digest = hashlib.sha256("\x00".join(self.ids + self.texts).encode("utf-8")).hexdigest()  # hash of all IDs and texts
        identity = getattr(self.embedder, "identity", type(self.embedder).__name__)  # model identity, or class name for a fake embedder
        return json.dumps({"model": identity, "docs": digest, "count": len(self.ids)}, sort_keys=True)  # stable string: model + documents + count

    def _load_or_build(self, numpy):  # reuse the cache when it still matches, else recompute
        if self.cache_file and self.cache_file.is_file():  # a cache file exists
            try:  # reading it might fail
                with numpy.load(self.cache_file, allow_pickle=False) as data:  # open without allowing pickled code
                    if str(data["signature"]) == self._signature:  # same model, same documents
                        self.from_cache = True  # record that we used it
                        return data["matrix"]  # reuse the stored vectors
            except Exception:  # corrupt or unreadable file
                pass  # unreadable or stale cache: rebuild it
        matrix = self.embedder.encode(self.texts) if self.texts else numpy.zeros((0, 1), dtype="float32")  # compute vectors (empty array for no texts)
        if self.cache_file:  # a cache location was given
            try:  # writing might fail
                numpy.savez_compressed(self.cache_file, matrix=matrix, signature=self._signature)  # save vectors with their signature
            except OSError:  # read-only folder and similar
                pass  # caching is an optimisation only
        return matrix  # the vectors

    def _nearest_positions(self, query, k):  # positions of the k most similar documents
        numpy = _numpy()  # numpy available check (raises if not)
        if not len(self.ids):  # nothing indexed
            return []  # no candidates
        vector = self.embedder.encode([query])[0]  # vector for the question
        scores = self.matrix @ vector  # cosine similarity against every document
        # Stable, deterministic order: score descending, then id.
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))  # ties broken by ID so results repeat
        return order[:k]  # only the top k positions; scores are not returned

    def nearest_ids(self, query, k=5):  # public: candidate IDs only
        """Candidate IDs, most similar first. No scores are exposed."""
        return [self.ids[i] for i in self._nearest_positions(query, k)]  # map positions to IDs

    def candidates(self, query, k=5):  # public: candidate records
        """Candidate payloads (e.g. interaction records) for the nearest documents."""
        return [self.payloads[i] for i in self._nearest_positions(query, k)]  # map positions to stored records


def build_interaction_index(records, embedder, state_path=None):  # index over interaction records
    """Index of interaction records, cached beside the state file when one is given."""
    texts = [interaction_text(r) for r in records]  # one whole document per record
    cache = cache_path(state_path, "interactions") if state_path else None  # cache file, if a state path is known
    return EmbeddingIndex([r["id"] for r in records], texts, embedder, payloads=list(records),  # IDs, texts and the records themselves
                          cache_file=cache, kind="interactions")  # cache location and index kind


def build_hpo_index(terms, embedder, state_path=None):  # separate index over HPO terms
    """Separate index of HPO terms: [(hp_id, text)] where text is the label plus synonyms."""
    cache = cache_path(state_path, "hpo") if state_path else None  # cache file, if a state path is known
    return EmbeddingIndex([t[0] for t in terms], [t[1] for t in terms], embedder, cache_file=cache, kind="hpo")  # IDs and texts only; no payloads


def build_name_suggester(drug_names, embedder, state_path=None):  # "did you mean" for drug names
    """A suggester(text) -> [names] for the resolver: nearest drug names, top-k, never auto-chosen."""
    cache = cache_path(state_path, "drugs") if state_path else None  # cache file, if a state path is known
    index = EmbeddingIndex(list(drug_names), list(drug_names), embedder, cache_file=cache, kind="drugs")  # each name is both ID and text
    return lambda text: index.nearest_ids(text, k=3)  # returns the 3 nearest names as suggestions only


# ---------------------------------------------------------------------------
# FILE SUMMARY: Wraps the all-MiniLM-L6-v2 embedding model and a cosine nearest-neighbour index with an on-disk
# vector cache. It only proposes candidate records, "did you mean" drug names and HPO term candidates; it never
# decides that an interaction exists or its severity, and no similarity score is exposed.
# ASSIGNMENT TOOL: Embeddings in a limited role (the modified Meeting 7 / Meeting 8 embedding work). One
# interaction row is one whole document; no splitting of records (RAG) is implemented.
# ---------------------------------------------------------------------------
