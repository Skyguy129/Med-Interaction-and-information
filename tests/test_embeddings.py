# Module docstring (two lines): tests tier two, embeddings in a limited role; a fake embedder keeps tests offline.
"""Tests for tier two: embeddings in their limited role. A fake embedder stands in for the model,
so these run offline; one optional test uses the real model when it is installed and downloadable."""

import json  # writes small HPO and interaction files for the HPO test
import tempfile  # throwaway folders for cache files and test databases
import unittest  # standard-library test framework
from pathlib import Path  # path objects for temp files
from unittest import mock  # patches the embedder loader to simulate failures

from ddi_backend import embeddings, hpo, ingest, load_data, registry  # modules under test and their helpers
from ddi_backend.checker import Checker  # the deterministic checker the question answerer sits on
from ddi_backend.retrieval import QuestionAnswerer  # free-text question handling that may use embeddings
from tests import support  # shared helpers: open the real database
from tests.support import ddi_row  # builds one interaction row for a test database

try:  # numpy is optional (tier two only)
    import numpy  # needed by the fake embedder and the real index
except ImportError:  # tier two switches itself off without numpy
    numpy = None  # remember that it is missing so tests can skip

needs_numpy = unittest.skipUnless(numpy is not None, "numpy is not installed")  # decorator: skip a test when numpy is absent


class FakeEmbedder:  # stand-in for the real model so tests run offline
    # Docstring below says it is a hashed bag of words that, like the real model, ignores the word "not".
    """Hashed bag of words. Like the real model in our Meeting 8 test, it ignores the word 'not'."""
    identity = "fake-embedder-v1"  # name stored with the cache so a model change is detected
    DIM = 64  # number of vector slots

    def __init__(self):  # sets up the call counter
        self.calls = 0  # counts encode calls so tests can show the cache avoided recomputing

    def encode(self, texts):  # turns a list of texts into one unit-length vector each
        self.calls += 1  # record the call
        out = numpy.zeros((len(texts), self.DIM), dtype="float32")  # one row of zeros per text
        for row, text in enumerate(texts):  # each text in turn
            for word in text.casefold().replace("\n", " ").split():  # lower-case words split on whitespace
                word = word.strip(".,:;?!()")  # drop punctuation around the word
                if word and word != "not":  # skip empty strings and the word "not", copying the real model's blindness
                    out[row, hash(word) % self.DIM] += 1.0  # count the word into a slot picked by its hash
        norms = numpy.linalg.norm(out, axis=1, keepdims=True)  # length of each vector
        norms[norms == 0] = 1.0  # avoid dividing by zero for empty texts
        return out / norms  # scale to unit length so dot product equals cosine similarity


class ExplodingEmbedder:  # an embedder that fails the test if it is ever used
    identity = "must-not-be-used"  # label, never stored anywhere in the tests

    def encode(self, texts):  # any use is an error
        raise AssertionError("the embedder must not be used on this path")


def records():  # helper: load all real interaction records
    conn = support.open_real()  # open the real database
    try:  # make sure it is closed even if loading fails
        return ingest.load_interactions(conn)  # all 180 records
    finally:
        conn.close()  # always close the connection


def has_key(value, name):  # helper: does this nested structure contain a dictionary key called name anywhere?
    if isinstance(value, dict):  # dictionaries: check their keys, then recurse into their values
        return name in value or any(has_key(v, name) for v in value.values())
    if isinstance(value, (list, tuple)):  # lists and tuples: recurse into each item
        return any(has_key(v, name) for v in value)
    return False  # plain values have no keys


@needs_numpy  # skip the whole class when numpy is missing
class TestEmbeddingIndex(unittest.TestCase):  # tests building, caching and querying the embedding index
    def test_nearest_ids_are_candidates_without_scores(self):  # search returns record IDs only
        index = embeddings.build_interaction_index(records(), FakeEmbedder())  # index over all records
        ids = index.nearest_ids("warfarin ibuprofen bleeding", k=3)  # three nearest records
        self.assertEqual(len(ids), 3)  # three came back
        self.assertTrue(all(i.startswith("ddi:") for i in ids))  # all are interaction record IDs
        self.assertEqual(ids, index.nearest_ids("warfarin ibuprofen bleeding", k=3))  # deterministic
        self.assertTrue(all(isinstance(i, str) for i in ids))  # IDs only, never (id, score)

    def test_vectors_are_cached_beside_the_state_file_and_reused(self):  # the cache is written once and reused
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            state = Path(tmp) / "agent_state.db"  # where the state file would be
            first = FakeEmbedder()  # embedder for the first build
            index = embeddings.build_interaction_index(records(), first, state)  # first build computes vectors
            self.assertFalse(index.from_cache)  # nothing was cached yet
            cache = embeddings.cache_path(state, "interactions")  # where the cache file should land
            self.assertTrue(cache.is_file())  # it was written
            self.assertEqual(cache.parent, state.parent)  # beside the state file
            second = FakeEmbedder()  # fresh embedder for the second build
            again = embeddings.build_interaction_index(records(), second, state)  # second build
            self.assertTrue(again.from_cache)  # loaded from the cache
            self.assertEqual(second.calls, 0)  # computed once, not on every start
            self.assertEqual(index.nearest_ids("fluconazole", 3), again.nearest_ids("fluconazole", 3))  # same answers either way

    def test_cache_is_rebuilt_when_the_model_or_the_data_changes(self):  # a stale cache must not be trusted
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            state = Path(tmp) / "agent_state.db"  # state file location
            embeddings.build_interaction_index(records(), FakeEmbedder(), state)  # write the first cache

            class OtherModel(FakeEmbedder):  # a "different model" that only differs in its name
                identity = "fake-embedder-v2"

            self.assertFalse(embeddings.build_interaction_index(records(), OtherModel(), state).from_cache)  # new model name: rebuild
            changed = records()  # a fresh copy of the records
            changed[0] = dict(changed[0], effect="A different effect text")  # edit one record's effect text
            self.assertFalse(embeddings.build_interaction_index(changed, OtherModel(), state).from_cache)  # changed data: rebuild

    def test_a_corrupt_cache_is_ignored(self):  # a broken cache file is treated as missing
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            state = Path(tmp) / "agent_state.db"  # state file location
            embeddings.cache_path(state, "interactions").write_bytes(b"not an npz file")  # plant garbage as the cache
            index = embeddings.build_interaction_index(records(), FakeEmbedder(), state)  # build anyway
            self.assertFalse(index.from_cache)  # garbage was not used

    def test_a_duplicated_record_changes_nothing(self):  # repeated text must not skew results
        base = records()  # the real records
        duplicate = dict(base[3], id="ddi:9999")  # a copy of one record under a new ID
        single = embeddings.build_interaction_index(base, FakeEmbedder())  # index without the copy
        doubled = embeddings.build_interaction_index(base + [duplicate], FakeEmbedder())  # index with the copy
        self.assertEqual(len(single.ids), len(doubled.ids))  # the copy was dropped
        for query in ("bleeding risk", "warfarin", "serotonin"):  # a few sample queries
            self.assertEqual(single.nearest_ids(query, 5), doubled.nearest_ids(query, 5))  # same neighbours with or without it
        self.assertNotIn("ddi:9999", doubled.ids)  # the duplicate ID never appears

    def test_one_record_is_one_document(self):  # each record is embedded whole
        recs = records()  # the real records
        index = embeddings.build_interaction_index(recs, FakeEmbedder())  # build the index
        self.assertEqual(len(index.ids), len({embeddings.interaction_text(r).casefold() for r in recs}))  # one entry per distinct record text
        self.assertEqual(index.matrix.shape[0], len(index.ids))  # one vector row per ID

    def test_hpo_has_its_own_index_and_cache(self):  # the HPO vocabulary is indexed separately
        conn = support.open_real_full()  # real database including the HPO tables
        self.addCleanup(conn.close)  # close afterwards
        terms = hpo.documents(conn)[:300]  # the first 300 HPO term documents, to keep the test quick
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            state = Path(tmp) / "agent_state.db"  # state file location
            embeddings.build_interaction_index(records(), FakeEmbedder(), state)  # also write the interaction cache
            index = embeddings.build_hpo_index(terms, FakeEmbedder(), state)  # build the HPO index
            self.assertTrue(all(i.startswith("HP:") for i in index.ids))  # its IDs are HPO term IDs
            self.assertNotEqual(embeddings.cache_path(state, "hpo"), embeddings.cache_path(state, "interactions"))  # a separate cache file
            self.assertTrue(embeddings.cache_path(state, "hpo").is_file())  # and it was written

    def test_name_suggester_returns_real_drug_names_only(self):  # did-you-mean never invents names
        names = ["Warfarin", "Ibuprofen", "Fluconazole"]  # the known drug names
        suggest = embeddings.build_name_suggester(names, FakeEmbedder())  # build the suggester
        self.assertTrue(set(suggest("warfarin tablets")) <= set(names))  # every suggestion is a known name
        self.assertLessEqual(len(suggest("anything")), 3)  # and there are at most three


@needs_numpy  # skip the class when numpy is missing
class TestEmbeddingsNeverDecide(unittest.TestCase):  # similarity may find candidates but never decide an answer
    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.conn = support.open_real_full()  # real database with HPO tables
        cls.checker = Checker(cls.conn)  # deterministic checker over it
        cls.index = embeddings.build_interaction_index(ingest.load_interactions(cls.conn), FakeEmbedder())  # shared index

    @classmethod
    def tearDownClass(cls):  # runs once after the class
        cls.conn.close()  # close the shared connection

    def test_resolved_questions_never_touch_the_embedder(self):  # exact lookups skip embeddings completely
        index = embeddings.build_interaction_index(records(), FakeEmbedder())  # a fresh index
        index.embedder = ExplodingEmbedder()  # swap in an embedder that fails if called
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, embeddings=index)  # answerer using that index
        pair = qa.ask("Do warfarin and ibuprofen interact?")  # a pair question
        self.assertEqual(pair["results"][0]["source"]["record_ids"], ["ddi:1"])  # answered from the database
        avoid = qa.ask("Which pain reliever should I not take with warfarin?")  # a negation question
        self.assertEqual(avoid["intent"], "avoid")  # handled as an avoid-list question
        self.assertNotEqual(avoid["avoid"]["avoid"][0]["drug"], "Acetaminophen")  # the top avoid entry is not acetaminophen

    def test_similarity_is_blind_to_negation_so_the_rules_handle_it(self):  # why rules, not embeddings, handle "not"
        # The Meeting 8 failure: an embedding ignores 'not', so "not take" and "take" rank the same records.
        with_not = self.index.nearest_ids("Which pain reliever should I not take with warfarin", k=5)  # neighbours with "not"
        without = self.index.nearest_ids("Which pain reliever should I take with warfarin", k=5)  # neighbours without it
        self.assertEqual(with_not, without)  # identical, showing the blindness
        # The explicit negation rule is what changes the answer.
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, embeddings=self.index)  # answerer with the index
        self.assertEqual(qa.ask("Which pain reliever should I not take with warfarin")["intent"], "avoid")  # "not" gives avoid
        self.assertEqual(qa.ask("Which pain reliever should I take with warfarin")["intent"], "drug")  # no "not" gives a drug question

    def test_free_text_with_no_drug_returns_labelled_candidates_without_scores(self):  # nonsense falls to candidates
        qa = QuestionAnswerer(self.checker, hpo_conn=None, embeddings=self.index)  # answerer with the index, no HPO
        answer = qa.ask("xyzzy plugh frobnicate")  # text that matches no drug
        self.assertEqual(answer["status"], "candidates_only")  # labelled as candidates, not an answer
        self.assertTrue(answer["candidates"])  # some candidates exist
        self.assertIn("embedding_candidates_only", answer["rules_fired"])  # the named rule is listed
        self.assertIn("does not decide", " ".join(answer["notes"]))  # a note says similarity does not decide
        for banned in ("score", "similarity", "confidence", "probability"):  # keys that would present a score as confidence
            self.assertFalse(has_key(answer, banned), banned)  # none may appear anywhere in the answer
        self.assertNotIn("results", answer)  # a candidate list is not a documented answer

    def test_candidates_still_pass_the_eligibility_filter(self):  # candidates cannot bypass the filter
        # A registry in which the only source is archived, so every record is ineligible.
        stale = registry.build_registry({"approved_authority_classes": ["course_designated_dataset"],
                                         "sources": {"kaggle_ddi": {"status": "archived", "authority_class": "course_designated_dataset", "scope": "exact_pair"}}})
        checker = Checker(self.conn, registry=stale)  # checker using that registry
        qa = QuestionAnswerer(checker, hpo_conn=None, embeddings=self.index)  # answerer using the index
        answer = qa.ask("xyzzy plugh frobnicate")  # nonsense question
        self.assertEqual(answer["status"], "no_match")  # all candidates were filtered out
        self.assertNotIn("candidates", answer)  # so none are shown

    def test_without_an_index_the_answer_is_no_match_not_a_guess(self):  # no embeddings means an honest "no match"
        answer = QuestionAnswerer(self.checker, hpo_conn=None).ask("xyzzy plugh frobnicate")  # no index given
        self.assertEqual(answer["status"], "no_match")  # no guess is made

    def test_hpo_embedding_candidates_only_help_find_effect_text(self):  # HPO helps search but the answer is the effect text
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            tmp = Path(tmp)  # use it as a Path from here on
            hpo_path = tmp / "hp.json"  # where the tiny HPO file goes
            iri = "http://purl.obolibrary.org/obo/"  # prefix HPO uses for its term identifiers
            # Write a minimal HPO file with a single term, Vertigo.
            hpo_path.write_text(json.dumps({"graphs": [{"nodes": [
                {"id": iri + "HP_0000001", "lbl": "Vertigo", "type": "CLASS", "meta": {}}], "edges": []}]}))
            ddi = tmp / "ddi.json"  # where the tiny interaction file goes
            # Write one interaction whose effect text mentions vertigo.
            ddi.write_text(json.dumps({"drug_interactions": {"all": [
                ddi_row("A", "B", "Moderate", effect="Causes vertigo in some patients")]}}))
            db = tmp / "t.db"  # the test database path
            conn = load_data.connect(db)  # create it with the real schema
            load_data.load_ddi(conn, ddi)  # load the interaction file
            load_data.load_hpo(conn, hpo_path)  # load the HPO file
            conn.close()  # finish writing
            ro = ingest.open_readonly(db)  # reopen read-only like the application
            try:  # make sure the connection is closed at the end
                class StubIndex:  # fake HPO index that always suggests the Vertigo term
                    def nearest_ids(self, word, k=3):  # ignores the word and returns the one known term ID
                        return ["HP:0000001"]

                qa = QuestionAnswerer(Checker(ro), hpo_conn=ro, hpo_index=StubIndex())  # answerer using the stub HPO index
                answer = qa.ask("Which drugs cause wooziness?")  # a lay word that is not in the HPO labels
                self.assertEqual([i["id"] for i in answer["interactions"]], ["ddi:1"])  # found the interaction
                self.assertEqual(answer["interactions"][0]["matched_on"], "effect text")  # matched on the effect text itself
                self.assertEqual(answer["interactions"][0]["hpo_term_helped"]["label"], "Vertigo")  # the HPO term only helped find it
            finally:
                ro.close()  # close the read-only connection


class TestFallbackWithoutTierTwo(unittest.TestCase):  # tests that tier one keeps working when embeddings are unavailable
    def test_load_embedder_returns_none_when_the_model_cannot_load(self):  # model failure switches embeddings off
        # Make constructing the real model class raise EmbeddingUnavailable, as a failed download would.
        with mock.patch.object(embeddings, "SentenceTransformerEmbedder",
                               side_effect=embeddings.EmbeddingUnavailable("no network")):
            self.assertIsNone(embeddings.load_embedder())  # the loader returns None instead of raising

    def test_load_embedder_returns_none_without_numpy(self):  # a missing numpy switches embeddings off
        with mock.patch.object(embeddings, "_numpy", side_effect=embeddings.EmbeddingUnavailable("no numpy")):  # numpy import fails
            self.assertIsNone(embeddings.load_embedder())  # loader returns None

    def test_sentence_transformer_failure_becomes_embedding_unavailable(self):  # import errors are wrapped
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):  # makes importing it fail
            with self.assertRaises(embeddings.EmbeddingUnavailable):  # the wrapper error is raised
                embeddings.SentenceTransformerEmbedder()

    def test_tier_one_works_with_no_embeddings_and_falls_back_to_difflib(self):  # the core never depends on tier two
        conn = support.open_real()  # open the real database
        self.addCleanup(conn.close)  # close afterwards
        qa = QuestionAnswerer(Checker(conn))  # answerer with no embeddings at all
        self.assertEqual(qa.ask("Do warfarin and ibuprofen interact?")["results"][0]["severity"], "Major")  # exact lookup still works
        typo = qa.ask("Does Warfrin interact with Ibuprofen?")  # a misspelled drug name
        self.assertIn("Warfarin", typo["clarifications"][0]["did_you_mean"])  # difflib, no model

    def test_pinned_model_is_the_one_the_class_notebooks_used(self):  # the model choice is fixed and reproducible
        self.assertTrue(embeddings.MODEL_NAME.endswith("all-MiniLM-L6-v2"))  # the class notebooks' model
        self.assertTrue(embeddings.MODEL_REVISION)  # a pinned revision is set


class TestRealModel(unittest.TestCase):  # optional test using the actual sentence-transformers model
    # Docstring below: runs only when the model package is installed and loadable.
    """Runs only when sentence-transformers is installed and the model can be loaded."""

    @classmethod
    def setUpClass(cls):  # runs once for the class
        cls.embedder = embeddings.load_embedder() if numpy is not None else None  # try to load the real model
        if cls.embedder is None:  # model not available
            raise unittest.SkipTest("sentence-transformers model is not available")  # skip the whole class

    def test_real_embedder_finds_candidates_and_caches(self):  # smoke test with the real model
        with tempfile.TemporaryDirectory() as tmp:  # temporary folder
            state = Path(tmp) / "agent_state.db"  # state file location
            index = embeddings.build_interaction_index(records(), self.embedder, state)  # build with the real model
            ids = index.nearest_ids("blood thinner and nonsteroidal pain medicine bleeding", k=5)  # a descriptive query
            self.assertEqual(len(ids), 5)  # five candidates came back
            self.assertTrue(embeddings.build_interaction_index(records(), self.embedder, state).from_cache)  # second build uses the cache


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests ddi_backend/embeddings.py and its use by the question answerer. A fake hashed bag-of-words
# embedder keeps everything offline: the tests check candidate-only output with no scores, caching and cache rebuilds,
# one-record-one-document, a separate HPO index, real drug names only in did-you-mean suggestions, that embeddings
# never decide an answer and are blind to "not", that candidates still pass the eligibility filter, and that the
# system falls back to exact names plus difflib when the model or numpy is missing. One optional test uses the real model.
# ASSIGNMENT TOOL: Embeddings in a limited role (the modified Meeting 7 embedding neighbours and Meeting 8
# semantic-search failures), with the eligibility filter and the rule-based checker as the deciders.
# ---------------------------------------------------------------------------
