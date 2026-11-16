"""Tests for tier two: embeddings in their limited role. A fake embedder stands in for the model,
so these run offline; one optional test uses the real model when it is installed and downloadable."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ddi_backend import embeddings, hpo, ingest, load_data, registry
from ddi_backend.checker import Checker
from ddi_backend.retrieval import QuestionAnswerer
from tests import support
from tests.support import ddi_row

try:
    import numpy
except ImportError:  # tier two switches itself off without numpy
    numpy = None

needs_numpy = unittest.skipUnless(numpy is not None, "numpy is not installed")


class FakeEmbedder:
    """Hashed bag of words. Like the real model in our Meeting 8 test, it ignores the word 'not'."""
    identity = "fake-embedder-v1"
    DIM = 64

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = numpy.zeros((len(texts), self.DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in text.casefold().replace("\n", " ").split():
                word = word.strip(".,:;?!()")
                if word and word != "not":
                    out[row, hash(word) % self.DIM] += 1.0
        norms = numpy.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class ExplodingEmbedder:
    identity = "must-not-be-used"

    def encode(self, texts):
        raise AssertionError("the embedder must not be used on this path")


def records():
    conn = support.open_real()
    try:
        return ingest.load_interactions(conn)
    finally:
        conn.close()


def has_key(value, name):
    if isinstance(value, dict):
        return name in value or any(has_key(v, name) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_key(v, name) for v in value)
    return False


@needs_numpy
class TestEmbeddingIndex(unittest.TestCase):
    def test_nearest_ids_are_candidates_without_scores(self):
        index = embeddings.build_interaction_index(records(), FakeEmbedder())
        ids = index.nearest_ids("warfarin ibuprofen bleeding", k=3)
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(i.startswith("ddi:") for i in ids))
        self.assertEqual(ids, index.nearest_ids("warfarin ibuprofen bleeding", k=3))  # deterministic
        self.assertTrue(all(isinstance(i, str) for i in ids))  # IDs only, never (id, score)

    def test_vectors_are_cached_beside_the_state_file_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "agent_state.db"
            first = FakeEmbedder()
            index = embeddings.build_interaction_index(records(), first, state)
            self.assertFalse(index.from_cache)
            cache = embeddings.cache_path(state, "interactions")
            self.assertTrue(cache.is_file())
            self.assertEqual(cache.parent, state.parent)
            second = FakeEmbedder()
            again = embeddings.build_interaction_index(records(), second, state)
            self.assertTrue(again.from_cache)
            self.assertEqual(second.calls, 0)  # computed once, not on every start
            self.assertEqual(index.nearest_ids("fluconazole", 3), again.nearest_ids("fluconazole", 3))

    def test_cache_is_rebuilt_when_the_model_or_the_data_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "agent_state.db"
            embeddings.build_interaction_index(records(), FakeEmbedder(), state)

            class OtherModel(FakeEmbedder):
                identity = "fake-embedder-v2"

            self.assertFalse(embeddings.build_interaction_index(records(), OtherModel(), state).from_cache)
            changed = records()
            changed[0] = dict(changed[0], effect="A different effect text")
            self.assertFalse(embeddings.build_interaction_index(changed, OtherModel(), state).from_cache)

    def test_a_corrupt_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "agent_state.db"
            embeddings.cache_path(state, "interactions").write_bytes(b"not an npz file")
            index = embeddings.build_interaction_index(records(), FakeEmbedder(), state)
            self.assertFalse(index.from_cache)

    def test_a_duplicated_record_changes_nothing(self):
        base = records()
        duplicate = dict(base[3], id="ddi:9999")
        single = embeddings.build_interaction_index(base, FakeEmbedder())
        doubled = embeddings.build_interaction_index(base + [duplicate], FakeEmbedder())
        self.assertEqual(len(single.ids), len(doubled.ids))
        for query in ("bleeding risk", "warfarin", "serotonin"):
            self.assertEqual(single.nearest_ids(query, 5), doubled.nearest_ids(query, 5))
        self.assertNotIn("ddi:9999", doubled.ids)

    def test_one_record_is_one_document(self):
        recs = records()
        index = embeddings.build_interaction_index(recs, FakeEmbedder())
        self.assertEqual(len(index.ids), len({embeddings.interaction_text(r).casefold() for r in recs}))
        self.assertEqual(index.matrix.shape[0], len(index.ids))

    def test_hpo_has_its_own_index_and_cache(self):
        conn = support.open_real_full()
        self.addCleanup(conn.close)
        terms = hpo.documents(conn)[:300]
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "agent_state.db"
            embeddings.build_interaction_index(records(), FakeEmbedder(), state)
            index = embeddings.build_hpo_index(terms, FakeEmbedder(), state)
            self.assertTrue(all(i.startswith("HP:") for i in index.ids))
            self.assertNotEqual(embeddings.cache_path(state, "hpo"), embeddings.cache_path(state, "interactions"))
            self.assertTrue(embeddings.cache_path(state, "hpo").is_file())

    def test_name_suggester_returns_real_drug_names_only(self):
        names = ["Warfarin", "Ibuprofen", "Fluconazole"]
        suggest = embeddings.build_name_suggester(names, FakeEmbedder())
        self.assertTrue(set(suggest("warfarin tablets")) <= set(names))
        self.assertLessEqual(len(suggest("anything")), 3)


@needs_numpy
class TestEmbeddingsNeverDecide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real_full()
        cls.checker = Checker(cls.conn)
        cls.index = embeddings.build_interaction_index(ingest.load_interactions(cls.conn), FakeEmbedder())

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_resolved_questions_never_touch_the_embedder(self):
        index = embeddings.build_interaction_index(records(), FakeEmbedder())
        index.embedder = ExplodingEmbedder()
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, embeddings=index)
        pair = qa.ask("Do warfarin and ibuprofen interact?")
        self.assertEqual(pair["results"][0]["source"]["record_ids"], ["ddi:1"])
        avoid = qa.ask("Which pain reliever should I not take with warfarin?")
        self.assertEqual(avoid["intent"], "avoid")
        self.assertNotEqual(avoid["avoid"]["avoid"][0]["drug"], "Acetaminophen")

    def test_similarity_is_blind_to_negation_so_the_rules_handle_it(self):
        # The Meeting 8 failure: an embedding ignores 'not', so "not take" and "take" rank the same records.
        with_not = self.index.nearest_ids("Which pain reliever should I not take with warfarin", k=5)
        without = self.index.nearest_ids("Which pain reliever should I take with warfarin", k=5)
        self.assertEqual(with_not, without)
        # The explicit negation rule is what changes the answer.
        qa = QuestionAnswerer(self.checker, hpo_conn=self.conn, embeddings=self.index)
        self.assertEqual(qa.ask("Which pain reliever should I not take with warfarin")["intent"], "avoid")
        self.assertEqual(qa.ask("Which pain reliever should I take with warfarin")["intent"], "drug")

    def test_free_text_with_no_drug_returns_labelled_candidates_without_scores(self):
        qa = QuestionAnswerer(self.checker, hpo_conn=None, embeddings=self.index)
        answer = qa.ask("xyzzy plugh frobnicate")
        self.assertEqual(answer["status"], "candidates_only")
        self.assertTrue(answer["candidates"])
        self.assertIn("embedding_candidates_only", answer["rules_fired"])
        self.assertIn("does not decide", " ".join(answer["notes"]))
        for banned in ("score", "similarity", "confidence", "probability"):
            self.assertFalse(has_key(answer, banned), banned)
        self.assertNotIn("results", answer)  # a candidate list is not a documented answer

    def test_candidates_still_pass_the_eligibility_filter(self):
        stale = registry.build_registry({"approved_authority_classes": ["course_designated_dataset"],
                                         "sources": {"kaggle_ddi": {"status": "archived", "authority_class": "course_designated_dataset", "scope": "exact_pair"}}})
        checker = Checker(self.conn, registry=stale)
        qa = QuestionAnswerer(checker, hpo_conn=None, embeddings=self.index)
        answer = qa.ask("xyzzy plugh frobnicate")
        self.assertEqual(answer["status"], "no_match")
        self.assertNotIn("candidates", answer)

    def test_without_an_index_the_answer_is_no_match_not_a_guess(self):
        answer = QuestionAnswerer(self.checker, hpo_conn=None).ask("xyzzy plugh frobnicate")
        self.assertEqual(answer["status"], "no_match")

    def test_hpo_embedding_candidates_only_help_find_effect_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hpo_path = tmp / "hp.json"
            iri = "http://purl.obolibrary.org/obo/"
            hpo_path.write_text(json.dumps({"graphs": [{"nodes": [
                {"id": iri + "HP_0000001", "lbl": "Vertigo", "type": "CLASS", "meta": {}}], "edges": []}]}))
            ddi = tmp / "ddi.json"
            ddi.write_text(json.dumps({"drug_interactions": {"all": [
                ddi_row("A", "B", "Moderate", effect="Causes vertigo in some patients")]}}))
            db = tmp / "t.db"
            conn = load_data.connect(db)
            load_data.load_ddi(conn, ddi)
            load_data.load_hpo(conn, hpo_path)
            conn.close()
            ro = ingest.open_readonly(db)
            try:
                class StubIndex:
                    def nearest_ids(self, word, k=3):
                        return ["HP:0000001"]

                qa = QuestionAnswerer(Checker(ro), hpo_conn=ro, hpo_index=StubIndex())
                answer = qa.ask("Which drugs cause wooziness?")
                self.assertEqual([i["id"] for i in answer["interactions"]], ["ddi:1"])
                self.assertEqual(answer["interactions"][0]["matched_on"], "effect text")
                self.assertEqual(answer["interactions"][0]["hpo_term_helped"]["label"], "Vertigo")
            finally:
                ro.close()


class TestFallbackWithoutTierTwo(unittest.TestCase):
    def test_load_embedder_returns_none_when_the_model_cannot_load(self):
        with mock.patch.object(embeddings, "SentenceTransformerEmbedder",
                               side_effect=embeddings.EmbeddingUnavailable("no network")):
            self.assertIsNone(embeddings.load_embedder())

    def test_load_embedder_returns_none_without_numpy(self):
        with mock.patch.object(embeddings, "_numpy", side_effect=embeddings.EmbeddingUnavailable("no numpy")):
            self.assertIsNone(embeddings.load_embedder())

    def test_sentence_transformer_failure_becomes_embedding_unavailable(self):
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(embeddings.EmbeddingUnavailable):
                embeddings.SentenceTransformerEmbedder()

    def test_tier_one_works_with_no_embeddings_and_falls_back_to_difflib(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        qa = QuestionAnswerer(Checker(conn))
        self.assertEqual(qa.ask("Do warfarin and ibuprofen interact?")["results"][0]["severity"], "Major")
        typo = qa.ask("Does Warfrin interact with Ibuprofen?")
        self.assertIn("Warfarin", typo["clarifications"][0]["did_you_mean"])  # difflib, no model

    def test_pinned_model_is_the_one_the_class_notebooks_used(self):
        self.assertTrue(embeddings.MODEL_NAME.endswith("all-MiniLM-L6-v2"))
        self.assertTrue(embeddings.MODEL_REVISION)


class TestRealModel(unittest.TestCase):
    """Runs only when sentence-transformers is installed and the model can be loaded."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = embeddings.load_embedder() if numpy is not None else None
        if cls.embedder is None:
            raise unittest.SkipTest("sentence-transformers model is not available")

    def test_real_embedder_finds_candidates_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "agent_state.db"
            index = embeddings.build_interaction_index(records(), self.embedder, state)
            ids = index.nearest_ids("blood thinner and nonsteroidal pain medicine bleeding", k=5)
            self.assertEqual(len(ids), 5)
            self.assertTrue(embeddings.build_interaction_index(records(), self.embedder, state).from_cache)


if __name__ == "__main__":
    unittest.main()
