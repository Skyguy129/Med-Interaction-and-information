"""Tests for the eligibility filter. Non-kaggle sources here are made up for the tests."""

import tempfile
import unittest
from pathlib import Path

from ddi_backend import ingest, registry, rules
from ddi_backend.checker import DOCUMENTED, EVIDENCE_INELIGIBLE, NO_DOCUMENTED, Checker
from ddi_backend.common import NOT_SAFE_NOTE
from ddi_backend.eligibility import MONOGRAPH_RECORD, EligibilityFilter
from tests import support
from tests.support import ddi_row


def make_registry(**sources):
    seed = {"approved_authority_classes": ["course_designated_dataset", "demo_reference"], "sources": sources}
    return registry.build_registry(seed)


KAGGLE = {"status": "current", "authority_class": "course_designated_dataset", "scope": "exact_pair"}
PAIR = ("Warfarin", "Ibuprofen")


def record(key="kaggle_ddi", a="Warfarin", b="Ibuprofen", **extra):
    return {"id": f"{key}:1", "source_key": key, "drug_a": a, "drug_b": b, "severity": "Major",
            "mechanism": "m", "effect": "e", "safer_alternative": "", "rationale": "r", "reference": "ref",
            "alternative": {"text": "", "type": "management_action", "drug_names": []}, **extra}


class TestEvaluate(unittest.TestCase):
    def verdict(self, source_meta, rec=None, **kwargs):
        sources = {} if source_meta is None else {"kaggle_ddi": source_meta}
        return EligibilityFilter(make_registry(**sources)).evaluate(rec or record(), PAIR, **kwargs)

    def test_current_approved_exact_pair_is_eligible(self):
        v = self.verdict(KAGGLE)
        self.assertTrue(v.eligible)
        self.assertEqual(v.role, "direct")
        self.assertEqual(v.metadata["authority_class"], "course_designated_dataset")

    def test_stale_source_is_excluded(self):
        for status in ("archived", "unknown"):
            v = self.verdict({**KAGGLE, "status": status})
            with self.subTest(status=status):
                self.assertFalse(v.eligible)
                self.assertIn(status, v.reason)

    def test_missing_authority_is_excluded_and_not_filled_in(self):
        v = self.verdict({"status": "current", "scope": "exact_pair"})
        self.assertFalse(v.eligible)
        self.assertIn("missing authority", v.reason)
        self.assertIsNone(v.metadata["authority_class"])

    def test_unapproved_authority_is_excluded(self):
        v = self.verdict({**KAGGLE, "authority_class": "vocabulary"})
        self.assertFalse(v.eligible)
        self.assertIn("not approved", v.reason)

    def test_scope_none_is_never_interaction_evidence(self):
        v = self.verdict({**KAGGLE, "scope": "none"})
        self.assertFalse(v.eligible)
        self.assertIn("scope 'none'", v.reason)

    def test_unregistered_or_unsourced_record_is_excluded(self):
        self.assertIn("not in the registry", self.verdict(None).reason)
        self.assertIn("no source key", self.verdict(KAGGLE, record(key=None)).reason)

    def test_record_for_a_different_pair_is_not_evidence_for_this_one(self):
        # The Meeting 18 distractor: the warfarin + acetaminophen row must not answer warfarin + ibuprofen.
        v = self.verdict(KAGGLE, record(b="Acetaminophen"))
        self.assertEqual(v.role, "context")
        self.assertTrue(v.eligible)  # shares warfarin, so it is context, never a direct answer
        unrelated = self.verdict(KAGGLE, record(a="Zinc", b="Iron Sulfate"))
        self.assertFalse(unrelated.eligible)

    def test_monograph_record_must_cover_the_drug(self):
        flt = EligibilityFilter(make_registry(kaggle_ddi={**KAGGLE, "scope": "single_drug"}))
        ok = flt.evaluate({"id": "m:1", "source_key": "kaggle_ddi", "drug": "Warfarin"},
                          record_type=MONOGRAPH_RECORD, drug_name="warfarin")
        wrong = flt.evaluate({"id": "m:2", "source_key": "kaggle_ddi", "drug": "Ibuprofen"},
                             record_type=MONOGRAPH_RECORD, drug_name="Warfarin")
        self.assertEqual((ok.eligible, ok.role), (True, "context"))
        self.assertFalse(wrong.eligible)


class TestRealRegistry(unittest.TestCase):
    def test_the_filter_never_over_excludes_the_real_dataset(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        flt = EligibilityFilter(registry.load_registry())
        excluded = []
        for rec in ingest.load_interactions(conn):
            ev = flt.evaluate(rec, (rec["drug_a"], rec["drug_b"]))
            if not ev.eligible:
                excluded.append((rec["id"], ev.reason))
        self.assertEqual(excluded, [])


class TestCheckerWithEligibility(unittest.TestCase):
    def build(self, rows, reg, extra=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), rows), ingest.CORE_TABLES)
        self.addCleanup(conn.close)
        return Checker(conn, registry=reg, extra_evidence=extra)

    def result(self, checker):
        return checker.check(["Warfarin"], "Ibuprofen")["results"][0]

    def test_stale_only_evidence_is_evidence_ineligible_not_no_interaction(self):
        checker = self.build([ddi_row(*PAIR, "Major")], make_registry(kaggle_ddi={**KAGGLE, "status": "archived"}))
        r = self.result(checker)
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)
        self.assertNotEqual(r["status"], NO_DOCUMENTED)
        self.assertIsNone(r["severity"])
        self.assertTrue(r["needs_review"])
        self.assertIn("archived", r["message"])
        self.assertIn("not a statement that there is no interaction", r["message"])
        self.assertIn(NOT_SAFE_NOTE, r["notes"])
        self.assertIn(rules.RULE_EVIDENCE_INELIGIBLE, r["rules_fired"])
        self.assertEqual(r["excluded_evidence"][0]["record_id"], "ddi:1")
        self.assertEqual(r["records"], [])  # the excluded record's content is not displayed as evidence

    def test_missing_authority_is_evidence_ineligible(self):
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi={"status": "current", "scope": "exact_pair"}))
        r = self.result(checker)
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)
        self.assertIn("missing authority", r["message"])

    def test_empty_registry_abstains_instead_of_answering(self):
        r = self.result(self.build([ddi_row(*PAIR, "Major")], make_registry()))
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)

    def test_pairs_with_no_rows_stay_no_documented_even_with_a_bad_registry(self):
        r = self.result(self.build([ddi_row("Warfarin", "Zinc", "Major"), ddi_row("Ibuprofen", "Zinc", "Minor")],
                                   make_registry()))
        self.assertEqual(r["status"], NO_DOCUMENTED)

    def test_ineligible_extra_source_is_excluded_and_the_rest_still_answers(self):
        stale = record(key="old_source")
        stale["severity"] = "Minor"
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi=KAGGLE, old_source={**KAGGLE, "status": "archived"}),
                             extra=lambda names: [stale])
        r = self.result(checker)
        self.assertEqual((r["status"], r["severity"]), (DOCUMENTED, "Major"))
        self.assertEqual([x["id"] for x in r["records"]], ["ddi:1"])
        self.assertEqual(r["excluded_evidence"][0]["record_id"], "old_source:1")
        self.assertTrue(r["needs_review"])
        self.assertTrue(any("excluded_record old_source:1" in reason for reason in r["review_reasons"]))
        self.assertFalse(r["severity_conflict"])  # the excluded Minor row does not create a conflict

    def test_eligible_second_source_can_conflict_in_severity(self):
        other = record(key="second")
        other["severity"] = "Minor"
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi=KAGGLE, second={**KAGGLE}), extra=lambda names: [other])
        r = self.result(checker)
        self.assertEqual(r["severity"], "Major")
        self.assertTrue(r["severity_conflict"])

    def test_context_evidence_never_establishes_an_interaction_alone(self):
        context = record(key="info_page", b="Aspirin", id="info:1")  # names warfarin only
        reg = make_registry(kaggle_ddi=KAGGLE, info_page={"status": "current", "authority_class": "demo_reference", "scope": "single_drug"})
        checker = self.build([ddi_row("Zinc", "Iron Sulfate", "Minor"), ddi_row("Warfarin", "Zinc", "Minor"),
                              ddi_row("Ibuprofen", "Zinc", "Minor")], reg, extra=lambda names: [context])
        r = self.result(checker)
        self.assertEqual(r["status"], NO_DOCUMENTED)
        self.assertEqual(r["context_evidence"][0]["id"], "info:1")
        self.assertIn(NOT_SAFE_NOTE, r["notes"])

    def test_context_evidence_is_attached_to_a_documented_pair(self):
        context = record(key="info_page", b="Aspirin", id="info:1")
        reg = make_registry(kaggle_ddi=KAGGLE, info_page={"status": "current", "authority_class": "demo_reference", "scope": "single_drug"})
        checker = self.build([ddi_row(*PAIR, "Major")], reg, extra=lambda names: [context])
        r = self.result(checker)
        self.assertEqual(r["status"], DOCUMENTED)
        self.assertEqual([c["id"] for c in r["context_evidence"]], ["info:1"])
        roles = {e["record_id"]: e["role"] for e in r["evidence"]}
        self.assertEqual(roles, {"ddi:1": "direct", "info:1": "context"})

    def test_eligibility_can_be_turned_off(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), [ddi_row(*PAIR, "Major")]), ingest.CORE_TABLES)
        self.addCleanup(conn.close)
        r = Checker(conn, eligibility=False).check(["Warfarin"], "Ibuprofen")["results"][0]
        self.assertEqual(r["status"], DOCUMENTED)
        self.assertNotIn("evidence", r)

    def test_real_data_reports_evidence_metadata(self):
        conn = support.open_real()
        self.addCleanup(conn.close)
        r = Checker(conn).check(["Warfarin"], "Ibuprofen")["results"][0]
        self.assertEqual(r["status"], DOCUMENTED)
        self.assertIn(rules.RULE_ELIGIBILITY, r["rules_fired"])
        ev = r["evidence"][0]
        self.assertEqual((ev["eligible"], ev["role"], ev["source_key"]), (True, "direct", "kaggle_ddi"))
        self.assertIsNone(ev["metadata"]["effective_date"])


if __name__ == "__main__":
    unittest.main()
