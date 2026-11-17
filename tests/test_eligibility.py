# Module docstring: says this file tests the eligibility filter and that non-kaggle sources are invented for the tests.
"""Tests for the eligibility filter. Non-kaggle sources here are made up for the tests."""

import tempfile  # makes throwaway folders so each test can build its own small database
import unittest  # standard-library test framework
from pathlib import Path  # path objects for the temporary folder

from ddi_backend import ingest, registry, rules  # database access, source registry, and the named rule constants
from ddi_backend.checker import DOCUMENTED, EVIDENCE_INELIGIBLE, NO_DOCUMENTED, Checker  # pair statuses and the main checker
from ddi_backend.common import NOT_SAFE_NOTE  # the "not found is not the same as safe" wording the notes must carry
from ddi_backend.eligibility import MONOGRAPH_RECORD, EligibilityFilter  # the filter under test and the single-drug record kind
from tests import support  # shared helpers: build a small database, open the real one
from tests.support import ddi_row  # builds one interaction row for a test database


def make_registry(**sources):  # builds a source registry from keyword arguments, one per source key
    # Seed with two approved authority classes (the second is a made-up one for the tests) plus the given sources.
    seed = {"approved_authority_classes": ["course_designated_dataset", "demo_reference"], "sources": sources}
    return registry.build_registry(seed)  # turns the seed dictionary into a real registry object


# Metadata for a good source: current, approved authority class, and applies to exact drug pairs.
KAGGLE = {"status": "current", "authority_class": "course_designated_dataset", "scope": "exact_pair"}
PAIR = ("Warfarin", "Ibuprofen")  # the pair most tests ask about


def record(key="kaggle_ddi", a="Warfarin", b="Ibuprofen", **extra):  # builds a fake interaction record, overridable by field
    # Start of the record dict: ID built from the source key, then the source and the two drug names.
    return {"id": f"{key}:1", "source_key": key, "drug_a": a, "drug_b": b, "severity": "Major",
            # Filler text fields and the reference.
            "mechanism": "m", "effect": "e", "safer_alternative": "", "rationale": "r", "reference": "ref",
            # An empty management-action alternative; **extra lets a test override any field above.
            "alternative": {"text": "", "type": "management_action", "drug_names": []}, **extra}


class TestEvaluate(unittest.TestCase):  # tests EligibilityFilter.evaluate on single records
    def verdict(self, source_meta, rec=None, **kwargs):  # helper: run the filter for a source with the given metadata
        # With None the source is left out of the registry entirely; otherwise it is registered as kaggle_ddi.
        sources = {} if source_meta is None else {"kaggle_ddi": source_meta}
        # Evaluate the given (or default) record against the Warfarin/Ibuprofen pair and return the verdict.
        return EligibilityFilter(make_registry(**sources)).evaluate(rec or record(), PAIR, **kwargs)

    def test_current_approved_exact_pair_is_eligible(self):  # the happy path
        v = self.verdict(KAGGLE)  # evaluate with fully good metadata
        self.assertTrue(v.eligible)  # it counts as evidence
        self.assertEqual(v.role, "direct")  # and directly answers this pair
        self.assertEqual(v.metadata["authority_class"], "course_designated_dataset")  # the verdict reports the authority it used

    def test_stale_source_is_excluded(self):  # a source that is not current must not count
        for status in ("archived", "unknown"):  # two kinds of not-current status
            v = self.verdict({**KAGGLE, "status": status})  # same metadata but with the bad status
            with self.subTest(status=status):  # report each status separately if one fails
                self.assertFalse(v.eligible)  # excluded
                self.assertIn(status, v.reason)  # and the reason names the status

    def test_missing_authority_is_excluded_and_not_filled_in(self):  # absent metadata is never guessed
        v = self.verdict({"status": "current", "scope": "exact_pair"})  # no authority_class key at all
        self.assertFalse(v.eligible)  # excluded
        self.assertIn("missing authority", v.reason)  # reason says authority is missing
        self.assertIsNone(v.metadata["authority_class"])  # the code left it empty rather than filling in a default

    def test_unapproved_authority_is_excluded(self):  # an authority class not on the approved list fails
        v = self.verdict({**KAGGLE, "authority_class": "vocabulary"})  # vocabulary is not approved as evidence
        self.assertFalse(v.eligible)  # excluded
        self.assertIn("not approved", v.reason)  # reason says so

    def test_scope_none_is_never_interaction_evidence(self):  # a source with scope none cannot support an interaction
        v = self.verdict({**KAGGLE, "scope": "none"})  # otherwise-good metadata with scope none
        self.assertFalse(v.eligible)  # excluded
        self.assertIn("scope 'none'", v.reason)  # reason quotes the scope

    def test_unregistered_or_unsourced_record_is_excluded(self):  # two ways a record can lack a usable source
        self.assertIn("not in the registry", self.verdict(None).reason)  # source key not registered
        self.assertIn("no source key", self.verdict(KAGGLE, record(key=None)).reason)  # record has no source key

    def test_record_for_a_different_pair_is_not_evidence_for_this_one(self):  # a record about another pair must not answer
        # The Meeting 18 distractor: the warfarin + acetaminophen row must not answer warfarin + ibuprofen.
        v = self.verdict(KAGGLE, record(b="Acetaminophen"))  # record is for warfarin + acetaminophen
        self.assertEqual(v.role, "context")  # it only counts as context
        self.assertTrue(v.eligible)  # shares warfarin, so it is context, never a direct answer
        unrelated = self.verdict(KAGGLE, record(a="Zinc", b="Iron Sulfate"))  # a record sharing no drug with the pair
        self.assertFalse(unrelated.eligible)  # not eligible at all

    def test_monograph_record_must_cover_the_drug(self):  # single-drug records must be about the drug asked
        # Registry whose source is a single-drug source instead of an exact-pair one.
        flt = EligibilityFilter(make_registry(kaggle_ddi={**KAGGLE, "scope": "single_drug"}))
        # A record about Warfarin, evaluated for the drug warfarin (case differs on purpose).
        ok = flt.evaluate({"id": "m:1", "source_key": "kaggle_ddi", "drug": "Warfarin"},
                          record_type=MONOGRAPH_RECORD, drug_name="warfarin")
        # A record about Ibuprofen, evaluated for Warfarin, which it does not cover.
        wrong = flt.evaluate({"id": "m:2", "source_key": "kaggle_ddi", "drug": "Ibuprofen"},
                             record_type=MONOGRAPH_RECORD, drug_name="Warfarin")
        self.assertEqual((ok.eligible, ok.role), (True, "context"))  # the matching one is eligible context
        self.assertFalse(wrong.eligible)  # the mismatched one is excluded


class TestRealRegistry(unittest.TestCase):  # runs the filter against the real data and the real registry
    def test_the_filter_never_over_excludes_the_real_dataset(self):  # every real record must pass
        conn = support.open_real()  # opens the real medications database
        self.addCleanup(conn.close)  # close it when the test ends
        flt = EligibilityFilter(registry.load_registry())  # filter using the shipped sources_seed.json
        excluded = []  # collects any record the filter wrongly drops
        for rec in ingest.load_interactions(conn):  # every interaction record in the database
            ev = flt.evaluate(rec, (rec["drug_a"], rec["drug_b"]))  # evaluate it for its own pair
            if not ev.eligible:  # if it was excluded
                excluded.append((rec["id"], ev.reason))  # remember which one and why
        self.assertEqual(excluded, [])  # none should have been excluded


class TestCheckerWithEligibility(unittest.TestCase):  # tests how the Checker reports eligibility outcomes
    def build(self, rows, reg, extra=None):  # helper: build a Checker over a small temporary database
        tmp = tempfile.TemporaryDirectory()  # temporary folder for the database file
        self.addCleanup(tmp.cleanup)  # delete the folder after the test
        # Build a database from the rows and open it read-only, as the application does.
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), rows), ingest.CORE_TABLES)
        self.addCleanup(conn.close)  # close the connection after the test
        return Checker(conn, registry=reg, extra_evidence=extra)  # checker using this registry and optional extra source

    def result(self, checker):  # helper: check warfarin against ibuprofen and return the one pair result
        return checker.check(["Warfarin"], "Ibuprofen")["results"][0]

    def test_stale_only_evidence_is_evidence_ineligible_not_no_interaction(self):  # the key safeguard
        # The only row comes from a source marked archived.
        checker = self.build([ddi_row(*PAIR, "Major")], make_registry(kaggle_ddi={**KAGGLE, "status": "archived"}))
        r = self.result(checker)  # result for the pair
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)  # status says the evidence is ineligible
        self.assertNotEqual(r["status"], NO_DOCUMENTED)  # and it must not be reported as "no interaction"
        self.assertIsNone(r["severity"])  # no severity is shown from excluded evidence
        self.assertTrue(r["needs_review"])  # the pair is routed to review
        self.assertIn("archived", r["message"])  # the message gives the reason
        self.assertIn("not a statement that there is no interaction", r["message"])  # and warns it is not a safety statement
        self.assertIn(NOT_SAFE_NOTE, r["notes"])  # the standard not-safe note is attached
        self.assertIn(rules.RULE_EVIDENCE_INELIGIBLE, r["rules_fired"])  # the named rule is listed as fired
        self.assertEqual(r["excluded_evidence"][0]["record_id"], "ddi:1")  # the dropped record is listed by ID
        self.assertEqual(r["records"], [])  # the excluded record's content is not displayed as evidence

    def test_missing_authority_is_evidence_ineligible(self):  # same outcome when authority metadata is absent
        # Registry whose source has no authority class.
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi={"status": "current", "scope": "exact_pair"}))
        r = self.result(checker)  # result for the pair
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)  # ineligible, not "no interaction"
        self.assertIn("missing authority", r["message"])  # message names the missing authority

    def test_empty_registry_abstains_instead_of_answering(self):  # no registered sources means no answer
        r = self.result(self.build([ddi_row(*PAIR, "Major")], make_registry()))  # registry with no sources
        self.assertEqual(r["status"], EVIDENCE_INELIGIBLE)  # the checker abstains

    def test_pairs_with_no_rows_stay_no_documented_even_with_a_bad_registry(self):  # no rows is a different case
        # The database has rows for other pairs (both drugs exist) but none for warfarin + ibuprofen.
        r = self.result(self.build([ddi_row("Warfarin", "Zinc", "Major"), ddi_row("Ibuprofen", "Zinc", "Minor")],
                                   make_registry()))  # and the registry is empty
        self.assertEqual(r["status"], NO_DOCUMENTED)  # there is nothing to exclude, so no documented interaction

    def test_ineligible_extra_source_is_excluded_and_the_rest_still_answers(self):  # one bad source does not sink the answer
        stale = record(key="old_source")  # a record from a second source
        stale["severity"] = "Minor"  # with a different severity than the main row
        # Main source is good; the extra one is archived. extra supplies the stale record.
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi=KAGGLE, old_source={**KAGGLE, "status": "archived"}),
                             extra=lambda names: [stale])
        r = self.result(checker)  # result for the pair
        self.assertEqual((r["status"], r["severity"]), (DOCUMENTED, "Major"))  # still documented, headline Major
        self.assertEqual([x["id"] for x in r["records"]], ["ddi:1"])  # only the eligible record is shown
        self.assertEqual(r["excluded_evidence"][0]["record_id"], "old_source:1")  # the stale one is listed as excluded
        self.assertTrue(r["needs_review"])  # an exclusion routes the pair to review
        # One of the review reasons mentions the excluded record.
        self.assertTrue(any("excluded_record old_source:1" in reason for reason in r["review_reasons"]))
        self.assertFalse(r["severity_conflict"])  # the excluded Minor row does not create a conflict

    def test_eligible_second_source_can_conflict_in_severity(self):  # two eligible sources that disagree
        other = record(key="second")  # a record from a second source
        other["severity"] = "Minor"  # disagreeing severity
        # Both sources are good, so both records count.
        checker = self.build([ddi_row(*PAIR, "Major")],
                             make_registry(kaggle_ddi=KAGGLE, second={**KAGGLE}), extra=lambda names: [other])
        r = self.result(checker)  # result for the pair
        self.assertEqual(r["severity"], "Major")  # the headline is the highest severity
        self.assertTrue(r["severity_conflict"])  # and the disagreement is flagged

    def test_context_evidence_never_establishes_an_interaction_alone(self):  # context records cannot create an interaction
        context = record(key="info_page", b="Aspirin", id="info:1")  # names warfarin only
        # Registry with the real source plus a single-drug "info page" source.
        reg = make_registry(kaggle_ddi=KAGGLE, info_page={"status": "current", "authority_class": "demo_reference", "scope": "single_drug"})
        # Database has rows for other pairs only, so warfarin + ibuprofen has no row of its own.
        checker = self.build([ddi_row("Zinc", "Iron Sulfate", "Minor"), ddi_row("Warfarin", "Zinc", "Minor"),
                              ddi_row("Ibuprofen", "Zinc", "Minor")], reg, extra=lambda names: [context])
        r = self.result(checker)  # result for the pair
        self.assertEqual(r["status"], NO_DOCUMENTED)  # context alone does not make it a documented interaction
        self.assertEqual(r["context_evidence"][0]["id"], "info:1")  # but the context record is still shown
        self.assertIn(NOT_SAFE_NOTE, r["notes"])  # and the not-safe note stays attached

    def test_context_evidence_is_attached_to_a_documented_pair(self):  # context rides along with a real answer
        context = record(key="info_page", b="Aspirin", id="info:1")  # a single-drug context record
        # Same two-source registry as above.
        reg = make_registry(kaggle_ddi=KAGGLE, info_page={"status": "current", "authority_class": "demo_reference", "scope": "single_drug"})
        checker = self.build([ddi_row(*PAIR, "Major")], reg, extra=lambda names: [context])  # now the pair has its own row
        r = self.result(checker)  # result for the pair
        self.assertEqual(r["status"], DOCUMENTED)  # documented thanks to the direct row
        self.assertEqual([c["id"] for c in r["context_evidence"]], ["info:1"])  # context record listed separately
        roles = {e["record_id"]: e["role"] for e in r["evidence"]}  # map each evidence record to its role
        self.assertEqual(roles, {"ddi:1": "direct", "info:1": "context"})  # direct row is direct, info page is context

    def test_eligibility_can_be_turned_off(self):  # the filter is optional
        tmp = tempfile.TemporaryDirectory()  # temporary folder for a small database
        self.addCleanup(tmp.cleanup)  # remove it afterwards
        # Build and open the database by hand (this test does not use the build helper).
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), [ddi_row(*PAIR, "Major")]), ingest.CORE_TABLES)
        self.addCleanup(conn.close)  # close the connection afterwards
        r = Checker(conn, eligibility=False).check(["Warfarin"], "Ibuprofen")["results"][0]  # checker with the filter off
        self.assertEqual(r["status"], DOCUMENTED)  # still answers
        self.assertNotIn("evidence", r)  # but carries no evidence-metadata block

    def test_real_data_reports_evidence_metadata(self):  # end-to-end on the real database
        conn = support.open_real()  # open the real medications database
        self.addCleanup(conn.close)  # close it afterwards
        r = Checker(conn).check(["Warfarin"], "Ibuprofen")["results"][0]  # default checker, filter on
        self.assertEqual(r["status"], DOCUMENTED)  # warfarin + ibuprofen is documented
        self.assertIn(rules.RULE_ELIGIBILITY, r["rules_fired"])  # the eligibility rule shows in the fired list
        ev = r["evidence"][0]  # the first evidence entry
        self.assertEqual((ev["eligible"], ev["role"], ev["source_key"]), (True, "direct", "kaggle_ddi"))  # eligible, direct, from kaggle_ddi
        self.assertIsNone(ev["metadata"]["effective_date"])  # the unknown effective date stays empty, not guessed


if __name__ == "__main__":  # only when the file is run directly
    unittest.main()  # run the tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the evidence eligibility filter (ddi_backend/eligibility.py) and how the Checker reports its
# results. It covers stale, unapproved, missing-authority and scope-none sources, records about the wrong pair,
# context versus direct evidence, the "evidence_ineligible is never no interaction" rule, severity conflicts across
# sources, switching the filter off, and a check that the real dataset is never wrongly excluded.
# ASSIGNMENT TOOL: Metadata-aware eligibility filtering (adopted), working with the knowledge graph and rule-based
# reasoning in the checker.
# ---------------------------------------------------------------------------
