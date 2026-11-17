# Docstring describing the module: tests for the rule layer and the deterministic checking flow.
"""Tests for the rule layer and the deterministic checking flow."""

import hashlib  # fingerprints the database file to prove it is never written
import tempfile  # temporary folders for synthetic databases
import unittest  # the test framework
from pathlib import Path  # path objects

from ddi_backend import ingest, load_data, rules  # database access, the loader helpers and the named rules
from ddi_backend.checker import DOCUMENTED, NO_DOCUMENTED, Checker  # status constants and the checker under test
from ddi_backend.common import NOT_SAFE_NOTE  # the standard "this does not mean it is safe" sentence
from ddi_backend.graph import Graph  # the knowledge graph class (imported for reference)
from tests import support  # shared database builders
from tests.support import ddi_row  # builds one fake interaction row

# The five dataset rows from the RAG handout, confirmed against the real database.
FIXTURE_ROWS = {  # record id -> (severity, drug a, drug b)
    "ddi:1": ("Major", "Warfarin", "Ibuprofen"),  # warfarin + ibuprofen
    "ddi:12": ("Major", "Warfarin", "Fluconazole"),  # warfarin + fluconazole
    "ddi:20": ("Major", "Warfarin", "Amiodarone"),  # warfarin + amiodarone
    "ddi:26": ("Major", "Warfarin", "Fluoxetine"),  # warfarin + fluoxetine
    "ddi:73": ("Moderate", "Venlafaxine", "Ibuprofen"),  # the one Moderate example
}


# Base class that opens the real database once and offers a one-pair helper.
class RealDbTestCase(unittest.TestCase):
    @classmethod  # runs once for the class
    def setUpClass(cls):
        cls.conn = support.open_real()  # open the real database read-only
        cls.checker = Checker(cls.conn)  # a checker over it

    @classmethod  # runs once after the class
    def tearDownClass(cls):
        cls.conn.close()  # release the connection

    def pair(self, a, b, **kwargs):  # check drug a (current) against drug b (new)
        return self.checker.check([a], b, **kwargs)["results"][0]  # return just the single pair result


# Confirms the test fixtures really match the loaded data.
class TestFixturesMatchTheDatabase(RealDbTestCase):
    def test_handout_rows(self):
        records = {r["id"]: r for r in ingest.load_interactions(self.conn)}  # every record keyed by id
        for record_id, (severity, a, b) in FIXTURE_ROWS.items():  # each fixture
            with self.subTest(record_id=record_id):  # report each separately
                r = records[record_id]  # the stored record
                self.assertEqual((r["severity"], r["drug_a"], r["drug_b"]), (severity, a, b))  # same as the handout

    def test_contradictory_warfarin_rows(self):
        # ddi:1 names acetaminophen as the safer alternative, yet warfarin + acetaminophen has its own row.
        records = {r["id"]: r for r in ingest.load_interactions(self.conn)}  # every record keyed by id
        self.assertTrue(records["ddi:1"]["safer_alternative"].startswith("Acetaminophen"))  # the suggested alternative
        self.assertEqual((records["ddi:69"]["drug_a"], records["ddi:69"]["drug_b"]), ("Warfarin", "Acetaminophen"))  # its own row


# Tests looking up one pair of drugs.
class TestPairLookup(RealDbTestCase):
    def test_either_order_returns_the_same_rows(self):
        forward, backward = self.pair("Warfarin", "Ibuprofen"), self.pair("ibuprofen", "WARFARIN")  # swap order and case
        self.assertEqual(forward["status"], DOCUMENTED)  # documented
        self.assertEqual([r["id"] for r in forward["records"]], [r["id"] for r in backward["records"]])  # same rows
        self.assertEqual([r["id"] for r in forward["records"]], ["ddi:1"])  # and it is ddi:1

    def test_record_comes_back_whole(self):
        result = self.pair("Warfarin", "Amiodarone")  # a documented pair
        record = result["records"][0]  # its record
        for field in ("severity", "mechanism", "effect", "safer_alternative", "rationale", "reference"):  # every field
            self.assertTrue(record[field], field)  # is filled in
        self.assertEqual(result["severity"], "Major")  # headline severity
        self.assertEqual(result["effect"], record["effect"])  # the effect is the record's own text
        self.assertEqual(result["source"]["record_ids"], ["ddi:20"])  # the source names the record id
        self.assertEqual(result["source"]["citations"][0]["reference"], record["reference"])  # and its reference

    def test_lookup_agrees_with_the_graph(self):
        graph = self.checker.graph  # the knowledge graph
        a, b = graph.drug_id("Warfarin"), graph.drug_id("Fluconazole")  # node ids of the two drugs
        # The direct database query and the graph return the same record ids.
        self.assertEqual([r["id"] for r in ingest.pair_records(self.conn, a, b)],
                         [r["id"] for r in graph.pair(a, b)])

    def test_pair_with_no_row_is_not_called_safe(self):
        result = self.pair("Warfarin", "Docusate")  # both drugs exist but no row links them
        self.assertEqual(result["status"], NO_DOCUMENTED)  # no documented interaction
        self.assertIsNone(result["severity"])  # no severity invented
        self.assertIn(NOT_SAFE_NOTE, result["notes"])  # the "not the same as safe" note is attached
        self.assertIn(rules.RULE_ABSENCE_NOT_SAFETY, result["rules_fired"])  # and the matching rule fired
        # Apart from the allowed sentence, the message must not use the word "safe".
        self.assertNotIn("safe", result.get("message", "").replace("not mean the combination is safe", ""))

    def test_drugs_missing_from_the_data_are_unknown_not_safe(self):
        result = self.pair("Cabergoline", "Rucaparib")  # neither is in the data
        self.assertEqual(result["status"], "unknown_drug")  # unknown, not safe
        self.assertEqual(len(result["unresolved"]), 2)  # both names are listed as unresolved
        self.assertIn(NOT_SAFE_NOTE, result["notes"])  # with the not-safe note

    def test_abbreviation_gets_a_question_not_a_guess(self):
        result = self.pair("MS", "Warfarin")  # "MS" could mean several drugs
        self.assertEqual(result["status"], "ambiguous_name")  # flagged ambiguous
        self.assertIn("Which medication", result["message"])  # asks the user to clarify


# Tests pairs that appear twice in the data with different severities.
class TestRepeatedPair(RealDbTestCase):
    def test_clopidogrel_omeprazole_headlines_major_and_flags_conflict(self):
        result = self.pair("Clopidogrel", "Omeprazole")  # listed as Major and as Moderate
        self.assertEqual(result["severity"], "Major")  # the headline is the highest severity
        self.assertTrue(result["severity_conflict"])  # the disagreement is flagged
        self.assertEqual(result["severities"], ["Major", "Moderate"])  # both severities are shown
        self.assertEqual(len(result["records"]), 2)  # every row is returned
        self.assertTrue(result["needs_review"])  # routed to review
        self.assertTrue(any(reason.startswith("severity_conflict") for reason in result["review_reasons"]))  # with a reason
        self.assertIn(rules.RULE_REPEATED_PAIR, result["rules_fired"])  # the repeated-pair rule fired

    def test_every_repeated_pair_in_the_data_reports_its_highest_severity(self):
        order = {"Major": 0, "Moderate": 1, "Minor": 2}  # lower number means more severe
        repeated = load_data.find_repeated_pairs(self.conn)  # all pairs listed more than once
        self.assertEqual(len(repeated), 8)  # the data has 8 of them
        for a, b, severities in repeated:  # check each one
            found = severities.split(" / ")  # the severities as a list
            result = self.pair(a, b)  # run the check
            with self.subTest(pair=(a, b)):  # report each pair separately
                self.assertEqual(result["severity"], min(found, key=order.get))  # headline = most severe
                self.assertEqual(result["severity_conflict"], len(set(found)) > 1)  # conflict only if they differ
                self.assertEqual(len(result["records"]), len(found))  # every row returned


# Tests the whole multi-medication check flow.
class TestMainFlow(RealDbTestCase):
    def test_major_interactions_sort_first(self):
        out = self.checker.check(["Venlafaxine", "Docusate", "Warfarin"], "Ibuprofen")  # three current drugs
        statuses = [(r["pair"][0], r["status"], r["severity"]) for r in out["results"]]  # summarise each result
        self.assertEqual(statuses, [  # the order must be Major, Moderate, then no record
            ("Warfarin", DOCUMENTED, "Major"),
            ("Venlafaxine", DOCUMENTED, "Moderate"),
            ("Docusate", NO_DOCUMENTED, None),
        ])
        self.assertEqual(out["summary"]["major_documented"], 1)  # the summary counts one Major

    def test_current_medications_can_also_be_checked_against_each_other(self):
        without = self.checker.check(["Warfarin", "Fluconazole"], "Docusate")  # default: only against the new drug
        with_pairs = self.checker.check(["Warfarin", "Fluconazole"], "Docusate", include_current_pairs=True)  # plus each other
        self.assertEqual(len(without["results"]), 2)  # two new-drug pairs
        self.assertEqual(len(with_pairs["results"]), 3)  # plus the current-vs-current pair
        among = [r for r in with_pairs["results"] if r["kind"] == "current_vs_current"]  # that extra pair
        self.assertEqual((among[0]["status"], among[0]["severity"]), (DOCUMENTED, "Major"))  # warfarin + fluconazole

    def test_doses_and_details_are_echoed_never_recommended(self):
        # A dose inside the name, a free-text detail, and an explicit dose on the new drug.
        out = self.checker.check([{"name": "Warfarin 5 mg", "details": "evening"}], {"name": "Ibuprofen", "dose": "400 mg"})
        self.assertEqual(out["inputs"]["current"][0]["dose"], "5 mg")  # the dose was split off the name
        self.assertEqual(out["inputs"]["current"][0]["details"], "evening")  # the detail is kept
        self.assertEqual(out["inputs"]["new"]["dose"], "400 mg")  # the new drug's dose is kept
        self.assertEqual(out["results"][0]["doses"], {"Warfarin": "5 mg", "Ibuprofen": "400 mg"})  # echoed in the result
        self.assertNotIn("recommended_dose", str(out))  # no dose advice anywhere

    def test_medication_listed_twice_or_as_the_new_drug_is_not_self_checked(self):
        out = self.checker.check(["Warfarin", "warfarin 5 mg", "Ibuprofen"], "Ibuprofen")  # repeats and the new drug listed as current
        self.assertEqual([r["pair"] for r in out["results"]], [["Warfarin", "Ibuprofen"]])  # only one real pair
        self.assertEqual(len(out["notes"]), 2)  # and two notes explaining what was skipped

    def test_unknown_new_drug_never_reads_as_safe(self):
        out = self.checker.check(["Warfarin"], "Cabergoline")  # the new drug is not in the data
        self.assertEqual(out["results"][0]["status"], "unknown_drug")  # the pair result says unknown
        self.assertEqual(out["inputs"]["new"]["status"], "unknown_drug")  # and so does the input echo

    def test_response_carries_the_disclaimer_and_the_rules_that_fired(self):
        out = self.checker.check(["Warfarin"], "Ibuprofen")  # a simple check
        self.assertIn("not medical advice", out["disclaimer"])  # the disclaimer is present
        for rule in (rules.RULE_EXACT_NAME, rules.RULE_PAIR_LOOKUP, rules.RULE_SEVERITY_PRECEDENCE):  # expected rules
            self.assertIn(rule, out["rules_fired"])  # each is listed

    def test_database_is_unchanged_by_checking(self):
        path = support.real_db_path()  # the database file
        before = hashlib.sha256(path.read_bytes()).hexdigest()  # fingerprint before
        self.checker.check(["Warfarin"], "Ibuprofen", include_current_pairs=True)  # run a check
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)  # the file is identical


# Tests the safer-alternative handling and the cross-check against other drugs.
class TestAlternatives(RealDbTestCase):
    def test_management_action_is_shown_as_written(self):
        alt = self.pair("Warfarin", "Amiodarone")["alternatives"][0]  # the stored "alternative"
        self.assertEqual((alt["type"], alt["text"]), ("management_action", "Reduce warfarin dose by 30-50%"))  # an action, as written
        self.assertEqual(alt["cross_check"], [])  # no drug name, so nothing to cross-check
        self.assertTrue(alt["source"]["reference"])  # the source reference is shown

    def test_alternative_drug_with_its_own_interaction_is_warned_and_flagged(self):
        result = self.pair("Warfarin", "Ibuprofen")  # alternative is acetaminophen
        alt = result["alternatives"][0]  # the alternative
        self.assertEqual(alt["type"], "alternative_drug")  # typed as a drug
        flagged = [f for f in alt["cross_check"] if f["interactions"]]  # entries with their own interactions
        self.assertEqual(flagged[0]["alternative"], "Acetaminophen")  # it is acetaminophen
        self.assertEqual(flagged[0]["interactions"][0]["record_id"], "ddi:69")  # warfarin + acetaminophen, ddi:69
        self.assertIn("Warning", flagged[0]["warning"])  # a warning is shown
        self.assertTrue(result["needs_review"])  # and the pair is routed to review
        self.assertIn(rules.RULE_ALTERNATIVE_CROSS_CHECK, result["rules_fired"])  # the cross-check rule fired

    def test_alternative_is_checked_against_the_users_other_medications(self):
        # Venlafaxine + ibuprofen suggests acetaminophen; the user also takes warfarin.
        out = self.checker.check(["Venlafaxine", "Warfarin"], "Ibuprofen")  # two current drugs
        result = next(r for r in out["results"] if r["pair"][0] == "Venlafaxine")  # the venlafaxine result
        hits = [i for f in result["alternatives"][0]["cross_check"] for i in f["interactions"]]  # all cross-check hits
        # One hit must be with warfarin, found among the user's current medications.
        self.assertIn(("Warfarin", "current_medication"), [(h["with"], h["scope"]) for h in hits])

    def test_alternative_not_in_the_data_is_not_called_safe(self):
        out = self.pair("Warfarin", "Ibuprofen")["alternatives"][0]["cross_check"]  # cross-check entries
        missing = next(f for f in out if not f["in_data"])  # the synonym that is not in our data
        self.assertIn("does not mean it is safe", missing["note"])  # its note refuses to call it safe


# Tests the "which drugs should I avoid with X" list.
class TestAvoidList(RealDbTestCase):
    def test_warfarin_avoid_list_is_ordered_by_severity_and_unfiltered_by_class(self):
        out = self.checker.avoid("warfarin")  # the avoid list
        listed = out["avoid"]  # its entries
        ranks = [{"Major": 0, "Moderate": 1}[e["severity"]] for e in listed]  # severity as a number
        self.assertEqual(ranks, sorted(ranks))  # already in order, Major first
        names = [e["drug"] for e in listed]  # the drug names
        self.assertTrue({"Ibuprofen", "Fluconazole", "Amiodarone", "Fluoxetine"} <= set(names))  # key drugs are present
        self.assertNotEqual(names[0], "Acetaminophen")  # a Moderate edge, listed after every Major
        self.assertLess(names.index("Ibuprofen"), names.index("Acetaminophen"))  # Major ibuprofen before Moderate acetaminophen
        self.assertIn("classes are not in our data", " ".join(out["notes"]))  # the class caveat is stated
        self.assertIn(rules.RULE_AVOID_LIST, out["rules_fired"])  # the avoid-list rule fired

    def test_minor_edges_are_not_listed(self):
        graph = self.checker.graph  # the knowledge graph
        for entry in rules.avoid_list(graph, graph.drug_id("Warfarin")):  # each entry in warfarin's list
            self.assertIn(entry["severity"], ("Major", "Moderate"))  # only Major or Moderate appear

    def test_unknown_drug_gets_suggestions_instead_of_a_list(self):
        out = self.checker.avoid("Warfrin")  # a typo
        self.assertEqual(out["status"], "unknown_drug")  # not auto-resolved
        self.assertIn("Warfarin", out["did_you_mean"])  # a suggestion instead


# Tests the single-drug view.
class TestDrugView(RealDbTestCase):
    def test_returns_name_and_interactions_only(self):
        out = self.checker.drug("warfarin")  # look up one drug
        self.assertEqual(out["drug"]["name"], "Warfarin")  # its canonical name
        self.assertEqual(out["interactions"][0]["severity"], "Major")  # interactions are listed worst first
        self.assertTrue(all(v == "not available in our data" for v in out["not_available"].values()))  # monograph fields absent
        self.assertNotIn("uses", out["drug"])  # no invented "uses" field


# Tests duplicate records and conflicts using small synthetic databases.
class TestDuplicatesAndConflictsSynthetic(unittest.TestCase):
    def build(self, rows):  # helper: make a checker over a synthetic database
        tmp = tempfile.TemporaryDirectory()  # temporary folder
        self.addCleanup(tmp.cleanup)  # delete it afterwards
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), rows), ingest.CORE_TABLES)  # build and open read-only
        self.addCleanup(conn.close)  # close it afterwards
        return Checker(conn)  # the checker over it

    def test_a_duplicated_record_changes_nothing(self):
        base = [ddi_row("Warfarin", "Ibuprofen", "Major"), ddi_row("Warfarin", "Fluconazole", "Moderate")]  # two rows
        single = self.build(base)  # database without a duplicate
        doubled = self.build(base + [dict(base[0])])  # database with the first row repeated exactly
        a, b = single.check(["Warfarin"], "Ibuprofen")["results"][0], doubled.check(["Warfarin"], "Ibuprofen")["results"][0]  # both results
        for key in ("status", "severity", "severity_conflict", "severities"):  # the headline fields
            self.assertEqual(a[key], b[key])  # must be identical
        self.assertEqual(len(b["records"]), 1)  # the duplicate is collapsed to one record
        self.assertEqual(b["duplicate_record_ids"], ["ddi:3"])  # and the dropped id is listed
        self.assertEqual(single.avoid("Warfarin")["avoid"], doubled.avoid("Warfarin")["avoid"])  # avoid list is unaffected

    def test_same_content_in_reversed_order_is_still_a_duplicate(self):
        checker = self.build([ddi_row("A", "B", "Major"), ddi_row("B", "A", "Major")])  # same row with drugs swapped
        result = checker.check(["A"], "B")["results"][0]  # check the pair
        self.assertFalse(result["severity_conflict"])  # no conflict, it is one fact
        self.assertEqual(len(result["records"]), 1)  # collapsed to one record

    def test_minor_only_pair_is_documented_and_headlined(self):
        checker = self.build([ddi_row("A", "B", "Minor")])  # a single Minor row
        result = checker.check(["A"], "B")["results"][0]  # check the pair
        self.assertEqual((result["status"], result["severity"]), (DOCUMENTED, "Minor"))  # documented with Minor headline

    def test_severity_precedence_rule(self):
        rows = [{"id": "ddi:1", "severity": "Minor"}, {"id": "ddi:2", "severity": "Major"}, {"id": "ddi:3", "severity": "Moderate"}]  # unordered rows
        headline, ordered = rules.severity_precedence(rows)  # apply the rule
        self.assertEqual(headline["id"], "ddi:2")  # the Major row is the headline
        self.assertEqual([r["severity"] for r in ordered], ["Major", "Moderate", "Minor"])  # sorted most severe first
        self.assertEqual(rules.repeated_pair_conflict(rows)["severities"], ["Major", "Moderate", "Minor"])  # conflict lists all
        self.assertEqual(rules.severity_precedence([]), (None, []))  # empty input gives no headline

    def test_graph_is_built_from_interactions_only(self):
        graph = self.build([ddi_row("A", "B", "Major"), ddi_row("B", "C", "Minor")]).graph  # a three-drug graph
        self.assertEqual(graph.edge_count(), 2)  # two edges, one per interaction
        self.assertEqual([graph.name(i) for i in graph.neighbors(graph.drug_id("B"))], ["A", "C"])  # B touches A and C


if __name__ == "__main__":  # only when run as a script
    unittest.main()  # run the tests in this file


# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the rule layer and the deterministic main flow in checker.py against the real database and small
# synthetic ones: pair lookup in either order, "not found" never reading as safe, repeated pairs headlining the
# highest severity, Major-first sorting, dose echoing, safer-alternative cross-checks, the avoid list and the drug view.
# ASSIGNMENT TOOL: knowledge graph with rule-based reasoning (the core engine), including severity precedence,
# repeated-pair conflict detection and the alternative cross-check rules.
# ---------------------------------------------------------------------------
