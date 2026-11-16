"""Tests for the rule layer and the deterministic checking flow."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from ddi_backend import ingest, load_data, rules
from ddi_backend.checker import DOCUMENTED, NO_DOCUMENTED, Checker
from ddi_backend.common import NOT_SAFE_NOTE
from ddi_backend.graph import Graph
from tests import support
from tests.support import ddi_row

# The five dataset rows from the RAG handout, confirmed against the real database.
FIXTURE_ROWS = {
    "ddi:1": ("Major", "Warfarin", "Ibuprofen"),
    "ddi:12": ("Major", "Warfarin", "Fluconazole"),
    "ddi:20": ("Major", "Warfarin", "Amiodarone"),
    "ddi:26": ("Major", "Warfarin", "Fluoxetine"),
    "ddi:73": ("Moderate", "Venlafaxine", "Ibuprofen"),
}


class RealDbTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = support.open_real()
        cls.checker = Checker(cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def pair(self, a, b, **kwargs):
        return self.checker.check([a], b, **kwargs)["results"][0]


class TestFixturesMatchTheDatabase(RealDbTestCase):
    def test_handout_rows(self):
        records = {r["id"]: r for r in ingest.load_interactions(self.conn)}
        for record_id, (severity, a, b) in FIXTURE_ROWS.items():
            with self.subTest(record_id=record_id):
                r = records[record_id]
                self.assertEqual((r["severity"], r["drug_a"], r["drug_b"]), (severity, a, b))

    def test_contradictory_warfarin_rows(self):
        # ddi:1 names acetaminophen as the safer alternative, yet warfarin + acetaminophen has its own row.
        records = {r["id"]: r for r in ingest.load_interactions(self.conn)}
        self.assertTrue(records["ddi:1"]["safer_alternative"].startswith("Acetaminophen"))
        self.assertEqual((records["ddi:69"]["drug_a"], records["ddi:69"]["drug_b"]), ("Warfarin", "Acetaminophen"))


class TestPairLookup(RealDbTestCase):
    def test_either_order_returns_the_same_rows(self):
        forward, backward = self.pair("Warfarin", "Ibuprofen"), self.pair("ibuprofen", "WARFARIN")
        self.assertEqual(forward["status"], DOCUMENTED)
        self.assertEqual([r["id"] for r in forward["records"]], [r["id"] for r in backward["records"]])
        self.assertEqual([r["id"] for r in forward["records"]], ["ddi:1"])

    def test_record_comes_back_whole(self):
        result = self.pair("Warfarin", "Amiodarone")
        record = result["records"][0]
        for field in ("severity", "mechanism", "effect", "safer_alternative", "rationale", "reference"):
            self.assertTrue(record[field], field)
        self.assertEqual(result["severity"], "Major")
        self.assertEqual(result["effect"], record["effect"])
        self.assertEqual(result["source"]["record_ids"], ["ddi:20"])
        self.assertEqual(result["source"]["citations"][0]["reference"], record["reference"])

    def test_lookup_agrees_with_the_graph(self):
        graph = self.checker.graph
        a, b = graph.drug_id("Warfarin"), graph.drug_id("Fluconazole")
        self.assertEqual([r["id"] for r in ingest.pair_records(self.conn, a, b)],
                         [r["id"] for r in graph.pair(a, b)])

    def test_pair_with_no_row_is_not_called_safe(self):
        result = self.pair("Warfarin", "Docusate")
        self.assertEqual(result["status"], NO_DOCUMENTED)
        self.assertIsNone(result["severity"])
        self.assertIn(NOT_SAFE_NOTE, result["notes"])
        self.assertIn(rules.RULE_ABSENCE_NOT_SAFETY, result["rules_fired"])
        self.assertNotIn("safe", result.get("message", "").replace("not mean the combination is safe", ""))

    def test_drugs_missing_from_the_data_are_unknown_not_safe(self):
        result = self.pair("Cabergoline", "Rucaparib")
        self.assertEqual(result["status"], "unknown_drug")
        self.assertEqual(len(result["unresolved"]), 2)
        self.assertIn(NOT_SAFE_NOTE, result["notes"])

    def test_abbreviation_gets_a_question_not_a_guess(self):
        result = self.pair("MS", "Warfarin")
        self.assertEqual(result["status"], "ambiguous_name")
        self.assertIn("Which medication", result["message"])


class TestRepeatedPair(RealDbTestCase):
    def test_clopidogrel_omeprazole_headlines_major_and_flags_conflict(self):
        result = self.pair("Clopidogrel", "Omeprazole")
        self.assertEqual(result["severity"], "Major")
        self.assertTrue(result["severity_conflict"])
        self.assertEqual(result["severities"], ["Major", "Moderate"])
        self.assertEqual(len(result["records"]), 2)  # every row is returned
        self.assertTrue(result["needs_review"])
        self.assertTrue(any(reason.startswith("severity_conflict") for reason in result["review_reasons"]))
        self.assertIn(rules.RULE_REPEATED_PAIR, result["rules_fired"])

    def test_every_repeated_pair_in_the_data_reports_its_highest_severity(self):
        order = {"Major": 0, "Moderate": 1, "Minor": 2}
        repeated = load_data.find_repeated_pairs(self.conn)
        self.assertEqual(len(repeated), 8)
        for a, b, severities in repeated:
            found = severities.split(" / ")
            result = self.pair(a, b)
            with self.subTest(pair=(a, b)):
                self.assertEqual(result["severity"], min(found, key=order.get))
                self.assertEqual(result["severity_conflict"], len(set(found)) > 1)
                self.assertEqual(len(result["records"]), len(found))


class TestMainFlow(RealDbTestCase):
    def test_major_interactions_sort_first(self):
        out = self.checker.check(["Venlafaxine", "Docusate", "Warfarin"], "Ibuprofen")
        statuses = [(r["pair"][0], r["status"], r["severity"]) for r in out["results"]]
        self.assertEqual(statuses, [
            ("Warfarin", DOCUMENTED, "Major"),
            ("Venlafaxine", DOCUMENTED, "Moderate"),
            ("Docusate", NO_DOCUMENTED, None),
        ])
        self.assertEqual(out["summary"]["major_documented"], 1)

    def test_current_medications_can_also_be_checked_against_each_other(self):
        without = self.checker.check(["Warfarin", "Fluconazole"], "Docusate")
        with_pairs = self.checker.check(["Warfarin", "Fluconazole"], "Docusate", include_current_pairs=True)
        self.assertEqual(len(without["results"]), 2)
        self.assertEqual(len(with_pairs["results"]), 3)
        among = [r for r in with_pairs["results"] if r["kind"] == "current_vs_current"]
        self.assertEqual((among[0]["status"], among[0]["severity"]), (DOCUMENTED, "Major"))

    def test_doses_and_details_are_echoed_never_recommended(self):
        out = self.checker.check([{"name": "Warfarin 5 mg", "details": "evening"}], {"name": "Ibuprofen", "dose": "400 mg"})
        self.assertEqual(out["inputs"]["current"][0]["dose"], "5 mg")
        self.assertEqual(out["inputs"]["current"][0]["details"], "evening")
        self.assertEqual(out["inputs"]["new"]["dose"], "400 mg")
        self.assertEqual(out["results"][0]["doses"], {"Warfarin": "5 mg", "Ibuprofen": "400 mg"})
        self.assertNotIn("recommended_dose", str(out))

    def test_medication_listed_twice_or_as_the_new_drug_is_not_self_checked(self):
        out = self.checker.check(["Warfarin", "warfarin 5 mg", "Ibuprofen"], "Ibuprofen")
        self.assertEqual([r["pair"] for r in out["results"]], [["Warfarin", "Ibuprofen"]])
        self.assertEqual(len(out["notes"]), 2)

    def test_unknown_new_drug_never_reads_as_safe(self):
        out = self.checker.check(["Warfarin"], "Cabergoline")
        self.assertEqual(out["results"][0]["status"], "unknown_drug")
        self.assertEqual(out["inputs"]["new"]["status"], "unknown_drug")

    def test_response_carries_the_disclaimer_and_the_rules_that_fired(self):
        out = self.checker.check(["Warfarin"], "Ibuprofen")
        self.assertIn("not medical advice", out["disclaimer"])
        for rule in (rules.RULE_EXACT_NAME, rules.RULE_PAIR_LOOKUP, rules.RULE_SEVERITY_PRECEDENCE):
            self.assertIn(rule, out["rules_fired"])

    def test_database_is_unchanged_by_checking(self):
        path = support.real_db_path()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.checker.check(["Warfarin"], "Ibuprofen", include_current_pairs=True)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


class TestAlternatives(RealDbTestCase):
    def test_management_action_is_shown_as_written(self):
        alt = self.pair("Warfarin", "Amiodarone")["alternatives"][0]
        self.assertEqual((alt["type"], alt["text"]), ("management_action", "Reduce warfarin dose by 30-50%"))
        self.assertEqual(alt["cross_check"], [])
        self.assertTrue(alt["source"]["reference"])

    def test_alternative_drug_with_its_own_interaction_is_warned_and_flagged(self):
        result = self.pair("Warfarin", "Ibuprofen")
        alt = result["alternatives"][0]
        self.assertEqual(alt["type"], "alternative_drug")
        flagged = [f for f in alt["cross_check"] if f["interactions"]]
        self.assertEqual(flagged[0]["alternative"], "Acetaminophen")
        self.assertEqual(flagged[0]["interactions"][0]["record_id"], "ddi:69")
        self.assertIn("Warning", flagged[0]["warning"])
        self.assertTrue(result["needs_review"])
        self.assertIn(rules.RULE_ALTERNATIVE_CROSS_CHECK, result["rules_fired"])

    def test_alternative_is_checked_against_the_users_other_medications(self):
        # Venlafaxine + ibuprofen suggests acetaminophen; the user also takes warfarin.
        out = self.checker.check(["Venlafaxine", "Warfarin"], "Ibuprofen")
        result = next(r for r in out["results"] if r["pair"][0] == "Venlafaxine")
        hits = [i for f in result["alternatives"][0]["cross_check"] for i in f["interactions"]]
        self.assertIn(("Warfarin", "current_medication"), [(h["with"], h["scope"]) for h in hits])

    def test_alternative_not_in_the_data_is_not_called_safe(self):
        out = self.pair("Warfarin", "Ibuprofen")["alternatives"][0]["cross_check"]
        missing = next(f for f in out if not f["in_data"])
        self.assertIn("does not mean it is safe", missing["note"])


class TestAvoidList(RealDbTestCase):
    def test_warfarin_avoid_list_is_ordered_by_severity_and_unfiltered_by_class(self):
        out = self.checker.avoid("warfarin")
        listed = out["avoid"]
        ranks = [{"Major": 0, "Moderate": 1}[e["severity"]] for e in listed]
        self.assertEqual(ranks, sorted(ranks))
        names = [e["drug"] for e in listed]
        self.assertTrue({"Ibuprofen", "Fluconazole", "Amiodarone", "Fluoxetine"} <= set(names))
        self.assertNotEqual(names[0], "Acetaminophen")  # a Moderate edge, listed after every Major
        self.assertLess(names.index("Ibuprofen"), names.index("Acetaminophen"))
        self.assertIn("classes are not in our data", " ".join(out["notes"]))
        self.assertIn(rules.RULE_AVOID_LIST, out["rules_fired"])

    def test_minor_edges_are_not_listed(self):
        graph = self.checker.graph
        for entry in rules.avoid_list(graph, graph.drug_id("Warfarin")):
            self.assertIn(entry["severity"], ("Major", "Moderate"))

    def test_unknown_drug_gets_suggestions_instead_of_a_list(self):
        out = self.checker.avoid("Warfrin")
        self.assertEqual(out["status"], "unknown_drug")
        self.assertIn("Warfarin", out["did_you_mean"])


class TestDrugView(RealDbTestCase):
    def test_returns_name_and_interactions_only(self):
        out = self.checker.drug("warfarin")
        self.assertEqual(out["drug"]["name"], "Warfarin")
        self.assertEqual(out["interactions"][0]["severity"], "Major")
        self.assertTrue(all(v == "not available in our data" for v in out["not_available"].values()))
        self.assertNotIn("uses", out["drug"])


class TestDuplicatesAndConflictsSynthetic(unittest.TestCase):
    def build(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = ingest.open_readonly(support.build_db(Path(tmp.name), rows), ingest.CORE_TABLES)
        self.addCleanup(conn.close)
        return Checker(conn)

    def test_a_duplicated_record_changes_nothing(self):
        base = [ddi_row("Warfarin", "Ibuprofen", "Major"), ddi_row("Warfarin", "Fluconazole", "Moderate")]
        single = self.build(base)
        doubled = self.build(base + [dict(base[0])])
        a, b = single.check(["Warfarin"], "Ibuprofen")["results"][0], doubled.check(["Warfarin"], "Ibuprofen")["results"][0]
        for key in ("status", "severity", "severity_conflict", "severities"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(len(b["records"]), 1)
        self.assertEqual(b["duplicate_record_ids"], ["ddi:3"])
        self.assertEqual(single.avoid("Warfarin")["avoid"], doubled.avoid("Warfarin")["avoid"])

    def test_same_content_in_reversed_order_is_still_a_duplicate(self):
        checker = self.build([ddi_row("A", "B", "Major"), ddi_row("B", "A", "Major")])
        result = checker.check(["A"], "B")["results"][0]
        self.assertFalse(result["severity_conflict"])
        self.assertEqual(len(result["records"]), 1)

    def test_minor_only_pair_is_documented_and_headlined(self):
        checker = self.build([ddi_row("A", "B", "Minor")])
        result = checker.check(["A"], "B")["results"][0]
        self.assertEqual((result["status"], result["severity"]), (DOCUMENTED, "Minor"))

    def test_severity_precedence_rule(self):
        rows = [{"id": "ddi:1", "severity": "Minor"}, {"id": "ddi:2", "severity": "Major"}, {"id": "ddi:3", "severity": "Moderate"}]
        headline, ordered = rules.severity_precedence(rows)
        self.assertEqual(headline["id"], "ddi:2")
        self.assertEqual([r["severity"] for r in ordered], ["Major", "Moderate", "Minor"])
        self.assertEqual(rules.repeated_pair_conflict(rows)["severities"], ["Major", "Moderate", "Minor"])
        self.assertEqual(rules.severity_precedence([]), (None, []))

    def test_graph_is_built_from_interactions_only(self):
        graph = self.build([ddi_row("A", "B", "Major"), ddi_row("B", "C", "Minor")]).graph
        self.assertEqual(graph.edge_count(), 2)
        self.assertEqual([graph.name(i) for i in graph.neighbors(graph.drug_id("B"))], ["A", "C"])


if __name__ == "__main__":
    unittest.main()
