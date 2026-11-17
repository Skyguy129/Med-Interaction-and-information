# Docstring describing the module: tests for the HTTP API, run in-process with no server or network.
"""Tests for the HTTP API (FastAPI TestClient; no server, no network)."""

import hashlib  # fingerprints the database file to prove it is never written
import json  # turns the stub model's reply into a JSON string
import unittest  # the test framework

try:  # FastAPI is optional, so importing its test client may fail
    from fastapi.testclient import TestClient  # in-process client that calls the app without a server
except ImportError:  # the CLI works without the web framework
    TestClient = None  # remember that it is missing so the tests can be skipped

from ddi_backend.api import create_app  # builds the FastAPI application under test
from ddi_backend.llm import Provider  # the language model provider interface the stub implements
from tests import support  # shared database builders

# A decorator that skips a test or class when FastAPI/httpx are not installed.
needs_fastapi = unittest.skipUnless(TestClient is not None, "fastapi/httpx are not installed")


# A scripted stand-in for the language model: always returns the same reply and counts how often it was called.
class StubProvider(Provider):
    def __init__(self, reply):
        self.reply, self.calls = reply, 0  # store the canned reply and start the call counter at zero

    def complete(self, messages):  # the one method the provider interface requires
        self.calls += 1  # record that the model was asked
        return json.dumps(self.reply)  # return the canned reply as JSON text


@needs_fastapi  # skip all tests in the class if FastAPI is missing
# Base class that creates a fresh app, client and cleanup for every test.
class ApiTestCase(unittest.TestCase):
    provider = None  # no language model unless a subclass supplies one

    def setUp(self):
        # Build the app over the full (DDI + HPO) database with throwaway in-memory state.
        self.app = create_app(db_path=support.real_full_db_path(), in_memory_state=True, provider=self.provider)
        self.client = TestClient(self.app)  # an in-process HTTP client for that app
        self.addCleanup(self.app.state.backend.close)  # close the backend's connections when the test ends

    def stage(self, **body):  # helper that posts a review-queue request
        payload = {"drug_pair": ["Warfarin", "Ibuprofen"], "request_id": "req-1", "approved": True}  # a valid default
        payload.update(body)  # let each test override fields
        return self.client.post("/review-queue/requests", json=payload)  # send it and return the response


# Tests for the read-only routes (health, check, drugs, avoid, ask).
class TestReadRoutes(ApiTestCase):
    def test_health_lists_the_tiers(self):
        data = self.client.get("/health").json()  # ask which tiers are available
        self.assertEqual(data["status"], "ok")  # the service is up
        self.assertTrue(data["tiers"]["core"])  # the deterministic core is available
        self.assertFalse(data["tiers"]["language_model"])  # no model was supplied
        self.assertIn("not medical advice", data["disclaimer"])  # the disclaimer is included

    def test_check_returns_one_result_per_pair_major_first(self):
        # A list of current drugs (one with a dose), a new drug, and a brief detail level.
        body = {"current": ["Docusate", {"name": "Warfarin", "dose": "5 mg"}], "new": "Ibuprofen",
                "options": {"detail_level": "brief"}}
        response = self.client.post("/check", json=body)  # call the check route
        self.assertEqual(response.status_code, 200)  # success
        data = response.json()  # the response body
        # Warfarin + Ibuprofen is Major, so it must come before the pair with no record.
        self.assertEqual([r["status"] for r in data["results"]], ["documented_interaction", "no_documented_interaction"])
        first = data["results"][0]  # the Major pair
        for key in ("severity", "mechanism", "effect", "alternatives", "rules_fired", "source", "text"):  # needed fields
            self.assertIn(key, first)  # each one is present
        self.assertEqual(first["severity"], "Major")  # headline severity
        self.assertEqual(first["doses"], {"Warfarin": "5 mg", "Ibuprofen": None})  # the dose is echoed back only
        self.assertIn("not medical advice", data["disclaimer"])  # disclaimer present
        self.assertIn("does not mean the combination is safe", data["results"][1]["text"])  # no row is not called safe

    def test_check_options(self):
        # Ask for the current drugs to be checked against each other as well.
        body = {"current": ["Warfarin", "Fluconazole"], "new": "Docusate", "options": {"include_current_pairs": True}}
        self.assertEqual(len(self.client.post("/check", json=body).json()["results"]), 3)  # 2 new pairs + 1 current pair

    def test_check_validates_the_request(self):
        self.assertEqual(self.client.post("/check", json={"current": ["Warfarin"]}).status_code, 422)  # "new" missing
        bad = {"current": [], "new": "X", "options": {"detail_level": "enormous"}}  # invalid detail level
        self.assertEqual(self.client.post("/check", json=bad).status_code, 422)  # rejected

    def test_unknown_names_are_reported_not_called_safe(self):
        data = self.client.post("/check", json={"current": ["Cabergoline"], "new": "Rucaparib"}).json()  # not in data
        self.assertEqual(data["results"][0]["status"], "unknown_drug")  # reported as unknown

    def test_drug_route(self):
        data = self.client.get("/drugs/warfarin").json()  # case-insensitive lookup
        self.assertEqual(data["drug"]["name"], "Warfarin")  # the canonical name
        self.assertEqual(data["interactions"][0]["severity"], "Major")  # the worst interaction comes first
        self.assertEqual(data["not_available"]["side_effects"], "not available in our data")  # no monograph data

    def test_drug_route_unknown_and_ambiguous(self):
        unknown = self.client.get("/drugs/Warfrin")  # a typo
        self.assertEqual(unknown.status_code, 404)  # not found, never auto-corrected
        self.assertIn("Warfarin", unknown.json()["did_you_mean"])  # but a suggestion is offered
        self.assertEqual(self.client.get("/drugs/Valpro").status_code, 409)  # a partial name matching several is a conflict

    def test_avoid_route(self):
        data = self.client.get("/avoid/warfarin").json()  # the avoid list for warfarin
        self.assertEqual(data["avoid"][0]["severity"], "Major")  # Major entries come first
        self.assertIn("classes are not in our data", " ".join(data["notes"]))  # the class caveat is stated
        self.assertEqual(self.client.get("/avoid/Advil").status_code, 404)  # a brand name is unknown

    def test_ask_route(self):
        # The warfarin question from the handout should be understood as an "avoid" question.
        data = self.client.post("/ask", json={"question": "Which pain reliever should I not take with warfarin?"}).json()
        self.assertEqual(data["intent"], "avoid")  # classified as an avoid-list question
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)  # empty question rejected

    def test_the_api_serves_no_user_interface(self):
        for route in self.app.routes:  # look at every registered route
            self.assertNotIn("static", getattr(route, "name", ""))  # none serves static (UI) files
        response = self.client.get("/health")  # fetch a route
        self.assertTrue(response.headers["content-type"].startswith("application/json"))  # it returns JSON, not pages


# Tests for the review-queue routes and the trace route.
class TestReviewRoutes(ApiTestCase):
    def test_request_needs_a_request_id_and_explicit_approval(self):
        # Missing request_id is rejected.
        self.assertEqual(self.client.post("/review-queue/requests",
                                          json={"drug_pair": ["Warfarin", "Ibuprofen"], "approved": True}).status_code, 422)
        # Missing approved is rejected.
        self.assertEqual(self.client.post("/review-queue/requests",
                                          json={"drug_pair": ["Warfarin", "Ibuprofen"], "request_id": "r"}).status_code, 422)
        self.assertEqual(self.client.get("/review-queue").json()["review_queue"], [])  # nothing was written

    def test_declined_approval_writes_nothing(self):
        response = self.stage(approved=False)  # the user declines
        self.assertEqual(response.status_code, 403)  # forbidden
        self.assertEqual(response.json()["status"], "approval_required")  # the gate that stopped it
        self.assertFalse(response.json()["action_occurred"])  # the action did not happen
        self.assertEqual(self.client.get("/review-queue").json()["review_queue"], [])  # the queue is still empty

    def test_approved_request_is_staged_and_listed_as_simulated(self):
        response = self.stage()  # a valid, approved request
        self.assertEqual(response.status_code, 201)  # created
        body = response.json()  # the response body
        self.assertEqual(body["status"], "staged_in_simulation")  # staged
        self.assertTrue(body["simulated"])  # labelled simulated
        queue = self.client.get("/review-queue").json()  # read the queue back
        self.assertTrue(queue["simulated"])  # the queue is labelled simulated
        self.assertIn("SIMULATED", queue["note"])  # with an explicit note
        self.assertEqual(queue["review_queue"][0]["evidence_id"], "ddi:1")  # holds the evidence id
        self.assertEqual(queue["review_queue"][0]["purpose"], "clinician review only")  # and the purpose

    def test_gate_failures_map_to_http_errors_and_write_nothing(self):
        self.assertEqual(self.stage(target="medication_order").status_code, 403)  # out of scope is forbidden
        self.assertEqual(self.stage(drug_pair=["Warfarin", "NoSuchDrug"]).status_code, 404)  # missing evidence is 404
        self.assertEqual(self.stage().status_code, 201)  # a valid request succeeds
        self.assertEqual(self.stage().status_code, 409)  # the request ID is used
        self.assertEqual(len(self.client.get("/review-queue").json()["review_queue"]), 1)  # only the success is queued

    def test_trace_route(self):
        self.stage(request_id="req-9", approved=False)  # first attempt, denied
        self.stage(request_id="req-9")  # second attempt, approved
        data = self.client.get("/trace/req-9").json()  # read the trace for that id
        self.assertEqual([a["gate"] for a in data["attempts"]], ["approval_required", "allowed"])  # both attempts' gates
        self.assertEqual([a["action_occurred"] for a in data["attempts"]], [False, True])  # only the second happened
        self.assertIn("therefore the action did occur", data["text"])  # readable text for the success
        self.assertEqual(self.client.get("/trace/never-seen").status_code, 404)  # unknown id is 404

    def test_the_interaction_database_is_never_written(self):
        path = support.real_full_db_path()  # the database file
        before = hashlib.sha256(path.read_bytes()).hexdigest()  # fingerprint before
        self.stage()  # stage a request (writes only to the state file)
        self.client.post("/check", json={"current": ["Warfarin"], "new": "Ibuprofen"})  # run a check
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)  # file unchanged


# Tests the "explain" option when no language model is configured.
class TestExplainOption(ApiTestCase):
    def test_deterministic_only_when_no_provider(self):
        body = {"current": ["Warfarin"], "new": "Fluconazole", "options": {"explain": True}}  # ask for an explanation
        data = self.client.post("/check", json=body).json()  # call check
        self.assertEqual(data["explanations"][0]["generated_by"], "template")  # the template composer wrote it
        self.assertEqual(data["explanations"][0]["records"][0]["id"], "ddi:12")  # next to its source record
        self.assertFalse(data["settings"]["language_model"])  # the model is reported as off


@needs_fastapi  # skip if FastAPI is missing
# Tests the "explain" option with a scripted language model.
class TestExplainWithAProvider(unittest.TestCase):
    def test_validated_model_text_sits_next_to_the_source_record(self):
        record_claims = [  # the claims the stub model will make, each pointing at a record id
            {"record_id": "ddi:12", "field": "severity", "value": "Major"},
        ]
        conn = support.open_real()  # open the real database to read the true record
        # Load every interaction and pick ddi:12 (the import is done inline with __import__).
        record = next(r for r in __import__("ddi_backend.ingest", fromlist=["x"]).load_interactions(conn) if r["id"] == "ddi:12")
        conn.close()  # done with this connection
        # Add claims for the mechanism and effect that match the stored record exactly.
        record_claims += [{"record_id": "ddi:12", "field": f, "value": record[f]} for f in ("mechanism", "effect")]
        # The stub's structured reply: a summary using only record facts, the claims, and no clarifying question.
        provider = StubProvider({"summary": f"Warfarin with Fluconazole is documented as a Major interaction. {record['effect']}.",
                                 "claims": record_claims, "clarifying_question": None})
        app = create_app(db_path=support.real_full_db_path(), in_memory_state=True, provider=provider)  # app with the stub
        self.addCleanup(app.state.backend.close)  # close it when done
        client = TestClient(app)  # in-process client
        data = client.post("/check", json={"current": ["Warfarin"], "new": "Fluconazole",
                                           "options": {"explain": True, "detail_level": "standard"}}).json()  # explain
        explanation = data["explanations"][0]  # the first explanation
        self.assertEqual(explanation["generated_by"], "language_model")  # the validated model text was used
        self.assertEqual(explanation["records"][0]["id"], "ddi:12")  # shown next to its source record
        self.assertEqual(provider.calls, 1)  # the model was called exactly once
        self.assertTrue(data["settings"]["language_model"])  # the model is reported as on


if __name__ == "__main__":  # only when run as a script
    unittest.main()  # run the tests in this file


# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the HTTP API using FastAPI's in-process test client: health, check, drug, avoid and ask routes,
# input validation, the review-queue and trace routes with their HTTP status mapping, the guarantee that the
# interaction database is never written, and the optional "explain" option with and without a (stub) language model.
# ASSIGNMENT TOOL: supporting infrastructure (not one of the assignment tools) that exposes all of them over HTTP:
# the knowledge graph and rules, the guarded agent with its simulated review queue, and the language model's explain mode.
# ---------------------------------------------------------------------------
