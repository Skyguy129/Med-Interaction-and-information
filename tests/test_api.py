"""Tests for the HTTP API (FastAPI TestClient; no server, no network)."""

import hashlib
import json
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # the CLI works without the web framework
    TestClient = None

from ddi_backend.api import create_app
from ddi_backend.llm import Provider
from tests import support

needs_fastapi = unittest.skipUnless(TestClient is not None, "fastapi/httpx are not installed")


class StubProvider(Provider):
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, messages):
        self.calls += 1
        return json.dumps(self.reply)


@needs_fastapi
class ApiTestCase(unittest.TestCase):
    provider = None

    def setUp(self):
        self.app = create_app(db_path=support.real_full_db_path(), in_memory_state=True, provider=self.provider)
        self.client = TestClient(self.app)
        self.addCleanup(self.app.state.backend.close)

    def stage(self, **body):
        payload = {"drug_pair": ["Warfarin", "Ibuprofen"], "request_id": "req-1", "approved": True}
        payload.update(body)
        return self.client.post("/review-queue/requests", json=payload)


class TestReadRoutes(ApiTestCase):
    def test_health_lists_the_tiers(self):
        data = self.client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["tiers"]["core"])
        self.assertFalse(data["tiers"]["language_model"])
        self.assertIn("not medical advice", data["disclaimer"])

    def test_check_returns_one_result_per_pair_major_first(self):
        body = {"current": ["Docusate", {"name": "Warfarin", "dose": "5 mg"}], "new": "Ibuprofen",
                "options": {"detail_level": "brief"}}
        response = self.client.post("/check", json=body)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual([r["status"] for r in data["results"]], ["documented_interaction", "no_documented_interaction"])
        first = data["results"][0]
        for key in ("severity", "mechanism", "effect", "alternatives", "rules_fired", "source", "text"):
            self.assertIn(key, first)
        self.assertEqual(first["severity"], "Major")
        self.assertEqual(first["doses"], {"Warfarin": "5 mg", "Ibuprofen": None})
        self.assertIn("not medical advice", data["disclaimer"])
        self.assertIn("does not mean the combination is safe", data["results"][1]["text"])

    def test_check_options(self):
        body = {"current": ["Warfarin", "Fluconazole"], "new": "Docusate", "options": {"include_current_pairs": True}}
        self.assertEqual(len(self.client.post("/check", json=body).json()["results"]), 3)

    def test_check_validates_the_request(self):
        self.assertEqual(self.client.post("/check", json={"current": ["Warfarin"]}).status_code, 422)
        bad = {"current": [], "new": "X", "options": {"detail_level": "enormous"}}
        self.assertEqual(self.client.post("/check", json=bad).status_code, 422)

    def test_unknown_names_are_reported_not_called_safe(self):
        data = self.client.post("/check", json={"current": ["Cabergoline"], "new": "Rucaparib"}).json()
        self.assertEqual(data["results"][0]["status"], "unknown_drug")

    def test_drug_route(self):
        data = self.client.get("/drugs/warfarin").json()
        self.assertEqual(data["drug"]["name"], "Warfarin")
        self.assertEqual(data["interactions"][0]["severity"], "Major")
        self.assertEqual(data["not_available"]["side_effects"], "not available in our data")

    def test_drug_route_unknown_and_ambiguous(self):
        unknown = self.client.get("/drugs/Warfrin")
        self.assertEqual(unknown.status_code, 404)
        self.assertIn("Warfarin", unknown.json()["did_you_mean"])
        self.assertEqual(self.client.get("/drugs/Valpro").status_code, 409)

    def test_avoid_route(self):
        data = self.client.get("/avoid/warfarin").json()
        self.assertEqual(data["avoid"][0]["severity"], "Major")
        self.assertIn("classes are not in our data", " ".join(data["notes"]))
        self.assertEqual(self.client.get("/avoid/Advil").status_code, 404)

    def test_ask_route(self):
        data = self.client.post("/ask", json={"question": "Which pain reliever should I not take with warfarin?"}).json()
        self.assertEqual(data["intent"], "avoid")
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)

    def test_the_api_serves_no_user_interface(self):
        for route in self.app.routes:
            self.assertNotIn("static", getattr(route, "name", ""))
        response = self.client.get("/health")
        self.assertTrue(response.headers["content-type"].startswith("application/json"))


class TestReviewRoutes(ApiTestCase):
    def test_request_needs_a_request_id_and_explicit_approval(self):
        self.assertEqual(self.client.post("/review-queue/requests",
                                          json={"drug_pair": ["Warfarin", "Ibuprofen"], "approved": True}).status_code, 422)
        self.assertEqual(self.client.post("/review-queue/requests",
                                          json={"drug_pair": ["Warfarin", "Ibuprofen"], "request_id": "r"}).status_code, 422)
        self.assertEqual(self.client.get("/review-queue").json()["review_queue"], [])

    def test_declined_approval_writes_nothing(self):
        response = self.stage(approved=False)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["status"], "approval_required")
        self.assertFalse(response.json()["action_occurred"])
        self.assertEqual(self.client.get("/review-queue").json()["review_queue"], [])

    def test_approved_request_is_staged_and_listed_as_simulated(self):
        response = self.stage()
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["status"], "staged_in_simulation")
        self.assertTrue(body["simulated"])
        queue = self.client.get("/review-queue").json()
        self.assertTrue(queue["simulated"])
        self.assertIn("SIMULATED", queue["note"])
        self.assertEqual(queue["review_queue"][0]["evidence_id"], "ddi:1")
        self.assertEqual(queue["review_queue"][0]["purpose"], "clinician review only")

    def test_gate_failures_map_to_http_errors_and_write_nothing(self):
        self.assertEqual(self.stage(target="medication_order").status_code, 403)
        self.assertEqual(self.stage(drug_pair=["Warfarin", "NoSuchDrug"]).status_code, 404)
        self.assertEqual(self.stage().status_code, 201)
        self.assertEqual(self.stage().status_code, 409)  # the request ID is used
        self.assertEqual(len(self.client.get("/review-queue").json()["review_queue"]), 1)

    def test_trace_route(self):
        self.stage(request_id="req-9", approved=False)
        self.stage(request_id="req-9")
        data = self.client.get("/trace/req-9").json()
        self.assertEqual([a["gate"] for a in data["attempts"]], ["approval_required", "allowed"])
        self.assertEqual([a["action_occurred"] for a in data["attempts"]], [False, True])
        self.assertIn("therefore the action did occur", data["text"])
        self.assertEqual(self.client.get("/trace/never-seen").status_code, 404)

    def test_the_interaction_database_is_never_written(self):
        path = support.real_full_db_path()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.stage()
        self.client.post("/check", json={"current": ["Warfarin"], "new": "Ibuprofen"})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


class TestExplainOption(ApiTestCase):
    def test_deterministic_only_when_no_provider(self):
        body = {"current": ["Warfarin"], "new": "Fluconazole", "options": {"explain": True}}
        data = self.client.post("/check", json=body).json()
        self.assertEqual(data["explanations"][0]["generated_by"], "template")
        self.assertEqual(data["explanations"][0]["records"][0]["id"], "ddi:12")
        self.assertFalse(data["settings"]["language_model"])


@needs_fastapi
class TestExplainWithAProvider(unittest.TestCase):
    def test_validated_model_text_sits_next_to_the_source_record(self):
        record_claims = [
            {"record_id": "ddi:12", "field": "severity", "value": "Major"},
        ]
        conn = support.open_real()
        record = next(r for r in __import__("ddi_backend.ingest", fromlist=["x"]).load_interactions(conn) if r["id"] == "ddi:12")
        conn.close()
        record_claims += [{"record_id": "ddi:12", "field": f, "value": record[f]} for f in ("mechanism", "effect")]
        provider = StubProvider({"summary": f"Warfarin with Fluconazole is documented as a Major interaction. {record['effect']}.",
                                 "claims": record_claims, "clarifying_question": None})
        app = create_app(db_path=support.real_full_db_path(), in_memory_state=True, provider=provider)
        self.addCleanup(app.state.backend.close)
        client = TestClient(app)
        data = client.post("/check", json={"current": ["Warfarin"], "new": "Fluconazole",
                                           "options": {"explain": True, "detail_level": "standard"}}).json()
        explanation = data["explanations"][0]
        self.assertEqual(explanation["generated_by"], "language_model")
        self.assertEqual(explanation["records"][0]["id"], "ddi:12")
        self.assertEqual(provider.calls, 1)
        self.assertTrue(data["settings"]["language_model"])


if __name__ == "__main__":
    unittest.main()
