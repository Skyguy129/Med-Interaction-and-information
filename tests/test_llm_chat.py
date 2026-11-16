"""Tests for database-grounded API chat without network requests."""

import contextlib
import io
import json
import os
import unittest
from unittest import mock

import ddi_backend.llm_chat as llm_chat


INTERACTION = {
    "id": "ddi:1",
    "source": "drug interaction",
    "severity": "Major",
    "drug_a": "Warfarin",
    "drug_b": "Ibuprofen",
    "effect": "Increased bleeding risk",
}


class TestGetReply(unittest.TestCase):
    def test_posts_database_candidates_as_json_selection_request(self):
        response = io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": '{"record_ids":["ddi:1"]}'}}]}
            ).encode("utf-8")
        )
        messages = [{"role": "user", "content": "select records"}]

        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response) as urlopen:
            reply = llm_chat.get_reply(messages, "test-key", "test-model")

        self.assertEqual(reply, '{"record_ids":["ddi:1"]}')
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, llm_chat.API_URL)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_reports_http_error(self):
        error = llm_chat.HTTPError(
            llm_chat.API_URL,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"invalid key"}'),
        )
        self.addCleanup(error.close)
        with mock.patch("ddi_backend.llm_chat.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                llm_chat.get_reply([], "bad-key")

    def test_rejects_response_without_reply(self):
        response = io.BytesIO(b'{"choices":[]}')
        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "did not contain"):
                llm_chat.get_reply([], "test-key")


class TestRecordSelection(unittest.TestCase):
    def test_messages_include_only_the_query_and_retrieved_records(self):
        messages = llm_chat.build_selection_messages(
            "warfarin", [INTERACTION]
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["query"], "warfarin")
        self.assertEqual(payload["candidate_database_records"], [INTERACTION])

    def test_selected_results_are_original_database_rows(self):
        selected = llm_chat.select_records(
            '{"record_ids":["ddi:1"]}', [INTERACTION]
        )

        self.assertEqual(selected, [INTERACTION])

    def test_rejects_prose_or_unknown_ids_instead_of_displaying_them(self):
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            llm_chat.select_records("This is an answer.", [INTERACTION])
        with self.assertRaisesRegex(RuntimeError, "not supplied"):
            llm_chat.select_records('{"record_ids":["made-up"]}', [INTERACTION])


class TestMain(unittest.TestCase):
    def test_model_can_select_only_supplied_database_rows(self):
        prompts = iter(["What about warfarin?", "/exit"])
        output = io.StringIO()
        conn = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch("ddi_backend.llm_chat.open_database", return_value=conn),
            mock.patch("ddi_backend.llm_chat.search_records", return_value=[INTERACTION]) as search,
            mock.patch(
                "ddi_backend.llm_chat.get_reply",
                return_value='{"record_ids":["ddi:1"]}',
            ) as get_reply,
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),
            contextlib.redirect_stdout(output),
        ):
            result = llm_chat.main([])

        self.assertEqual(result, 0)
        search.assert_called_once_with(conn, "What about warfarin?", 12)
        sent_messages = get_reply.call_args.args[0]
        self.assertEqual(len(sent_messages), 2)
        self.assertIn("candidate_database_records", sent_messages[1]["content"])
        self.assertIn("Warfarin", output.getvalue())
        self.assertIn("Increased bleeding risk", output.getvalue())
        self.assertNotIn("This is a generated answer", output.getvalue())
        conn.close.assert_called_once()

    def test_skips_api_when_database_has_no_matches(self):
        prompts = iter(["unmatched query", "/exit"])
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch("ddi_backend.llm_chat.open_database", return_value=mock.Mock()),
            mock.patch("ddi_backend.llm_chat.search_records", return_value=[]),
            mock.patch("ddi_backend.llm_chat.get_reply") as get_reply,
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),
            contextlib.redirect_stdout(output),
        ):
            result = llm_chat.main([])

        self.assertEqual(result, 0)
        get_reply.assert_not_called()
        self.assertIn("No matching records found", output.getvalue())

    def test_requires_api_key_after_database_is_opened(self):
        output = io.StringIO()
        conn = mock.Mock()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("ddi_backend.llm_chat.open_database", return_value=conn),
            contextlib.redirect_stderr(output),
        ):
            result = llm_chat.main([])

        self.assertEqual(result, 1)
        self.assertIn("OPENAI_API_KEY", output.getvalue())
        conn.close.assert_called_once()

    def test_reports_database_validation_error(self):
        output = io.StringIO()
        with (
            mock.patch("ddi_backend.llm_chat.open_database", side_effect=RuntimeError("load both files")),
            contextlib.redirect_stderr(output),
        ):
            result = llm_chat.main([])

        self.assertEqual(result, 1)
        self.assertIn("load both files", output.getvalue())


if __name__ == "__main__":
    unittest.main()
