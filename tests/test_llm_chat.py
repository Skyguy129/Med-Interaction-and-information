# Docstring: says this file tests the database-grounded API chat without any network requests.
"""Tests for database-grounded API chat without network requests."""

import contextlib  # redirect_stdout / redirect_stderr to capture printed output
import io  # BytesIO / StringIO for fake responses and captured output
import json  # build and read JSON payloads
import os  # patch environment variables such as the API key
import unittest  # standard test framework
from unittest import mock  # replace network, database and input calls with fakes

import ddi_backend.llm_chat as llm_chat  # the module under test (selector-mode chat)


# A fake database row shaped like a real interaction record, used as the candidate record in the tests.
INTERACTION = {
    "id": "ddi:1",  # record ID in the ddi:<n> format
    "source": "drug interaction",  # which table it came from
    "severity": "Major",  # documented severity
    "drug_a": "Warfarin",  # first drug of the pair
    "drug_b": "Ibuprofen",  # second drug of the pair
    "effect": "Increased bleeding risk",  # stored clinical effect text
}


class TestGetReply(unittest.TestCase):  # tests for the function that calls the chat API
    def test_posts_database_candidates_as_json_selection_request(self):  # checks the exact request that is sent
        response = io.BytesIO(  # a fake HTTP response body held in memory
            json.dumps(
                # shaped like a chat completion whose content selects record ddi:1
                {"choices": [{"message": {"content": '{"record_ids":["ddi:1"]}'}}]}
            ).encode("utf-8")
        )
        messages = [{"role": "user", "content": "select records"}]  # the messages we will send

        # replace the real network call with one that returns the fake response
        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response) as urlopen:
            reply = llm_chat.get_reply(messages, "test-key", "test-model")  # call with a fake key and model

        self.assertEqual(reply, '{"record_ids":["ddi:1"]}')  # the reply text was extracted correctly
        request = urlopen.call_args.args[0]  # the request object that would have been sent
        self.assertEqual(request.full_url, llm_chat.API_URL)  # it went to the configured API address
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")  # the key was sent as a bearer token
        payload = json.loads(request.data)  # decode the JSON body that was sent
        self.assertEqual(payload["model"], "test-model")  # the model name was included
        self.assertEqual(payload["messages"], messages)  # the messages were passed through unchanged
        self.assertEqual(payload["response_format"], {"type": "json_object"})  # JSON output was requested

    def test_reports_http_error(self):  # an HTTP failure becomes a readable RuntimeError
        error = llm_chat.HTTPError(  # build a fake 401 error
            llm_chat.API_URL,  # URL that "failed"
            401,  # status code
            "Unauthorized",  # status text
            hdrs=None,  # no headers needed
            fp=io.BytesIO(b'{"error":"invalid key"}'),  # error body
        )
        self.addCleanup(error.close)  # close the error object when the test ends
        with mock.patch("ddi_backend.llm_chat.urlopen", side_effect=error):  # make the network call raise it
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):  # the message must mention the status
                llm_chat.get_reply([], "bad-key")

    def test_rejects_response_without_reply(self):  # a response with no choices is an error
        response = io.BytesIO(b'{"choices":[]}')  # fake body with an empty choices list
        with mock.patch("ddi_backend.llm_chat.urlopen", return_value=response):  # return it from the fake network
            with self.assertRaisesRegex(RuntimeError, "did not contain"):  # error says there was no reply
                llm_chat.get_reply([], "test-key")


class TestRecordSelection(unittest.TestCase):  # tests for building the prompt and validating the model's choice
    def test_messages_include_only_the_query_and_retrieved_records(self):  # the model sees nothing else
        messages = llm_chat.build_selection_messages(
            "warfarin", [INTERACTION]  # the user's query and the candidate records
        )

        self.assertEqual(len(messages), 2)  # exactly a system message and a user message
        self.assertEqual(messages[0]["role"], "system")  # first is the system instruction
        payload = json.loads(messages[1]["content"])  # the user message holds JSON
        self.assertEqual(payload["query"], "warfarin")  # it contains the query
        self.assertEqual(payload["candidate_database_records"], [INTERACTION])  # and only the candidate records

    def test_selected_results_are_original_database_rows(self):  # the output is the stored row, not model text
        selected = llm_chat.select_records(
            '{"record_ids":["ddi:1"]}', [INTERACTION]  # a valid model reply and the candidates
        )

        self.assertEqual(selected, [INTERACTION])  # the stored row comes back unchanged

    def test_rejects_prose_or_unknown_ids_instead_of_displaying_them(self):  # invalid replies show nothing
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):  # prose is not JSON
            llm_chat.select_records("This is an answer.", [INTERACTION])
        with self.assertRaisesRegex(RuntimeError, "not supplied"):  # an ID that was never a candidate
            llm_chat.select_records('{"record_ids":["made-up"]}', [INTERACTION])


class TestMain(unittest.TestCase):  # tests for the interactive chat loop
    def test_model_can_select_only_supplied_database_rows(self):  # full happy path with fakes
        prompts = iter(["What about warfarin?", "/exit"])  # scripted user input: a question then exit
        output = io.StringIO()  # buffer for printed output
        conn = mock.Mock()  # fake database connection
        with (  # patch the key, database, search, model call and input
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),  # pretend a key is set
            mock.patch("ddi_backend.llm_chat.open_database", return_value=conn),  # fake database
            # search returns the one candidate; "search" records the call
            mock.patch("ddi_backend.llm_chat.search_records", return_value=[INTERACTION]) as search,
            mock.patch(
                "ddi_backend.llm_chat.get_reply",  # the model call
                return_value='{"record_ids":["ddi:1"]}',  # the model selects ddi:1
            ) as get_reply,
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),  # feed the scripted prompts
            contextlib.redirect_stdout(output),  # capture output
        ):
            result = llm_chat.main([])  # run the chat; returns an exit code

        self.assertEqual(result, 0)  # success
        search.assert_called_once_with(conn, "What about warfarin?", 12)  # search got the question and limit 12
        sent_messages = get_reply.call_args.args[0]  # what was sent to the model
        self.assertEqual(len(sent_messages), 2)  # a system message and a user message
        self.assertIn("candidate_database_records", sent_messages[1]["content"])  # candidates were included
        self.assertIn("Warfarin", output.getvalue())  # the stored drug name was printed
        self.assertIn("Increased bleeding risk", output.getvalue())  # the stored effect was printed
        self.assertNotIn("This is a generated answer", output.getvalue())  # no model-written prose is shown
        conn.close.assert_called_once()  # the connection was closed

    def test_skips_api_when_database_has_no_matches(self):  # no candidates means no paid API call
        prompts = iter(["unmatched query", "/exit"])  # scripted input
        output = io.StringIO()  # buffer for printed output
        with (  # patch the key, database, search, model call and input
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),  # pretend a key is set
            mock.patch("ddi_backend.llm_chat.open_database", return_value=mock.Mock()),  # fake database
            mock.patch("ddi_backend.llm_chat.search_records", return_value=[]),  # search finds nothing
            mock.patch("ddi_backend.llm_chat.get_reply") as get_reply,  # watch the model call
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),  # feed the scripted prompts
            contextlib.redirect_stdout(output),  # capture output
        ):
            result = llm_chat.main([])  # run the chat

        self.assertEqual(result, 0)  # success
        get_reply.assert_not_called()  # the model was never called
        self.assertIn("No matching records found", output.getvalue())  # the user is told nothing matched

    def test_requires_api_key_after_database_is_opened(self):  # missing key is an error, and the database is closed
        output = io.StringIO()  # buffer for the error stream
        conn = mock.Mock()  # fake database connection
        with (  # patch an empty environment and the database
            mock.patch.dict(os.environ, {}, clear=True),  # no environment variables, so no key
            mock.patch("ddi_backend.llm_chat.open_database", return_value=conn),  # fake database
            contextlib.redirect_stderr(output),  # capture the error output
        ):
            result = llm_chat.main([])  # run the chat

        self.assertEqual(result, 1)  # failure exit code
        self.assertIn("OPENAI_API_KEY", output.getvalue())  # the message names the missing variable
        conn.close.assert_called_once()  # the database connection was still closed

    def test_reports_database_validation_error(self):  # a database problem is reported
        output = io.StringIO()  # buffer for the error stream
        with (  # patch the database opener to fail
            # opening the database raises a RuntimeError with a message
            mock.patch("ddi_backend.llm_chat.open_database", side_effect=RuntimeError("load both files")),
            contextlib.redirect_stderr(output),  # capture the error output
        ):
            result = llm_chat.main([])  # run the chat

        self.assertEqual(result, 1)  # failure exit code
        self.assertIn("load both files", output.getvalue())  # the message reached the user


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the selector-mode chat (ddi_backend/llm_chat.py) with no network. It checks the request
# sent to the model (key, model, JSON mode), that the model sees only the question and candidate records, that
# only IDs from the candidate list are accepted and the stored rows are printed, that no candidates means no API
# call, and that a missing key or database error is reported cleanly.
# ASSIGNMENT TOOL: the language model tool, selector mode (the model picks record IDs and the program prints the
# stored rows only).
# ---------------------------------------------------------------------------
