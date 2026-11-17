# Docstring: says this file tests the offline command line that only looks things up in the database.
"""Tests for the offline database-only lookup CLI."""

import contextlib  # redirect_stdout / redirect_stderr to capture printed output
import io  # StringIO buffers that receive the captured output
import unittest  # standard test framework
from unittest import mock  # patch the database, search and input so nothing real is touched

import ddi_backend.toy_llm as toy_llm  # the module under test (the database-only lookup CLI)


# A fake database row shaped like a real interaction record; the "search" stub returns it.
INTERACTION = {
    "id": "ddi:1",  # record ID in the ddi:<n> format
    "source": "drug interaction",  # which table the record came from
    "severity": "Major",  # documented severity
    "drug_a": "Warfarin",  # first drug of the pair
    "drug_b": "Ibuprofen",  # second drug of the pair
    "effect": "Increased bleeding risk",  # stored clinical effect text
}


class TestToyCli(unittest.TestCase):  # groups the tests for toy_llm.main
    def test_shows_exact_database_record_and_closes_connection(self):  # happy path: matching record is printed
        prompts = iter(["What interactions involve warfarin?", "/exit"])  # scripted user input: one question then exit
        output = io.StringIO()  # buffer that captures what the CLI prints
        conn = mock.Mock()  # fake database connection so we can check it is closed
        with (  # patch everything the CLI touches for the duration of the call
            # open_database returns our fake connection; open_db records how it was called
            mock.patch("ddi_backend.toy_llm.open_database", return_value=conn) as open_db,
            # search_records returns the one fake record; "search" records the call arguments
            mock.patch("ddi_backend.toy_llm.search_records", return_value=[INTERACTION]) as search,
            # each time input() is called, hand back the next scripted prompt
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),
            # capture standard output into the buffer
            contextlib.redirect_stdout(output),
        ):
            result = toy_llm.main([])  # run the CLI with no arguments; returns its exit code

        self.assertEqual(result, 0)  # exit code 0 means success
        open_db.assert_called_once()  # the database was opened exactly once
        # the search got the connection, the user's exact text and the default limit of 12 records
        search.assert_called_once_with(conn, "What interactions involve warfarin?", 12)
        self.assertIn("Warfarin", output.getvalue())  # the stored drug name was printed
        self.assertIn("Increased bleeding risk", output.getvalue())  # the stored effect text was printed
        self.assertNotIn("Assistant:", output.getvalue())  # no generated "assistant" answer is shown, only the row
        conn.close.assert_called_once()  # the connection was closed after use

    def test_does_not_invent_an_answer_when_database_has_no_match(self):  # empty search must not produce an answer
        prompts = iter(["Explain an unrelated topic", "/exit"])  # scripted input: an off-topic question then exit
        output = io.StringIO()  # buffer that captures printed output
        with (  # patch the database, search and input
            mock.patch("ddi_backend.toy_llm.open_database", return_value=mock.Mock()),  # fake connection
            mock.patch("ddi_backend.toy_llm.search_records", return_value=[]),  # search finds nothing
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),  # feed the scripted prompts
            contextlib.redirect_stdout(output),  # capture output
        ):
            result = toy_llm.main([])  # run the CLI

        self.assertEqual(result, 0)  # still exits cleanly
        self.assertIn("No matching records found", output.getvalue())  # says so instead of making something up

    def test_database_validation_error_is_reported(self):  # a bad or missing database is reported, not hidden
        output = io.StringIO()  # buffer that captures the error stream
        with (  # patch the database opener to fail
            # opening the database raises a RuntimeError with a message
            mock.patch("ddi_backend.toy_llm.open_database", side_effect=RuntimeError("load both files")),
            contextlib.redirect_stderr(output),  # capture standard error
        ):
            result = toy_llm.main([])  # run the CLI

        self.assertEqual(result, 1)  # exit code 1 means failure
        self.assertIn("load both files", output.getvalue())  # the error message reached the user


if __name__ == "__main__":  # only when run directly as a script
    unittest.main()  # run all tests in this file

# ---------------------------------------------------------------------------
# FILE SUMMARY: Tests the offline "toy" lookup command line (ddi_backend/toy_llm.py), which reads a question,
# searches the interaction database and prints only the stored rows. The tests prove it shows the exact database
# record, never invents an answer when nothing matches, closes its connection, and reports database errors.
# ASSIGNMENT TOOL: the language model tool, in its deterministic selector-style role (database-grounded lookup
# that prints stored rows only and never generates facts); the toy CLI is the no-network ancestor of that mode.
# ---------------------------------------------------------------------------
