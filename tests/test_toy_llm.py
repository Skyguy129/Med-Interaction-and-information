"""Tests for the offline database-only lookup CLI."""

import contextlib
import io
import unittest
from unittest import mock

import ddi_backend.toy_llm as toy_llm


INTERACTION = {
    "id": "ddi:1",
    "source": "drug interaction",
    "severity": "Major",
    "drug_a": "Warfarin",
    "drug_b": "Ibuprofen",
    "effect": "Increased bleeding risk",
}


class TestToyCli(unittest.TestCase):
    def test_shows_exact_database_record_and_closes_connection(self):
        prompts = iter(["What interactions involve warfarin?", "/exit"])
        output = io.StringIO()
        conn = mock.Mock()
        with (
            mock.patch("ddi_backend.toy_llm.open_database", return_value=conn) as open_db,
            mock.patch("ddi_backend.toy_llm.search_records", return_value=[INTERACTION]) as search,
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),
            contextlib.redirect_stdout(output),
        ):
            result = toy_llm.main([])

        self.assertEqual(result, 0)
        open_db.assert_called_once()
        search.assert_called_once_with(conn, "What interactions involve warfarin?", 12)
        self.assertIn("Warfarin", output.getvalue())
        self.assertIn("Increased bleeding risk", output.getvalue())
        self.assertNotIn("Assistant:", output.getvalue())
        conn.close.assert_called_once()

    def test_does_not_invent_an_answer_when_database_has_no_match(self):
        prompts = iter(["Explain an unrelated topic", "/exit"])
        output = io.StringIO()
        with (
            mock.patch("ddi_backend.toy_llm.open_database", return_value=mock.Mock()),
            mock.patch("ddi_backend.toy_llm.search_records", return_value=[]),
            mock.patch("builtins.input", side_effect=lambda _: next(prompts)),
            contextlib.redirect_stdout(output),
        ):
            result = toy_llm.main([])

        self.assertEqual(result, 0)
        self.assertIn("No matching records found", output.getvalue())

    def test_database_validation_error_is_reported(self):
        output = io.StringIO()
        with (
            mock.patch("ddi_backend.toy_llm.open_database", side_effect=RuntimeError("load both files")),
            contextlib.redirect_stderr(output),
        ):
            result = toy_llm.main([])

        self.assertEqual(result, 1)
        self.assertIn("load both files", output.getvalue())


if __name__ == "__main__":
    unittest.main()
