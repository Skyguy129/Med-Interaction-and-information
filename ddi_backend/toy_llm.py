"""Offline database-backed alternative to the API chat.

This program is a lookup demo, not a real LLM. It answers by showing matching
rows from the DDI and HPO tables loaded by ``load_data.py``. It does not train
on built-in examples, read other corpora, or generate facts.

Load both source databases first:
    python -m ddi_backend.load_data load

Then run:
    python -m ddi_backend.toy_llm
    python -m ddi_backend.toy_llm --db path\\to\\medications.db --limit 20

Type /quit or /exit to stop.
"""

import argparse
import sqlite3
import sys

import ddi_backend.load_data as load_data
from ddi_backend.database_context import (
    configure_console_output,
    format_record,
    open_database,
    search_records,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search only the DDI and HPO records loaded by load_data.py."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(load_data.DEFAULT_DB),
        help="loaded SQLite database (default: medications.db in the repository root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="maximum matching records to show per prompt (default: 12)",
    )
    return parser


def main(argv=None):
    configure_console_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be greater than zero")

    try:
        conn = open_database(args.db)
    except (OSError, RuntimeError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        "Offline database lookup (not a real LLM). Answers are exact records "
        "from the loaded DDI/HPO database. Type /quit or /exit to stop."
    )
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if prompt.casefold() in {"/quit", "/exit"}:
                return 0
            if not prompt:
                continue

            try:
                records = search_records(conn, prompt, args.limit)
            except (OSError, RuntimeError, sqlite3.Error) as error:
                print(f"Error: {error}", file=sys.stderr)
                continue
            if not records:
                print("No matching records found in the loaded DDI/HPO database.")
                continue

            print("Database records:")
            for record in records:
                print(format_record(record))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
