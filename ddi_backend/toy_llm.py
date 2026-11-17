# Module docstring: explains this is a database lookup demo, not a real language model, and how to run it.
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

import argparse  # command-line option parsing
import sqlite3  # only needed for the sqlite3.Error exception type
import sys  # stderr for error messages and exit codes

import ddi_backend.load_data as load_data  # supplies the default database path (DEFAULT_DB)
from ddi_backend.database_context import (  # shared read-only database helpers
    configure_console_output,  # makes the console print unicode safely
    format_record,  # turns one stored row into printable text
    open_database,  # opens medications.db read-only
    search_records,  # simple lookup of rows that match a prompt
)


def build_parser():  # builds the command-line option parser
    parser = argparse.ArgumentParser(  # create the parser
        description="Search only the DDI and HPO records loaded by load_data.py."  # text shown by --help
    )
    parser.add_argument(  # option: which database file to read
        "--db",  # flag name
        type=str,  # value is a path string
        default=str(load_data.DEFAULT_DB),  # default is medications.db in the repository root
        help="loaded SQLite database (default: medications.db in the repository root)",  # help text
    )
    parser.add_argument(  # option: how many rows to print per prompt
        "--limit",  # flag name
        type=int,  # value must be an integer
        default=12,  # show up to 12 rows by default
        help="maximum matching records to show per prompt (default: 12)",  # help text
    )
    return parser  # hand the finished parser back to main()


def main(argv=None):  # entry point; argv lets tests pass arguments directly
    configure_console_output()  # make sure printing does not fail on odd characters
    parser = build_parser()  # build the option parser
    args = parser.parse_args(argv)  # read the options (from argv or the real command line)
    if args.limit < 1:  # a limit of zero or less makes no sense
        parser.error("--limit must be greater than zero")  # print usage and exit with an error

    try:  # opening the database can fail
        conn = open_database(args.db)  # open the loaded database read-only
    except (OSError, RuntimeError, sqlite3.Error) as error:  # missing file, partial database, SQLite problem
        print(f"Error: {error}", file=sys.stderr)  # report the problem on stderr
        return 1  # non-zero exit code means failure

    print(  # one-time banner explaining what this program is
        "Offline database lookup (not a real LLM). Answers are exact records "  # first half of the banner
        "from the loaded DDI/HPO database. Type /quit or /exit to stop."  # second half of the banner
    )
    try:  # make sure the connection is always closed
        while True:  # the interactive prompt loop
            try:  # reading input can be interrupted
                prompt = input("You: ").strip()  # read one line from the user and trim spaces
            except (EOFError, KeyboardInterrupt):  # end of input or Ctrl+C
                print()  # finish the line neatly
                return 0  # leave quietly with success
            if prompt.casefold() in {"/quit", "/exit"}:  # the two stop commands
                return 0  # exit the loop and the program
            if not prompt:  # an empty line
                continue  # ask again

            try:  # the search itself can fail
                records = search_records(conn, prompt, args.limit)  # find matching rows, at most --limit of them
            except (OSError, RuntimeError, sqlite3.Error) as error:  # database error during the search
                print(f"Error: {error}", file=sys.stderr)  # report it
                continue  # keep the session alive
            if not records:  # nothing matched
                print("No matching records found in the loaded DDI/HPO database.")  # say so plainly
                continue  # ask again

            print("Database records:")  # heading before the rows
            for record in records:  # go through each matching row
                print(format_record(record))  # print the stored values only
    finally:  # runs on every exit path
        conn.close()  # release the database connection


if __name__ == "__main__":  # only when run as a script, not when imported
    sys.exit(main())  # run main() and pass its return value as the exit code


# ---------------------------------------------------------------------------
# FILE SUMMARY: An offline, interactive lookup demo that prints stored DDI and HPO rows matching what the user
# types. It generates nothing and calls no model; every line shown is a stored record, so it is a safe demo of
# reading the database directly. It shares its database helpers with llm_chat.py.
# ASSIGNMENT TOOL: Supporting infrastructure (not one of the assignment tools): an offline stand-in for the
# language model selector chat (llm_chat.py), kept so the course's "toy LLM" idea has a no-key, no-network version.
# ---------------------------------------------------------------------------
