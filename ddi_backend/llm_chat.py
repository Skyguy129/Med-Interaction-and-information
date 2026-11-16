"""A small terminal chat client for the OpenAI Chat Completions API.

The client only displays rows retrieved from the DDI/HPO SQLite database
created by ``load_data.py``. The API model selects relevant record IDs; the
program renders the selected records directly rather than displaying generated
prose.

Load both source databases first:
    python -m ddi_backend.load_data load

Set OPENAI_API_KEY in the environment, then run:
    python -m ddi_backend.llm_chat

The optional OPENAI_MODEL environment variable selects the model. Use --db to
select another loaded database. Type /quit or /exit to end the chat.
"""

import argparse
import json
import os
import sqlite3
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ddi_backend.load_data as load_data
from ddi_backend.database_context import (
    configure_console_output,
    format_record,
    open_database,
    search_records,
)

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_MESSAGE = (
    "You are a database record selector, not a conversational answerer. "
    "The only permitted source is the candidate records supplied by the "
    "application from its loaded DDI and HPO SQLite database. Return a JSON "
    "object with exactly one key, record_ids, whose value is a list of IDs "
    "selected from those candidates that directly match the user's query. "
    "Do not add facts, explanations, or IDs that are not in the candidates. "
    "Return an empty list when none of the candidates directly match."
)


def get_reply(messages, api_key, model=DEFAULT_MODEL):
    """Ask the API to select candidate database record IDs as JSON."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        message = f"API request failed with HTTP {error.code}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to the LLM API: {error.reason}") from error
    except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Could not read a valid response from the LLM API: {error}") from error

    try:
        reply = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The LLM API response did not contain an assistant reply.") from error

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("The LLM API returned an empty assistant reply.")
    return reply.strip()


def build_selection_messages(prompt, records):
    """Send only the query and retrieved database rows to the model."""
    candidates = [
        {key: value for key, value in record.items() if value is not None}
        for record in records
    ]
    user_message = {
        "query": prompt,
        "candidate_database_records": candidates,
    }
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": json.dumps(user_message, ensure_ascii=False)},
    ]


def select_records(reply, records):
    """Validate the model's selection and return only original database rows."""
    try:
        selection = json.loads(reply)
    except json.JSONDecodeError as error:
        raise RuntimeError("The model returned invalid JSON; no answer was shown.") from error

    if (
        not isinstance(selection, dict)
        or set(selection) != {"record_ids"}
        or not isinstance(selection["record_ids"], list)
        or any(not isinstance(record_id, str) for record_id in selection["record_ids"])
    ):
        raise RuntimeError(
            "The model returned an invalid record selection; no answer was shown."
        )

    records_by_id = {record["id"]: record for record in records}
    unknown_ids = set(selection["record_ids"]) - records_by_id.keys()
    if unknown_ids:
        raise RuntimeError(
            "The model selected a record that was not supplied by the database; "
            "no answer was shown."
        )

    selected_ids = set(selection["record_ids"])
    return [record for record in records if record["id"] in selected_ids]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Use the LLM only to select matching records from the loaded database."
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
        help="maximum database candidates per prompt (default: 12)",
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. Set it in your terminal environment "
            "before starting the chat.",
            file=sys.stderr,
        )
        conn.close()
        return 1

    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    print(
        f"Database-grounded chat started using {model}. The API can select "
        "records only; displayed answers are exact database rows. "
        "Type /quit or /exit to stop."
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
                if not records:
                    print("No matching records found in the loaded DDI/HPO database.")
                    continue

                messages = build_selection_messages(prompt, records)
                reply = get_reply(messages, api_key, model)
                selected = select_records(reply, records)
            except (RuntimeError, sqlite3.Error) as error:
                print(f"Error: {error}", file=sys.stderr)
                continue

            if not selected:
                print("No supplied database records directly matched that query.")
                continue
            print("Selected database records:")
            for record in selected:
                print(format_record(record))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
