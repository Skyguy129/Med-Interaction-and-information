# Module docstring: explains the selector-mode chat where the model only picks record IDs.
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

import argparse  # command-line options
import json  # builds the request body and parses replies
import os  # reads OPENAI_API_KEY and OPENAI_MODEL
import sqlite3  # sqlite3.Error is caught in the chat loop
import sys  # stderr and exit codes
from urllib.error import HTTPError, URLError  # error types for failed web requests
from urllib.request import Request, urlopen  # standard-library web client (no extra packages)

import ddi_backend.load_data as load_data  # supplies the default database path
from ddi_backend.database_context import (  # shared read-only database helpers
    configure_console_output,  # safe console printing
    format_record,  # prints one stored row
    open_database,  # opens medications.db read-only
    search_records,  # finds candidate rows for a prompt
)

API_URL = "https://api.openai.com/v1/chat/completions"  # the provider endpoint
DEFAULT_MODEL = "gpt-4o-mini"  # used when OPENAI_MODEL is not set
SYSTEM_MESSAGE = (  # the rule sent to the model: select IDs only, add nothing
    "You are a database record selector, not a conversational answerer. "  # role: selector only
    "The only permitted source is the candidate records supplied by the "  # sole source of information
    "application from its loaded DDI and HPO SQLite database. Return a JSON "  # candidates come from our database
    "object with exactly one key, record_ids, whose value is a list of IDs "  # required output shape
    "selected from those candidates that directly match the user's query. "  # only candidate IDs allowed
    "Do not add facts, explanations, or IDs that are not in the candidates. "  # no new facts
    "Return an empty list when none of the candidates directly match."  # empty answer is acceptable
)


def get_reply(messages, api_key, model=DEFAULT_MODEL):  # call the API and return its text
    """Ask the API to select candidate database record IDs as JSON."""
    body = json.dumps(  # turn the request into JSON text
        {  # request fields
            "model": model,  # which model to use
            "messages": messages,  # system and user messages
            "response_format": {"type": "json_object"},  # ask for a JSON object back
        }
    ).encode("utf-8")  # the web client needs bytes
    request = Request(  # build the HTTP request
        API_URL,  # where to send it
        data=body,  # the JSON body
        headers={  # request headers
            "Authorization": f"Bearer {api_key}",  # the key comes from the environment, never from code
            "Content-Type": "application/json",  # we send JSON
            "Accept": "application/json",  # we want JSON back
        },
        method="POST",  # sending a body needs POST
    )

    try:  # the network call can fail in several ways
        with urlopen(request, timeout=60) as response:  # send it, waiting at most 60 seconds
            result = json.loads(response.read().decode("utf-8"))  # parse the JSON reply
    except HTTPError as error:  # the server answered with an error code
        detail = error.read().decode("utf-8", errors="replace").strip()  # read the error body for context
        message = f"API request failed with HTTP {error.code}"  # start the message with the code
        if detail:  # the server said more
            message += f": {detail}"  # include it
        raise RuntimeError(message) from error  # report as one error type
    except URLError as error:  # could not reach the server
        raise RuntimeError(f"Could not connect to the LLM API: {error.reason}") from error  # report the reason
    except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:  # slow or unreadable reply
        raise RuntimeError(f"Could not read a valid response from the LLM API: {error}") from error  # report it

    try:  # dig the reply text out of the response
        reply = result["choices"][0]["message"]["content"]  # standard Chat Completions shape
    except (KeyError, IndexError, TypeError) as error:  # unexpected shape
        raise RuntimeError("The LLM API response did not contain an assistant reply.") from error  # report it

    if not isinstance(reply, str) or not reply.strip():  # empty or non-text reply
        raise RuntimeError("The LLM API returned an empty assistant reply.")  # refuse it
    return reply.strip()  # the model's JSON text


def build_selection_messages(prompt, records):  # make the messages sent to the model
    """Send only the query and retrieved database rows to the model."""
    candidates = [  # drop empty fields so the model sees only stored values
        {key: value for key, value in record.items() if value is not None}  # one cleaned record
        for record in records  # every candidate row
    ]
    user_message = {  # the user turn
        "query": prompt,  # what the user asked
        "candidate_database_records": candidates,  # the only records the model may pick from
    }
    return [  # the two messages
        {"role": "system", "content": SYSTEM_MESSAGE},  # the selector rule
        {"role": "user", "content": json.dumps(user_message, ensure_ascii=False)},  # query plus candidates as JSON
    ]


def select_records(reply, records):  # validate the model's choice
    """Validate the model's selection and return only original database rows."""
    try:  # the reply must be JSON
        selection = json.loads(reply)  # parse it
    except json.JSONDecodeError as error:  # not JSON
        raise RuntimeError("The model returned invalid JSON; no answer was shown.") from error  # show nothing

    if (  # strict shape check
        not isinstance(selection, dict)  # must be an object
        or set(selection) != {"record_ids"}  # with exactly one key, record_ids
        or not isinstance(selection["record_ids"], list)  # whose value is a list
        or any(not isinstance(record_id, str) for record_id in selection["record_ids"])  # of strings
    ):
        raise RuntimeError(  # wrong shape: show nothing
            "The model returned an invalid record selection; no answer was shown."
        )

    records_by_id = {record["id"]: record for record in records}  # look up candidates by ID
    unknown_ids = set(selection["record_ids"]) - records_by_id.keys()  # any ID that was not a candidate
    if unknown_ids:  # the model invented or guessed an ID
        raise RuntimeError(  # show nothing
            "The model selected a record that was not supplied by the database; "
            "no answer was shown."
        )

    selected_ids = set(selection["record_ids"])  # the IDs the model chose
    return [record for record in records if record["id"] in selected_ids]  # return the stored rows, in their original order


def build_parser():  # command-line options
    parser = argparse.ArgumentParser(  # create the parser
        description="Use the LLM only to select matching records from the loaded database."  # --help text
    )
    parser.add_argument(  # database option
        "--db",  # flag name
        type=str,  # path string
        default=str(load_data.DEFAULT_DB),  # default database file
        help="loaded SQLite database (default: medications.db in the repository root)",  # --help text
    )
    parser.add_argument(  # candidate-count option
        "--limit",  # flag name
        type=int,  # integer
        default=12,  # default number of candidates
        help="maximum database candidates per prompt (default: 12)",  # --help text
    )
    return parser  # give the parser to main()


def main(argv=None):  # entry point for the selector chat
    configure_console_output()  # safe console printing
    parser = build_parser()  # build the parser
    args = parser.parse_args(argv)  # read options
    if args.limit < 1:  # must be positive
        parser.error("--limit must be greater than zero")  # print usage and exit

    try:  # opening the database can fail
        conn = open_database(args.db)  # open read-only
    except (OSError, RuntimeError, sqlite3.Error) as error:  # missing or partial database
        print(f"Error: {error}", file=sys.stderr)  # report it
        return 1  # failure exit code

    api_key = os.environ.get("OPENAI_API_KEY")  # the key lives only in the environment
    if not api_key:  # no key
        print(  # explain what to do
            "OPENAI_API_KEY is not set. Set it in your terminal environment "
            "before starting the chat.",
            file=sys.stderr,  # error output
        )
        conn.close()  # release the database
        return 1  # failure exit code

    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)  # chosen model, or the default
    print(  # startup banner
        f"Database-grounded chat started using {model}. The API can select "  # which model is in use
        "records only; displayed answers are exact database rows. "  # reminder of the safeguard
        "Type /quit or /exit to stop."  # how to stop
    )
    try:  # always close the database
        while True:  # chat loop
            try:  # input can be interrupted
                prompt = input("You: ").strip()  # read a line
            except (EOFError, KeyboardInterrupt):  # end of input or Ctrl+C
                print()  # tidy the line
                return 0  # leave quietly

            if prompt.casefold() in {"/quit", "/exit"}:  # stop commands
                return 0  # leave
            if not prompt:  # blank line
                continue  # ask again

            try:  # lookup, model call and validation
                records = search_records(conn, prompt, args.limit)  # candidate rows from the database
                if not records:  # no candidates
                    print("No matching records found in the loaded DDI/HPO database.")  # no API call is made
                    continue  # ask again

                messages = build_selection_messages(prompt, records)  # package question and candidates
                reply = get_reply(messages, api_key, model)  # ask the model to choose IDs
                selected = select_records(reply, records)  # validate and map IDs back to stored rows
            except (RuntimeError, sqlite3.Error) as error:  # any failure in the steps above
                print(f"Error: {error}", file=sys.stderr)  # report it, show nothing else
                continue  # keep the session alive

            if not selected:  # the model chose nothing
                print("No supplied database records directly matched that query.")  # say so
                continue  # ask again
            print("Selected database records:")  # heading
            for record in selected:  # each chosen row
                print(format_record(record))  # print the stored values only
    finally:  # runs on every exit
        conn.close()  # release the database


if __name__ == "__main__":  # script entry only
    sys.exit(main())  # run and pass the result as the exit code


# ---------------------------------------------------------------------------
# FILE SUMMARY: A terminal chat in which the language model may only choose record IDs from database candidates.
# The program validates the reply (exact key, list of strings, every ID a candidate) and then prints the stored
# rows, so no generated text is ever shown. With no candidates there is no API call; invalid output shows nothing.
# ASSIGNMENT TOOL: The language model, selector mode (the unchanged first of its two modes; the second is the
# explain mode in llm.py).
# ---------------------------------------------------------------------------
