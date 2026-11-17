# Module docstring: lists every command the CLI offers and states that review commands are simulated.
"""Command-line interface for the DDI backend.

    python -m ddi_backend check --current "Warfarin 5 mg" --current Omeprazole --new Ibuprofen
    python -m ddi_backend check --current Warfarin --current Fluconazole --new Ibuprofen --include-current-pairs
    python -m ddi_backend drug warfarin
    python -m ddi_backend avoid warfarin
    python -m ddi_backend ask "Which pain reliever should I not take with warfarin?"
    python -m ddi_backend review stage --pair Warfarin Ibuprofen --request-id req-1 --approve
    python -m ddi_backend review list
    python -m ddi_backend trace req-1
    python -m ddi_backend serve
    python -m ddi_backend export-graph

This is a class prototype and not medical advice. Everything the review commands do is simulated.
"""

import argparse  # builds the command and option parser
import json  # prints structured results with --json
import sys  # stderr for errors and the exit code
from pathlib import Path  # path-typed options

from ddi_backend import composer, llm, load_data, state  # template text, model settings, default db path, state file
from ddi_backend.agent import format_trace  # readable trace of a review request
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE  # standard disclaimer texts
from ddi_backend.database_context import configure_console_output, format_record  # console setup, record printing
from ddi_backend.service import EMBEDDING_MODES, App, default_embedding_mode  # shared setup and embedding choices


def build_parser():
    # Options shared by every command; add_help=False so it can be used as a parent parser.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=load_data.DEFAULT_DB,
                        help="medications.db built by load_data.py (opened read-only)")  # database file
    common.add_argument("--state", type=Path, default=state.DEFAULT_STATE_DB,
                        help="state file for the review queue and request log (default: agent_state.db)")  # writes
    common.add_argument("--memory", action="store_true", help="keep the state in memory instead of the state file")
    common.add_argument("--json", action="store_true", help="print the structured result as JSON")  # machine output
    common.add_argument("--detail", choices=("brief", "standard", "detailed"), default="standard",
                        help="how much of each record to put in the plain-language text")  # text detail level
    common.add_argument("--context", choices=llm.CONTEXT_LEVELS, default="full_context",
                        help="what the language model may see besides the records")  # model context level
    common.add_argument("--embeddings", choices=EMBEDDING_MODES, default=default_embedding_mode(),
                        help="auto tries the embedding model and falls back to exact names if it is unavailable")
    common.add_argument("--no-symptom-mapping", action="store_true", help="do not use HPO to expand symptom words")

    # The main parser; the first docstring line becomes the help description.
    parser = argparse.ArgumentParser(prog="python -m ddi_backend", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)  # one subcommand must be given

    # check: new drug against current drugs.
    p = sub.add_parser("check", parents=[common], help="check a new medication against the current ones")
    p.add_argument("--current", action="append", default=[], help="a current medication; repeat for each (doses allowed)")
    p.add_argument("--new", required=True, help="the medication being considered")
    p.add_argument("--include-current-pairs", action="store_true", help="also check the current medications against each other")
    p.add_argument("--explain", action="store_true", help="use the language model (if a key is set) to write the explanation")
    p.set_defaults(func=cmd_check)  # remember which function runs this command

    # drug: one drug's interactions.
    p = sub.add_parser("drug", parents=[common], help="a drug's name and the interactions it appears in")
    p.add_argument("name")  # the drug name
    p.set_defaults(func=cmd_drug)

    # avoid: the avoid list for a drug.
    p = sub.add_parser("avoid", parents=[common], help="every drug with a Major or Moderate interaction with a drug")
    p.add_argument("name")  # the drug name
    p.set_defaults(func=cmd_avoid)

    # ask: free-text question.
    p = sub.add_parser("ask", parents=[common], help="ask a free-text question")
    p.add_argument("question")  # the question text
    p.set_defaults(func=cmd_ask)

    # review: group of simulated review-queue commands.
    review = sub.add_parser("review", help="the simulated pharmacist-review queue")
    rsub = review.add_subparsers(dest="review_command", required=True)  # review needs stage or list
    p = rsub.add_parser("stage", parents=[common], help="stage a documented interaction (needs --approve and --request-id)")
    p.add_argument("--pair", nargs=2, required=True, metavar=("DRUG_A", "DRUG_B"))  # the two drugs
    p.add_argument("--request-id", required=True)  # unique id for this request
    p.add_argument("--approve", action="store_true", help="the user's explicit approval; without it nothing is written")
    p.set_defaults(func=cmd_review_stage)
    p = rsub.add_parser("list", parents=[common], help="list the review queue")
    p.set_defaults(func=cmd_review_list)

    # trace: what happened for one request.
    p = sub.add_parser("trace", parents=[common], help="what happened for one request: call, gate, tool result, state change")
    p.add_argument("request_id")
    p.set_defaults(func=cmd_trace)

    # serve: run the HTTP API.
    p = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")  # local only by default
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    # export-graph: offline export for the deferred neural model.
    p = sub.add_parser("export-graph", parents=[common], help="offline graph export and pair-level split (trains nothing)")
    p.add_argument("--out", type=Path, default=load_data.ROOT / "data" / "graph_export")  # output folder
    p.set_defaults(func=cmd_export_graph)

    # health: which tiers are available.
    p = sub.add_parser("health", parents=[common], help="which tiers are available")
    p.set_defaults(func=cmd_health)
    return parser  # the finished parser


def make_app(args):
    # Build the shared backend (database, checker, agent, state, optional tiers) from the parsed options.
    return App(db_path=args.db, state_path=args.state, symptom_mapping=not args.no_symptom_mapping,
               embeddings=args.embeddings, settings=llm.Settings(args.context, args.detail),
               in_memory_state=args.memory)


def emit(args, payload, text):
    if args.json:  # structured output requested
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))  # default=str handles odd values
    else:
        print(text)  # human-readable output


# ---- commands ---------------------------------------------------------------
def cmd_check(args, app):
    # Deterministic check of the new drug against the current ones.
    out = app.checker.check(args.current, args.new, include_current_pairs=args.include_current_pairs)
    if args.explain:  # language-model (or template fallback) explanation requested
        explained = app.explainer.explain_check(out)  # one validated explanation per pair
        out["explanations"] = explained  # include them in the JSON result too
        blocks = []  # one text block per pair
        for item in explained:
            block = [item["text"], f"(written by: {item['generated_by']})"]  # the text and who wrote it
            for record in item["records"]:
                block.append(format_record(record))  # show the source record beside the text
            blocks.append("\n".join(block))
        text = "\n\n".join(blocks + [DISCLAIMER])  # all blocks plus the disclaimer
    else:
        text = composer.compose_check(out, args.detail)  # template plain-language text
        if out["notes"]:  # put any notes before the text
            text = "\n".join(out["notes"]) + "\n\n" + text
    emit(args, out, text)  # print as text or JSON
    return 0  # success


def cmd_drug(args, app):
    out = app.checker.drug(args.name)  # exact-name lookup of the drug and its interactions
    if out["status"] != "resolved":  # unknown or ambiguous name
        # Show the message and any did-you-mean suggestions.
        emit(args, out, (out.get("message") or "") + (
            f"\nDid you mean: {', '.join(out['did_you_mean'])}?" if out.get("did_you_mean") else ""))
        return 1  # failure
    lines = [f"{out['drug']['name']}: {len(out['interactions'])} documented interaction(s), most severe first."]
    for record in out["interactions"]:  # one line per documented interaction
        lines.append(f"  [{record['severity']}] {record['drug_a']} + {record['drug_b']} ({record['id']}): {record['effect']}")
    # Be explicit about what our data cannot answer, so silence is not read as "none".
    lines.append("Uses, side effects, warnings, contraindications, ingredients and food interactions: "
                 "not available in our data.")
    lines.append(DISCLAIMER)  # standard disclaimer
    emit(args, out, "\n".join(lines))
    return 0  # success


def cmd_avoid(args, app):
    out = app.checker.avoid(args.name)  # drugs with a Major or Moderate edge to this drug
    emit(args, out, composer.compose_avoid(out, args.detail) + "\n" + DISCLAIMER)
    return 0 if out["status"] == "resolved" else 1  # failure if the name did not resolve


def cmd_ask(args, app):
    answer = app.qa.ask(args.question)  # the question handler decides how to answer
    lines = []  # output lines, built in order
    for item in answer.get("clarifications", []):  # names that could not be resolved
        lines.append(item["question"] or "")  # the clarifying question
        if item.get("did_you_mean"):
            lines.append("  Did you mean: " + ", ".join(item["did_you_mean"]) + "?")  # suggestions
    if answer.get("clarifying_question"):  # a general question back to the user
        lines.append(answer["clarifying_question"])
    if answer.get("results"):  # the question was about specific pairs
        lines.append(composer.compose_check({"results": answer["results"]}, args.detail))
    elif answer.get("avoid"):  # the question asked what to avoid
        lines.append(composer.compose_avoid(answer["avoid"], args.detail))
    elif answer.get("interactions"):  # a drug's interactions or a symptom search
        for record in answer["interactions"]:
            # Say when the match came from the effect or mechanism text.
            via = f" [matched on the {record['matched_on']}]" if record.get("matched_on") else ""
            lines.append(f"  [{record['severity']}] {record['drug_a']} + {record['drug_b']} ({record['id']}){via}: {record['effect']}")
    elif answer.get("candidates"):  # similarity candidates only
        lines.append("Candidate records (not an answer):")  # make clear they do not decide anything
        lines += [format_record(r) for r in answer["candidates"]]
    for note in answer["notes"]:  # always show the notes
        lines.append(note)
    if answer.get("phenotype_records"):  # symptom-vocabulary matches, shown separately
        lines.append(answer["phenotype_note"] + " " + ", ".join(p["label"] for p in answer["phenotype_records"][:5]))
    lines.append(DISCLAIMER)
    emit(args, answer, "\n".join(line for line in lines if line))  # drop empty lines
    return 0


def cmd_review_stage(args, app):
    # Ask the guarded agent to stage the review; nothing is written unless approved and every gate passes.
    outcome = app.agent.request_review(args.pair, args.request_id, approved=args.approve)
    summary = [SIMULATED_NOTE, f"Result: {outcome['status']}",
               f"Review queue: {len(outcome['state_before']['review_queue'])} -> {len(outcome['state_after']['review_queue'])}"]
    if not args.approve:  # explain why nothing happened
        summary.append("Nothing was written: pass --approve to give the explicit approval.")
    emit(args, outcome, "\n".join(summary))
    return 0 if outcome["status"] == "staged_in_simulation" else 1  # success only when staged


def cmd_review_list(args, app):
    rows = app.state.queue_rows()  # staged notes from the state file
    lines = [SIMULATED_NOTE, f"{len(rows)} note(s) in the review queue."]
    for row in rows:  # one line per staged note
        lines.append(f"  {row['request_id']}: {row['evidence_id']} {' + '.join(row['drug_pair'])} ({row['purpose']})")
    emit(args, {"simulated": True, "review_queue": rows}, "\n".join(lines))
    return 0


def cmd_trace(args, app):
    attempts = app.state.attempts(args.request_id)  # logged attempts for this request id
    emit(args, {"simulated": True, "request_id": args.request_id, "attempts": attempts},
         format_trace(args.request_id, attempts))  # readable trace text
    return 0 if attempts else 1  # failure when nothing is logged


def cmd_serve(args, app):
    try:
        import uvicorn  # the web server, only needed here
    except ImportError:  # the API extras are not installed
        print("The HTTP API needs fastapi and uvicorn: pip install -r requirements.txt", file=sys.stderr)
        return 1
    from ddi_backend.api import create_app  # imported late so the core works without fastapi
    app.close()  # the API builds its own backend, so release this one
    # Start the server with a fresh backend that allows use from several threads.
    uvicorn.run(create_app(db_path=args.db, state_path=args.state, symptom_mapping=not args.no_symptom_mapping,
                           embeddings=args.embeddings, in_memory_state=args.memory),
                host=args.host, port=args.port)
    return 0


def cmd_export_graph(args, app):
    from ddi_backend import models  # offline tooling only; never part of the user path

    nodes, edges = models.export_graph(app.conn, args.out)  # write nodes.csv and edges.csv
    paths = models.write_split(app.conn, args.out)  # write the pair-level train and test files
    print(f"Wrote {nodes}, {edges}, {paths['train']}, {paths['test']}")  # list the files
    print("Nothing was trained. Any future prediction must be labelled " + repr(models.PREDICTED_LABEL) + ".")
    return 0


def cmd_health(args, app):
    health = app.health()  # which tiers are available
    emit(args, health, "\n".join(f"{k}: {v}" for k, v in health["tiers"].items()) + "\n" + DISCLAIMER)
    return 0


def main(argv=None):
    configure_console_output()  # safe console encoding
    args = build_parser().parse_args(argv)  # parse the command line
    try:
        app = make_app(args)  # open the database and build everything
    except (OSError, RuntimeError, ValueError) as error:  # missing database, bad option and the like
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        return args.func(args, app)  # run the chosen command
    except BrokenPipeError:  # output piped to something that closed early (e.g. head)
        return 0
    except (OSError, RuntimeError, ValueError) as error:  # report expected failures cleanly
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            app.close()  # always release the database and state file
        except Exception:
            pass  # never let cleanup hide the real result


# Only run when executed as a script.
if __name__ == "__main__":
    sys.exit(main())  # run main and pass its return code to the shell

# ---------------------------------------------------------------------------
# FILE SUMMARY: The command-line front end. It defines the commands (check, drug, avoid, ask, review stage/list, trace,
# serve, export-graph, health), builds the shared App, calls the checker, question handler, composer, explainer and
# guarded agent, and prints plain text or JSON. It holds no decision logic of its own.
# ASSIGNMENT TOOL: Supporting infrastructure (user entry point) over the knowledge graph + rules core, the language
# model explain mode, the guarded agent with its simulated review queue, and the deferred graph export.
# ---------------------------------------------------------------------------
