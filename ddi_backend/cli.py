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

import argparse
import json
import sys
from pathlib import Path

from ddi_backend import composer, llm, load_data, state
from ddi_backend.agent import format_trace
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE
from ddi_backend.database_context import configure_console_output, format_record
from ddi_backend.service import EMBEDDING_MODES, App, default_embedding_mode


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=load_data.DEFAULT_DB,
                        help="medications.db built by load_data.py (opened read-only)")
    common.add_argument("--state", type=Path, default=state.DEFAULT_STATE_DB,
                        help="state file for the review queue and request log (default: agent_state.db)")
    common.add_argument("--memory", action="store_true", help="keep the state in memory instead of the state file")
    common.add_argument("--json", action="store_true", help="print the structured result as JSON")
    common.add_argument("--detail", choices=("brief", "standard", "detailed"), default="standard",
                        help="how much of each record to put in the plain-language text")
    common.add_argument("--context", choices=llm.CONTEXT_LEVELS, default="full_context",
                        help="what the language model may see besides the records")
    common.add_argument("--embeddings", choices=EMBEDDING_MODES, default=default_embedding_mode(),
                        help="auto tries the embedding model and falls back to exact names if it is unavailable")
    common.add_argument("--no-symptom-mapping", action="store_true", help="do not use HPO to expand symptom words")

    parser = argparse.ArgumentParser(prog="python -m ddi_backend", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", parents=[common], help="check a new medication against the current ones")
    p.add_argument("--current", action="append", default=[], help="a current medication; repeat for each (doses allowed)")
    p.add_argument("--new", required=True, help="the medication being considered")
    p.add_argument("--include-current-pairs", action="store_true", help="also check the current medications against each other")
    p.add_argument("--explain", action="store_true", help="use the language model (if a key is set) to write the explanation")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("drug", parents=[common], help="a drug's name and the interactions it appears in")
    p.add_argument("name")
    p.set_defaults(func=cmd_drug)

    p = sub.add_parser("avoid", parents=[common], help="every drug with a Major or Moderate interaction with a drug")
    p.add_argument("name")
    p.set_defaults(func=cmd_avoid)

    p = sub.add_parser("ask", parents=[common], help="ask a free-text question")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    review = sub.add_parser("review", help="the simulated pharmacist-review queue")
    rsub = review.add_subparsers(dest="review_command", required=True)
    p = rsub.add_parser("stage", parents=[common], help="stage a documented interaction (needs --approve and --request-id)")
    p.add_argument("--pair", nargs=2, required=True, metavar=("DRUG_A", "DRUG_B"))
    p.add_argument("--request-id", required=True)
    p.add_argument("--approve", action="store_true", help="the user's explicit approval; without it nothing is written")
    p.set_defaults(func=cmd_review_stage)
    p = rsub.add_parser("list", parents=[common], help="list the review queue")
    p.set_defaults(func=cmd_review_list)

    p = sub.add_parser("trace", parents=[common], help="what happened for one request: call, gate, tool result, state change")
    p.add_argument("request_id")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("export-graph", parents=[common], help="offline graph export and pair-level split (trains nothing)")
    p.add_argument("--out", type=Path, default=load_data.ROOT / "data" / "graph_export")
    p.set_defaults(func=cmd_export_graph)

    p = sub.add_parser("health", parents=[common], help="which tiers are available")
    p.set_defaults(func=cmd_health)
    return parser


def make_app(args):
    return App(db_path=args.db, state_path=args.state, symptom_mapping=not args.no_symptom_mapping,
               embeddings=args.embeddings, settings=llm.Settings(args.context, args.detail),
               in_memory_state=args.memory)


def emit(args, payload, text):
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(text)


# ---- commands ---------------------------------------------------------------
def cmd_check(args, app):
    out = app.checker.check(args.current, args.new, include_current_pairs=args.include_current_pairs)
    if args.explain:
        explained = app.explainer.explain_check(out)
        out["explanations"] = explained
        blocks = []
        for item in explained:
            block = [item["text"], f"(written by: {item['generated_by']})"]
            for record in item["records"]:
                block.append(format_record(record))
            blocks.append("\n".join(block))
        text = "\n\n".join(blocks + [DISCLAIMER])
    else:
        text = composer.compose_check(out, args.detail)
        if out["notes"]:
            text = "\n".join(out["notes"]) + "\n\n" + text
    emit(args, out, text)
    return 0


def cmd_drug(args, app):
    out = app.checker.drug(args.name)
    if out["status"] != "resolved":
        emit(args, out, (out.get("message") or "") + (
            f"\nDid you mean: {', '.join(out['did_you_mean'])}?" if out.get("did_you_mean") else ""))
        return 1
    lines = [f"{out['drug']['name']}: {len(out['interactions'])} documented interaction(s), most severe first."]
    for record in out["interactions"]:
        lines.append(f"  [{record['severity']}] {record['drug_a']} + {record['drug_b']} ({record['id']}): {record['effect']}")
    lines.append("Uses, side effects, warnings, contraindications, ingredients and food interactions: "
                 "not available in our data.")
    lines.append(DISCLAIMER)
    emit(args, out, "\n".join(lines))
    return 0


def cmd_avoid(args, app):
    out = app.checker.avoid(args.name)
    emit(args, out, composer.compose_avoid(out, args.detail) + "\n" + DISCLAIMER)
    return 0 if out["status"] == "resolved" else 1


def cmd_ask(args, app):
    answer = app.qa.ask(args.question)
    lines = []
    for item in answer.get("clarifications", []):
        lines.append(item["question"] or "")
        if item.get("did_you_mean"):
            lines.append("  Did you mean: " + ", ".join(item["did_you_mean"]) + "?")
    if answer.get("clarifying_question"):
        lines.append(answer["clarifying_question"])
    if answer.get("results"):
        lines.append(composer.compose_check({"results": answer["results"]}, args.detail))
    elif answer.get("avoid"):
        lines.append(composer.compose_avoid(answer["avoid"], args.detail))
    elif answer.get("interactions"):
        for record in answer["interactions"]:
            via = f" [matched on the {record['matched_on']}]" if record.get("matched_on") else ""
            lines.append(f"  [{record['severity']}] {record['drug_a']} + {record['drug_b']} ({record['id']}){via}: {record['effect']}")
    elif answer.get("candidates"):
        lines.append("Candidate records (not an answer):")
        lines += [format_record(r) for r in answer["candidates"]]
    for note in answer["notes"]:
        lines.append(note)
    if answer.get("phenotype_records"):
        lines.append(answer["phenotype_note"] + " " + ", ".join(p["label"] for p in answer["phenotype_records"][:5]))
    lines.append(DISCLAIMER)
    emit(args, answer, "\n".join(line for line in lines if line))
    return 0


def cmd_review_stage(args, app):
    outcome = app.agent.request_review(args.pair, args.request_id, approved=args.approve)
    summary = [SIMULATED_NOTE, f"Result: {outcome['status']}",
               f"Review queue: {len(outcome['state_before']['review_queue'])} -> {len(outcome['state_after']['review_queue'])}"]
    if not args.approve:
        summary.append("Nothing was written: pass --approve to give the explicit approval.")
    emit(args, outcome, "\n".join(summary))
    return 0 if outcome["status"] == "staged_in_simulation" else 1


def cmd_review_list(args, app):
    rows = app.state.queue_rows()
    lines = [SIMULATED_NOTE, f"{len(rows)} note(s) in the review queue."]
    for row in rows:
        lines.append(f"  {row['request_id']}: {row['evidence_id']} {' + '.join(row['drug_pair'])} ({row['purpose']})")
    emit(args, {"simulated": True, "review_queue": rows}, "\n".join(lines))
    return 0


def cmd_trace(args, app):
    attempts = app.state.attempts(args.request_id)
    emit(args, {"simulated": True, "request_id": args.request_id, "attempts": attempts},
         format_trace(args.request_id, attempts))
    return 0 if attempts else 1


def cmd_serve(args, app):
    try:
        import uvicorn
    except ImportError:
        print("The HTTP API needs fastapi and uvicorn: pip install -r requirements.txt", file=sys.stderr)
        return 1
    from ddi_backend.api import create_app
    app.close()
    uvicorn.run(create_app(db_path=args.db, state_path=args.state, symptom_mapping=not args.no_symptom_mapping,
                           embeddings=args.embeddings, in_memory_state=args.memory),
                host=args.host, port=args.port)
    return 0


def cmd_export_graph(args, app):
    from ddi_backend import models  # offline tooling only; never part of the user path

    nodes, edges = models.export_graph(app.conn, args.out)
    paths = models.write_split(app.conn, args.out)
    print(f"Wrote {nodes}, {edges}, {paths['train']}, {paths['test']}")
    print("Nothing was trained. Any future prediction must be labelled " + repr(models.PREDICTED_LABEL) + ".")
    return 0


def cmd_health(args, app):
    health = app.health()
    emit(args, health, "\n".join(f"{k}: {v}" for k, v in health["tiers"].items()) + "\n" + DISCLAIMER)
    return 0


def main(argv=None):
    configure_console_output()
    args = build_parser().parse_args(argv)
    try:
        app = make_app(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        return args.func(args, app)
    except BrokenPipeError:
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
