#!/usr/bin/env python3
"""Full system test: ask a question in the terminal and watch the whole system answer it.

    python full_system_test.py
    python full_system_test.py --embeddings auto      # also try the embedding model (tier two)
    python full_system_test.py --no-llm               # deterministic answers only

For every question it runs the real pipeline and prints each step:

  1. name resolution        exact, case-insensitive drug names (typos and abbreviations get questions)
  2. question rules         negation, ambiguity, population context, topics our data does not cover
  3. lookup and rules       direct pair lookup, severity precedence, avoid lists, symptom words through HPO
  4. eligibility filter     only evidence from a current source with an approved authority counts
  5. language model         writes the plain-language response from the eligible records only;
                            a validator rejects any new fact and falls back to the template text
  6. the answer             shown next to the source records it came from

With OPENAI_API_KEY set (or typed when asked) the model writes the response. Without a key the template
composer writes it, and everything else is identical. The model never decides anything and never writes
to a database. Class prototype, not medical advice.

Inside the chat, type /help for commands (/meds, /new, /detail, /context, /llm, /steps, /stage, /trace).
"""

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddi_backend import composer, llm, load_data, state  # noqa: E402
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE, severity_rank  # noqa: E402
from ddi_backend.database_context import configure_console_output, format_record  # noqa: E402
from ddi_backend.agent import format_trace  # noqa: E402
from ddi_backend.service import App  # noqa: E402

HELP = """Commands
  /meds A, B 5 mg, C     set your current medications (doses allowed); a bare drug name is then checked against them
  /meds                  show them;  /meds clear  removes them
  /new DRUG              check one new medication against your current ones
  /detail brief|standard|detailed     how much of each record goes into the text
  /context names_only|full_context    what the language model may see besides the records
  /llm on|off            use or skip the language model for this session
  /steps on|off          show or hide the pipeline steps
  /stage A B ID          stage a documented interaction in the SIMULATED review queue (asks you to approve)
  /trace ID              show what happened for one review request
  /help, /quit
Anything else is treated as a question, for example:
  Do warfarin and ibuprofen interact?     Which pain reliever should I not take with warfarin?
  What does MS interact with?             Which drugs cause bleeding?     Can I take Advil with warfarin?"""

NARRATE_SYSTEM = (
    "You write a short plain-language answer for a class prototype that is not medical advice. You have no tools "
    "and no internet, and you must not use anything you know or remember. The only permitted source is the records "
    "in the user message. Return a JSON object with exactly the keys summary (string) and record_ids (list of IDs "
    "taken from the records). Use the records' own wording; add no facts, names, numbers, doses, advice or "
    "reassurance, and never say a combination is safe."
)


class Session:
    def __init__(self, app, provider, show_steps=True):
        self.app = app
        self.provider = provider
        self.use_llm = provider is not None
        self.show_steps = show_steps
        self.meds = []
        self.settings = app.settings


def step(session, number, title, detail=""):
    if session.show_steps:
        print(f"  [{number}] {title}" + (f": {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Writing the response
# --------------------------------------------------------------------------
def make_explainer(session):
    provider = session.provider if session.use_llm else None
    return llm.Explainer(provider, session.settings, session.app.explainer.known_drug_names)


def validate_narration(reply, records, known_names):
    """Check a list-style narration against the records it was given. Returns (parsed, problems)."""
    try:
        parsed = json.loads(reply)
    except (TypeError, json.JSONDecodeError):
        return None, ["reply is not valid JSON"]
    if not isinstance(parsed, dict) or set(parsed) != {"summary", "record_ids"}:
        return None, ["reply must have exactly summary and record_ids"]
    summary, ids = parsed["summary"], parsed["record_ids"]
    if not isinstance(summary, str) or not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        return None, ["summary or record_ids has the wrong type"]
    supplied = {r["id"]: r for r in records}
    problems = [f"cites {i!r}, which was not supplied" for i in ids if i not in supplied]
    corpus = llm._record_text(records)
    corpus_words = {w.casefold() for w in llm._WORD.findall(corpus)}
    severities = {r["severity"].casefold() for r in records}
    for word in llm._WORD.findall(summary):
        lower = word.casefold()
        if lower in llm._SEVERITY_WORDS:
            if lower not in severities:
                problems.append(f"states severity {word}, which the records do not")
        elif len(lower) >= 4 and lower not in llm.ALLOWED_PLAIN_WORDS and lower not in corpus_words:
            problems.append(f"uses {word!r}, which is not in the records")
    problems += [f"contains the number {n}, which is not in the records"
                 for n in llm._NUMBER.findall(summary) if n not in corpus]
    allowed = {llm._norm(r["drug_a"]) for r in records} | {llm._norm(r["drug_b"]) for r in records}
    for r in records:
        allowed |= {llm._norm(n) for n in r["alternative"]["drug_names"]}
    text = llm._norm(summary)
    for name in known_names:
        key = llm._norm(name)
        if key and key not in allowed and re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", text):
            problems.append(f"names {name}, which is not in the records")
    if len(summary) > 1200:
        problems.append("summary is too long")
    return (None, problems) if problems else (parsed, [])


def narrate(session, question, records, fallback_text):
    """The model's plain-language answer for a list-style result, or the template text. Returns (text, how, why)."""
    if not (session.use_llm and session.provider and records):
        why = "language model is off" if not session.use_llm else "no language model configured"
        return fallback_text, "template", why if records else "no eligible records, so no model call"
    payload = {"question": question, "detail_level": session.settings.detail_level,
               "records": [{k: r.get(k) for k in ("id", "severity", "drug_a", "drug_b", "mechanism", "effect")
                            if r.get(k) is not None} for r in records]}
    try:
        reply = session.provider.complete([{"role": "system", "content": NARRATE_SYSTEM},
                                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    except RuntimeError as error:
        return fallback_text, "template", f"language model unavailable: {error}"
    parsed, problems = validate_narration(reply, records, session.app.explainer.known_drug_names)
    if problems:
        return fallback_text, "template", "reply failed validation: " + "; ".join(problems[:3])
    return parsed["summary"], "language model", "reply passed validation"


def records_for(answer, graph):
    """Records behind a non-pair answer, most severe first."""
    seen, out = set(), []
    for rid in [rid for e in answer.get("avoid", {}).get("avoid", []) for rid in e["record_ids"][:1]]:
        rec = next((r for r in graph.records if r["id"] == rid), None)
        if rec and rid not in seen:
            seen.add(rid)
            out.append(rec)
    for rec in answer.get("interactions", []):
        if rec["id"] not in seen:
            seen.add(rec["id"])
            out.append(rec)
    return sorted(out, key=lambda r: (severity_rank(r["severity"]), r["id"]))


# --------------------------------------------------------------------------
# One question through the whole system
# --------------------------------------------------------------------------
def answer_question(session, question):
    app = session.app
    print("\nProcessing")
    with app.lock:
        answer = app.qa.ask(question)

    # Step 1-2: names and question rules
    mentions = [m["name"] for m in answer.get("inputs", {}).get("current", []) if m.get("name")]
    if answer.get("inputs", {}).get("new", {}).get("name"):
        mentions.append(answer["inputs"]["new"]["name"])
    step(session, 1, "names found", ", ".join(mentions) or answer.get("drug") or "none resolved")
    for item in answer.get("clarifications", []):
        step(session, 1, f"could not resolve '{item['input']}'", item["status"] +
             (f" (did you mean: {', '.join(item['did_you_mean'])})" if item.get("did_you_mean") else ""))
    step(session, 2, "question type", answer.get("intent", "?") +
         (f" (negation words: {', '.join(answer['negation_words'])})" if answer.get("negation_words") else ""))

    # A single new drug plus current medications is the main "should I start this?" flow.
    if answer.get("intent") == "drug" and session.meds:
        step(session, 3, "your current medications are set", f"checking {answer['drug']} against {len(session.meds)} of them")
        with app.lock:
            out = app.checker.check(session.meds, answer["drug"])
        answer.update(status="answered", intent="pair", results=out["results"], inputs=out["inputs"])
        answer["rules_fired"] = list(dict.fromkeys(answer["rules_fired"] + out["rules_fired"]))
        answer["notes"] += out["notes"]

    step(session, 3, "lookup and rules fired", ", ".join(answer["rules_fired"]) or "none")
    step(session, 4, "eligibility filter", "applied to every record before it can count as evidence")
    for note in answer["notes"]:
        step(session, "!", "note", note)

    print()
    if answer.get("results"):
        explainer = make_explainer(session)
        names = [m["name"] for m in answer.get("inputs", {}).get("current", []) if m.get("name")]
        names.append(answer["inputs"]["new"].get("name"))
        shown = []
        for result in answer["results"]:
            others = [n for n in session.meds if n and n.split()[0] not in result["pair"]] or \
                [n for n in names if n and n not in result["pair"]]
            explained = explainer.explain_pair(result, other_medications=others)
            how = "language model" if explained["generated_by"] == "language_model" else "template"
            step(session, 5, f"response for {' + '.join(result['pair'])}",
                 f"{how}" + (f" ({explained['fallback_reason']})" if explained.get("fallback_reason") and how == "template" else ""))
            shown.append(explained)
        print()
        for explained in shown:
            print(f"== {' + '.join(explained['pair'])}: {explained['status']} ==")
            print(explained["text"])
            for record in explained["records"]:
                print(format_record(record))
            if explained.get("clarifying_question"):
                print(explained["clarifying_question"])
            print()
    elif answer.get("avoid") or answer.get("interactions"):
        records = records_for(answer, app.checker.graph)
        fallback = (composer.compose_avoid(answer["avoid"], session.settings.detail_level) if answer.get("avoid")
                    else "\n".join(f"[{r['severity']}] {r['drug_a']} + {r['drug_b']} ({r['id']}): {r['effect']}"
                                   for r in records))
        text, how, why = narrate(session, question, records, fallback)
        step(session, 5, "response", f"{how} ({why})")
        print(f"\n{text}")
        if how == "language model":
            print("\nStructured result (what the rules found, shown beside the model's wording):")
            print(fallback)
        if answer.get("phenotype_records"):
            print(answer["phenotype_note"] + " " + ", ".join(p["label"] for p in answer["phenotype_records"][:5]))
        print("\nSource records:")
        for record in records[:8]:
            print(format_record(record))
        if len(records) > 8:
            print(f"... and {len(records) - 8} more (use /detail detailed or the CLI for the full list).")
    else:
        step(session, 5, "response", "template (nothing to look up yet)")
        for item in answer.get("clarifications", []):
            print(item["question"] or "")
            if item.get("did_you_mean"):
                print("  Did you mean: " + ", ".join(item["did_you_mean"]) + "?")
        if answer.get("clarifying_question"):
            print(answer["clarifying_question"])
        if not answer.get("clarifications") and not answer.get("clarifying_question"):
            print(answer["notes"][-1] if answer["notes"] else "Nothing documented in our data matched.")
    for note in answer["notes"]:
        print("Note:", note)
    print("\n" + DISCLAIMER)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def parse_meds(text):
    return [m.strip() for m in re.split(r"[,;]", text) if m.strip()]


def handle_command(session, line, input_fn):
    """Returns False to quit."""
    command, _, rest = line.partition(" ")
    command, rest = command.casefold(), rest.strip()
    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        print(HELP)
    elif command == "/meds":
        if rest.casefold() == "clear":
            session.meds = []
        elif rest:
            session.meds = parse_meds(rest)
        print("Current medications:", "; ".join(session.meds) if session.meds else "none set")
    elif command == "/new":
        if not rest:
            print("Usage: /new DRUG")
        else:
            check_new(session, rest)
    elif command == "/detail":
        if rest in composer.DETAIL_LEVELS:
            session.settings = llm.Settings(session.settings.context_level, rest)
            print("detail_level =", rest)
        else:
            print("Choose one of:", ", ".join(composer.DETAIL_LEVELS))
    elif command == "/context":
        if rest in llm.CONTEXT_LEVELS:
            session.settings = llm.Settings(rest, session.settings.detail_level)
            print("context_level =", rest)
        else:
            print("Choose one of:", ", ".join(llm.CONTEXT_LEVELS))
    elif command == "/llm":
        if rest == "on" and session.provider is None:
            print("No language model is configured (set OPENAI_API_KEY and restart).")
        elif rest in ("on", "off"):
            session.use_llm = rest == "on"
            print("language model:", "on" if session.use_llm else "off")
        else:
            print("Usage: /llm on|off")
    elif command == "/steps":
        session.show_steps = rest != "off"
        print("pipeline steps:", "shown" if session.show_steps else "hidden")
    elif command == "/stage":
        parts = rest.split()
        if len(parts) != 3:
            print("Usage: /stage DRUG_A DRUG_B REQUEST_ID")
        else:
            print(SIMULATED_NOTE)
            approved = input_fn("Do you approve staging this interaction for simulated pharmacist review? (yes/no) ").strip().casefold() == "yes"
            with session.app.lock:
                outcome = session.app.agent.request_review(parts[:2], parts[2], approved=approved)
            print(f"Result: {outcome['status']}  (queue {len(outcome['state_before']['review_queue'])} -> "
                  f"{len(outcome['state_after']['review_queue'])})")
    elif command == "/trace":
        print(format_trace(rest, session.app.state.attempts(rest)) if rest else "Usage: /trace REQUEST_ID")
    else:
        print("Unknown command. Type /help.")
    return True


def check_new(session, drug):
    if not session.meds:
        print("Set your current medications first, for example: /meds Warfarin 5 mg, Omeprazole")
        return
    app = session.app
    print("\nProcessing")
    with app.lock:
        out = app.checker.check(session.meds, drug)
    step(session, 1, "names", ", ".join(m["name"] or m["input"] for m in out["inputs"]["current"]) + f" | new: {out['inputs']['new']['name'] or drug}")
    step(session, 3, "rules fired", ", ".join(out["rules_fired"]))
    explainer = make_explainer(session)
    print()
    for explained in explainer.explain_check(out):
        how = "language model" if explained["generated_by"] == "language_model" else "template"
        step(session, 5, f"response for {' + '.join(explained['pair'])}", how +
             (f" ({explained['fallback_reason']})" if how == "template" and explained.get("fallback_reason") else ""))
        print(f"\n== {' + '.join(explained['pair'])}: {explained['status']} ==")
        print(explained["text"])
        for record in explained["records"]:
            print(format_record(record))
    for note in out["notes"]:
        print("Note:", note)
    print("\n" + DISCLAIMER)


# --------------------------------------------------------------------------
# Start-up and the loop
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)
    parser.add_argument("--state", type=Path, default=state.DEFAULT_STATE_DB)
    parser.add_argument("--memory", action="store_true", help="keep the review queue in memory only")
    parser.add_argument("--embeddings", choices=("off", "auto"), default="off")
    parser.add_argument("--no-llm", action="store_true", help="never use a language model")
    parser.add_argument("--no-key-prompt", action="store_true", help="do not ask for an API key when none is set")
    parser.add_argument("--detail", choices=composer.DETAIL_LEVELS, default="standard")
    parser.add_argument("--context", choices=llm.CONTEXT_LEVELS, default="full_context")
    parser.add_argument("--hide-steps", action="store_true")
    parser.add_argument("--meds", help="current medications, comma separated, to start with")
    return parser


def main(argv=None, input_fn=input, key_fn=getpass.getpass):
    configure_console_output()
    args = build_parser().parse_args(argv)
    provider = None
    if not args.no_llm:
        provider = llm.provider_from_env()
        if provider is None and not args.no_key_prompt:
            key = key_fn("OPENAI_API_KEY is not set. Paste a key for this session only (Enter to skip): ").strip()
            if key:
                provider = llm.OpenAIProvider(key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    try:
        app = App(db_path=args.db, state_path=args.state, embeddings=args.embeddings,
                  settings=llm.Settings(args.context, args.detail), provider=provider,
                  in_memory_state=args.memory)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    session = Session(app, provider, show_steps=not args.hide_steps)
    if args.meds:
        session.meds = parse_meds(args.meds)
    health = app.health()["tiers"]
    print("Full system test (class prototype, not medical advice)")
    print(f"  database: {args.db}")
    print(f"  tiers: core on | embeddings {'on' if health['embeddings'] else 'off'} | "
          f"symptom mapping {'on' if health['symptom_mapping'] else 'off'} | "
          f"language model {'on (' + provider.model + ')' if provider else 'off (template responses)'}")
    if session.meds:
        print("  current medications:", "; ".join(session.meds))
    print("Ask a question, or type /help. Type /quit to leave.\n")
    try:
        while True:
            try:
                line = input_fn("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not handle_command(session, line, input_fn):
                        return 0
                else:
                    answer_question(session, line)
            except (RuntimeError, ValueError) as error:
                print(f"Error: {error}", file=sys.stderr)
            print()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
