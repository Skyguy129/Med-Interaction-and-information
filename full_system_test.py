#!/usr/bin/env python3
# Module docstring: explains the interactive test and the six pipeline steps it prints for every question.
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

import argparse  # command-line options
import getpass  # reads an API key without echoing it
import json  # builds and parses the model's JSON messages
import os  # reads OPENAI_MODEL
import re  # splits the medication list and matches names in text
import sys  # stderr, sys.path, exit code
from pathlib import Path  # file paths

# Put the repository root on the import path so the package imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Backend pieces this script drives (imports follow the path edit, hence noqa).
from ddi_backend import composer, llm, load_data, state  # noqa: E402  # template text, model code, db path, state file
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE, severity_rank  # noqa: E402  # texts and severity ordering
from ddi_backend.database_context import configure_console_output, format_record  # noqa: E402  # console, records
from ddi_backend.agent import format_trace  # noqa: E402  # readable review-request trace
from ddi_backend.service import App  # noqa: E402  # shared setup: database, checker, qa, agent, state

# The in-chat help text, printed by /help. (Multi-line string, so its notes are here, not inside it.)
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

# System prompt for list-style answers: no tools, no outside knowledge, JSON out, records' own wording only.
NARRATE_SYSTEM = (
    "You write a short plain-language answer for a class prototype that is not medical advice. You have no tools "  # role
    "and no internet, and you must not use anything you know or remember. The only permitted source is the records "
    "in the user message. Return a JSON object with exactly the keys summary (string) and record_ids (list of IDs "  # format
    "taken from the records). Use the records' own wording; add no facts, names, numbers, doses, advice or "
    "reassurance, and never say a combination is safe."  # never present absence as safety
)


class Session:  # holds everything that changes during one chat
    def __init__(self, app, provider, show_steps=True):
        self.app = app  # the shared backend
        self.provider = provider  # the language-model provider, or None
        self.use_llm = provider is not None  # model use starts on only when a provider exists
        self.show_steps = show_steps  # whether to print the pipeline steps
        self.meds = []  # the user's current medications as typed
        self.settings = app.settings  # detail and context levels, changeable with commands


def step(session, number, title, detail=""):
    if session.show_steps:  # only print when steps are shown
        print(f"  [{number}] {title}" + (f": {detail}" if detail else ""))  # e.g. "  [3] lookup and rules fired: ..."


# --------------------------------------------------------------------------
# Writing the response
# --------------------------------------------------------------------------
def make_explainer(session):
    provider = session.provider if session.use_llm else None  # no provider means template-only text
    return llm.Explainer(provider, session.settings, session.app.explainer.known_drug_names)  # validated explainer


def validate_narration(reply, records, known_names):
    # Docstring: this is the safety check for list-style model answers.
    """Check a list-style narration against the records it was given. Returns (parsed, problems)."""
    try:
        parsed = json.loads(reply)  # the reply must be JSON
    except (TypeError, json.JSONDecodeError):
        return None, ["reply is not valid JSON"]
    if not isinstance(parsed, dict) or set(parsed) != {"summary", "record_ids"}:  # exactly these two keys
        return None, ["reply must have exactly summary and record_ids"]
    summary, ids = parsed["summary"], parsed["record_ids"]  # pull out the two fields
    if not isinstance(summary, str) or not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        return None, ["summary or record_ids has the wrong type"]  # wrong types or no cited ids
    supplied = {r["id"]: r for r in records}  # the records the model was allowed to use, by id
    problems = [f"cites {i!r}, which was not supplied" for i in ids if i not in supplied]  # citations must be real
    corpus = llm._record_text(records)  # all text of the supplied records
    corpus_words = {w.casefold() for w in llm._WORD.findall(corpus)}  # every word the records contain
    severities = {r["severity"].casefold() for r in records}  # severities the records really have
    for word in llm._WORD.findall(summary):  # check each word of the model's summary
        lower = word.casefold()
        if lower in llm._SEVERITY_WORDS:  # a severity word must match a real record
            if lower not in severities:
                problems.append(f"states severity {word}, which the records do not")
        elif len(lower) >= 4 and lower not in llm.ALLOWED_PLAIN_WORDS and lower not in corpus_words:
            problems.append(f"uses {word!r}, which is not in the records")  # a new content word is a new fact
    # Any number in the summary must appear in the records.
    problems += [f"contains the number {n}, which is not in the records"
                 for n in llm._NUMBER.findall(summary) if n not in corpus]
    # Drug names the summary may mention: those in the records and their alternatives.
    allowed = {llm._norm(r["drug_a"]) for r in records} | {llm._norm(r["drug_b"]) for r in records}
    for r in records:
        allowed |= {llm._norm(n) for n in r["alternative"]["drug_names"]}
    text = llm._norm(summary)  # normalised summary for name matching
    for name in known_names:  # every drug in the database
        key = llm._norm(name)
        # A known drug name that is not in the records must not appear as a whole word.
        if key and key not in allowed and re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", text):
            problems.append(f"names {name}, which is not in the records")
    if len(summary) > 1200:  # keep answers short
        problems.append("summary is too long")
    return (None, problems) if problems else (parsed, [])  # reject on any problem


def narrate(session, question, records, fallback_text):
    # Docstring: the model's answer when allowed and valid, else the template text.
    """The model's plain-language answer for a list-style result, or the template text. Returns (text, how, why)."""
    if not (session.use_llm and session.provider and records):  # no model call is possible or needed
        why = "language model is off" if not session.use_llm else "no language model configured"
        return fallback_text, "template", why if records else "no eligible records, so no model call"
    # What the model may see: the question and a trimmed copy of each record (absent fields left out).
    payload = {"question": question, "detail_level": session.settings.detail_level,
               "records": [{k: r.get(k) for k in ("id", "severity", "drug_a", "drug_b", "mechanism", "effect")
                            if r.get(k) is not None} for r in records]}
    try:
        # One call with the strict system prompt and the records as JSON.
        reply = session.provider.complete([{"role": "system", "content": NARRATE_SYSTEM},
                                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    except RuntimeError as error:  # network or key problem
        return fallback_text, "template", f"language model unavailable: {error}"
    parsed, problems = validate_narration(reply, records, session.app.explainer.known_drug_names)  # safety check
    if problems:  # any new fact means the template text is used instead
        return fallback_text, "template", "reply failed validation: " + "; ".join(problems[:3])
    return parsed["summary"], "language model", "reply passed validation"  # the validated model text


def records_for(answer, graph):
    # Docstring: gathers the records behind avoid lists or interaction lists, worst first.
    """Records behind a non-pair answer, most severe first."""
    seen, out = set(), []  # ids already added, and the ordered records
    # For each avoid-list entry take its first record id, and look the full record up in the graph.
    for rid in [rid for e in answer.get("avoid", {}).get("avoid", []) for rid in e["record_ids"][:1]]:
        rec = next((r for r in graph.records if r["id"] == rid), None)
        if rec and rid not in seen:  # skip unknown or repeated ids
            seen.add(rid)
            out.append(rec)
    for rec in answer.get("interactions", []):  # records returned directly by the question handler
        if rec["id"] not in seen:
            seen.add(rec["id"])
            out.append(rec)
    return sorted(out, key=lambda r: (severity_rank(r["severity"]), r["id"]))  # Major first, then by id


# --------------------------------------------------------------------------
# One question through the whole system
# --------------------------------------------------------------------------
def answer_question(session, question):
    app = session.app  # shorthand
    print("\nProcessing")
    with app.lock:  # one use of the shared connection at a time
        answer = app.qa.ask(question)  # steps 1 to 4 happen inside the question handler

    # Step 1-2: names and question rules
    mentions = [m["name"] for m in answer.get("inputs", {}).get("current", []) if m.get("name")]  # resolved current drugs
    if answer.get("inputs", {}).get("new", {}).get("name"):  # plus the new drug if resolved
        mentions.append(answer["inputs"]["new"]["name"])
    step(session, 1, "names found", ", ".join(mentions) or answer.get("drug") or "none resolved")
    for item in answer.get("clarifications", []):  # names that did not resolve
        step(session, 1, f"could not resolve '{item['input']}'", item["status"] +
             (f" (did you mean: {', '.join(item['did_you_mean'])})" if item.get("did_you_mean") else ""))
    step(session, 2, "question type", answer.get("intent", "?") +
         (f" (negation words: {', '.join(answer['negation_words'])})" if answer.get("negation_words") else ""))

    # A single new drug plus current medications is the main "should I start this?" flow.
    if answer.get("intent") == "drug" and session.meds:
        step(session, 3, "your current medications are set", f"checking {answer['drug']} against {len(session.meds)} of them")
        with app.lock:
            out = app.checker.check(session.meds, answer["drug"])  # pair check against the saved medications
        answer.update(status="answered", intent="pair", results=out["results"], inputs=out["inputs"])  # reuse as pair answer
        answer["rules_fired"] = list(dict.fromkeys(answer["rules_fired"] + out["rules_fired"]))  # merge, keep order
        answer["notes"] += out["notes"]  # keep the checker's notes too

    step(session, 3, "lookup and rules fired", ", ".join(answer["rules_fired"]) or "none")
    step(session, 4, "eligibility filter", "applied to every record before it can count as evidence")
    for note in answer["notes"]:  # show notes as they apply
        step(session, "!", "note", note)

    print()
    if answer.get("results"):  # pair results: one explanation per pair
        explainer = make_explainer(session)
        names = [m["name"] for m in answer.get("inputs", {}).get("current", []) if m.get("name")]  # current drug names
        names.append(answer["inputs"]["new"].get("name"))  # plus the new drug
        shown = []  # explanations to print after the steps
        for result in answer["results"]:
            # Other medications for context: saved ones not in this pair, else resolved names not in this pair.
            # The line continues with a backslash, so this note sits above it.
            others = [n for n in session.meds if n and n.split()[0] not in result["pair"]] or \
                [n for n in names if n and n not in result["pair"]]
            explained = explainer.explain_pair(result, other_medications=others)  # model or template text
            how = "language model" if explained["generated_by"] == "language_model" else "template"
            step(session, 5, f"response for {' + '.join(result['pair'])}",
                 f"{how}" + (f" ({explained['fallback_reason']})" if explained.get("fallback_reason") and how == "template" else ""))
            shown.append(explained)
        print()
        for explained in shown:  # print each pair's answer beside its source records
            print(f"== {' + '.join(explained['pair'])}: {explained['status']} ==")  # heading with the pair status
            print(explained["text"])  # the plain-language response
            for record in explained["records"]:
                print(format_record(record))  # the source record
            if explained.get("clarifying_question"):  # missing patient context becomes a question
                print(explained["clarifying_question"])
            print()
    elif answer.get("avoid") or answer.get("interactions"):  # list-style answers
        records = records_for(answer, app.checker.graph)  # the records behind the answer
        # Template text used when the model is off or rejected.
        fallback = (composer.compose_avoid(answer["avoid"], session.settings.detail_level) if answer.get("avoid")
                    else "\n".join(f"[{r['severity']}] {r['drug_a']} + {r['drug_b']} ({r['id']}): {r['effect']}"
                                   for r in records))
        text, how, why = narrate(session, question, records, fallback)  # model text if valid, else fallback
        step(session, 5, "response", f"{how} ({why})")
        print(f"\n{text}")
        if how == "language model":  # show the rules' result beside the model's wording
            print("\nStructured result (what the rules found, shown beside the model's wording):")
            print(fallback)
        if answer.get("phenotype_records"):  # symptom-vocabulary matches, shown separately
            print(answer["phenotype_note"] + " " + ", ".join(p["label"] for p in answer["phenotype_records"][:5]))
        print("\nSource records:")
        for record in records[:8]:  # show up to eight sources
            print(format_record(record))
        if len(records) > 8:  # say how many were left out
            print(f"... and {len(records) - 8} more (use /detail detailed or the CLI for the full list).")
    else:  # nothing to look up: clarifications, or nothing matched
        step(session, 5, "response", "template (nothing to look up yet)")
        for item in answer.get("clarifications", []):  # questions about unresolved names
            print(item["question"] or "")
            if item.get("did_you_mean"):
                print("  Did you mean: " + ", ".join(item["did_you_mean"]) + "?")
        if answer.get("clarifying_question"):
            print(answer["clarifying_question"])
        if not answer.get("clarifications") and not answer.get("clarifying_question"):
            # Never imply safety: say nothing documented matched.
            print(answer["notes"][-1] if answer["notes"] else "Nothing documented in our data matched.")
    for note in answer["notes"]:  # repeat the notes after the answer
        print("Note:", note)
    print("\n" + DISCLAIMER)  # always end with the disclaimer


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def parse_meds(text):
    # Split on commas or semicolons and drop blanks, keeping doses attached to names.
    return [m.strip() for m in re.split(r"[,;]", text) if m.strip()]


def handle_command(session, line, input_fn):
    # Docstring: the return value tells the loop whether to keep going.
    """Returns False to quit."""
    command, _, rest = line.partition(" ")  # first word is the command, the rest is its argument
    command, rest = command.casefold(), rest.strip()  # commands are case-insensitive
    if command in ("/quit", "/exit"):
        return False  # tell the loop to stop
    if command == "/help":
        print(HELP)
    elif command == "/meds":  # set, show or clear the medication list
        if rest.casefold() == "clear":
            session.meds = []
        elif rest:
            session.meds = parse_meds(rest)
        print("Current medications:", "; ".join(session.meds) if session.meds else "none set")
    elif command == "/new":  # check one new drug against the saved medications
        if not rest:
            print("Usage: /new DRUG")
        else:
            check_new(session, rest)
    elif command == "/detail":  # change how much record text goes into the answer
        if rest in composer.DETAIL_LEVELS:
            session.settings = llm.Settings(session.settings.context_level, rest)  # keep the other setting
            print("detail_level =", rest)
        else:
            print("Choose one of:", ", ".join(composer.DETAIL_LEVELS))
    elif command == "/context":  # change what the model may see
        if rest in llm.CONTEXT_LEVELS:
            session.settings = llm.Settings(rest, session.settings.detail_level)  # keep the other setting
            print("context_level =", rest)
        else:
            print("Choose one of:", ", ".join(llm.CONTEXT_LEVELS))
    elif command == "/llm":  # turn the model on or off for this session
        if rest == "on" and session.provider is None:  # cannot turn on without a key
            print("No language model is configured (set OPENAI_API_KEY and restart).")
        elif rest in ("on", "off"):
            session.use_llm = rest == "on"
            print("language model:", "on" if session.use_llm else "off")
        else:
            print("Usage: /llm on|off")
    elif command == "/steps":  # show or hide the pipeline steps
        session.show_steps = rest != "off"
        print("pipeline steps:", "shown" if session.show_steps else "hidden")
    elif command == "/stage":  # ask the guarded agent to stage a review (simulated)
        parts = rest.split()
        if len(parts) != 3:  # needs two drugs and a request id
            print("Usage: /stage DRUG_A DRUG_B REQUEST_ID")
        else:
            print(SIMULATED_NOTE)  # make clear this is simulated
            # Only an explicit "yes" counts as approval.
            approved = input_fn("Do you approve staging this interaction for simulated pharmacist review? (yes/no) ").strip().casefold() == "yes"
            with session.app.lock:
                outcome = session.app.agent.request_review(parts[:2], parts[2], approved=approved)  # gates run inside
            print(f"Result: {outcome['status']}  (queue {len(outcome['state_before']['review_queue'])} -> "
                  f"{len(outcome['state_after']['review_queue'])})")  # status and queue size before and after
    elif command == "/trace":  # show what happened for one request
        print(format_trace(rest, session.app.state.attempts(rest)) if rest else "Usage: /trace REQUEST_ID")
    else:
        print("Unknown command. Type /help.")
    return True  # keep running


def check_new(session, drug):
    if not session.meds:  # nothing to check against
        print("Set your current medications first, for example: /meds Warfarin 5 mg, Omeprazole")
        return
    app = session.app
    print("\nProcessing")
    with app.lock:
        out = app.checker.check(session.meds, drug)  # deterministic pair checks for the new drug
    step(session, 1, "names", ", ".join(m["name"] or m["input"] for m in out["inputs"]["current"]) + f" | new: {out['inputs']['new']['name'] or drug}")
    step(session, 3, "rules fired", ", ".join(out["rules_fired"]))
    explainer = make_explainer(session)  # model or template writer
    print()
    for explained in explainer.explain_check(out):  # one explanation per pair
        how = "language model" if explained["generated_by"] == "language_model" else "template"
        step(session, 5, f"response for {' + '.join(explained['pair'])}", how +
             (f" ({explained['fallback_reason']})" if how == "template" and explained.get("fallback_reason") else ""))
        print(f"\n== {' + '.join(explained['pair'])}: {explained['status']} ==")  # heading with the pair status
        print(explained["text"])  # the plain-language response
        for record in explained["records"]:
            print(format_record(record))  # the source record beside it
    for note in out["notes"]:
        print("Note:", note)
    print("\n" + DISCLAIMER)


# --------------------------------------------------------------------------
# Start-up and the loop
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # help text from the docstring's first line
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)  # database file
    parser.add_argument("--state", type=Path, default=state.DEFAULT_STATE_DB)  # review-queue state file
    parser.add_argument("--memory", action="store_true", help="keep the review queue in memory only")
    parser.add_argument("--embeddings", choices=("off", "auto"), default="off")  # tier two on or off
    parser.add_argument("--no-llm", action="store_true", help="never use a language model")
    parser.add_argument("--no-key-prompt", action="store_true", help="do not ask for an API key when none is set")
    parser.add_argument("--detail", choices=composer.DETAIL_LEVELS, default="standard")  # starting detail level
    parser.add_argument("--context", choices=llm.CONTEXT_LEVELS, default="full_context")  # starting context level
    parser.add_argument("--hide-steps", action="store_true")  # start with the steps hidden
    parser.add_argument("--meds", help="current medications, comma separated, to start with")
    return parser


# input_fn and key_fn can be replaced so tests can script the conversation.
def main(argv=None, input_fn=input, key_fn=getpass.getpass):
    configure_console_output()  # safe console encoding
    args = build_parser().parse_args(argv)  # read the options
    provider = None  # no language model unless one is configured
    if not args.no_llm:
        provider = llm.provider_from_env()  # use OPENAI_API_KEY if it is set
        if provider is None and not args.no_key_prompt:  # offer to take a key for this session only
            key = key_fn("OPENAI_API_KEY is not set. Paste a key for this session only (Enter to skip): ").strip()
            if key:
                provider = llm.OpenAIProvider(key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))  # key is never stored
    try:
        # Build the shared backend with the chosen options.
        app = App(db_path=args.db, state_path=args.state, embeddings=args.embeddings,
                  settings=llm.Settings(args.context, args.detail), provider=provider,
                  in_memory_state=args.memory)
    except (OSError, RuntimeError) as error:  # e.g. medications.db missing
        print(f"Error: {error}", file=sys.stderr)
        return 1
    session = Session(app, provider, show_steps=not args.hide_steps)  # start the chat state
    if args.meds:  # medications given on the command line
        session.meds = parse_meds(args.meds)
    health = app.health()["tiers"]  # which tiers are working
    print("Full system test (class prototype, not medical advice)")
    print(f"  database: {args.db}")
    print(f"  tiers: core on | embeddings {'on' if health['embeddings'] else 'off'} | "
          f"symptom mapping {'on' if health['symptom_mapping'] else 'off'} | "
          f"language model {'on (' + provider.model + ')' if provider else 'off (template responses)'}")
    if session.meds:
        print("  current medications:", "; ".join(session.meds))
    print("Ask a question, or type /help. Type /quit to leave.\n")
    try:
        while True:  # the chat loop
            try:
                line = input_fn("You: ").strip()  # read one line
            except (EOFError, KeyboardInterrupt):  # Ctrl-D, Ctrl-C or end of scripted input
                print()
                return 0
            if not line:  # ignore blank lines
                continue
            try:
                if line.startswith("/"):  # a command
                    if not handle_command(session, line, input_fn):  # False means quit
                        return 0
                else:
                    answer_question(session, line)  # anything else is a question
            except (RuntimeError, ValueError) as error:  # keep the chat going after an expected error
                print(f"Error: {error}", file=sys.stderr)
            print()
    finally:
        app.close()  # always release the database and state file


# Only run when executed as a script.
if __name__ == "__main__":
    sys.exit(main())  # run the chat and pass its return code to the shell

# ---------------------------------------------------------------------------
# FILE SUMMARY: An interactive terminal run of the whole system. For each typed question it calls the real question
# handler and checker, prints the six pipeline steps (names, question rules, lookup and rules, eligibility filter,
# response writer, answer with source records), optionally lets the language model word list-style answers after a
# validator rejects any new fact, and supports commands to set medications, change settings and stage a simulated review.
# ASSIGNMENT TOOL: Supporting infrastructure that demonstrates all of them end to end: the knowledge graph + rules core,
# eligibility filtering, the language model (explain/narrate with validation), embeddings (optional tier) and the guarded
# agent with its simulated review queue.
# ---------------------------------------------------------------------------
