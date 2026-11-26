# Module docstring: what the script writes and the three ways to run it (no model, Claude, simulated stand-in).
"""Run the evaluation and write eval/report.md and eval/results.json.

    python eval/run_eval.py                          # deterministic backend and the filter check only
    python eval/run_eval.py --llm claude             # also the 2x2 language-model conditions (needs ANTHROPIC_API_KEY)
    python eval/run_eval.py --llm simulated          # scripted stand-in model, to see the report shape (SIMULATED)

The language-model conditions are names-only vs full-context and brief vs detailed, all on the same pairs.
"""

import argparse  # parses command-line options
import json  # writes results.json
import sys  # stderr for errors, sys.path, exit code
from pathlib import Path  # file paths

# Put the repository root on the import path so `ddi_backend` imports work when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Backend modules used by the evaluation (imports follow the path edit, hence noqa).
from ddi_backend import evaluation, ingest, llm, load_data, registry  # noqa: E402
from ddi_backend.checker import Checker  # noqa: E402  # the deterministic pair checker being evaluated
from ddi_backend.database_context import configure_console_output  # noqa: E402  # safe console encoding on Windows

HERE = Path(__file__).resolve().parent  # the eval/ folder, the default place for the outputs


def main(argv=None):
    configure_console_output()  # make printing non-ASCII text safe in the console
    # Use the first docstring line as the help text.
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)  # which medications.db to evaluate
    parser.add_argument("--llm", choices=("none", "claude", "simulated"), default="none")  # model condition to run
    parser.add_argument("--simulated-mode", choices=("oracle", "noisy"), default="noisy")  # how the stand-in behaves
    parser.add_argument("--negatives", type=int, default=60, help="pairs with no documented row to add")  # no-row pairs
    parser.add_argument("--seed", type=int, default=0)  # seed so the sampled pairs are repeatable
    # Optional cap on the pairs scored by the model, to save API calls.
    parser.add_argument("--limit", type=int, help="score only a seeded sample of this many pairs (saves API calls)")
    parser.add_argument("--out", type=Path, default=HERE)  # output folder
    args = parser.parse_args(argv)  # read the options (argv=None means sys.argv)

    try:
        conn = ingest.open_readonly(args.db, ingest.CORE_TABLES)  # open the database read-only, checking the tables
    except RuntimeError as error:  # the database is missing or incomplete
        print(f"Error: {error}", file=sys.stderr)  # report on stderr
        return 1  # failure exit code
    try:
        reg = registry.load_registry()  # source registry used by the eligibility filter
        checker = Checker(conn, registry=reg)  # the backend under test
        provider, label = None, "no language model"  # default: deterministic only
        if args.llm == "claude":  # real model requested
            provider = llm.provider_from_env()  # build the provider from ANTHROPIC_API_KEY, or None
            if provider is None:  # no key available
                print("Error: --llm claude needs ANTHROPIC_API_KEY in the environment.", file=sys.stderr)
                return 1  # stop, since the real model cannot run
            label = f"Claude ({provider.model})"  # label the report with the model name
        elif args.llm == "simulated":  # scripted stand-in requested
            # The stand-in answers from the known truth (oracle) or with seeded mistakes (noisy).
            provider = evaluation.SimulatedProvider(evaluation.ground_truth(checker), args.simulated_mode, args.seed)
            label = f"SIMULATED {args.simulated_mode} stand-in, not a real model"  # clearly mark it as simulated
        # Run every check and collect the metrics into one results dictionary.
        results = evaluation.run_evaluation(checker, reg, provider, args.negatives, args.seed, args.limit, label)
    finally:
        conn.close()  # always release the database

    args.out.mkdir(parents=True, exist_ok=True)  # make sure the output folder exists
    suffix = "" if args.llm == "none" else f"_{args.llm}"  # keep model runs from overwriting the base report
    report = args.out / f"report{suffix}.md"  # markdown report path
    report.write_text(evaluation.format_report(results), encoding="utf-8")  # write the human-readable report
    # Write the raw numbers as JSON (default=str handles values JSON cannot encode).
    (args.out / f"results{suffix}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {report}")  # tell the user where the report went
    return 0  # success exit code


# Only run when executed as a script.
if __name__ == "__main__":
    sys.exit(main())  # run main and pass its return code to the shell

# ---------------------------------------------------------------------------
# FILE SUMMARY: Command-line driver for the evaluation. It opens the database, builds the checker, optionally a real or
# simulated language-model provider, runs ddi_backend.evaluation on the same set of pairs, and writes report.md and
# results.json (suffixed with the model mode when one is used).
# ASSIGNMENT TOOL: Evaluation harness (missed Major interactions, false interactions, wrong severity, unsupported
# claims, the 2x2 language-model conditions).
# ---------------------------------------------------------------------------
