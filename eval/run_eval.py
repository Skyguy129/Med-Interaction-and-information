"""Run the evaluation and write eval/report.md and eval/results.json.

    python eval/run_eval.py                          # deterministic backend and the filter check only
    python eval/run_eval.py --llm openai             # also the 2x2 language-model conditions (needs OPENAI_API_KEY)
    python eval/run_eval.py --llm simulated          # scripted stand-in model, to see the report shape (SIMULATED)

The language-model conditions are names-only vs full-context and brief vs detailed, all on the same pairs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddi_backend import evaluation, ingest, llm, load_data, registry  # noqa: E402
from ddi_backend.checker import Checker  # noqa: E402
from ddi_backend.database_context import configure_console_output  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(argv=None):
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=load_data.DEFAULT_DB)
    parser.add_argument("--llm", choices=("none", "openai", "simulated"), default="none")
    parser.add_argument("--simulated-mode", choices=("oracle", "noisy"), default="noisy")
    parser.add_argument("--negatives", type=int, default=60, help="pairs with no documented row to add")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="score only a seeded sample of this many pairs (saves API calls)")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv)

    try:
        conn = ingest.open_readonly(args.db, ingest.CORE_TABLES)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        reg = registry.load_registry()
        checker = Checker(conn, registry=reg)
        provider, label = None, "no language model"
        if args.llm == "openai":
            provider = llm.provider_from_env()
            if provider is None:
                print("Error: --llm openai needs OPENAI_API_KEY in the environment.", file=sys.stderr)
                return 1
            label = f"openai {provider.model}"
        elif args.llm == "simulated":
            provider = evaluation.SimulatedProvider(evaluation.ground_truth(checker), args.simulated_mode, args.seed)
            label = f"SIMULATED {args.simulated_mode} stand-in, not a real model"
        results = evaluation.run_evaluation(checker, reg, provider, args.negatives, args.seed, args.limit, label)
    finally:
        conn.close()

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.llm == "none" else f"_{args.llm}"
    report = args.out / f"report{suffix}.md"
    report.write_text(evaluation.format_report(results), encoding="utf-8")
    (args.out / f"results{suffix}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
