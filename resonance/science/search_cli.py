from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from resonance.science.ledger import DEFAULT_LEDGER_PATH
from resonance.science.program_search import DEFAULT_BEAM_WIDTH, DEFAULT_BUDGET, load_seed_hypotheses, run_program_search
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded exploration/tuning scientific program search.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=f"science artifact root; defaults to {DEFAULT_ARTIFACT_ROOT}",
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"scientific ledger path; defaults to {DEFAULT_LEDGER_PATH}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="fit, tune-rank, and ledger a bounded search")
    run.add_argument("--snapshot", required=True, help="snapshot id to search without blind access")
    run.add_argument(
        "--seed-hypothesis",
        action="append",
        required=True,
        help="path to a hypothesis JSON file; may be repeated and may contain one object or a list",
    )
    run.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    run.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    run.add_argument("--complexity-penalty", type=float, default=0.001)
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument(
        "--no-ledger",
        action="store_true",
        help="run locally without appending a program_search_completed ledger event",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        seeds = load_seed_hypotheses([Path(path) for path in args.seed_hypothesis])
        result = run_program_search(
            seeds,
            snapshot_id=args.snapshot,
            budget=args.budget,
            beam_width=args.beam_width,
            complexity_penalty=args.complexity_penalty,
            random_seed=args.random_seed,
            artifact_root=Path(args.artifact_root),
            ledger_path=Path(args.ledger),
            record_ledger=not args.no_ledger,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
