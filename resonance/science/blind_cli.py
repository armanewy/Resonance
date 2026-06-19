from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from resonance.science.blind_evaluator import BlindEvaluationError, evaluate_preregistration
from resonance.science.ledger import DEFAULT_LEDGER_PATH
from resonance.science.preregistration import load_preregistration
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run sealed one-shot blind scientific evaluation.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=f"root directory for sealed snapshot and metrics artifacts; defaults to {DEFAULT_ARTIFACT_ROOT}",
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"scientific ledger path; defaults to {DEFAULT_LEDGER_PATH}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash", help="calculate a preregistration hash")
    hash_parser.add_argument("preregistration_file", help="JSON preregistration file")

    evaluate_parser = subparsers.add_parser("evaluate", help="spend the one blind evaluation")
    evaluate_parser.add_argument("preregistration_hash", help="64-character preregistration hash")
    evaluate_parser.add_argument("preregistration_file", help="JSON preregistration file")

    args = parser.parse_args(argv)
    if args.command == "hash":
        preregistration = _read_preregistration(args.preregistration_file)
        print(preregistration.preregistration_hash())
        return 0
    if args.command == "evaluate":
        try:
            result = evaluate_preregistration(
                _read_preregistration(args.preregistration_file),
                args.preregistration_hash,
                artifact_root=Path(args.artifact_root),
                ledger_path=Path(args.ledger),
            )
        except BlindEvaluationError as exc:
            print(f"Blind evaluation refused: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _read_preregistration(path: str | Path):
    return load_preregistration(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    raise SystemExit(main())
