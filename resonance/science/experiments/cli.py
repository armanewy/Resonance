from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from resonance.science.experiments.evaluator import ExperimentEvaluationError, evaluate_experiment
from resonance.science.experiments.runner import (
    ExperimentRunnerError,
    begin_block,
    confirm_condition,
    end_block,
    experiment_status,
    preregister_experiment,
    start_experiment,
)
from resonance.science.ledger import DEFAULT_LEDGER_PATH
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT
from resonance.storage import DEFAULT_DB_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except (
        ExperimentEvaluationError,
        ExperimentRunnerError,
        FileNotFoundError,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"Experiment command failed: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run low-risk manual controlled experiments.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=f"content-addressed artifact root; defaults to {DEFAULT_ARTIFACT_ROOT}",
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"scientific ledger path; defaults to {DEFAULT_LEDGER_PATH}",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Resonance SQLite database path; defaults to {DEFAULT_DB_PATH}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preregister = subparsers.add_parser("preregister", help="freeze a manual experiment spec")
    preregister.add_argument("spec_json")

    for command in ("start", "begin-block", "confirm-condition", "status", "evaluate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("experiment_id")

    end = subparsers.add_parser("end-block")
    end.add_argument("experiment_id")
    end.add_argument("--missed", action="store_true", help="record the active block as missed")
    end.add_argument(
        "--noncompliant",
        action="store_true",
        help="record the active block as noncompliant and exclude it from evaluation",
    )
    end.add_argument("--note", default="", help="optional missed/noncompliance note")

    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = Path(args.artifact_root)
    ledger_path = Path(args.ledger)
    if args.command == "preregister":
        return preregister_experiment(args.spec_json, artifact_root=artifact_root, ledger_path=ledger_path)
    if args.command == "start":
        return start_experiment(args.experiment_id, artifact_root=artifact_root, ledger_path=ledger_path)
    if args.command == "begin-block":
        return begin_block(args.experiment_id, artifact_root=artifact_root, ledger_path=ledger_path)
    if args.command == "confirm-condition":
        return confirm_condition(args.experiment_id, artifact_root=artifact_root, ledger_path=ledger_path)
    if args.command == "end-block":
        return end_block(
            args.experiment_id,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            missed=args.missed,
            compliant=not args.noncompliant,
            note=args.note,
        )
    if args.command == "status":
        return experiment_status(args.experiment_id, artifact_root=artifact_root)
    if args.command == "evaluate":
        return evaluate_experiment(
            args.experiment_id,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            db_path=Path(args.db),
        )
    raise ExperimentRunnerError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
