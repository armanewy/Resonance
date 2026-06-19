from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from resonance.science.ledger import (
    DEFAULT_LEDGER_PATH,
    LedgerError,
    read_entries,
    verify_ledger,
    verify_ledger_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the tamper-evident scientific ledger.")
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"Ledger JSONL path. Defaults to {DEFAULT_LEDGER_PATH}.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify ledger sequence and hashes.")
    verify_parser.add_argument(
        "--artifact-root",
        help="Also verify path-bearing artifacts relative to this root.",
    )

    show_parser = subparsers.add_parser("show", help="Show recent verified ledger entries.")
    show_parser.add_argument("--limit", type=_positive_int, default=20, help="Number of entries to show.")

    args = parser.parse_args(argv)
    ledger_path = Path(args.ledger)

    if args.command == "verify":
        verification = verify_ledger(ledger_path)
        errors = list(verification.errors)
        if verification.valid and args.artifact_root is not None:
            errors.extend(verify_ledger_artifacts(ledger_path, artifact_root=args.artifact_root))
        if errors:
            print("Ledger verification failed:")
            for error in errors:
                print(f"- {error}")
            return 1
        head = verification.head_hash or "none"
        print(f"Ledger verified: {verification.entry_count} entries; head={head}")
        if args.artifact_root is not None:
            print("Ledger artifacts verified")
        return 0

    if args.command == "show":
        try:
            entries = read_entries(ledger_path, limit=args.limit)
        except LedgerError as exc:
            print(f"Ledger verification failed: {exc}")
            return 1
        for entry in entries:
            print(json.dumps(entry, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

