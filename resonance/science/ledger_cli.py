from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from resonance.science.ledger import DEFAULT_LEDGER_PATH, LedgerError, read_entries, verify_ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the tamper-evident scientific ledger.")
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"Ledger JSONL path. Defaults to {DEFAULT_LEDGER_PATH}.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Verify ledger sequence and hashes.")

    show_parser = subparsers.add_parser("show", help="Show recent verified ledger entries.")
    show_parser.add_argument("--limit", type=_positive_int, default=20, help="Number of entries to show.")

    args = parser.parse_args(argv)
    ledger_path = Path(args.ledger)

    if args.command == "verify":
        verification = verify_ledger(ledger_path)
        if verification.valid:
            head = verification.head_hash or "none"
            print(f"Ledger verified: {verification.entry_count} entries; head={head}")
            return 0
        print("Ledger verification failed:")
        for error in verification.errors:
            print(f"- {error}")
        return 1

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

