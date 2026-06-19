from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from resonance.analysis.scanner import finding_to_dict, scan_correlations
from resonance.storage import DEFAULT_DB_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the conservative local Resonance correlation scanner.")
    parser.add_argument("--hours", type=_positive_float, required=True, help="Lookback window in hours.")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute findings without writing them.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of JSON lines.")
    args = parser.parse_args(argv)

    findings = scan_correlations(
        Path(args.database),
        hours=args.hours,
        dry_run=args.dry_run,
    )
    rows = [finding_to_dict(finding) for finding in findings]
    if not rows:
        return 0

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
