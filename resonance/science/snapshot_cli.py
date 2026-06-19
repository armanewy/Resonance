from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from resonance.science.snapshots import (
    DEFAULT_ARTIFACT_ROOT,
    create_snapshot,
    parse_metric_csv,
)
from resonance.storage import DEFAULT_DB_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create sealed scientific data snapshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a reproducible snapshot")
    create.add_argument("--db", default=str(DEFAULT_DB_PATH), help="source SQLite database path")
    create.add_argument("--hours", type=int, required=True, help="lookback window ending at latest row")
    create.add_argument(
        "--metrics",
        required=True,
        help="comma-separated metric names, for example tcp_latency_ms,dns_latency_ms",
    )
    create.add_argument(
        "--max-lag-seconds",
        type=int,
        required=True,
        help="maximum searched lag; used as the split embargo duration",
    )
    create.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="root directory for content-addressed snapshot artifacts",
    )

    args = parser.parse_args(argv)
    if args.command == "create":
        manifest = create_snapshot(
            db_path=Path(args.db),
            hours=args.hours,
            metrics=parse_metric_csv(args.metrics),
            max_lag_seconds=args.max_lag_seconds,
            artifact_root=Path(args.artifact_root),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
