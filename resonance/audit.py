from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Iterable, Sequence

from resonance.storage import DEFAULT_DB_PATH
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


COVERAGE_WARNING_PERCENT = 80.0
EVENT_UNITS = {"boolean", "code"}
EVENT_METRIC_SUFFIXES = ("_success", "_plugged", "_code")


def audit_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    if hours <= 0:
        raise ValueError("hours must be greater than 0")

    end_utc = ensure_utc(now or utc_now()).replace(microsecond=0)
    start_utc = end_utc - timedelta(hours=hours)
    interval_seconds = (end_utc - start_utc).total_seconds()
    database_path = _display_database_path(db_path)
    report: dict[str, Any] = {
        "database_path": database_path,
        "database_exists": _database_exists(db_path),
        "audit_interval": {
            "start_utc": to_utc_iso(start_utc),
            "end_utc": to_utc_iso(end_utc),
            "hours": _clean_number(hours),
        },
        "total_measurements": 0,
        "total_collector_errors": 0,
        "stale_metrics": [],
        "metrics_with_less_than_80_percent_coverage": [],
        "metrics": [],
    }

    if not report["database_exists"]:
        return report

    conn = _connect_read_only(db_path)
    try:
        if not _table_exists(conn, "measurements"):
            return report

        rows = _fetch_measurements(conn, start_utc, end_utc)
        report["total_measurements"] = len(rows)
        if _table_exists(conn, "collector_errors"):
            report["total_collector_errors"] = _count_collector_errors(conn, start_utc, end_utc)
    finally:
        conn.close()

    metrics = [
        _summarize_metric(metric, metric_rows, start_utc, end_utc, interval_seconds)
        for metric, metric_rows in sorted(_group_by_metric(rows).items())
    ]
    report["metrics"] = metrics
    report["stale_metrics"] = [metric["metric"] for metric in metrics if metric["is_stale"]]
    report["metrics_with_less_than_80_percent_coverage"] = [
        metric["metric"]
        for metric in metrics
        if metric["coverage_percentage"] is not None
        and metric["coverage_percentage"] < COVERAGE_WARNING_PERCENT
    ]
    return report


def format_report(report: dict[str, Any]) -> str:
    interval = report["audit_interval"]
    stale_metrics = _format_name_list(report["stale_metrics"])
    low_coverage_metrics = _format_name_list(report["metrics_with_less_than_80_percent_coverage"])
    lines = [
        "Resonance Data Audit",
        f"Database path: {report['database_path']}",
        f"Audit interval: {interval['start_utc']} to {interval['end_utc']} ({interval['hours']:g} hours)",
        f"Total measurements: {report['total_measurements']}",
        f"Total collector errors: {report['total_collector_errors']}",
        f"Stale metrics: {stale_metrics}",
        f"Metrics with less than 80% coverage: {low_coverage_metrics}",
    ]

    if not report["database_exists"]:
        lines.append("Database not found; no measurements audited.")
        return "\n".join(lines)

    if not report["metrics"]:
        lines.append("No measurements found in the requested interval.")
        return "\n".join(lines)

    for metric in report["metrics"]:
        lines.extend(
            [
                "",
                f"{metric['metric']}:",
                f"  Sample count: {metric['sample_count']}",
                f"  Earliest timestamp: {_format_optional(metric['earliest_timestamp_utc'])}",
                f"  Latest timestamp: {_format_optional(metric['latest_timestamp_utc'])}",
                f"  Age of latest sample: {_format_duration(metric['latest_sample_age_seconds'])}",
                f"  Median observed sampling interval: {_format_duration(metric['median_sampling_interval_seconds'])}",
                f"  Expected sample count: {_format_optional(metric['expected_sample_count'])}",
                f"  Approximate coverage: {_format_percentage(metric['coverage_percentage'])}",
                f"  Longest gap: {_format_duration(metric['longest_gap_seconds'])}",
                f"  Duplicate timestamp count: {metric['duplicate_timestamp_count']}",
                f"  Null/non-numeric count: {metric['null_or_non_numeric_count']}",
                f"  Source values encountered: {_format_name_list(metric['source_values'])}",
            ]
        )
        if metric["minimum"] is not None:
            lines.append(
                "  Values: "
                f"min={metric['minimum']:g}, "
                f"median={metric['median']:g}, "
                f"max={metric['maximum']:g}, "
                f"std_dev={metric['standard_deviation']:g}"
            )
        elif metric["value_distribution"]:
            distribution = ", ".join(
                f"{value}={count}" for value, count in metric["value_distribution"].items()
            )
            lines.append(f"  Value distribution: {distribution}")
        else:
            lines.append("  Values: n/a")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Resonance measurement data quality.")
    parser.add_argument("--hours", type=_positive_float, default=24.0, help="Hours to audit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    args = parser.parse_args(argv)

    try:
        report = audit_database(args.database, hours=args.hours, now=now)
    except sqlite3.Error as exc:
        parser.exit(2, f"Could not audit database: {exc}\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


def _connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _database_exists(db_path: str | Path) -> bool:
    if str(db_path) == ":memory:":
        return True
    return Path(db_path).exists()


def _display_database_path(db_path: str | Path) -> str:
    if str(db_path) == ":memory:":
        return ":memory:"
    return str(Path(db_path).resolve())


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _fetch_measurements(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT timestamp_utc, metric, value, unit, source
            FROM measurements
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?
            ORDER BY metric ASC, timestamp_utc ASC, id ASC
            """,
            (to_utc_iso(start_utc), to_utc_iso(end_utc)),
        )
    )


def _count_collector_errors(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS error_count
        FROM collector_errors
        WHERE timestamp_utc >= ? AND timestamp_utc <= ?
        """,
        (to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchone()
    return int(row["error_count"])


def _group_by_metric(rows: Iterable[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metric"])].append(row)
    return dict(grouped)


def _summarize_metric(
    metric: str,
    rows: list[sqlite3.Row],
    start_utc: datetime,
    end_utc: datetime,
    interval_seconds: float,
) -> dict[str, Any]:
    timestamps: list[datetime] = []
    numeric_values: list[float] = []
    source_values: set[str] = set()
    units: set[str] = set()
    invalid_value_count = 0

    for row in rows:
        timestamp = _parse_timestamp(row["timestamp_utc"])
        if timestamp is not None:
            timestamps.append(timestamp)
        source_values.add(str(row["source"]))
        units.add(str(row["unit"]))
        value = _parse_numeric(row["value"])
        if value is None:
            invalid_value_count += 1
        else:
            numeric_values.append(value)

    unique_timestamps = sorted(set(timestamps))
    duplicate_timestamp_count = len(timestamps) - len(unique_timestamps)
    intervals = _intervals_between(unique_timestamps)
    median_interval_seconds = _clean_number(median(intervals)) if intervals else None
    expected_sample_count = _expected_sample_count(interval_seconds, median_interval_seconds)
    coverage_percentage = _coverage_percentage(len(unique_timestamps), expected_sample_count)
    longest_gap = _longest_gap(unique_timestamps)
    latest_timestamp = unique_timestamps[-1] if unique_timestamps else None
    latest_sample_age_seconds = (
        _clean_number(max(0.0, (end_utc - latest_timestamp).total_seconds()))
        if latest_timestamp
        else None
    )
    categorical = _is_categorical_metric(metric, units, numeric_values)
    value_stats = _value_statistics(numeric_values, categorical)
    value_distribution = _value_distribution(numeric_values) if categorical else {}

    return {
        "metric": metric,
        "sample_count": len(rows),
        "distinct_timestamp_count": len(unique_timestamps),
        "earliest_timestamp_utc": to_utc_iso(unique_timestamps[0]) if unique_timestamps else None,
        "latest_timestamp_utc": to_utc_iso(latest_timestamp) if latest_timestamp else None,
        "latest_sample_age_seconds": latest_sample_age_seconds,
        "median_sampling_interval_seconds": median_interval_seconds,
        "expected_sample_count": expected_sample_count,
        "coverage_percentage": coverage_percentage,
        "longest_gap_seconds": longest_gap["seconds"],
        "longest_gap_start_utc": longest_gap["start_utc"],
        "longest_gap_end_utc": longest_gap["end_utc"],
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "null_or_non_numeric_count": invalid_value_count,
        "minimum": value_stats["minimum"],
        "median": value_stats["median"],
        "maximum": value_stats["maximum"],
        "standard_deviation": value_stats["standard_deviation"],
        "value_distribution": value_distribution,
        "source_values": sorted(source_values),
        "is_stale": _is_stale(latest_sample_age_seconds, median_interval_seconds, interval_seconds),
    }


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        return parse_utc(str(value))
    except (TypeError, ValueError):
        return None


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _intervals_between(timestamps: list[datetime]) -> list[float]:
    return [
        (current - previous).total_seconds()
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]


def _expected_sample_count(
    interval_seconds: float,
    median_interval_seconds: float | int | None,
) -> int | None:
    if median_interval_seconds is None or median_interval_seconds <= 0:
        return None
    return int(math.floor(interval_seconds / float(median_interval_seconds))) + 1


def _coverage_percentage(distinct_timestamp_count: int, expected_sample_count: int | None) -> float | None:
    if not expected_sample_count:
        return None
    return _clean_number(min(100.0, (distinct_timestamp_count / expected_sample_count) * 100.0))


def _longest_gap(timestamps: list[datetime]) -> dict[str, Any]:
    if len(timestamps) < 2:
        return {"seconds": None, "start_utc": None, "end_utc": None}
    previous, current = max(
        zip(timestamps, timestamps[1:]),
        key=lambda pair: (pair[1] - pair[0]).total_seconds(),
    )
    return {
        "seconds": _clean_number((current - previous).total_seconds()),
        "start_utc": to_utc_iso(previous),
        "end_utc": to_utc_iso(current),
    }


def _is_categorical_metric(metric: str, units: set[str], values: list[float]) -> bool:
    normalized_units = {unit.lower() for unit in units}
    if normalized_units & EVENT_UNITS:
        return True
    if metric.endswith(EVENT_METRIC_SUFFIXES):
        return True
    return bool(values) and set(values).issubset({0.0, 1.0})


def _value_statistics(values: list[float], categorical: bool) -> dict[str, float | None]:
    if not values or categorical:
        return {
            "minimum": None,
            "median": None,
            "maximum": None,
            "standard_deviation": None,
        }
    return {
        "minimum": _clean_number(min(values)),
        "median": _clean_number(median(values)),
        "maximum": _clean_number(max(values)),
        "standard_deviation": _clean_number(pstdev(values) if len(values) > 1 else 0.0),
    }


def _value_distribution(values: list[float]) -> dict[str, int]:
    counter = Counter(values)
    return {_format_value(value): count for value, count in sorted(counter.items())}


def _is_stale(
    latest_sample_age_seconds: float | int | None,
    median_interval_seconds: float | int | None,
    interval_seconds: float,
) -> bool:
    if latest_sample_age_seconds is None:
        return False
    if median_interval_seconds is not None and median_interval_seconds > 0:
        return latest_sample_age_seconds > median_interval_seconds * 2
    return latest_sample_age_seconds > interval_seconds / 2


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _clean_number(value: float | int) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 6)


def _format_name_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _format_percentage(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:g}%"


def _format_duration(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:g}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:g}m"
    hours = minutes / 60
    return f"{hours:g}h"


def _format_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


if __name__ == "__main__":
    raise SystemExit(main())
