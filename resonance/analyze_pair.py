from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from resonance.analysis import (
    TRANSFORMS,
    PairAnalysis,
    ValidationResult,
    align_series,
    apply_transform,
    chronological_holdout_validation,
    lagged_spearman,
    max_lag_block_permutation_test,
    window_stability,
)
from resonance.analysis.contracts import AlignedPair
from resonance.storage import DEFAULT_DB_PATH
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


MIN_ALIGNED_POINTS = 30
MIN_OVERLAP = 30
HOLDOUT_FRACTION = 0.25
WINDOW_COUNT = 4
PERMUTATIONS = 199


def analyze_pair(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    x_metric: str,
    y_metric: str,
    hours: float,
    transform_name: str,
    max_lag_minutes: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if hours <= 0:
        raise ValueError("hours must be greater than 0")
    if max_lag_minutes < 0:
        raise ValueError("max_lag_minutes must be non-negative")
    if transform_name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {transform_name}")

    end_utc = ensure_utc(now or utc_now()).replace(microsecond=0)
    start_utc = end_utc - timedelta(hours=hours)
    base_report = _base_report(
        db_path,
        x_metric=x_metric,
        y_metric=y_metric,
        hours=hours,
        start_utc=start_utc,
        end_utc=end_utc,
        transform_name=transform_name,
        max_lag_minutes=max_lag_minutes,
    )

    if not _database_exists(db_path):
        return _insufficient(base_report, "database not found")

    try:
        conn = _connect_read_only(db_path)
    except sqlite3.Error as exc:
        return _insufficient(base_report, f"could not open database read-only: {exc}")

    try:
        if not _table_exists(conn, "measurements"):
            return _insufficient(base_report, "measurements table not found")
        rows = _fetch_metric_rows(conn, start_utc, end_utc, (x_metric, y_metric))
    finally:
        conn.close()

    x_series = _series_from_rows(rows, x_metric)
    y_series = _series_from_rows(rows, y_metric)
    if x_series.empty or y_series.empty:
        missing = []
        if x_series.empty:
            missing.append(x_metric)
        if y_series.empty:
            missing.append(y_metric)
        return _insufficient(base_report, f"no measurements for metric(s): {', '.join(missing)}")

    try:
        raw_pair = align_series(x_series, y_series, min_points=MIN_ALIGNED_POINTS)
        transformed_pair = _transform_aligned_pair(raw_pair, transform_name)
        max_lag_steps = _max_lag_steps(transformed_pair.cadence_seconds, max_lag_minutes)
        if max_lag_steps == 0 and max_lag_minutes > 0:
            return _insufficient(
                base_report,
                "max lag is shorter than the aligned cadence; no lag scan can be performed",
            )

        discovery_frame = _discovery_frame(transformed_pair.frame, HOLDOUT_FRACTION)
        lag_result = lagged_spearman(
            discovery_frame,
            max_lag_steps=max_lag_steps,
            min_overlap=MIN_OVERLAP,
        )
        if lag_result.best_rho is None:
            return _insufficient(
                base_report,
                "no discovery lag met the minimum overlap and variance requirements",
            )

        candidate_lags = range(-max_lag_steps, max_lag_steps + 1)
        validation_frame = _records_from_frame(transformed_pair.frame)
        holdout = chronological_holdout_validation(
            validation_frame,
            candidate_lag_steps=(lag_result.best_lag_steps,),
            holdout_fraction=HOLDOUT_FRACTION,
            min_overlap=MIN_OVERLAP,
        )
        p_value = max_lag_block_permutation_test(
            validation_frame,
            candidate_lag_steps=candidate_lags,
            permutations=PERMUTATIONS,
            min_overlap=MIN_OVERLAP,
        )
        stability = window_stability(
            validation_frame,
            lag_steps=lag_result.best_lag_steps,
            window_count=WINDOW_COUNT,
            min_overlap=MIN_OVERLAP,
        )
    except ValueError as exc:
        return _insufficient(base_report, str(exc))

    validation_result = ValidationResult(
        permutation_p_value=p_value,
        holdout_rho=holdout.holdout_rho,
        holdout_overlap=holdout.holdout_overlap,
        sign_stability=stability.sign_stability,
        window_scores=stability.window_scores,
        warnings=tuple(dict.fromkeys((*holdout.warnings, *stability.warnings))),
    )
    analysis = PairAnalysis(
        aligned_pair=transformed_pair,
        transform_name=transform_name,
        lag_result=lag_result,
        validation_result=validation_result,
    )
    return _analysis_report(base_report, analysis)


def format_report(report: dict[str, Any]) -> str:
    interval = report["interval"]
    lines = [
        "Resonance Pair Analysis",
        f"Database path: {report['database_path']}",
        f"Metrics: {report['x_metric']} -> {report['y_metric']}",
        f"Interval: {interval['start_utc']} to {interval['end_utc']} ({interval['hours']:g} hours)",
        f"Transform: {report['transform']}",
    ]

    if report["status"] != "ok":
        lines.extend(
            [
                f"Status: {report['status']}",
                f"Reason: {report['reason']}",
                "No association estimate was produced.",
                report["causation_warning"],
            ]
        )
        return "\n".join(lines)

    aligned = report["aligned"]
    lag = report["lag"]
    validation = report["validation"]
    lines.extend(
        [
            f"Aligned observations: {aligned['observation_count']} at {_format_duration(aligned['cadence_seconds'])} cadence",
            f"Coverage: {report['x_metric']}={_format_percentage(aligned['x_coverage'])}, "
            f"{report['y_metric']}={_format_percentage(aligned['y_coverage'])}",
            "Best discovery lag: "
            f"{_format_signed_duration(lag['best_lag_seconds'])} "
            f"({lag['best_lag_steps']} steps), "
            f"Spearman rho={_format_optional_float(lag['best_rho'])}, "
            f"overlap={lag['best_overlap']}",
            "Validation: "
            f"holdout rho={_format_optional_float(validation['holdout_rho'])} "
            f"(overlap={validation['holdout_overlap']}), "
            f"block permutation p={_format_optional_float(validation['permutation_p_value'])}, "
            f"window sign stability={_format_optional_float(validation['sign_stability'])}",
        ]
    )
    if validation["warnings"]:
        lines.append(f"Warnings: {'; '.join(validation['warnings'])}")
    lines.append(report["causation_warning"])
    lines.append("Positive lag means changes in X precede aligned changes in Y.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one requested Resonance metric pair.")
    parser.add_argument("--x", required=True, help="Metric name to treat as X.")
    parser.add_argument("--y", required=True, help="Metric name to treat as Y.")
    parser.add_argument("--hours", type=_positive_float, required=True, help="Lookback window in hours.")
    parser.add_argument(
        "--transform",
        choices=sorted(TRANSFORMS),
        default="raw",
        help="Transform to apply after alignment.",
    )
    parser.add_argument(
        "--max-lag-minutes",
        type=_non_negative_float,
        required=True,
        help="Maximum lag to scan in each direction.",
    )
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--now-utc",
        help="Override the analysis end time with an ISO UTC timestamp, for reproducible runs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        analysis_now = parse_utc(args.now_utc) if args.now_utc else now
        report = analyze_pair(
            args.database,
            x_metric=args.x,
            y_metric=args.y,
            hours=args.hours,
            transform_name=args.transform,
            max_lag_minutes=args.max_lag_minutes,
            now=analysis_now,
        )
    except ValueError as exc:
        parser.exit(2, f"Could not analyze pair: {exc}\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report["status"] == "ok" else 1


def _base_report(
    db_path: str | Path,
    *,
    x_metric: str,
    y_metric: str,
    hours: float,
    start_utc: datetime,
    end_utc: datetime,
    transform_name: str,
    max_lag_minutes: float,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "database_path": _display_database_path(db_path),
        "x_metric": x_metric,
        "y_metric": y_metric,
        "interval": {
            "start_utc": to_utc_iso(start_utc),
            "end_utc": to_utc_iso(end_utc),
            "hours": _clean_number(hours),
        },
        "transform": transform_name,
        "max_lag_minutes": _clean_number(max_lag_minutes),
        "causation_warning": "This is an association analysis only; it does not establish causation.",
    }


def _analysis_report(base_report: dict[str, Any], analysis: PairAnalysis) -> dict[str, Any]:
    aligned = analysis.aligned_pair
    lag = analysis.lag_result
    validation = analysis.validation_result
    best_score = _score_for_lag(lag.scores, lag.best_lag_steps)
    report = dict(base_report)
    report.update(
        {
            "status": "ok",
            "aligned": {
                "cadence_seconds": aligned.cadence_seconds,
                "observation_count": len(aligned.frame),
                "x_coverage": _clean_number(aligned.x_coverage),
                "y_coverage": _clean_number(aligned.y_coverage),
                "start_utc": to_utc_iso(aligned.start_utc),
                "end_utc": to_utc_iso(aligned.end_utc),
            },
            "lag": {
                "best_lag_steps": lag.best_lag_steps,
                "best_lag_seconds": lag.best_lag_seconds,
                "best_lag_minutes": _clean_number(lag.best_lag_seconds / 60),
                "best_rho": _clean_optional_float(lag.best_rho),
                "best_overlap": best_score["overlap_count"] if best_score else None,
                "score_count": len(lag.scores),
                "selected_on": "chronological discovery subset",
            },
            "validation": {
                "holdout_rho": _clean_optional_float(validation.holdout_rho),
                "holdout_overlap": validation.holdout_overlap,
                "permutation_p_value": _clean_optional_float(validation.permutation_p_value),
                "sign_stability": _clean_optional_float(validation.sign_stability),
                "window_scores": _json_safe(validation.window_scores),
                "warnings": list(validation.warnings),
            },
        }
    )
    return report


def _insufficient(base_report: dict[str, Any], reason: str) -> dict[str, Any]:
    report = dict(base_report)
    report.update({"status": "insufficient_data", "reason": reason})
    return report


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
    return str(db_path) == ":memory:" or Path(db_path).exists()


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


def _fetch_metric_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    metrics: Sequence[str],
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in metrics)
    return list(
        conn.execute(
            f"""
            SELECT timestamp_utc, metric, value
            FROM measurements
            WHERE timestamp_utc >= ?
              AND timestamp_utc <= ?
              AND metric IN ({placeholders})
            ORDER BY timestamp_utc ASC, metric ASC, id ASC
            """,
            (to_utc_iso(start_utc), to_utc_iso(end_utc), *metrics),
        )
    )


def _series_from_rows(rows: Sequence[sqlite3.Row], metric: str) -> pd.Series:
    timestamps: list[datetime] = []
    values: list[float] = []
    for row in rows:
        if row["metric"] != metric:
            continue
        try:
            value = float(row["value"])
            timestamp = parse_utc(str(row["timestamp_utc"]))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            timestamps.append(timestamp)
            values.append(value)
    if not timestamps:
        return pd.Series(dtype=float, name=metric)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name=metric, dtype=float)


def _transform_aligned_pair(pair: AlignedPair, transform_name: str) -> AlignedPair:
    frame = pair.frame
    x = frame["x"].rename(pair.x_metric)
    y = frame["y"].rename(pair.y_metric)
    transform_kwargs: dict[str, Any] = {}
    if transform_name == "calendar_residual":
        transform_kwargs["cadence_seconds"] = pair.cadence_seconds

    transformed_frame = pd.concat(
        (
            apply_transform(transform_name, x, **transform_kwargs).rename("x"),
            apply_transform(transform_name, y, **transform_kwargs).rename("y"),
        ),
        axis=1,
    ).dropna(how="any")

    if len(transformed_frame) < MIN_ALIGNED_POINTS:
        raise ValueError(
            f"insufficient transformed observations: got {len(transformed_frame)}, "
            f"need {MIN_ALIGNED_POINTS}"
        )

    return AlignedPair(
        x_metric=pair.x_metric,
        y_metric=pair.y_metric,
        cadence_seconds=pair.cadence_seconds,
        frame=transformed_frame,
        x_coverage=pair.x_coverage,
        y_coverage=pair.y_coverage,
        start_utc=transformed_frame.index[0].to_pydatetime(),
        end_utc=transformed_frame.index[-1].to_pydatetime(),
    )


def _max_lag_steps(cadence_seconds: int, max_lag_minutes: float) -> int:
    max_lag_seconds = int(max_lag_minutes * 60)
    return max_lag_seconds // cadence_seconds


def _discovery_frame(frame: pd.DataFrame, holdout_fraction: float) -> pd.DataFrame:
    split_index = max(1, min(len(frame) - 1, round(len(frame) * (1 - holdout_fraction))))
    return frame.iloc[:split_index]


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_utc": timestamp.to_pydatetime(),
            "x": row["x"],
            "y": row["y"],
        }
        for timestamp, row in frame.iterrows()
    ]


def _score_for_lag(scores: Sequence[dict[str, Any]], lag_steps: int) -> dict[str, Any] | None:
    for score in scores:
        if score["lag_steps"] == lag_steps:
            return score
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return to_utc_iso(value)
    if hasattr(value, "to_pydatetime"):
        return to_utc_iso(value.to_pydatetime())
    return value


def _positive_float(value: str) -> float:
    parsed = _float_arg(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = _float_arg(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _clean_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return _clean_number(value)


def _clean_number(value: float | int) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 6)


def _format_percentage(value: float | int) -> str:
    return f"{float(value) * 100:g}%"


def _format_optional_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


def _format_duration(seconds: float | int) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:g}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:g}m"
    return f"{minutes / 60:g}h"


def _format_signed_duration(seconds: float | int) -> str:
    prefix = "+" if seconds > 0 else ""
    return f"{prefix}{_format_duration(seconds)}"


if __name__ == "__main__":
    raise SystemExit(main())
