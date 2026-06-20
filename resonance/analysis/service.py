from __future__ import annotations

import math
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from resonance.analysis.alignment import align_series
from resonance.analysis.contracts import AlignedPair, PairAnalysis, ValidationResult
from resonance.analysis.correlation import lagged_spearman
from resonance.analysis.transforms import apply_transform
from resonance.analysis.validation import (
    chronological_holdout_validation,
    max_lag_block_permutation_test,
    window_stability,
)
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso


DEFAULT_MIN_ALIGNED_POINTS = 30
DEFAULT_MIN_OVERLAP = 30
DEFAULT_HOLDOUT_FRACTION = 0.25
DEFAULT_WINDOW_COUNT = 4
DEFAULT_PERMUTATIONS = 199


@dataclass(frozen=True)
class ValidationOptions:
    min_aligned_points: int = DEFAULT_MIN_ALIGNED_POINTS
    min_overlap: int = DEFAULT_MIN_OVERLAP
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION
    window_count: int = DEFAULT_WINDOW_COUNT
    permutations: int = DEFAULT_PERMUTATIONS
    permutation_block_size: int | None = None
    permutation_seed: int | None = None
    cadence_seconds: int | None = None


@dataclass(frozen=True)
class AnalyzableMetric:
    metric: str
    units: tuple[str, ...]
    sources: tuple[str, ...]
    sample_count: int
    cadence_seconds: int | None
    coverage: float | None
    start_utc: datetime | None
    end_utc: datetime | None
    warnings: tuple[str, ...] = ()
    display_name: str | None = None
    series_id: str | None = None
    geography_type: str | None = None
    geography_id: str | None = None
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MetricPairAnalysis(PairAnalysis):
    x_metric_summary: AnalyzableMetric
    y_metric_summary: AnalyzableMetric
    warnings: tuple[str, ...] = ()


def list_analyzable_metrics(
    database_path: str | Path,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[AnalyzableMetric, ...]:
    start, end = _normalize_interval(start_utc, end_utc)
    with _connect_read_only(database_path) as conn:
        _require_measurements_table(conn)
        rows = conn.execute(
            """
            SELECT metric, unit, source, timestamp_utc
            FROM measurements
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?
            ORDER BY metric ASC, timestamp_utc ASC, id ASC
            """,
            (to_utc_iso(start), to_utc_iso(end)),
        ).fetchall()
        public_rows = _fetch_public_summary_rows(conn, start, end)

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["metric"]), []).append(row)

    measurement_metrics = tuple(
        _metric_summary(metric, metric_rows, interval_start=start, interval_end=end)
        for metric, metric_rows in sorted(grouped.items())
    )
    public_metrics = tuple(
        _public_metric_summary(series_id, series_rows, interval_start=start, interval_end=end)
        for series_id, series_rows in sorted(public_rows.items())
    )
    return tuple(sorted((*measurement_metrics, *public_metrics), key=lambda metric: metric.display_name or metric.metric))


def analyze_metric_pair(
    database_path: str | Path,
    x_metric: str,
    y_metric: str,
    start_utc: datetime,
    end_utc: datetime,
    transform: str,
    max_lag_steps: int,
    validation_options: ValidationOptions | Mapping[str, Any] | None = None,
) -> PairAnalysis:
    if not x_metric:
        raise ValueError("x_metric is required")
    if not y_metric:
        raise ValueError("y_metric is required")
    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be non-negative")

    start, end = _normalize_interval(start_utc, end_utc)
    options = _validation_options(validation_options)
    _validate_options(options)

    with _connect_read_only(database_path) as conn:
        _require_measurements_table(conn)
        x_rows, x_summary = _fetch_analysis_rows(conn, start, end, x_metric)
        y_rows, y_summary = _fetch_analysis_rows(conn, start, end, y_metric)

    if not x_rows or not y_rows:
        missing = [metric for metric, metric_rows in ((x_metric, x_rows), (y_metric, y_rows)) if not metric_rows]
        raise ValueError(f"no measurements for metric(s): {', '.join(missing)}")

    x_series = _series_from_rows(x_rows, x_metric)
    y_series = _series_from_rows(y_rows, y_metric)

    raw_pair = align_series(
        x_series,
        y_series,
        cadence_seconds=options.cadence_seconds,
        min_points=options.min_aligned_points,
    )
    transformed_pair = _transform_aligned_pair(
        raw_pair,
        transform,
        min_points=options.min_aligned_points,
    )
    discovery_frame = _discovery_frame(transformed_pair.frame, options.holdout_fraction)
    lag_result = lagged_spearman(
        discovery_frame,
        max_lag_steps=max_lag_steps,
        min_overlap=options.min_overlap,
    )

    validation_frame = _records_from_frame(transformed_pair.frame)
    candidate_lags = range(-max_lag_steps, max_lag_steps + 1)
    holdout = chronological_holdout_validation(
        validation_frame,
        candidate_lag_steps=(lag_result.best_lag_steps,),
        holdout_fraction=options.holdout_fraction,
        min_overlap=options.min_overlap,
    )
    permutation_kwargs: dict[str, Any] = {}
    if options.permutation_block_size is not None:
        permutation_kwargs["block_size"] = options.permutation_block_size
    if options.permutation_seed is not None:
        permutation_kwargs["seed"] = options.permutation_seed
    p_value = max_lag_block_permutation_test(
        validation_frame,
        candidate_lag_steps=candidate_lags,
        permutations=options.permutations,
        min_overlap=options.min_overlap,
        **permutation_kwargs,
    )
    stability = window_stability(
        validation_frame,
        lag_steps=lag_result.best_lag_steps,
        window_count=options.window_count,
        min_overlap=options.min_overlap,
    )

    validation_result = ValidationResult(
        permutation_p_value=p_value,
        holdout_rho=holdout.holdout_rho,
        holdout_overlap=holdout.holdout_overlap,
        sign_stability=stability.sign_stability,
        window_scores=stability.window_scores,
        warnings=tuple(dict.fromkeys((*holdout.warnings, *stability.warnings))),
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *x_summary.warnings,
                *y_summary.warnings,
                *validation_result.warnings,
            )
        )
    )
    return MetricPairAnalysis(
        aligned_pair=transformed_pair,
        transform_name=transform,
        lag_result=lag_result,
        validation_result=validation_result,
        x_metric_summary=x_summary,
        y_metric_summary=y_summary,
        warnings=warnings,
    )


def _connect_read_only(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _require_measurements_table(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'measurements'
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ValueError("measurements table not found")


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def _normalize_interval(start_utc: datetime, end_utc: datetime) -> tuple[datetime, datetime]:
    start = ensure_utc(start_utc).replace(microsecond=0)
    end = ensure_utc(end_utc).replace(microsecond=0)
    if start >= end:
        raise ValueError("start_utc must be before end_utc")
    return start, end


def _validation_options(
    options: ValidationOptions | Mapping[str, Any] | None,
) -> ValidationOptions:
    if options is None:
        return ValidationOptions()
    if isinstance(options, ValidationOptions):
        return options
    return ValidationOptions(**dict(options))


def _validate_options(options: ValidationOptions) -> None:
    if options.min_aligned_points < 2:
        raise ValueError("min_aligned_points must be at least 2")
    if options.min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")
    if not 0 < options.holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if options.window_count <= 0:
        raise ValueError("window_count must be positive")
    if options.permutations <= 0:
        raise ValueError("permutations must be positive")
    if options.permutation_block_size is not None and options.permutation_block_size <= 0:
        raise ValueError("permutation_block_size must be positive")
    if options.cadence_seconds is not None and options.cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")


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
            SELECT id, timestamp_utc, metric, value, unit, source
            FROM measurements
            WHERE timestamp_utc >= ?
              AND timestamp_utc <= ?
              AND metric IN ({placeholders})
            ORDER BY timestamp_utc ASC, metric ASC, id ASC
            """,
            (to_utc_iso(start_utc), to_utc_iso(end_utc), *metrics),
        )
    )


def _fetch_analysis_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    identifier: str,
) -> tuple[list[Mapping[str, Any]], AnalyzableMetric]:
    if _is_registered_series(conn, identifier):
        rows = _fetch_public_rows(conn, start_utc, end_utc, identifier)
        summary = _public_metric_summary(identifier, rows, interval_start=start_utc, interval_end=end_utc)
        return rows, summary
    rows = [dict(row) for row in _fetch_metric_rows(conn, start_utc, end_utc, (identifier,))]
    summary = _metric_summary(identifier, rows, interval_start=start_utc, interval_end=end_utc)
    return rows, summary


def _is_registered_series(conn: sqlite3.Connection, identifier: str) -> bool:
    if not _has_table(conn, "series_registry"):
        return False
    row = conn.execute(
        "SELECT 1 FROM series_registry WHERE series_id = ? LIMIT 1",
        (identifier,),
    ).fetchone()
    return row is not None


def _fetch_public_summary_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, list[Mapping[str, Any]]]:
    if not (_has_table(conn, "public_observations") and _has_table(conn, "series_registry")):
        return {}
    rows = conn.execute(
        """
        SELECT o.series_id, o.valid_start_utc AS timestamp_utc, o.value, o.quality,
               o.source_revision, o.source_observation_key, s.source_id AS source,
               s.unit, s.display_name, s.geography_type, s.geography_id,
               s.lineage_id, s.cadence_seconds, s.quality_tier, s.metadata_json
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE o.valid_start_utc >= ?
          AND o.valid_start_utc <= ?
          AND NOT EXISTS (
              SELECT 1
              FROM public_observations newer
              WHERE newer.series_id = o.series_id
                AND newer.source_observation_key = o.source_observation_key
                AND newer.ingested_at_utc > o.ingested_at_utc
          )
        ORDER BY s.display_name ASC, o.valid_start_utc ASC
        """,
        (to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchall()
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["series_id"]), []).append(dict(row))
    return grouped


def _fetch_public_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    series_id: str,
) -> list[Mapping[str, Any]]:
    grouped = _fetch_public_summary_rows(conn, start_utc, end_utc)
    return grouped.get(series_id, [])


def _metric_summary(
    metric: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    interval_start: datetime,
    interval_end: datetime,
) -> AnalyzableMetric:
    timestamps = _timestamps_from_rows(rows)
    cadence_seconds = _infer_cadence_seconds(timestamps)
    units = tuple(sorted({str(row["unit"]) for row in rows}))
    sources = tuple(sorted({str(row["source"]) for row in rows}))
    warnings: list[str] = []
    if len(units) > 1:
        warnings.append(f"{metric} has multiple units")
    if len(sources) > 1:
        warnings.append(f"{metric} has multiple sources")
    if cadence_seconds is None:
        coverage = None
        warnings.append(f"{metric} cadence could not be inferred")
    else:
        coverage = _coverage(timestamps, cadence_seconds, interval_start, interval_end)

    return AnalyzableMetric(
        metric=metric,
        units=units,
        sources=sources,
        sample_count=len(rows),
        cadence_seconds=cadence_seconds,
        coverage=coverage,
        start_utc=timestamps[0] if timestamps else None,
        end_utc=timestamps[-1] if timestamps else None,
        warnings=tuple(warnings),
        display_name=metric,
    )


def _public_metric_summary(
    series_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    interval_start: datetime,
    interval_end: datetime,
) -> AnalyzableMetric:
    timestamps = _timestamps_from_rows(rows)
    first = rows[0] if rows else {}
    cadence_seconds = int(first["cadence_seconds"]) if first.get("cadence_seconds") is not None else _infer_cadence_seconds(timestamps)
    units = tuple(sorted({str(row["unit"]) for row in rows if row.get("unit") is not None}))
    sources = tuple(sorted({str(row["source"]) for row in rows if row.get("source") is not None}))
    warnings: list[str] = []
    if not rows:
        coverage = None
        warnings.append(f"{series_id} has no public observations in this interval")
    elif cadence_seconds is None or cadence_seconds <= 0:
        coverage = None
        warnings.append(f"{series_id} cadence could not be inferred")
    else:
        coverage = _coverage(timestamps, cadence_seconds, interval_start, interval_end)
    display_name = str(first.get("display_name") or series_id)
    geography_type = str(first.get("geography_type") or "") or None
    geography_id = str(first.get("geography_id") or "") or None
    if geography_id:
        display_name = f"{display_name} [{geography_id}]"
    return AnalyzableMetric(
        metric=series_id,
        units=units,
        sources=sources,
        sample_count=len(rows),
        cadence_seconds=cadence_seconds,
        coverage=coverage,
        start_utc=timestamps[0] if timestamps else None,
        end_utc=timestamps[-1] if timestamps else None,
        warnings=tuple(warnings),
        display_name=display_name,
        series_id=series_id,
        geography_type=geography_type,
        geography_id=geography_id,
        provenance={
            "source_id": first.get("source"),
            "lineage_id": first.get("lineage_id"),
            "quality_tier": first.get("quality_tier"),
            "metadata": json.loads(str(first.get("metadata_json") or "{}")),
        }
        if first
        else None,
    )


def _timestamps_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[datetime]:
    timestamps = []
    for row in rows:
        try:
            timestamps.append(parse_utc(str(row["timestamp_utc"])))
        except ValueError:
            continue
    return sorted(timestamps)


def _infer_cadence_seconds(timestamps: Sequence[datetime]) -> int | None:
    if len(timestamps) < 2:
        return None
    deltas = [
        int((right - left).total_seconds())
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    ]
    if not deltas:
        return None
    return max(1, int(round(float(pd.Series(deltas).median()))))


def _coverage(
    timestamps: Sequence[datetime],
    cadence_seconds: int,
    interval_start: datetime,
    interval_end: datetime,
) -> float:
    interval_seconds = max(0, int((interval_end - interval_start).total_seconds()))
    expected_count = interval_seconds // cadence_seconds + 1
    if expected_count <= 0:
        return 0.0
    bins = {
        int((timestamp - interval_start).total_seconds()) // cadence_seconds
        for timestamp in timestamps
        if interval_start <= timestamp <= interval_end
    }
    return min(1.0, len(bins) / expected_count)


def _series_from_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> pd.Series:
    timestamps: list[datetime] = []
    values: list[float] = []
    for row in rows:
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


def _transform_aligned_pair(pair: AlignedPair, transform_name: str, *, min_points: int) -> AlignedPair:
    transform_kwargs: dict[str, Any] = {}
    if transform_name == "calendar_residual":
        transform_kwargs["cadence_seconds"] = pair.cadence_seconds

    transformed_frame = pd.concat(
        (
            apply_transform(transform_name, pair.frame["x"].rename(pair.x_metric), **transform_kwargs).rename("x"),
            apply_transform(transform_name, pair.frame["y"].rename(pair.y_metric), **transform_kwargs).rename("y"),
        ),
        axis=1,
    ).dropna(how="any")
    if len(transformed_frame) < min_points:
        raise ValueError(
            f"insufficient transformed observations: got {len(transformed_frame)}, need {min_points}"
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


def _discovery_frame(frame: pd.DataFrame, holdout_fraction: float) -> pd.DataFrame:
    split_index = max(1, min(len(frame) - 1, round(len(frame) * (1 - holdout_fraction))))
    return frame.iloc[:split_index]


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"timestamp_utc": timestamp.to_pydatetime(), "x": row["x"], "y": row["y"]}
        for timestamp, row in frame.iterrows()
    ]
