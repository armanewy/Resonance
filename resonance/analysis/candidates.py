from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso


DEFAULT_MIN_OBSERVATIONS = 30
DEFAULT_MIN_COVERAGE = 0.8
DEFAULT_MIN_ALIGNED_BINS = 30

STATUS_UNITS = {"bool", "boolean", "code", "enum", "flag", "status"}
STATUS_TOKENS = (
    "success",
    "status",
    "state",
    "flag",
    "code",
    "plugged",
)
DIRECT_DERIVATION_KEYS = (
    "base_metric",
    "derived_from",
    "derived_from_metric",
    "direct_derivation_of",
    "input_metric",
    "input_metrics",
    "inputs",
    "parent_metric",
    "source_metric",
    "source_metrics",
)


@dataclass(frozen=True)
class CandidateOptions:
    min_observations: int = DEFAULT_MIN_OBSERVATIONS
    min_coverage: float = DEFAULT_MIN_COVERAGE
    min_aligned_bins: int = DEFAULT_MIN_ALIGNED_BINS


@dataclass(frozen=True)
class CandidateMetric:
    metric: str
    units: tuple[str, ...]
    sources: tuple[str, ...]
    sample_count: int
    numeric_count: int
    cadence_seconds: int | None
    coverage: float | None
    start_utc: datetime | None
    end_utc: datetime | None
    metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CandidatePair:
    x_metric: str
    y_metric: str
    cadence_seconds: int
    aligned_bins: int
    x_coverage: float
    y_coverage: float


@dataclass(frozen=True)
class CandidateRejection:
    metrics: tuple[str, ...]
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CandidateSelection:
    metrics: tuple[CandidateMetric, ...]
    pairs: tuple[CandidatePair, ...]
    rejections: tuple[CandidateRejection, ...]


def select_candidate_pairs(
    database_path: str | Path,
    start_utc: datetime,
    end_utc: datetime,
    *,
    metrics: Sequence[str] | None = None,
    options: CandidateOptions | Mapping[str, Any] | None = None,
) -> CandidateSelection:
    """Return conservative metric pairs eligible for later automatic analysis.

    This function only screens candidate pairs. It does not calculate correlation
    or write to SQLite.
    """

    start, end = _normalize_interval(start_utc, end_utc)
    resolved_options = _candidate_options(options)
    _validate_options(resolved_options)
    requested_metrics = _requested_metrics(metrics)

    with _connect_read_only(database_path) as conn:
        _require_measurements_table(conn)
        rows = _fetch_rows(conn, start, end, requested_metrics)

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["metric"]), []).append(row)

    rejections: list[CandidateRejection] = []
    rejections.extend(_duplicate_rejections(metrics))
    if requested_metrics is not None:
        for metric in requested_metrics:
            if metric not in grouped:
                rejections.append(
                    CandidateRejection((metric,), "no_observations", "no rows in requested interval")
                )

    all_metrics = tuple(
        _metric_profile(metric, metric_rows, interval_start=start, interval_end=end)
        for metric, metric_rows in sorted(grouped.items())
    )
    eligible: list[CandidateMetric] = []
    for metric in all_metrics:
        reason = _metric_rejection_reason(metric, resolved_options)
        if reason is None:
            eligible.append(metric)
        else:
            rejections.append(CandidateRejection((metric.metric,), reason))

    by_name = {metric.metric: metric for metric in eligible}
    pairs: list[CandidatePair] = []
    for left, right in _canonical_pair_names(tuple(by_name)):
        x_metric = by_name[left]
        y_metric = by_name[right]
        pair_rejection = _pair_rejection_reason(x_metric, y_metric, rows, resolved_options)
        if pair_rejection is not None:
            rejections.append(CandidateRejection((left, right), pair_rejection.reason, pair_rejection.detail))
            continue

        cadence = max(x_metric.cadence_seconds or 0, y_metric.cadence_seconds or 0)
        aligned_bins = _aligned_bin_count(rows, left, right, cadence)
        pairs.append(
            CandidatePair(
                x_metric=left,
                y_metric=right,
                cadence_seconds=cadence,
                aligned_bins=aligned_bins,
                x_coverage=float(x_metric.coverage or 0.0),
                y_coverage=float(y_metric.coverage or 0.0),
            )
        )

    return CandidateSelection(
        metrics=tuple(eligible),
        pairs=tuple(pairs),
        rejections=tuple(rejections),
    )


def _candidate_options(options: CandidateOptions | Mapping[str, Any] | None) -> CandidateOptions:
    if options is None:
        return CandidateOptions()
    if isinstance(options, CandidateOptions):
        return options
    return CandidateOptions(**dict(options))


def _validate_options(options: CandidateOptions) -> None:
    if options.min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if not 0 <= options.min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if options.min_aligned_bins < 2:
        raise ValueError("min_aligned_bins must be at least 2")


def _normalize_interval(start_utc: datetime, end_utc: datetime) -> tuple[datetime, datetime]:
    start = ensure_utc(start_utc).replace(microsecond=0)
    end = ensure_utc(end_utc).replace(microsecond=0)
    if start >= end:
        raise ValueError("start_utc must be before end_utc")
    return start, end


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


def _requested_metrics(metrics: Sequence[str] | None) -> tuple[str, ...] | None:
    if metrics is None:
        return None
    return tuple(dict.fromkeys(str(metric) for metric in metrics if str(metric)))


def _duplicate_rejections(metrics: Sequence[str] | None) -> list[CandidateRejection]:
    if metrics is None:
        return []
    seen: set[str] = set()
    rejections: list[CandidateRejection] = []
    for metric in metrics:
        metric_name = str(metric)
        if metric_name in seen:
            rejections.append(
                CandidateRejection(
                    (metric_name, metric_name),
                    "identical_metrics",
                    "duplicate requested metric would create a self-pair",
                )
            )
        seen.add(metric_name)
    return rejections


def _fetch_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    metrics: Sequence[str] | None,
) -> list[sqlite3.Row]:
    params: list[str] = [to_utc_iso(start_utc), to_utc_iso(end_utc)]
    metric_filter = ""
    if metrics:
        placeholders = ",".join("?" for _ in metrics)
        metric_filter = f" AND metric IN ({placeholders})"
        params.extend(metrics)
    return list(
        conn.execute(
            f"""
            SELECT id, timestamp_utc, metric, value, unit, source, metadata_json
            FROM measurements
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?{metric_filter}
            ORDER BY metric ASC, timestamp_utc ASC, id ASC
            """,
            params,
        )
    )


def _metric_profile(
    metric: str,
    rows: Sequence[sqlite3.Row],
    *,
    interval_start: datetime,
    interval_end: datetime,
) -> CandidateMetric:
    numeric_timestamps: list[datetime] = []
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    metadata = []

    for row in rows:
        timestamp = _optional_timestamp(row["timestamp_utc"])
        if timestamp is not None:
            start_utc = timestamp if start_utc is None else min(start_utc, timestamp)
            end_utc = timestamp if end_utc is None else max(end_utc, timestamp)
        if timestamp is not None and _optional_float(row["value"]) is not None:
            numeric_timestamps.append(timestamp)
        parsed_metadata = _parse_metadata(row["metadata_json"])
        if parsed_metadata:
            metadata.append(parsed_metadata)

    cadence_seconds = _infer_cadence_seconds(sorted(numeric_timestamps))
    coverage = None
    if cadence_seconds is not None:
        coverage = _coverage(sorted(numeric_timestamps), cadence_seconds, interval_start, interval_end)

    return CandidateMetric(
        metric=metric,
        units=tuple(sorted({str(row["unit"]) for row in rows})),
        sources=tuple(sorted({str(row["source"]) for row in rows})),
        sample_count=len(rows),
        numeric_count=len(numeric_timestamps),
        cadence_seconds=cadence_seconds,
        coverage=coverage,
        start_utc=start_utc,
        end_utc=end_utc,
        metadata=tuple(metadata),
    )


def _metric_rejection_reason(metric: CandidateMetric, options: CandidateOptions) -> str | None:
    if _is_status_metric(metric):
        return "status_or_flag_metric"
    if metric.numeric_count != metric.sample_count:
        return "non_numeric_series"
    if metric.numeric_count < options.min_observations:
        return "too_few_observations"
    if metric.cadence_seconds is None:
        return "cadence_unavailable"
    if metric.coverage is None or metric.coverage < options.min_coverage:
        return "low_coverage"
    return None


def _is_status_metric(metric: CandidateMetric) -> bool:
    units = {unit.lower() for unit in metric.units}
    if units & STATUS_UNITS:
        return True

    normalized_tokens = [token for token in metric.metric.lower().replace("-", "_").split("_") if token]
    if any(token in STATUS_TOKENS for token in normalized_tokens):
        return True

    for metadata in metric.metadata:
        metadata_type = str(metadata.get("type") or metadata.get("kind") or "").lower()
        if metadata_type in STATUS_UNITS:
            return True
    return False


def _canonical_pair_names(metric_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    names = sorted(dict.fromkeys(metric_names))
    return tuple((names[left], names[right]) for left in range(len(names)) for right in range(left + 1, len(names)))


def _pair_rejection_reason(
    x_metric: CandidateMetric,
    y_metric: CandidateMetric,
    rows: Sequence[sqlite3.Row],
    options: CandidateOptions,
) -> CandidateRejection | None:
    if x_metric.metric == y_metric.metric:
        return CandidateRejection((x_metric.metric, y_metric.metric), "identical_metrics")
    if _direct_derivation_between(x_metric, y_metric):
        return CandidateRejection(
            (x_metric.metric, y_metric.metric),
            "direct_derivation",
            "metadata identifies one metric as directly derived from the other",
        )

    cadence = max(x_metric.cadence_seconds or 0, y_metric.cadence_seconds or 0)
    aligned_bins = _aligned_bin_count(rows, x_metric.metric, y_metric.metric, cadence)
    if aligned_bins < options.min_aligned_bins:
        return CandidateRejection(
            (x_metric.metric, y_metric.metric),
            "too_few_aligned_bins",
            f"coarsest cadence produced {aligned_bins} aligned bins",
        )
    return None


def _direct_derivation_between(x_metric: CandidateMetric, y_metric: CandidateMetric) -> bool:
    return _metadata_names_metric(x_metric.metadata, y_metric.metric) or _metadata_names_metric(
        y_metric.metadata, x_metric.metric
    )


def _metadata_names_metric(metadata_values: Sequence[Mapping[str, Any]], metric: str) -> bool:
    target = metric.lower()
    for metadata in metadata_values:
        for key in DIRECT_DERIVATION_KEYS:
            if key in metadata and target in _metadata_metric_names(metadata[key]):
                return True
    return False


def _metadata_metric_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, Mapping):
        names = set()
        for nested_value in value.values():
            names.update(_metadata_metric_names(nested_value))
        return names
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        names = set()
        for item in value:
            names.update(_metadata_metric_names(item))
        return names
    return {str(value).lower()}


def _aligned_bin_count(
    rows: Sequence[sqlite3.Row],
    x_metric: str,
    y_metric: str,
    cadence_seconds: int,
) -> int:
    if cadence_seconds <= 0:
        return 0
    x_bins = _metric_bins(rows, x_metric, cadence_seconds)
    y_bins = _metric_bins(rows, y_metric, cadence_seconds)
    return len(x_bins & y_bins)


def _metric_bins(rows: Sequence[sqlite3.Row], metric: str, cadence_seconds: int) -> set[int]:
    bins: set[int] = set()
    for row in rows:
        if row["metric"] != metric or _optional_float(row["value"]) is None:
            continue
        timestamp = _optional_timestamp(row["timestamp_utc"])
        if timestamp is None:
            continue
        bins.add(math.floor(timestamp.timestamp() / cadence_seconds))
    return bins


def _parse_metadata(value: Any) -> Mapping[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _optional_timestamp(value: Any) -> datetime | None:
    try:
        return parse_utc(str(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


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


__all__ = [
    "CandidateMetric",
    "CandidateOptions",
    "CandidatePair",
    "CandidateRejection",
    "CandidateSelection",
    "select_candidate_pairs",
]
