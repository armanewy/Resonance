from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from resonance.analysis.alignment import align_series
from resonance.analysis.candidates import CandidateOptions
from resonance.analysis.contracts import AlignedPair, LagScanResult, ValidationResult
from resonance.analysis.correlation import lagged_spearman
from resonance.analysis.transforms import apply_transform
from resonance.analysis.validation import (
    chronological_holdout_validation,
    max_lag_block_permutation_test,
    window_stability,
)
from resonance.storage import (
    DEFAULT_DB_PATH,
    CorrelationFinding,
    ensure_database,
    upsert_correlation_findings,
)
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


DEFAULT_MIN_ALIGNED_OBSERVATIONS = 200
DEFAULT_MIN_COVERAGE = 0.8
DEFAULT_DISCOVERY_FRACTION = 0.70
DEFAULT_DISCOVERY_ABS_RHO = 0.65
DEFAULT_MAX_CORRECTED_Q = 0.01
DEFAULT_HOLDOUT_ABS_RHO = 0.40
DEFAULT_MIN_SIGN_STABILITY = 0.75
DEFAULT_MAX_FINDINGS = 5
DEFAULT_MAX_LAG_SECONDS = 3_600
DEFAULT_MIN_OVERLAP = 30
DEFAULT_WINDOW_COUNT = 4
DEFAULT_PERMUTATIONS = 199
DEFAULT_PERMUTATION_SEED = 20260619
DEFAULT_MULTIPLE_TESTING_METHOD = "by"

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
LOCAL_GEOGRAPHY_TYPES = {"configured_location", "device"}
REGIONAL_GEOGRAPHY_TYPES = {
    "balancing_authority",
    "city",
    "county",
    "metro",
    "regional_radius",
    "region",
    "state",
}


@dataclass(frozen=True)
class ScannerOptions:
    min_aligned_observations: int = DEFAULT_MIN_ALIGNED_OBSERVATIONS
    min_coverage: float = DEFAULT_MIN_COVERAGE
    discovery_fraction: float = DEFAULT_DISCOVERY_FRACTION
    min_discovery_abs_rho: float = DEFAULT_DISCOVERY_ABS_RHO
    max_corrected_q: float = DEFAULT_MAX_CORRECTED_Q
    min_holdout_abs_rho: float = DEFAULT_HOLDOUT_ABS_RHO
    min_sign_stability: float = DEFAULT_MIN_SIGN_STABILITY
    max_findings: int = DEFAULT_MAX_FINDINGS
    max_lag_seconds: int = DEFAULT_MAX_LAG_SECONDS
    min_overlap: int = DEFAULT_MIN_OVERLAP
    window_count: int = DEFAULT_WINDOW_COUNT
    permutations: int = DEFAULT_PERMUTATIONS
    permutation_block_size: int | None = None
    permutation_seed: int = DEFAULT_PERMUTATION_SEED
    calendar_min_history: int = 3
    calendar_timezone: str = "UTC"
    multiple_testing_method: str = DEFAULT_MULTIPLE_TESTING_METHOD


@dataclass(frozen=True)
class PairEvidence:
    pair: "ScannerCandidatePair"
    transform: str
    lag_result: LagScanResult
    validation_result: ValidationResult
    corrected_q: float
    aligned_pair: AlignedPair
    p_value: float | None
    discovery_overlap: int


@dataclass(frozen=True)
class ScannerCandidateSeries:
    identifier: str
    source_kind: str
    metric_name: str
    display_name: str
    unit: str
    source_id: str
    sample_count: int
    numeric_count: int
    cadence_seconds: int | None
    registry_cadence_seconds: int | None
    coverage: float | None
    start_utc: datetime | None
    end_utc: datetime | None
    geography_type: str | None
    geography_id: str | None
    lineage_id: str | None
    parent_series_id: str | None
    metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ScannerCandidatePair:
    x_metric: str
    y_metric: str
    cadence_seconds: int
    aligned_bins: int
    x_coverage: float
    y_coverage: float
    x_source_kind: str
    y_source_kind: str
    compatibility: Mapping[str, Any]

    @property
    def contains_public(self) -> bool:
        return "public" in {self.x_source_kind, self.y_source_kind}


@dataclass(frozen=True)
class ScannerCandidateRejection:
    metrics: tuple[str, ...]
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class ScannerCandidateSelection:
    series: tuple[ScannerCandidateSeries, ...]
    pairs: tuple[ScannerCandidatePair, ...]
    rejections: tuple[ScannerCandidateRejection, ...]


def scan_correlations(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    hours: float,
    dry_run: bool = False,
    now: datetime | None = None,
    options: ScannerOptions | Mapping[str, Any] | None = None,
) -> tuple[CorrelationFinding, ...]:
    if hours <= 0:
        raise ValueError("hours must be greater than 0")

    resolved_options = _scanner_options(options)
    _validate_options(resolved_options)
    database_path = Path(db_path)
    if str(database_path) != ":memory:" and not database_path.exists():
        return ()

    end_utc = ensure_utc(now or utc_now()).replace(microsecond=0)
    start_utc = end_utc - timedelta(hours=hours)

    selection = _select_scanner_candidate_pairs(
        database_path,
        start_utc,
        end_utc,
        include_public=dry_run,
        options=CandidateOptions(
            min_observations=resolved_options.min_aligned_observations,
            min_coverage=resolved_options.min_coverage,
            min_aligned_bins=resolved_options.min_aligned_observations,
        ),
    )
    if not selection.pairs:
        return ()

    evidence: list[PairEvidence] = []
    skipped_tests = 0
    for pair in selection.pairs:
        try:
            pair_evidence = _evaluate_pair(
                database_path,
                pair,
                start_utc,
                end_utc,
                resolved_options,
            )
        except ValueError:
            skipped_tests += 1
            continue
        evidence.append(pair_evidence)

    if not evidence:
        return ()

    total_tests = max(len(selection.pairs), len(evidence) + skipped_tests)
    q_values = _adjust_p_values(
        [item.p_value for item in evidence],
        total_tests=total_tests,
        method=resolved_options.multiple_testing_method,
    )
    corrected = [
        PairEvidence(
            pair=item.pair,
            transform=item.transform,
            lag_result=item.lag_result,
            validation_result=item.validation_result,
            corrected_q=q_value,
            aligned_pair=item.aligned_pair,
            p_value=item.p_value,
            discovery_overlap=item.discovery_overlap,
        )
        for item, q_value in zip(evidence, q_values, strict=True)
    ]

    promoted = [
        item for item in corrected if _passes_promotion_thresholds(item, resolved_options)
    ]
    promoted.sort(
        key=lambda item: (
            item.corrected_q,
            -abs(float(item.validation_result.holdout_rho or 0.0)),
            -abs(float(item.lag_result.best_rho or 0.0)),
            item.pair.x_metric,
            item.pair.y_metric,
        )
    )

    findings = tuple(
        _finding_from_evidence(
            item,
            first_seen_utc=end_utc,
            verified_utc=end_utc,
            options=resolved_options,
            total_tests=total_tests,
        )
        for item in promoted[: resolved_options.max_findings]
    )
    if findings and not dry_run:
        conn = ensure_database(database_path)
        try:
            upsert_correlation_findings(conn, findings)
        finally:
            conn.close()
    return findings


def finding_to_dict(finding: CorrelationFinding) -> dict[str, Any]:
    return {
        "x_metric": finding.x_metric,
        "y_metric": finding.y_metric,
        "transform": finding.transform,
        "lag_seconds": finding.lag_seconds,
        "discovery_rho": _clean_number(finding.discovery_rho),
        "holdout_rho": _clean_number(finding.holdout_rho),
        "corrected_q": _clean_number(finding.corrected_q),
        "stability": _clean_number(finding.stability),
        "overlap_count": finding.overlap_count,
        "first_seen_utc": to_utc_iso(finding.first_seen_utc),
        "last_verified_utc": to_utc_iso(finding.last_verified_utc),
        "status": finding.status,
        "evidence": _json_safe(finding.evidence),
    }


def _scanner_options(options: ScannerOptions | Mapping[str, Any] | None) -> ScannerOptions:
    if options is None:
        return ScannerOptions()
    if isinstance(options, ScannerOptions):
        return options
    return ScannerOptions(**dict(options))


def _validate_options(options: ScannerOptions) -> None:
    if options.min_aligned_observations < 2:
        raise ValueError("min_aligned_observations must be at least 2")
    if not 0 <= options.min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if not 0 < options.discovery_fraction < 1:
        raise ValueError("discovery_fraction must be between 0 and 1")
    if not 0 <= options.min_discovery_abs_rho <= 1:
        raise ValueError("min_discovery_abs_rho must be between 0 and 1")
    if not 0 <= options.max_corrected_q <= 1:
        raise ValueError("max_corrected_q must be between 0 and 1")
    if not 0 <= options.min_holdout_abs_rho <= 1:
        raise ValueError("min_holdout_abs_rho must be between 0 and 1")
    if not 0 <= options.min_sign_stability <= 1:
        raise ValueError("min_sign_stability must be between 0 and 1")
    if options.max_findings < 1:
        raise ValueError("max_findings must be at least 1")
    if options.max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be non-negative")
    if options.min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")
    if options.window_count <= 0:
        raise ValueError("window_count must be positive")
    if options.permutations <= 0:
        raise ValueError("permutations must be positive")
    if options.permutation_block_size is not None and options.permutation_block_size <= 0:
        raise ValueError("permutation_block_size must be positive")
    if options.calendar_min_history < 1:
        raise ValueError("calendar_min_history must be at least 1")
    if options.multiple_testing_method not in {"bh", "by"}:
        raise ValueError("multiple_testing_method must be 'bh' or 'by'")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(options.calendar_timezone)
    except Exception as exc:
        raise ValueError(f"calendar_timezone is not recognized: {options.calendar_timezone}") from exc


def _select_scanner_candidate_pairs(
    database_path: str | Path,
    start_utc: datetime,
    end_utc: datetime,
    *,
    include_public: bool,
    options: CandidateOptions | Mapping[str, Any] | None = None,
) -> ScannerCandidateSelection:
    start, end = _normalize_interval(start_utc, end_utc)
    resolved_options = _candidate_options(options)
    _validate_candidate_options(resolved_options)

    with _connect_read_only(Path(database_path)) as conn:
        _require_measurements_table(conn)
        rows = _fetch_scanner_rows(conn, start, end, include_public=include_public)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["metric"]), []).append(row)

    rejections: list[ScannerCandidateRejection] = []
    all_series = tuple(
        _scanner_series_profile(identifier, series_rows, interval_start=start, interval_end=end)
        for identifier, series_rows in sorted(grouped.items())
    )
    eligible: list[ScannerCandidateSeries] = []
    for series in all_series:
        reason = _series_rejection_reason(series, resolved_options)
        if reason is None:
            eligible.append(series)
        else:
            rejections.append(ScannerCandidateRejection((series.identifier,), reason))

    by_identifier = {series.identifier: series for series in eligible}
    pairs: list[ScannerCandidatePair] = []
    for left, right in _canonical_pair_names(tuple(by_identifier)):
        x_series = by_identifier[left]
        y_series = by_identifier[right]
        rejection = _scanner_pair_rejection_reason(x_series, y_series, rows, resolved_options)
        if rejection is not None:
            rejections.append(rejection)
            continue

        cadence = max(int(x_series.cadence_seconds or 0), int(y_series.cadence_seconds or 0))
        aligned_bins = _aligned_bin_count(rows, left, right, cadence)
        pairs.append(
            ScannerCandidatePair(
                x_metric=left,
                y_metric=right,
                cadence_seconds=cadence,
                aligned_bins=aligned_bins,
                x_coverage=float(x_series.coverage or 0.0),
                y_coverage=float(y_series.coverage or 0.0),
                x_source_kind=x_series.source_kind,
                y_source_kind=y_series.source_kind,
                compatibility={
                    "cadence": _cadence_relation(x_series, y_series),
                    "geography": _geography_relation(x_series, y_series),
                    "lineage": "independent",
                    "dry_run_only": x_series.source_kind == "public" or y_series.source_kind == "public",
                },
            )
        )

    return ScannerCandidateSelection(
        series=tuple(eligible),
        pairs=tuple(pairs),
        rejections=tuple(rejections),
    )


def _candidate_options(options: CandidateOptions | Mapping[str, Any] | None) -> CandidateOptions:
    if options is None:
        return CandidateOptions()
    if isinstance(options, CandidateOptions):
        return options
    return CandidateOptions(**dict(options))


def _validate_candidate_options(options: CandidateOptions) -> None:
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


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    return row is not None


def _fetch_scanner_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    *,
    include_public: bool,
) -> list[Mapping[str, Any]]:
    rows = _fetch_measurement_scanner_rows(conn, start_utc, end_utc)
    if include_public and _has_table(conn, "public_observations") and _has_table(conn, "series_registry"):
        rows.extend(_fetch_public_scanner_rows(conn, start_utc, end_utc))
    return rows


def _fetch_measurement_scanner_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
) -> list[Mapping[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.id, m.timestamp_utc, m.metric, m.value, m.unit, m.source,
               m.metadata_json AS row_metadata_json,
               s.series_id, s.source_id, s.metric_name, s.display_name,
               s.cadence_seconds AS registry_cadence_seconds,
               s.geography_type, s.geography_id, s.lineage_id, s.parent_series_id,
               s.metadata_json AS series_metadata_json
        FROM measurements m
        LEFT JOIN measurement_series_map msm
          ON msm.source = m.source AND msm.metric = m.metric
        LEFT JOIN series_registry s
          ON s.series_id = msm.series_id
        WHERE m.timestamp_utc >= ? AND m.timestamp_utc <= ?
        ORDER BY m.metric ASC, m.timestamp_utc ASC, m.id ASC
        """,
        (to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchall()
    normalized: list[Mapping[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        fallback_geography_type = "device" if source == "personal" else "configured_location"
        fallback_geography_id = "local_machine" if source == "personal" else "configured_location"
        normalized.append(
            {
                "metric": str(row["metric"]),
                "timestamp_utc": row["timestamp_utc"],
                "value": row["value"],
                "unit": row["unit"],
                "source_kind": "legacy",
                "source_id": row["source_id"] or source,
                "metric_name": row["metric_name"] or row["metric"],
                "display_name": row["display_name"] or row["metric"],
                "registry_cadence_seconds": row["registry_cadence_seconds"],
                "geography_type": row["geography_type"] or fallback_geography_type,
                "geography_id": row["geography_id"] or fallback_geography_id,
                "lineage_id": row["lineage_id"] or row["metric"],
                "parent_series_id": row["parent_series_id"],
                "series_metadata_json": row["series_metadata_json"] or "{}",
                "row_metadata_json": row["row_metadata_json"] or "{}",
            }
        )
    return normalized


def _fetch_public_scanner_rows(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
) -> list[Mapping[str, Any]]:
    rows = conn.execute(
        """
        SELECT o.series_id AS metric, o.valid_start_utc AS timestamp_utc, o.value,
               s.unit, s.source_id, s.metric_name, s.display_name,
               s.cadence_seconds AS registry_cadence_seconds,
               s.geography_type, s.geography_id, s.lineage_id, s.parent_series_id,
               s.metadata_json AS series_metadata_json,
               o.metadata_json AS row_metadata_json
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE o.valid_start_utc >= ?
          AND o.valid_start_utc <= ?
          AND NOT EXISTS (
              SELECT 1
              FROM public_observations newer
              WHERE newer.series_id = o.series_id
                AND newer.source_observation_key = o.source_observation_key
                AND (
                    newer.ingested_at_utc > o.ingested_at_utc
                    OR (
                        newer.ingested_at_utc = o.ingested_at_utc
                        AND newer.observation_id > o.observation_id
                    )
                )
          )
        ORDER BY o.series_id ASC, o.valid_start_utc ASC, o.observation_id ASC
        """,
        (to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchall()
    return [
        {
            "metric": str(row["metric"]),
            "timestamp_utc": row["timestamp_utc"],
            "value": row["value"],
            "unit": row["unit"],
            "source_kind": "public",
            "source_id": row["source_id"],
            "metric_name": row["metric_name"],
            "display_name": row["display_name"],
            "registry_cadence_seconds": row["registry_cadence_seconds"],
            "geography_type": row["geography_type"],
            "geography_id": row["geography_id"],
            "lineage_id": row["lineage_id"],
            "parent_series_id": row["parent_series_id"],
            "series_metadata_json": row["series_metadata_json"] or "{}",
            "row_metadata_json": row["row_metadata_json"] or "{}",
        }
        for row in rows
    ]


def _scanner_series_profile(
    identifier: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    interval_start: datetime,
    interval_end: datetime,
) -> ScannerCandidateSeries:
    numeric_timestamps: list[datetime] = []
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    metadata: list[Mapping[str, Any]] = []
    for row in rows:
        timestamp = _optional_timestamp(row["timestamp_utc"])
        if timestamp is not None:
            start_utc = timestamp if start_utc is None else min(start_utc, timestamp)
            end_utc = timestamp if end_utc is None else max(end_utc, timestamp)
        if timestamp is not None and _optional_float(row["value"]) is not None:
            numeric_timestamps.append(timestamp)
        for key in ("series_metadata_json", "row_metadata_json"):
            parsed = _parse_metadata(row.get(key))
            if parsed:
                metadata.append(parsed)

    first = rows[0] if rows else {}
    cadence_seconds = _infer_cadence_seconds(sorted(numeric_timestamps))
    coverage = None
    if cadence_seconds is not None:
        coverage = _coverage(sorted(numeric_timestamps), cadence_seconds, interval_start, interval_end)
    registry_cadences = {
        int(row["registry_cadence_seconds"])
        for row in rows
        if row.get("registry_cadence_seconds") not in (None, "")
    }
    return ScannerCandidateSeries(
        identifier=identifier,
        source_kind=str(first.get("source_kind") or "legacy"),
        metric_name=str(first.get("metric_name") or identifier),
        display_name=str(first.get("display_name") or identifier),
        unit=str(first.get("unit") or ""),
        source_id=str(first.get("source_id") or ""),
        sample_count=len(rows),
        numeric_count=len(numeric_timestamps),
        cadence_seconds=cadence_seconds,
        registry_cadence_seconds=next(iter(registry_cadences)) if len(registry_cadences) == 1 else None,
        coverage=coverage,
        start_utc=start_utc,
        end_utc=end_utc,
        geography_type=str(first.get("geography_type") or "") or None,
        geography_id=str(first.get("geography_id") or "") or None,
        lineage_id=str(first.get("lineage_id") or "") or None,
        parent_series_id=str(first.get("parent_series_id") or "") or None,
        metadata=tuple(metadata),
    )


def _series_rejection_reason(series: ScannerCandidateSeries, options: CandidateOptions) -> str | None:
    if _is_diagnostic_series(series):
        return "diagnostic_series"
    if _is_status_series(series):
        return "status_or_flag_series"
    if series.numeric_count != series.sample_count:
        return "non_numeric_series"
    if series.numeric_count < options.min_observations:
        return "too_few_observations"
    if series.cadence_seconds is None:
        return "cadence_unavailable"
    if series.registry_cadence_seconds and not _cadences_compatible(series.cadence_seconds, series.registry_cadence_seconds):
        return "incompatible_cadence"
    if series.coverage is None or series.coverage < options.min_coverage:
        return "low_coverage"
    return None


def _is_diagnostic_series(series: ScannerCandidateSeries) -> bool:
    for metadata in series.metadata:
        if metadata.get("analysis_eligible") is False:
            return True
        if metadata.get("diagnostic") is True and metadata.get("analysis_eligible") is not True:
            return True
    return False


def _is_status_series(series: ScannerCandidateSeries) -> bool:
    if series.unit.lower() in STATUS_UNITS:
        return True
    normalized_tokens = [
        token
        for token in series.metric_name.lower().replace("-", "_").split("_")
        if token
    ]
    if any(token in STATUS_TOKENS for token in normalized_tokens):
        return True
    for metadata in series.metadata:
        metadata_type = str(metadata.get("type") or metadata.get("kind") or "").lower()
        if metadata_type in STATUS_UNITS:
            return True
    return False


def _scanner_pair_rejection_reason(
    x_series: ScannerCandidateSeries,
    y_series: ScannerCandidateSeries,
    rows: Sequence[Mapping[str, Any]],
    options: CandidateOptions,
) -> ScannerCandidateRejection | None:
    metrics = (x_series.identifier, y_series.identifier)
    if x_series.identifier == y_series.identifier:
        return ScannerCandidateRejection(metrics, "identical_series")
    lineage_reason = _lineage_rejection_reason(x_series, y_series)
    if lineage_reason is not None:
        return ScannerCandidateRejection(metrics, lineage_reason, "series share direct or declared lineage")
    if not _geographies_compatible(x_series, y_series):
        return ScannerCandidateRejection(metrics, "incompatible_geography", _geography_relation(x_series, y_series))
    if not _cadences_compatible(int(x_series.cadence_seconds or 0), int(y_series.cadence_seconds or 0)):
        return ScannerCandidateRejection(metrics, "incompatible_cadence", _cadence_relation(x_series, y_series))

    cadence = max(int(x_series.cadence_seconds or 0), int(y_series.cadence_seconds or 0))
    aligned_bins = _aligned_bin_count(rows, x_series.identifier, y_series.identifier, cadence)
    if aligned_bins < options.min_aligned_bins:
        return ScannerCandidateRejection(
            metrics,
            "too_few_aligned_bins",
            f"coarsest compatible cadence produced {aligned_bins} aligned bins",
        )
    return None


def _lineage_rejection_reason(
    x_series: ScannerCandidateSeries,
    y_series: ScannerCandidateSeries,
) -> str | None:
    if x_series.parent_series_id and x_series.parent_series_id == y_series.identifier:
        return "direct_lineage"
    if y_series.parent_series_id and y_series.parent_series_id == x_series.identifier:
        return "direct_lineage"
    if x_series.lineage_id and x_series.lineage_id == y_series.lineage_id:
        return "shared_lineage"
    if _metadata_names_series(x_series.metadata, _series_identity_tokens(y_series)):
        return "direct_derivation"
    if _metadata_names_series(y_series.metadata, _series_identity_tokens(x_series)):
        return "direct_derivation"
    return None


def _series_identity_tokens(series: ScannerCandidateSeries) -> set[str]:
    tokens = {
        series.identifier,
        series.metric_name,
        series.lineage_id or "",
        series.parent_series_id or "",
    }
    for metadata in series.metadata:
        for key in ("eia_type", "eia_fuel_type", "metric_name", "series_id"):
            if key in metadata:
                tokens.update(_metadata_metric_names(metadata[key]))
    return {token.lower() for token in tokens if token}


def _metadata_names_series(metadata_values: Sequence[Mapping[str, Any]], tokens: set[str]) -> bool:
    for metadata in metadata_values:
        for key in DIRECT_DERIVATION_KEYS:
            if key in metadata and tokens & _metadata_metric_names(metadata[key]):
                return True
    return False


def _metadata_metric_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, Mapping):
        names: set[str] = set()
        for nested_value in value.values():
            names.update(_metadata_metric_names(nested_value))
        return names
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        names: set[str] = set()
        for item in value:
            names.update(_metadata_metric_names(item))
        return names
    return {str(value).lower()}


def _geographies_compatible(x_series: ScannerCandidateSeries, y_series: ScannerCandidateSeries) -> bool:
    if not (x_series.geography_type and x_series.geography_id and y_series.geography_type and y_series.geography_id):
        return False
    if (
        x_series.geography_type == y_series.geography_type
        and x_series.geography_id == y_series.geography_id
    ):
        return True
    if x_series.source_kind == "public" and y_series.source_kind == "public":
        return False
    types = {x_series.geography_type, y_series.geography_type}
    if types <= LOCAL_GEOGRAPHY_TYPES:
        return True
    return bool(types & LOCAL_GEOGRAPHY_TYPES and types & REGIONAL_GEOGRAPHY_TYPES)


def _geography_relation(x_series: ScannerCandidateSeries, y_series: ScannerCandidateSeries) -> str:
    if (
        x_series.geography_type == y_series.geography_type
        and x_series.geography_id == y_series.geography_id
    ):
        return "same_geography"
    types = {x_series.geography_type or "", y_series.geography_type or ""}
    if types <= LOCAL_GEOGRAPHY_TYPES:
        return "local_device_context"
    if types & LOCAL_GEOGRAPHY_TYPES and types & REGIONAL_GEOGRAPHY_TYPES:
        return "local_to_regional_context"
    return f"{x_series.geography_type}:{x_series.geography_id} vs {y_series.geography_type}:{y_series.geography_id}"


def _cadences_compatible(left: int, right: int) -> bool:
    if left <= 0 or right <= 0:
        return False
    coarser = max(left, right)
    finer = min(left, right)
    return coarser % finer == 0


def _cadence_relation(x_series: ScannerCandidateSeries, y_series: ScannerCandidateSeries) -> str:
    left = int(x_series.cadence_seconds or 0)
    right = int(y_series.cadence_seconds or 0)
    if left == right:
        return f"same:{left}"
    if _cadences_compatible(left, right):
        return f"commensurate:{min(left, right)}->{max(left, right)}"
    return f"incompatible:{left}:{right}"


def _canonical_pair_names(metric_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    names = sorted(dict.fromkeys(metric_names))
    return tuple((names[left], names[right]) for left in range(len(names)) for right in range(left + 1, len(names)))


def _evaluate_pair(
    database_path: Path,
    pair: ScannerCandidatePair,
    start_utc: datetime,
    end_utc: datetime,
    options: ScannerOptions,
) -> PairEvidence:
    rows = _fetch_series_rows(database_path, start_utc, end_utc, (pair.x_metric, pair.y_metric))
    x_series = _series_from_rows(rows, pair.x_metric)
    y_series = _series_from_rows(rows, pair.y_metric)
    if x_series.empty or y_series.empty:
        raise ValueError("pair has no numeric observations")

    raw_pair = align_series(
        x_series,
        y_series,
        cadence_seconds=pair.cadence_seconds,
        min_points=options.min_aligned_observations,
    )
    transformed_pair, transform = _preferred_transformed_pair(raw_pair, options)
    max_lag_steps = options.max_lag_seconds // transformed_pair.cadence_seconds
    discovery_frame = _discovery_frame(transformed_pair.frame, options.discovery_fraction)
    lag_result = lagged_spearman(
        discovery_frame,
        max_lag_steps=max_lag_steps,
        min_overlap=options.min_overlap,
    )
    if lag_result.best_rho is None:
        raise ValueError("no discovery lag met minimum overlap")

    discovery_records = _records_from_frame(discovery_frame)
    validation_records = _records_from_frame(transformed_pair.frame)
    candidate_lags = range(-max_lag_steps, max_lag_steps + 1)
    p_value = max_lag_block_permutation_test(
        discovery_records,
        candidate_lag_steps=candidate_lags,
        permutations=options.permutations,
        block_size=options.permutation_block_size,
        min_overlap=options.min_overlap,
        seed=options.permutation_seed,
    )
    holdout = chronological_holdout_validation(
        validation_records,
        candidate_lag_steps=(lag_result.best_lag_steps,),
        holdout_fraction=1 - options.discovery_fraction,
        min_overlap=options.min_overlap,
    )
    stability = window_stability(
        validation_records,
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
    best_score = _score_for_lag(lag_result.scores, lag_result.best_lag_steps)
    return PairEvidence(
        pair=pair,
        transform=transform,
        lag_result=lag_result,
        validation_result=validation_result,
        corrected_q=1.0,
        aligned_pair=transformed_pair,
        p_value=p_value,
        discovery_overlap=int(best_score["overlap_count"]) if best_score else 0,
    )


def _fetch_series_rows(
    database_path: Path,
    start_utc: datetime,
    end_utc: datetime,
    identifiers: Sequence[str],
) -> list[Mapping[str, Any]]:
    with _connect_read_only(database_path) as conn:
        rows: list[Mapping[str, Any]] = []
        placeholders = ",".join("?" for _ in identifiers)
        rows.extend(
            dict(row)
            for row in conn.execute(
                f"""
                SELECT timestamp_utc, metric, value
                FROM measurements
                WHERE timestamp_utc >= ?
                  AND timestamp_utc <= ?
                  AND metric IN ({placeholders})
                ORDER BY timestamp_utc ASC, metric ASC, id ASC
                """,
                (to_utc_iso(start_utc), to_utc_iso(end_utc), *identifiers),
            )
        )
        if _has_table(conn, "public_observations") and _has_table(conn, "series_registry"):
            rows.extend(
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT o.valid_start_utc AS timestamp_utc, o.series_id AS metric, o.value
                    FROM public_observations o
                    WHERE o.valid_start_utc >= ?
                      AND o.valid_start_utc <= ?
                      AND o.series_id IN ({placeholders})
                      AND NOT EXISTS (
                          SELECT 1
                          FROM public_observations newer
                          WHERE newer.series_id = o.series_id
                            AND newer.source_observation_key = o.source_observation_key
                            AND (
                                newer.ingested_at_utc > o.ingested_at_utc
                                OR (
                                    newer.ingested_at_utc = o.ingested_at_utc
                                    AND newer.observation_id > o.observation_id
                                )
                            )
                      )
                    ORDER BY o.valid_start_utc ASC, o.series_id ASC, o.observation_id ASC
                    """,
                    (to_utc_iso(start_utc), to_utc_iso(end_utc), *identifiers),
                )
            )
        return rows


def _aligned_bin_count(
    rows: Sequence[Mapping[str, Any]],
    x_metric: str,
    y_metric: str,
    cadence_seconds: int,
) -> int:
    if cadence_seconds <= 0:
        return 0
    x_bins = _metric_bins(rows, x_metric, cadence_seconds)
    y_bins = _metric_bins(rows, y_metric, cadence_seconds)
    return len(x_bins & y_bins)


def _metric_bins(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    cadence_seconds: int,
) -> set[int]:
    bins: set[int] = set()
    for row in rows:
        if row["metric"] != metric or _optional_float(row["value"]) is None:
            continue
        timestamp = _optional_timestamp(row["timestamp_utc"])
        if timestamp is None:
            continue
        bins.add(math.floor(timestamp.timestamp() / cadence_seconds))
    return bins


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    if str(database_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        uri = database_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _series_from_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> pd.Series:
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


def _preferred_transformed_pair(
    pair: AlignedPair,
    options: ScannerOptions,
) -> tuple[AlignedPair, str]:
    try:
        return (
            _transform_aligned_pair(
                pair,
                "calendar_residual",
                min_points=options.min_aligned_observations,
                calendar_min_history=options.calendar_min_history,
                calendar_timezone=options.calendar_timezone,
            ),
            "calendar_residual",
        )
    except ValueError:
        return (
            _transform_aligned_pair(
                pair,
                "first_difference",
                min_points=options.min_aligned_observations,
                calendar_min_history=options.calendar_min_history,
                calendar_timezone=options.calendar_timezone,
            ),
            "first_difference",
        )


def _transform_aligned_pair(
    pair: AlignedPair,
    transform_name: str,
    *,
    min_points: int,
    calendar_min_history: int,
    calendar_timezone: str,
) -> AlignedPair:
    transform_kwargs: dict[str, Any] = {}
    if transform_name == "calendar_residual":
        transform_kwargs["cadence_seconds"] = pair.cadence_seconds
        transform_kwargs["min_history"] = calendar_min_history
        transform_kwargs["timezone_name"] = calendar_timezone

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


def _discovery_frame(frame: pd.DataFrame, discovery_fraction: float) -> pd.DataFrame:
    split_index = max(1, min(len(frame) - 1, round(len(frame) * discovery_fraction)))
    return frame.iloc[:split_index]


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"timestamp_utc": timestamp.to_pydatetime(), "x": row["x"], "y": row["y"]}
        for timestamp, row in frame.iterrows()
    ]


def _score_for_lag(scores: Sequence[Mapping[str, Any]], lag_steps: int) -> Mapping[str, Any] | None:
    for score in scores:
        if score["lag_steps"] == lag_steps:
            return score
    return None


def _adjust_p_values(
    p_values: Sequence[float | None],
    *,
    total_tests: int,
    method: str = DEFAULT_MULTIPLE_TESTING_METHOD,
) -> tuple[float, ...]:
    """Adjust p-values with BH or the more conservative BY procedure.

    BY multiplies the BH correction by the harmonic sum of all tested
    hypotheses, which controls false discoveries under arbitrary dependence.
    """
    if total_tests <= 0:
        return ()
    if method not in {"bh", "by"}:
        raise ValueError("method must be 'bh' or 'by'")
    padded = [
        (1.0 if p_value is None else min(1.0, max(0.0, float(p_value))), index)
        for index, p_value in enumerate(p_values)
    ]
    padded.extend((1.0, index) for index in range(len(padded), total_tests))
    ordered = sorted(padded, key=lambda item: item[0])
    q_values = [1.0] * len(padded)
    running_min = 1.0
    dependency_factor = (
        sum(1.0 / index for index in range(1, total_tests + 1))
        if method == "by"
        else 1.0
    )
    for rank, (p_value, original_index) in reversed(list(enumerate(ordered, start=1))):
        running_min = min(
            running_min,
            p_value * total_tests * dependency_factor / rank,
        )
        q_values[original_index] = min(1.0, running_min)
    return tuple(q_values[: len(p_values)])


def _passes_promotion_thresholds(evidence: PairEvidence, options: ScannerOptions) -> bool:
    discovery_rho = evidence.lag_result.best_rho
    holdout_rho = evidence.validation_result.holdout_rho
    stability = evidence.validation_result.sign_stability
    if discovery_rho is None or holdout_rho is None or stability is None:
        return False
    if abs(discovery_rho) < options.min_discovery_abs_rho:
        return False
    if evidence.corrected_q > options.max_corrected_q:
        return False
    if abs(holdout_rho) < options.min_holdout_abs_rho:
        return False
    if _sign(discovery_rho) != _sign(holdout_rho):
        return False
    return stability >= options.min_sign_stability


def _finding_from_evidence(
    evidence: PairEvidence,
    *,
    first_seen_utc: datetime,
    verified_utc: datetime,
    options: ScannerOptions,
    total_tests: int,
) -> CorrelationFinding:
    validation = evidence.validation_result
    aligned = evidence.aligned_pair
    return CorrelationFinding(
        x_metric=evidence.pair.x_metric,
        y_metric=evidence.pair.y_metric,
        transform=evidence.transform,
        lag_seconds=evidence.lag_result.best_lag_seconds,
        discovery_rho=float(evidence.lag_result.best_rho),
        holdout_rho=float(validation.holdout_rho),
        corrected_q=evidence.corrected_q,
        stability=float(validation.sign_stability),
        overlap_count=validation.holdout_overlap,
        first_seen_utc=first_seen_utc,
        last_verified_utc=verified_utc,
        status="active",
        evidence={
            "cadence_seconds": aligned.cadence_seconds,
            "aligned_observation_count": len(aligned.frame),
            "aligned_start_utc": to_utc_iso(aligned.start_utc),
            "aligned_end_utc": to_utc_iso(aligned.end_utc),
            "x_coverage": _clean_number(aligned.x_coverage),
            "y_coverage": _clean_number(aligned.y_coverage),
            "discovery_overlap": evidence.discovery_overlap,
            "permutation_p_value": _clean_optional_number(evidence.p_value),
            "window_scores": _json_safe(validation.window_scores),
            "warnings": list(validation.warnings),
            "selected_on": f"first_{round(options.discovery_fraction * 100)}_percent",
            "validated_on": f"last_{round((1 - options.discovery_fraction) * 100)}_percent",
            "calendar_timezone": options.calendar_timezone,
            "multiple_testing": {
                "method": options.multiple_testing_method,
                "total_tests": total_tests,
            },
            "scanner_series": {
                "x_source_kind": evidence.pair.x_source_kind,
                "y_source_kind": evidence.pair.y_source_kind,
            },
            "pair_compatibility": _json_safe(evidence.pair.compatibility),
            "dry_run_only": evidence.pair.contains_public,
            "association_only": True,
        },
    )


def _sign(value: float | None) -> int | None:
    if value is None or value == 0:
        return None
    return 1 if value > 0 else -1


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
    if isinstance(value, float):
        return _clean_number(value)
    return value


def _clean_optional_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    return _clean_number(value)


def _clean_number(value: float | int) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 6)


__all__ = [
    "ScannerOptions",
    "finding_to_dict",
    "scan_correlations",
]
