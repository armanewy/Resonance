from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from resonance.analysis.alignment import align_series
from resonance.analysis.candidates import CandidateOptions, CandidatePair, select_candidate_pairs
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
    pair: CandidatePair
    transform: str
    lag_result: LagScanResult
    validation_result: ValidationResult
    corrected_q: float
    aligned_pair: AlignedPair
    p_value: float | None
    discovery_overlap: int


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

    selection = select_candidate_pairs(
        database_path,
        start_utc,
        end_utc,
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


def _evaluate_pair(
    database_path: Path,
    pair: CandidatePair,
    start_utc: datetime,
    end_utc: datetime,
    options: ScannerOptions,
) -> PairEvidence:
    rows = _fetch_metric_rows(database_path, start_utc, end_utc, (pair.x_metric, pair.y_metric))
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


def _fetch_metric_rows(
    database_path: Path,
    start_utc: datetime,
    end_utc: datetime,
    metrics: Sequence[str],
) -> list[sqlite3.Row]:
    with _connect_read_only(database_path) as conn:
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


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    if str(database_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        uri = database_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


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
