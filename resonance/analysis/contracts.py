from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, TypeAlias


FrameLike: TypeAlias = Any
TableLike: TypeAlias = tuple[Mapping[str, Any], ...]

ALIGNED_PAIR_COLUMNS = ("x", "y")
LAG_SCAN_SCORE_COLUMNS = ("lag_steps", "lag_seconds", "rho", "overlap_count")


@dataclass(frozen=True)
class AlignedPair:
    x_metric: str
    y_metric: str
    cadence_seconds: int
    frame: FrameLike
    x_coverage: float
    y_coverage: float
    start_utc: datetime
    end_utc: datetime


@dataclass(frozen=True)
class LagScanResult:
    scores: TableLike
    best_lag_steps: int
    best_lag_seconds: int
    best_rho: float | None


@dataclass(frozen=True)
class ValidationResult:
    permutation_p_value: float | None
    holdout_rho: float | None
    holdout_overlap: int
    sign_stability: float | None
    window_scores: TableLike = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PairAnalysis:
    aligned_pair: AlignedPair
    transform_name: str
    lag_result: LagScanResult
    validation_result: ValidationResult
