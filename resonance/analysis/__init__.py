"""Contracts for pairwise time-series analysis."""

from resonance.analysis.contracts import (
    ALIGNED_PAIR_COLUMNS,
    LAG_SCAN_SCORE_COLUMNS,
    AlignedPair,
    LagScanResult,
    PairAnalysis,
    ValidationResult,
)
from resonance.analysis.alignment import align_series
from resonance.analysis.transforms import (
    TRANSFORMS,
    apply_transform,
    calendar_residual,
    first_difference,
    raw,
    rolling_robust_zscore,
)

__all__ = [
    "ALIGNED_PAIR_COLUMNS",
    "LAG_SCAN_SCORE_COLUMNS",
    "AlignedPair",
    "LagScanResult",
    "PairAnalysis",
    "ValidationResult",
    "TRANSFORMS",
    "align_series",
    "apply_transform",
    "calendar_residual",
    "first_difference",
    "raw",
    "rolling_robust_zscore",
]
