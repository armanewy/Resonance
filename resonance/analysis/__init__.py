"""Contracts and helpers for pairwise time-series analysis."""

from resonance.analysis.contracts import (
    ALIGNED_PAIR_COLUMNS,
    LAG_SCAN_SCORE_COLUMNS,
    AlignedPair,
    LagScanResult,
    PairAnalysis,
    ValidationResult,
)
from resonance.analysis.alignment import align_series
from resonance.analysis.correlation import lagged_spearman
from resonance.analysis.transforms import (
    TRANSFORMS,
    apply_transform,
    calendar_residual,
    first_difference,
    raw,
    rolling_robust_zscore,
)
from resonance.analysis.validation import (
    chronological_holdout_validation,
    max_lag_block_permutation_test,
    window_stability,
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
    "chronological_holdout_validation",
    "first_difference",
    "lagged_spearman",
    "max_lag_block_permutation_test",
    "raw",
    "rolling_robust_zscore",
    "window_stability",
]
