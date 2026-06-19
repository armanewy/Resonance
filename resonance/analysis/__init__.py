"""Contracts for pairwise time-series analysis."""

from resonance.analysis.contracts import (
    ALIGNED_PAIR_COLUMNS,
    LAG_SCAN_SCORE_COLUMNS,
    AlignedPair,
    LagScanResult,
    PairAnalysis,
    ValidationResult,
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
    "chronological_holdout_validation",
    "max_lag_block_permutation_test",
    "window_stability",
]
