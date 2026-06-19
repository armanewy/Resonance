"""Contracts and helpers for pairwise time-series analysis."""

from resonance.analysis.contracts import (
    ALIGNED_PAIR_COLUMNS,
    LAG_SCAN_SCORE_COLUMNS,
    AlignedPair,
    LagScanResult,
    PairAnalysis,
    ValidationResult,
)
from resonance.analysis.correlation import lagged_spearman

__all__ = [
    "ALIGNED_PAIR_COLUMNS",
    "LAG_SCAN_SCORE_COLUMNS",
    "AlignedPair",
    "LagScanResult",
    "PairAnalysis",
    "ValidationResult",
    "lagged_spearman",
]
