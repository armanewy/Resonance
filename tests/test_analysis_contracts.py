from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime, timezone

import pytest

from resonance.analysis import (
    ALIGNED_PAIR_COLUMNS,
    LAG_SCAN_SCORE_COLUMNS,
    AlignedPair,
    LagScanResult,
    PairAnalysis,
    ValidationResult,
)
from resonance.analysis.contracts import TableLike


def test_analysis_contracts_import_from_package() -> None:
    assert ALIGNED_PAIR_COLUMNS == ("x", "y")
    assert LAG_SCAN_SCORE_COLUMNS == ("lag_steps", "lag_seconds", "rho", "overlap_count")
    assert is_dataclass(AlignedPair)
    assert is_dataclass(LagScanResult)
    assert is_dataclass(ValidationResult)
    assert is_dataclass(PairAnalysis)


def test_pair_analysis_contract_is_immutable_and_composable() -> None:
    aligned_pair = _aligned_pair()
    lag_result = LagScanResult(
        scores=(
            {"lag_steps": 0, "lag_seconds": 0, "rho": 0.25, "overlap_count": 12},
            {"lag_steps": 1, "lag_seconds": 300, "rho": 0.71, "overlap_count": 11},
        ),
        best_lag_steps=1,
        best_lag_seconds=300,
        best_rho=0.71,
    )
    validation_result = ValidationResult(
        permutation_p_value=0.04,
        holdout_rho=0.66,
        holdout_overlap=8,
        sign_stability=1.0,
        window_scores=({"window": "holdout", "rho": 0.66, "overlap_count": 8},),
        warnings=("small holdout",),
    )

    analysis = PairAnalysis(
        aligned_pair=aligned_pair,
        transform_name="identity",
        lag_result=lag_result,
        validation_result=validation_result,
    )

    assert analysis.aligned_pair.frame[0]["x"] == 1.0
    assert analysis.lag_result.best_lag_seconds == 300
    assert analysis.validation_result.holdout_overlap == 8
    with pytest.raises(FrozenInstanceError):
        analysis.transform_name = "zscore"  # type: ignore[misc]


def test_validation_defaults_are_empty_immutable_collections() -> None:
    validation_result = ValidationResult(
        permutation_p_value=None,
        holdout_rho=None,
        holdout_overlap=0,
        sign_stability=None,
    )

    assert validation_result.window_scores == ()
    assert validation_result.warnings == ()


def test_contract_examples_use_utc_aware_timestamps_and_expected_columns() -> None:
    aligned_pair = _aligned_pair()
    score = {"lag_steps": 0, "lag_seconds": 0, "rho": None, "overlap_count": 0}

    assert aligned_pair.start_utc.tzinfo is timezone.utc
    assert aligned_pair.end_utc.tzinfo is timezone.utc
    assert set(aligned_pair.frame[0]) == {"timestamp_utc", *ALIGNED_PAIR_COLUMNS}
    assert tuple(score) == LAG_SCAN_SCORE_COLUMNS


def _aligned_pair() -> AlignedPair:
    frame: TableLike = (
        {
            "timestamp_utc": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            "x": 1.0,
            "y": None,
        },
        {
            "timestamp_utc": datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
            "x": None,
            "y": 2.0,
        },
    )
    return AlignedPair(
        x_metric="cpu_percent",
        y_metric="temperature_2m",
        cadence_seconds=300,
        frame=frame,
        x_coverage=0.5,
        y_coverage=0.5,
        start_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
