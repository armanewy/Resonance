from __future__ import annotations

from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go

from resonance.analysis import AlignedPair, LagScanResult, PairAnalysis, ValidationResult
from resonance.ui import (
    aligned_transformed_timeline,
    lag_profile,
    lagged_scatter,
    stability_chart,
)


def test_aligned_transformed_timeline_shifts_y_and_keeps_missing_gaps() -> None:
    analysis = _analysis()

    fig = aligned_transformed_timeline(analysis)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "cpu_percent X (first_difference)"
    assert fig.data[1].name == "temperature_2m Y lag-aligned +10m (first_difference)"
    assert fig.data[0].connectgaps is False
    assert fig.data[1].connectgaps is False
    assert list(fig.data[0].y) == [1.0, 2.0, None, 4.0, 5.0]
    assert list(fig.data[1].y) == [20.0, 40.0, 50.0, None, None]
    assert fig.layout.xaxis.title.text == "Timestamp (UTC)"
    assert fig.layout.yaxis.title.text == "Transformed value"
    assert "selected lag +10m" in fig.layout.title.text


def test_lag_profile_marks_selected_lag_and_zero_lag_reference() -> None:
    analysis = _analysis()

    fig = lag_profile(analysis)

    assert len(fig.data) == 2
    assert fig.data[0].name == "Lag scores"
    assert list(fig.data[0].x) == [-10, 0, 10]
    assert list(fig.data[0].y) == [-0.1, 0.2, 0.8]
    assert fig.data[1].name == "Selected lag"
    assert list(fig.data[1].x) == [10]
    assert list(fig.data[1].y) == [0.8]
    assert fig.layout.xaxis.title.text == "Lag (minutes)"
    assert fig.layout.yaxis.title.text == "Spearman rho"
    assert any(shape.type == "line" and shape.x0 == 0 and shape.x1 == 0 for shape in fig.layout.shapes)


def test_lagged_scatter_pairs_selected_lag_and_exposes_hover_timestamps() -> None:
    analysis = _analysis()

    fig = lagged_scatter(analysis)

    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"
    assert list(fig.data[0].x) == [1.0, 2.0]
    assert list(fig.data[0].y) == [20.0, 40.0]
    assert fig.data[0].customdata[0][0] == _ts(0)
    assert fig.data[0].customdata[0][1] == _ts(2)
    assert "Spearman rho=0.8" in fig.layout.title.text
    assert "overlap=2" in fig.layout.title.text
    assert fig.layout.xaxis.title.text == "cpu_percent X (first_difference)"
    assert fig.layout.yaxis.title.text == "temperature_2m Y at selected lag (first_difference)"


def test_stability_chart_plots_windows_and_zero_reference() -> None:
    analysis = _analysis()

    fig = stability_chart(analysis)

    assert len(fig.data) == 1
    assert fig.data[0].name == "Window rho"
    assert list(fig.data[0].y) == [0.7, -0.2]
    assert list(fig.data[0].customdata) == [2, 2]
    assert fig.layout.xaxis.title.text == "Chronological window"
    assert fig.layout.yaxis.title.text == "Spearman rho"
    assert any(shape.type == "line" and shape.y0 == 0 and shape.y1 == 0 for shape in fig.layout.shapes)


def _analysis() -> PairAnalysis:
    frame = (
        {"timestamp_utc": _ts(0), "x": 1.0, "y": 10.0},
        {"timestamp_utc": _ts(1), "x": 2.0, "y": None},
        {"timestamp_utc": _ts(2), "x": None, "y": 20.0},
        {"timestamp_utc": _ts(3), "x": 4.0, "y": 40.0},
        {"timestamp_utc": _ts(4), "x": 5.0, "y": 50.0},
    )
    aligned_pair = AlignedPair(
        x_metric="cpu_percent",
        y_metric="temperature_2m",
        cadence_seconds=300,
        frame=frame,
        x_coverage=0.8,
        y_coverage=0.8,
        start_utc=_ts(0),
        end_utc=_ts(4),
    )
    lag_result = LagScanResult(
        scores=(
            {"lag_steps": -2, "lag_seconds": -600, "rho": -0.1, "overlap_count": 2},
            {"lag_steps": 0, "lag_seconds": 0, "rho": 0.2, "overlap_count": 3},
            {"lag_steps": 2, "lag_seconds": 600, "rho": 0.8, "overlap_count": 2},
        ),
        best_lag_steps=2,
        best_lag_seconds=600,
        best_rho=0.8,
    )
    validation_result = ValidationResult(
        permutation_p_value=0.04,
        holdout_rho=0.75,
        holdout_overlap=2,
        sign_stability=0.5,
        window_scores=(
            {"window_index": 0, "lag_steps": 2, "rho": 0.7, "overlap_count": 2, "start_utc": _ts(0), "end_utc": _ts(1)},
            {"window_index": 1, "lag_steps": 2, "rho": -0.2, "overlap_count": 2, "start_utc": _ts(2), "end_utc": _ts(4)},
        ),
    )
    return PairAnalysis(
        aligned_pair=aligned_pair,
        transform_name="first_difference",
        lag_result=lag_result,
        validation_result=validation_result,
    )


def _ts(offset: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * offset)
