from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from resonance.analysis.contracts import AlignedPair, LagScanResult, ValidationResult
from resonance.analysis.service import AnalyzableMetric, MetricPairAnalysis
from resonance.ui.pair_explorer import (
    coverage_rows,
    evidence_metrics,
    evidence_statement,
    max_lag_steps,
    metric_by_name,
    metric_names,
    pair_cadence_seconds,
    selected_interval,
    selected_max_lag,
    selected_transform,
)


def test_pair_explorer_options_match_analysis_service_names() -> None:
    assert selected_interval("6 hours") == timedelta(hours=6)
    assert selected_interval("24 hours") == timedelta(hours=24)
    assert selected_interval("7 days") == timedelta(days=7)
    assert selected_interval("30 days") == timedelta(days=30)
    assert selected_transform("raw") == "raw"
    assert selected_transform("first difference") == "first_difference"
    assert selected_transform("robust rolling z-score") == "rolling_robust_zscore"
    assert selected_transform("calendar residual") == "calendar_residual"
    assert selected_max_lag("1 hour") == timedelta(hours=1)


def test_metric_helpers_sort_and_lookup_metric_summaries() -> None:
    cpu = _summary("cpu_percent")
    tcp = _summary("tcp_latency_ms")

    assert metric_names([tcp, cpu]) == ("cpu_percent", "tcp_latency_ms")
    assert metric_by_name([tcp, cpu]) == {"tcp_latency_ms": tcp, "cpu_percent": cpu}


def test_metric_helpers_display_public_labels_but_return_internal_series() -> None:
    grid = _summary(
        "eia_grid_monitor:ISNE:system_load",
        display_name="ISO New England system load [ISNE]",
    )
    temp = _summary("weather_temperature_c")

    assert metric_names([grid, temp]) == ("ISO New England system load [ISNE]", "weather_temperature_c")
    by_label = metric_by_name([grid, temp])
    assert by_label["ISO New England system load [ISNE]"].metric == "eia_grid_monitor:ISNE:system_load"


def test_pair_cadence_and_max_lag_steps_are_explicit() -> None:
    x_summary = _summary("tcp_latency_ms", cadence_seconds=300)
    y_summary = _summary("cpu_percent", cadence_seconds=600)

    cadence_seconds = pair_cadence_seconds(x_summary, y_summary)

    assert cadence_seconds == 600
    assert max_lag_steps(timedelta(hours=1), cadence_seconds) == 6
    assert max_lag_steps(timedelta(minutes=5), cadence_seconds) == 0
    with pytest.raises(ValueError, match="cadence_seconds"):
        max_lag_steps(timedelta(hours=1), 0)


def test_pair_cadence_requires_both_metric_cadences() -> None:
    assert pair_cadence_seconds(
        _summary("x", cadence_seconds=300),
        _summary("y", cadence_seconds=None),
    ) is None


def test_coverage_rows_put_sample_counts_before_graphs() -> None:
    analysis = _analysis(best_lag_seconds=600)

    rows = coverage_rows(analysis)

    assert rows == [
        {"Series": "X", "Metric": "tcp_latency_ms", "Samples": 120, "Coverage": "90%", "Cadence": "5 minutes"},
        {"Series": "Y", "Metric": "cpu_percent", "Samples": 118, "Coverage": "80%", "Cadence": "5 minutes"},
        {
            "Series": "Aligned pair",
            "Metric": "tcp_latency_ms / cpu_percent",
            "Samples": 3,
            "Coverage": "n/a",
            "Cadence": "5 minutes",
        },
    ]


def test_evidence_metrics_and_statement_use_association_language() -> None:
    analysis = _analysis(best_lag_seconds=600)

    assert evidence_metrics(analysis) == {
        "Train rho": "0.81",
        "Best lag": "+10 minutes (positive lag)",
        "Permutation p-value": "0.01",
        "Holdout rho": "0.77",
        "Stability": "1",
    }
    statement = evidence_statement(analysis)
    assert statement == "Association: tcp_latency_ms and cpu_percent align at +10 minutes (positive lag)."
    assert "cause" not in statement.lower()


def test_negative_lag_statement_uses_signed_association_language() -> None:
    analysis = _analysis(best_lag_seconds=-300)

    assert evidence_metrics(analysis)["Best lag"] == "-5 minutes (negative lag)"
    assert evidence_statement(analysis) == "Association: tcp_latency_ms and cpu_percent align at -5 minutes (negative lag)."


def test_insufficient_evidence_statement_is_allowed() -> None:
    analysis = _analysis(best_lag_seconds=600, permutation_p_value=0.5)

    assert evidence_statement(analysis) == "Insufficient evidence for a stable association in this interval."


def _summary(
    metric: str,
    cadence_seconds: int | None = 300,
    display_name: str | None = None,
) -> AnalyzableMetric:
    return AnalyzableMetric(
        metric=metric,
        units=("unit",),
        sources=("synthetic",),
        sample_count=120 if metric == "tcp_latency_ms" else 118,
        cadence_seconds=cadence_seconds,
        coverage=0.9,
        start_utc=_ts(0),
        end_utc=_ts(2),
        display_name=display_name,
    )


def _analysis(
    *,
    best_lag_seconds: int,
    permutation_p_value: float = 0.01,
) -> MetricPairAnalysis:
    return MetricPairAnalysis(
        aligned_pair=AlignedPair(
            x_metric="tcp_latency_ms",
            y_metric="cpu_percent",
            cadence_seconds=300,
            frame=(
                {"timestamp_utc": _ts(0), "x": 1.0, "y": 3.0},
                {"timestamp_utc": _ts(1), "x": 2.0, "y": 4.0},
                {"timestamp_utc": _ts(2), "x": 3.0, "y": 5.0},
            ),
            x_coverage=0.9,
            y_coverage=0.8,
            start_utc=_ts(0),
            end_utc=_ts(2),
        ),
        transform_name="raw",
        lag_result=LagScanResult(
            scores=(
                {
                    "lag_steps": best_lag_seconds // 300,
                    "lag_seconds": best_lag_seconds,
                    "rho": 0.81,
                    "overlap_count": 40,
                },
            ),
            best_lag_steps=best_lag_seconds // 300,
            best_lag_seconds=best_lag_seconds,
            best_rho=0.81,
        ),
        validation_result=ValidationResult(
            permutation_p_value=permutation_p_value,
            holdout_rho=0.77,
            holdout_overlap=30,
            sign_stability=1.0,
            window_scores=(),
        ),
        x_metric_summary=_summary("tcp_latency_ms"),
        y_metric_summary=_summary("cpu_percent"),
    )


def _ts(offset: int) -> datetime:
    return datetime(2026, 6, 18, tzinfo=timezone.utc) + timedelta(minutes=5 * offset)
