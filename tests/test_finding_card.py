from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from resonance.analysis import AlignedPair, LagScanResult, PairAnalysis, ValidationResult
from resonance.storage import CorrelationFinding
from resonance.ui.finding_card import (
    association_statement,
    evidence_rows,
    render_finding_card,
    summary_metrics,
    warning_messages,
    weakest_windows,
)


def test_summary_metrics_show_verified_finding_fields_with_restrained_direction() -> None:
    finding = _finding(lag_seconds=600)

    rows = summary_metrics(finding)

    assert rows == {
        "Metrics": "tcp_latency_ms / cpu_percent",
        "Direction and lag": "tcp_latency_ms precedes cpu_percent in this dataset at +10 minutes",
        "Discovery rho": "0.8123",
        "Holdout rho": "0.71",
        "Corrected q": "0.004",
        "Stability": "0.75",
        "Aligned observations": "288",
        "Verified": "2026-06-19T12:00:00+00:00",
    }
    statement = association_statement(finding)
    assert statement == (
        "tcp_latency_ms and cpu_percent are associated; "
        "tcp_latency_ms precedes cpu_percent in this dataset."
    )
    assert "caus" not in statement.lower()


def test_negative_lag_direction_reverses_precedence_wording() -> None:
    finding = _finding(lag_seconds=-300)

    assert summary_metrics(finding)["Direction and lag"] == (
        "cpu_percent precedes tcp_latency_ms in this dataset at -5 minutes"
    )


def test_evidence_rows_include_interval_transform_coverage_validation_and_windows() -> None:
    finding = _finding(failed_validation_dimensions=("coverage", "holdout"))
    analysis = _analysis()

    rows = evidence_rows(finding, analysis)

    assert rows == [
        {"Evidence": "Data interval", "Value": "2026-06-18T12:00:00Z to 2026-06-19T12:00:00Z"},
        {"Evidence": "Transform", "Value": "first_difference"},
        {"Evidence": "Coverage", "Value": "tcp_latency_ms: 92%; cpu_percent: 86%"},
        {"Evidence": "Failed validation dimensions", "Value": "coverage, holdout"},
        {
            "Evidence": "Strongest supporting windows",
            "Value": "2026-06-18T12:00:00Z to 2026-06-18T18:00:00Z rho=0.91 n=80; "
            "2026-06-18T18:00:00Z to 2026-06-19T00:00:00Z rho=0.52 n=70; "
            "2026-06-19T00:00:00Z to 2026-06-19T06:00:00Z rho=0.11 n=68",
        },
        {
            "Evidence": "Weakest/counterexample windows",
            "Value": "relationship weakened: 2026-06-19T06:00:00Z to 2026-06-19T12:00:00Z rho=-0.22 n=66",
        },
        {"Evidence": "First seen", "Value": "2026-06-19T11:00:00+00:00"},
        {"Evidence": "Last verified", "Value": "2026-06-19T12:00:00+00:00"},
        {"Evidence": "Data split", "Value": "Selected on first_70_percent; validated on last_30_percent"},
    ]


def test_warning_messages_and_weak_windows_are_optional() -> None:
    finding = _finding(warnings=("small holdout",), window_scores=())
    analysis = _analysis(window_scores=())

    assert warning_messages(finding) == ("small holdout",)
    assert weakest_windows(finding, analysis) == "None available"


def test_render_finding_card_composes_metrics_evidence_and_four_charts() -> None:
    streamlit = _FakeStreamlit()

    render_finding_card(_finding(), _analysis(), streamlit=streamlit)

    assert streamlit.subheaders == ["tcp_latency_ms associated with cpu_percent"]
    assert streamlit.captions == [
        "tcp_latency_ms and cpu_percent are associated; tcp_latency_ms precedes cpu_percent in this dataset."
    ]
    assert len(streamlit.metric_calls) == 8
    assert [label for label, _value in streamlit.metric_calls[:2]] == ["Metrics", "Direction and lag"]
    assert len(streamlit.plotly_charts) == 4
    assert streamlit.expanders == ["Evidence"]
    assert len(streamlit.dataframes) == 1
    assert isinstance(streamlit.dataframes[0], pd.DataFrame)


def test_finding_card_source_avoids_prohibited_wording() -> None:
    source = Path("resonance/ui/finding_card.py").read_text(encoding="utf-8").lower()

    assert "caused" not in source


class _FakeColumn:
    def __init__(self, parent: _FakeStreamlit) -> None:
        self.parent = parent

    def metric(self, label: str, value: str) -> None:
        self.parent.metric_calls.append((label, value))


class _FakeExpander:
    def __init__(self, parent: _FakeStreamlit, label: str) -> None:
        self.parent = parent
        self.label = label

    def __enter__(self) -> _FakeStreamlit:
        self.parent.expanders.append(self.label)
        return self.parent

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.warnings: list[str] = []
        self.metric_calls: list[tuple[str, str]] = []
        self.plotly_charts: list[object] = []
        self.expanders: list[str] = []
        self.dataframes: list[object] = []

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def caption(self, value: str) -> None:
        self.captions.append(value)

    def warning(self, value: str) -> None:
        self.warnings.append(value)

    def columns(self, count: int) -> list[_FakeColumn]:
        return [_FakeColumn(self) for _ in range(count)]

    def plotly_chart(self, figure, *, use_container_width: bool) -> None:
        assert use_container_width is True
        self.plotly_charts.append(figure)

    def expander(self, label: str) -> _FakeExpander:
        return _FakeExpander(self, label)

    def dataframe(self, dataframe, *, use_container_width: bool, hide_index: bool) -> None:
        assert use_container_width is True
        assert hide_index is True
        self.dataframes.append(dataframe)


def _finding(
    *,
    lag_seconds: int = 600,
    failed_validation_dimensions: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    window_scores: tuple[dict[str, object], ...] | None = None,
) -> CorrelationFinding:
    return CorrelationFinding(
        x_metric="tcp_latency_ms",
        y_metric="cpu_percent",
        transform="first_difference",
        lag_seconds=lag_seconds,
        discovery_rho=0.81234,
        holdout_rho=0.71,
        corrected_q=0.004,
        stability=0.75,
        overlap_count=72,
        first_seen_utc=_ts(132),
        last_verified_utc=_ts(144),
        status="active",
        evidence={
            "aligned_observation_count": 288,
            "aligned_start_utc": "2026-06-18T12:00:00Z",
            "aligned_end_utc": "2026-06-19T12:00:00Z",
            "x_coverage": 0.92,
            "y_coverage": 0.86,
            "failed_validation_dimensions": list(failed_validation_dimensions),
            "window_scores": list(_default_window_scores() if window_scores is None else window_scores),
            "warnings": list(warnings),
            "selected_on": "first_70_percent",
            "validated_on": "last_30_percent",
            "association_only": True,
        },
    )


def _analysis(
    *,
    window_scores: tuple[dict[str, object], ...] | None = None,
) -> PairAnalysis:
    return PairAnalysis(
        aligned_pair=AlignedPair(
            x_metric="tcp_latency_ms",
            y_metric="cpu_percent",
            cadence_seconds=300,
            frame=(
                {"timestamp_utc": _ts(0), "x": 1.0, "y": 3.0},
                {"timestamp_utc": _ts(1), "x": 2.0, "y": 4.0},
                {"timestamp_utc": _ts(2), "x": 3.0, "y": 5.0},
                {"timestamp_utc": _ts(3), "x": 4.0, "y": 6.0},
            ),
            x_coverage=0.9,
            y_coverage=0.8,
            start_utc=_ts(0),
            end_utc=_ts(3),
        ),
        transform_name="first_difference",
        lag_result=LagScanResult(
            scores=(
                {"lag_steps": 0, "lag_seconds": 0, "rho": 0.2, "overlap_count": 4},
                {"lag_steps": 2, "lag_seconds": 600, "rho": 0.81234, "overlap_count": 2},
            ),
            best_lag_steps=2,
            best_lag_seconds=600,
            best_rho=0.81234,
        ),
        validation_result=ValidationResult(
            permutation_p_value=0.01,
            holdout_rho=0.71,
            holdout_overlap=72,
            sign_stability=0.75,
            window_scores=tuple(_default_window_scores() if window_scores is None else window_scores),
        ),
    )


def _default_window_scores() -> tuple[dict[str, object], ...]:
    return (
        {
            "window_index": 0,
            "rho": 0.91,
            "overlap_count": 80,
            "start_utc": "2026-06-18T12:00:00Z",
            "end_utc": "2026-06-18T18:00:00Z",
        },
        {
            "window_index": 1,
            "rho": 0.52,
            "overlap_count": 70,
            "start_utc": "2026-06-18T18:00:00Z",
            "end_utc": "2026-06-19T00:00:00Z",
        },
        {
            "window_index": 2,
            "rho": 0.11,
            "overlap_count": 68,
            "start_utc": "2026-06-19T00:00:00Z",
            "end_utc": "2026-06-19T06:00:00Z",
        },
        {
            "window_index": 3,
            "rho": -0.22,
            "overlap_count": 66,
            "start_utc": "2026-06-19T06:00:00Z",
            "end_utc": "2026-06-19T12:00:00Z",
        },
    )


def _ts(offset: int) -> datetime:
    return datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * offset)
