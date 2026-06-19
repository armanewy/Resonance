from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from resonance.analysis.contracts import PairAnalysis
from resonance.analysis.service import AnalyzableMetric, MetricPairAnalysis


PAIR_EXPLORER_INTERVALS: dict[str, timedelta] = {
    "6 hours": timedelta(hours=6),
    "24 hours": timedelta(hours=24),
    "7 days": timedelta(days=7),
    "30 days": timedelta(days=30),
}

PAIR_TRANSFORMS: dict[str, str] = {
    "raw": "raw",
    "first difference": "first_difference",
    "robust rolling z-score": "rolling_robust_zscore",
    "calendar residual": "calendar_residual",
}

PAIR_MAX_LAGS: dict[str, timedelta] = {
    "0 minutes": timedelta(minutes=0),
    "15 minutes": timedelta(minutes=15),
    "30 minutes": timedelta(minutes=30),
    "1 hour": timedelta(hours=1),
    "3 hours": timedelta(hours=3),
    "6 hours": timedelta(hours=6),
    "12 hours": timedelta(hours=12),
    "24 hours": timedelta(hours=24),
    "7 days": timedelta(days=7),
}


@dataclass(frozen=True)
class PairExplorerSelection:
    x_metric: str
    y_metric: str
    interval_label: str
    transform_label: str
    max_lag_label: str


def metric_names(metrics: Iterable[AnalyzableMetric]) -> tuple[str, ...]:
    return tuple(metric.metric for metric in sorted(metrics, key=lambda metric: metric.metric))


def metric_by_name(metrics: Iterable[AnalyzableMetric]) -> dict[str, AnalyzableMetric]:
    return {metric.metric: metric for metric in metrics}


def selected_transform(label: str) -> str:
    return PAIR_TRANSFORMS[label]


def selected_interval(label: str) -> timedelta:
    return PAIR_EXPLORER_INTERVALS[label]


def selected_max_lag(label: str) -> timedelta:
    return PAIR_MAX_LAGS[label]


def pair_cadence_seconds(
    x_summary: AnalyzableMetric,
    y_summary: AnalyzableMetric,
) -> int | None:
    cadences = [
        cadence
        for cadence in (x_summary.cadence_seconds, y_summary.cadence_seconds)
        if cadence is not None
    ]
    if len(cadences) != 2:
        return None
    return max(cadences)


def max_lag_steps(max_lag: timedelta, cadence_seconds: int) -> int:
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")
    return max(0, int(max_lag.total_seconds() // cadence_seconds))


def coverage_rows(analysis: MetricPairAnalysis) -> list[dict[str, Any]]:
    return [
        _coverage_row("X", analysis.x_metric_summary, analysis.aligned_pair.x_coverage),
        _coverage_row("Y", analysis.y_metric_summary, analysis.aligned_pair.y_coverage),
        {
            "Series": "Aligned pair",
            "Metric": f"{analysis.aligned_pair.x_metric} / {analysis.aligned_pair.y_metric}",
            "Samples": len(analysis.aligned_pair.frame),
            "Coverage": "n/a",
            "Cadence": _format_duration(analysis.aligned_pair.cadence_seconds),
        },
    ]


def evidence_metrics(analysis: PairAnalysis) -> dict[str, str]:
    validation = analysis.validation_result
    lag = analysis.lag_result
    return {
        "Train rho": _format_optional_float(lag.best_rho),
        "Best lag": best_lag_label(lag.best_lag_seconds),
        "Permutation p-value": _format_optional_float(validation.permutation_p_value),
        "Holdout rho": _format_optional_float(validation.holdout_rho),
        "Stability": _format_optional_float(validation.sign_stability),
    }


def evidence_statement(analysis: PairAnalysis) -> str:
    lag = analysis.lag_result
    validation = analysis.validation_result
    if not _has_sufficient_evidence(analysis):
        return "Insufficient evidence for a stable association in this interval."
    if lag.best_lag_seconds > 0:
        return (
            f"Association: {analysis.aligned_pair.x_metric} leads "
            f"{analysis.aligned_pair.y_metric} by {_format_duration(lag.best_lag_seconds)}."
        )
    if lag.best_lag_seconds < 0:
        return (
            f"Association: {analysis.aligned_pair.x_metric} follows "
            f"{analysis.aligned_pair.y_metric} by {_format_duration(abs(lag.best_lag_seconds))}."
        )
    return (
        f"Association: {analysis.aligned_pair.x_metric} and "
        f"{analysis.aligned_pair.y_metric} align at 0 seconds."
    )


def best_lag_label(seconds: int) -> str:
    duration = _format_signed_duration(seconds)
    if seconds > 0:
        return f"{duration} (X leads Y)"
    if seconds < 0:
        return f"{duration} (X follows Y)"
    return duration


def warning_messages(analysis: MetricPairAnalysis) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*analysis.warnings, *analysis.validation_result.warnings)))


def _coverage_row(
    series_label: str,
    summary: AnalyzableMetric,
    aligned_coverage: float,
) -> dict[str, Any]:
    coverage = aligned_coverage if aligned_coverage is not None else summary.coverage
    return {
        "Series": series_label,
        "Metric": summary.metric,
        "Samples": summary.sample_count,
        "Coverage": _format_percent(coverage),
        "Cadence": _format_duration(summary.cadence_seconds) if summary.cadence_seconds else "n/a",
    }


def _has_sufficient_evidence(analysis: PairAnalysis) -> bool:
    lag = analysis.lag_result
    validation = analysis.validation_result
    if lag.best_rho is None or validation.holdout_rho is None:
        return False
    if validation.permutation_p_value is None or validation.permutation_p_value > 0.05:
        return False
    if validation.sign_stability is None or validation.sign_stability < 0.75:
        return False
    return (lag.best_rho >= 0) == (validation.holdout_rho >= 0)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _format_signed_duration(seconds: int) -> str:
    if seconds == 0:
        return "0 seconds"
    sign = "+" if seconds > 0 else "-"
    return f"{sign}{_format_duration(abs(seconds))}"


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} seconds"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:g} minutes"
    if seconds < 86_400:
        hours = seconds / 3600
        return f"{hours:g} hours"
    days = seconds / 86_400
    return f"{days:g} days"


__all__ = [
    "PAIR_EXPLORER_INTERVALS",
    "PAIR_MAX_LAGS",
    "PAIR_TRANSFORMS",
    "PairExplorerSelection",
    "best_lag_label",
    "coverage_rows",
    "evidence_metrics",
    "evidence_statement",
    "max_lag_steps",
    "metric_by_name",
    "metric_names",
    "pair_cadence_seconds",
    "selected_interval",
    "selected_max_lag",
    "selected_transform",
    "warning_messages",
]
