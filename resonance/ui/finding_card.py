from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from resonance.analysis.contracts import PairAnalysis
from resonance.storage import CorrelationFinding
from resonance.ui.pair_charts import (
    aligned_transformed_timeline,
    lag_profile,
    lagged_scatter,
    stability_chart,
)


def render_finding_card(finding: CorrelationFinding, analysis: PairAnalysis, *, streamlit: Any) -> None:
    """Render a verified association finding with its evidence charts."""

    statement = association_statement(finding)
    streamlit.subheader(f"{finding.x_metric} associated with {finding.y_metric}")
    streamlit.caption(statement)

    for message in warning_messages(finding):
        streamlit.warning(message)

    columns = streamlit.columns(8)
    for column, (label, value) in zip(columns, summary_metrics(finding).items(), strict=False):
        column.metric(label, value)

    streamlit.plotly_chart(aligned_transformed_timeline(analysis), use_container_width=True)
    streamlit.plotly_chart(lag_profile(analysis), use_container_width=True)
    streamlit.plotly_chart(lagged_scatter(analysis), use_container_width=True)
    streamlit.plotly_chart(stability_chart(analysis), use_container_width=True)

    with streamlit.expander("Evidence"):
        streamlit.dataframe(pd.DataFrame(evidence_rows(finding, analysis)), use_container_width=True, hide_index=True)


def summary_metrics(finding: CorrelationFinding) -> dict[str, str]:
    evidence = _evidence(finding)
    return {
        "Metrics": f"{finding.x_metric} / {finding.y_metric}",
        "Direction and lag": direction_and_lag(finding),
        "Discovery rho": _format_optional_float(finding.discovery_rho),
        "Holdout rho": _format_optional_float(finding.holdout_rho),
        "Corrected q": _format_optional_float(finding.corrected_q),
        "Stability": _format_optional_float(finding.stability),
        "Aligned observations": _format_int(_evidence_int(evidence, "aligned_observation_count", finding.overlap_count)),
        "Verified": _format_timestamp(finding.last_verified_utc),
    }


def evidence_rows(finding: CorrelationFinding, analysis: PairAnalysis) -> list[dict[str, str]]:
    evidence = _evidence(finding)
    return [
        {"Evidence": "Data interval", "Value": data_interval(finding, analysis)},
        {"Evidence": "Transform", "Value": finding.transform},
        {"Evidence": "Coverage", "Value": coverage_label(finding, analysis)},
        {"Evidence": "Failed validation dimensions", "Value": failed_validation_dimensions(finding, analysis)},
        {"Evidence": "Strongest supporting windows", "Value": strongest_supporting_windows(finding, analysis)},
        {"Evidence": "Weakest/counterexample windows", "Value": weakest_windows(finding, analysis)},
        {"Evidence": "First seen", "Value": _format_timestamp(finding.first_seen_utc)},
        {"Evidence": "Last verified", "Value": _format_timestamp(finding.last_verified_utc)},
        {"Evidence": "Data split", "Value": _data_split_label(evidence)},
    ]


def association_statement(finding: CorrelationFinding) -> str:
    direction = direction_sentence(finding)
    return f"{finding.x_metric} and {finding.y_metric} are associated; {direction}."


def direction_and_lag(finding: CorrelationFinding) -> str:
    return f"{direction_sentence(finding)} at {_format_signed_duration(finding.lag_seconds)}"


def direction_sentence(finding: CorrelationFinding) -> str:
    if finding.lag_seconds > 0:
        return f"{finding.x_metric} precedes {finding.y_metric} in this dataset"
    if finding.lag_seconds < 0:
        return f"{finding.y_metric} precedes {finding.x_metric} in this dataset"
    return "No lead-lag direction in this dataset"


def warning_messages(finding: CorrelationFinding) -> tuple[str, ...]:
    raw_warnings = _evidence(finding).get("warnings", ())
    if not isinstance(raw_warnings, Sequence) or isinstance(raw_warnings, str):
        return ()
    return tuple(str(warning) for warning in raw_warnings if str(warning).strip())


def data_interval(finding: CorrelationFinding, analysis: PairAnalysis) -> str:
    evidence = _evidence(finding)
    start = evidence.get("aligned_start_utc", analysis.aligned_pair.start_utc)
    end = evidence.get("aligned_end_utc", analysis.aligned_pair.end_utc)
    return f"{_format_timestamp(start)} to {_format_timestamp(end)}"


def coverage_label(finding: CorrelationFinding, analysis: PairAnalysis) -> str:
    evidence = _evidence(finding)
    x_coverage = _optional_float(evidence.get("x_coverage", analysis.aligned_pair.x_coverage))
    y_coverage = _optional_float(evidence.get("y_coverage", analysis.aligned_pair.y_coverage))
    return f"{finding.x_metric}: {_format_percent(x_coverage)}; {finding.y_metric}: {_format_percent(y_coverage)}"


def failed_validation_dimensions(finding: CorrelationFinding, analysis: PairAnalysis) -> str:
    evidence = _evidence(finding)
    explicit = evidence.get("failed_validation_dimensions")
    if isinstance(explicit, Sequence) and not isinstance(explicit, str):
        values = tuple(str(item) for item in explicit if str(item).strip())
        return ", ".join(values) if values else "None recorded"
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    failed = []
    if finding.discovery_rho is None or not math.isfinite(float(finding.discovery_rho)):
        failed.append("discovery rho")
    if finding.holdout_rho is None or not math.isfinite(float(finding.holdout_rho)):
        failed.append("holdout rho")
    if finding.corrected_q is None or not math.isfinite(float(finding.corrected_q)):
        failed.append("corrected q")
    if finding.stability is None or not math.isfinite(float(finding.stability)):
        failed.append("stability")
    if analysis.validation_result.warnings:
        failed.extend(str(warning) for warning in analysis.validation_result.warnings)
    return ", ".join(dict.fromkeys(failed)) if failed else "None recorded"


def strongest_supporting_windows(finding: CorrelationFinding, analysis: PairAnalysis, *, limit: int = 3) -> str:
    reference_sign = _sign(finding.holdout_rho) or _sign(finding.discovery_rho)
    supporting = [
        (score, index)
        for index, score in enumerate(_window_scores(finding, analysis))
        if _sign(_optional_float(score.get("rho"))) == reference_sign
    ]
    supporting.sort(key=lambda item: abs(_optional_float(item[0].get("rho")) or 0.0), reverse=True)
    return _format_window_list(supporting[:limit]) if supporting else "None available"


def weakest_windows(finding: CorrelationFinding, analysis: PairAnalysis, *, limit: int = 3) -> str:
    reference_sign = _sign(finding.holdout_rho) or _sign(finding.discovery_rho)
    windows = tuple(enumerate(_window_scores(finding, analysis)))
    counterexamples = [
        (score, index)
        for index, score in windows
        if _optional_float(score.get("rho")) is not None
        and reference_sign is not None
        and _sign(_optional_float(score.get("rho"))) not in (None, reference_sign)
    ]
    if counterexamples:
        counterexamples.sort(key=lambda item: abs(_optional_float(item[0].get("rho")) or 0.0), reverse=True)
        return f"relationship weakened: {_format_window_list(counterexamples[:limit])}"

    weakest = [
        (score, index)
        for index, score in windows
        if _optional_float(score.get("rho")) is not None
    ]
    weakest.sort(key=lambda item: abs(_optional_float(item[0].get("rho")) or 0.0))
    return f"relationship weakened: {_format_window_list(weakest[:limit])}" if weakest else "None available"


def _window_scores(finding: CorrelationFinding, analysis: PairAnalysis) -> tuple[Mapping[str, Any], ...]:
    evidence_scores = _evidence(finding).get("window_scores")
    if isinstance(evidence_scores, Sequence) and not isinstance(evidence_scores, str):
        return tuple(score for score in evidence_scores if isinstance(score, Mapping))
    return tuple(analysis.validation_result.window_scores)


def _format_window_list(windows: Sequence[tuple[Mapping[str, Any], int]]) -> str:
    labels = []
    for score, index in windows:
        labels.append(
            f"{_window_label(score, index)} rho={_format_optional_float(_optional_float(score.get('rho')))} "
            f"n={_format_int(_evidence_int(score, 'overlap_count', 0))}"
        )
    return "; ".join(labels)


def _window_label(score: Mapping[str, Any], index: int) -> str:
    start = score.get("start_utc")
    end = score.get("end_utc")
    if start is not None and end is not None:
        return f"{_format_timestamp(start)} to {_format_timestamp(end)}"
    return f"Window {_evidence_int(score, 'window_index', index) + 1}"


def _data_split_label(evidence: Mapping[str, Any]) -> str:
    selected_on = evidence.get("selected_on")
    validated_on = evidence.get("validated_on")
    if selected_on and validated_on:
        return f"Selected on {selected_on}; validated on {validated_on}"
    return "Not recorded"


def _evidence(finding: CorrelationFinding) -> Mapping[str, Any]:
    return finding.evidence if isinstance(finding.evidence, Mapping) else {}


def _evidence_int(source: Mapping[str, Any] | CorrelationFinding, key: str, default: int) -> int:
    value = getattr(source, key, None) if isinstance(source, CorrelationFinding) else source.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sign(value: float | int | None) -> int | None:
    numeric = _optional_float(value)
    if numeric is None or numeric == 0:
        return None
    return 1 if numeric > 0 else -1


def _format_optional_float(value: float | int | None) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4g}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_signed_duration(seconds: int) -> str:
    if seconds == 0:
        return "0 seconds"
    sign = "+" if seconds > 0 else "-"
    absolute = abs(seconds)
    if absolute < 60:
        value = f"{absolute} seconds"
    elif absolute < 3600:
        value = f"{absolute / 60:g} minutes"
    elif absolute < 86_400:
        value = f"{absolute / 3600:g} hours"
    else:
        value = f"{absolute / 86_400:g} days"
    return f"{sign}{value}"


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = [
    "association_statement",
    "coverage_label",
    "data_interval",
    "direction_and_lag",
    "direction_sentence",
    "evidence_rows",
    "failed_validation_dimensions",
    "render_finding_card",
    "strongest_supporting_windows",
    "summary_metrics",
    "warning_messages",
    "weakest_windows",
]
