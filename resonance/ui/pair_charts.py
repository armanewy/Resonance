from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import plotly.graph_objects as go

from resonance.analysis.contracts import PairAnalysis


def aligned_transformed_timeline(
    source: PairAnalysis,
) -> go.Figure:
    """Plot X and Y shifted onto the selected lag without filling missing values."""

    aligned_pair = source.aligned_pair
    resolved_transform = source.transform_name
    resolved_lag_steps = source.lag_result.best_lag_steps
    resolved_lag_seconds = source.lag_result.best_lag_seconds
    rows = _coerce_rows(aligned_pair.frame)
    timestamps = [row.get("timestamp_utc") for row in rows]
    x_values = [_optional_float(row.get("x")) for row in rows]
    y_values = [_lag_aligned_y(rows, index, resolved_lag_steps) for index in range(len(rows))]
    lag_label = _format_signed_duration(resolved_lag_seconds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=x_values,
            mode="lines+markers",
            name=f"{aligned_pair.x_metric} X ({resolved_transform})",
            connectgaps=False,
            hovertemplate="UTC=%{x}<br>X=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_values,
            mode="lines+markers",
            name=f"{aligned_pair.y_metric} Y lag-aligned {lag_label} ({resolved_transform})",
            connectgaps=False,
            hovertemplate="UTC=%{x}<br>Lag-aligned Y=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Aligned transformed timeline - {resolved_transform}, selected lag {lag_label}",
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Transformed value",
        legend_title_text="Series",
    )
    return fig


def lag_profile(source: PairAnalysis) -> go.Figure:
    """Plot scanned lag scores and mark the selected lag."""

    lag_result = source.lag_result
    scores = sorted(lag_result.scores, key=lambda score: _numeric(score.get("lag_seconds"), 0.0))
    x_values = [_numeric(score.get("lag_seconds"), 0.0) / 60 for score in scores]
    y_values = [_optional_float(score.get("rho")) for score in scores]
    overlaps = [_numeric(score.get("overlap_count"), 0.0) for score in scores]
    best_score = _score_for_lag(lag_result.scores, lag_result.best_lag_steps)
    best_x = lag_result.best_lag_seconds / 60
    best_y = _optional_float(best_score.get("rho")) if best_score else lag_result.best_rho
    best_overlap = _numeric(best_score.get("overlap_count"), 0.0) if best_score else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="Lag scores",
            connectgaps=False,
            customdata=overlaps,
            hovertemplate="Lag=%{x:g} min<br>Spearman rho=%{y}<br>Overlap=%{customdata:g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_x],
            y=[best_y],
            mode="markers",
            name="Selected lag",
            marker={"size": 12, "symbol": "diamond"},
            customdata=[best_overlap],
            hovertemplate="Selected lag=%{x:g} min<br>Spearman rho=%{y}<br>Overlap=%{customdata:g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero lag")
    fig.update_layout(
        title="Lag profile",
        xaxis_title="Lag (minutes)",
        yaxis_title="Spearman rho",
        legend_title_text="Profile",
    )
    return fig


def lagged_scatter(
    source: PairAnalysis,
) -> go.Figure:
    """Plot X(t) against Y(t + selected lag) with paired timestamps in hover data."""

    aligned_pair = source.aligned_pair
    resolved_transform = source.transform_name
    resolved_lag_result = source.lag_result
    resolved_lag_steps = source.lag_result.best_lag_steps
    resolved_lag_seconds = source.lag_result.best_lag_seconds
    rows = _coerce_rows(aligned_pair.frame)
    pairs = _lagged_pairs(rows, resolved_lag_steps)
    best_score = _score_for_lag(resolved_lag_result.scores, resolved_lag_steps) if resolved_lag_result else None
    rho = (
        _optional_float(best_score.get("rho"))
        if best_score
        else (resolved_lag_result.best_rho if resolved_lag_result else None)
    )
    overlap = (
        int(_numeric(best_score.get("overlap_count"), 0.0))
        if best_score
        else len(pairs)
    )
    lag_label = _format_signed_duration(resolved_lag_seconds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[pair["x"] for pair in pairs],
            y=[pair["y"] for pair in pairs],
            mode="markers",
            name="Lagged pairs",
            customdata=[(pair["x_timestamp"], pair["y_timestamp"]) for pair in pairs],
            hovertemplate=(
                "X timestamp=%{customdata[0]}<br>"
                "Y timestamp=%{customdata[1]}<br>"
                "X=%{x}<br>"
                "Y=%{y}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Lagged scatter - lag {lag_label}, Spearman rho={_format_optional_float(rho)}, overlap={overlap}",
        xaxis_title=f"{aligned_pair.x_metric} X ({resolved_transform})",
        yaxis_title=f"{aligned_pair.y_metric} Y at selected lag ({resolved_transform})",
    )
    return fig


def stability_chart(source: PairAnalysis) -> go.Figure:
    """Plot chronological window correlations for the selected lag."""

    validation_result = source.validation_result
    scores = tuple(validation_result.window_scores)
    labels = [_window_label(score, index) for index, score in enumerate(scores)]
    y_values = [_optional_float(score.get("rho")) for score in scores]
    overlaps = [_numeric(score.get("overlap_count"), 0.0) for score in scores]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=y_values,
            mode="lines+markers",
            name="Window rho",
            connectgaps=False,
            customdata=overlaps,
            hovertemplate="Window=%{x}<br>Spearman rho=%{y}<br>Overlap=%{customdata:g}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero rho")
    fig.update_layout(
        title="Stability by chronological window",
        xaxis_title="Chronological window",
        yaxis_title="Spearman rho",
    )
    return fig


def _coerce_rows(frame: Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(frame, "iterrows"):
        return tuple(
            {"timestamp_utc": timestamp, "x": row["x"], "y": row["y"]}
            for timestamp, row in frame.iterrows()
        )

    rows = []
    for index, row in enumerate(frame):
        if isinstance(row, Mapping):
            timestamp = row.get("timestamp_utc", index)
            rows.append({"timestamp_utc": timestamp, "x": row.get("x"), "y": row.get("y")})
        else:
            rows.append({"timestamp_utc": index, "x": row[0], "y": row[1]})
    return tuple(rows)


def _lag_aligned_y(rows: Sequence[Mapping[str, Any]], index: int, lag_steps: int) -> float | None:
    y_index = index + lag_steps
    if y_index < 0 or y_index >= len(rows):
        return None
    return _optional_float(rows[y_index].get("y"))


def _lagged_pairs(rows: Sequence[Mapping[str, Any]], lag_steps: int) -> tuple[dict[str, Any], ...]:
    pairs = []
    for x_index, row in enumerate(rows):
        y_index = x_index + lag_steps
        if y_index < 0 or y_index >= len(rows):
            continue
        x_value = _optional_float(row.get("x"))
        y_value = _optional_float(rows[y_index].get("y"))
        if x_value is None or y_value is None:
            continue
        pairs.append(
            {
                "x": x_value,
                "y": y_value,
                "x_timestamp": row.get("timestamp_utc"),
                "y_timestamp": rows[y_index].get("timestamp_utc"),
            }
        )
    return tuple(pairs)


def _score_for_lag(scores: Sequence[Mapping[str, Any]], lag_steps: int) -> Mapping[str, Any] | None:
    for score in scores:
        if score.get("lag_steps") == lag_steps:
            return score
    return None


def _window_label(score: Mapping[str, Any], index: int) -> str:
    start = score.get("start_utc")
    end = score.get("end_utc")
    if start is not None and end is not None:
        return f"{_format_timestamp(start)} to {_format_timestamp(end)}"
    return f"Window {score.get('window_index', index) + 1}"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return None
    return numeric


def _numeric(value: Any, default: float) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _format_signed_duration(seconds: float | int) -> str:
    seconds = float(seconds)
    sign = "+" if seconds > 0 else ""
    absolute = abs(seconds)
    if absolute < 60:
        value = f"{absolute:g}s"
    else:
        minutes = absolute / 60
        value = f"{minutes:g}m" if minutes < 60 else f"{minutes / 60:g}h"
    return f"{sign}{value}" if seconds >= 0 else f"-{value}"


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = [
    "aligned_transformed_timeline",
    "lag_profile",
    "lagged_scatter",
    "stability_chart",
]
