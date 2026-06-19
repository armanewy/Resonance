from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from resonance.analysis.contracts import LagScanResult


def lagged_spearman(frame: Any, max_lag_steps: int, min_overlap: int = 30) -> LagScanResult:
    """Scan lagged Spearman correlations between frame columns x and y.

    Positive lags compare earlier X observations with later Y observations. The
    result reports association only; it does not imply causation.
    """
    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be non-negative")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")

    x_values = _extract_column(frame, "x")
    y_values = _extract_column(frame, "y")
    if len(x_values) != len(y_values):
        raise ValueError("x and y columns must have the same length")

    step_seconds = _infer_step_seconds(frame)
    scores: list[dict[str, float | int | None]] = []
    best_row: dict[str, float | int | None] | None = None

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        pairs = _lagged_pairs(x_values, y_values, lag_steps)
        rho = (
            _spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            if len(pairs) >= min_overlap
            else None
        )
        row: dict[str, float | int | None] = {
            "lag_steps": lag_steps,
            "lag_seconds": lag_steps * step_seconds,
            "rho": rho,
            "overlap_count": len(pairs),
        }
        scores.append(row)
        if rho is not None and _is_better_lag(row, best_row):
            best_row = row

    if best_row is None:
        return LagScanResult(scores=tuple(scores), best_lag_steps=0, best_lag_seconds=0, best_rho=None)

    return LagScanResult(
        scores=tuple(scores),
        best_lag_steps=int(best_row["lag_steps"]),
        best_lag_seconds=int(best_row["lag_seconds"]),
        best_rho=float(best_row["rho"]),
    )


def _extract_column(frame: Any, column: str) -> list[Any]:
    if hasattr(frame, "columns") and column in frame.columns:
        return list(frame[column])
    if isinstance(frame, Mapping) and column in frame:
        return list(frame[column])

    try:
        return [row[column] for row in frame]
    except (KeyError, TypeError) as exc:
        raise ValueError("frame must contain x and y columns") from exc


def _lagged_pairs(x_values: list[Any], y_values: list[Any], lag_steps: int) -> list[tuple[float, float]]:
    if lag_steps > 0:
        shifted_x = x_values[:-lag_steps]
        shifted_y = y_values[lag_steps:]
    elif lag_steps < 0:
        offset = -lag_steps
        shifted_x = x_values[offset:]
        shifted_y = y_values[:-offset]
    else:
        shifted_x = x_values
        shifted_y = y_values

    pairs = []
    for raw_x, raw_y in zip(shifted_x, shifted_y):
        x_value = _coerce_finite_float(raw_x)
        y_value = _coerce_finite_float(raw_y)
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    return pairs


def _coerce_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _spearman(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return _pearson(_ranks(x_values), _ranks(y_values))


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    index = 0
    while index < len(ordered):
        end = index + 1
        value = values[ordered[index]]
        while end < len(ordered) and values[ordered[end]] == value:
            end += 1
        average_rank = (index + 1 + end) / 2
        for ordered_index in ordered[index:end]:
            ranks[ordered_index] = average_rank
        index = end
    return ranks


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    numerator = sum(x_value * y_value for x_value, y_value in zip(x_centered, y_centered))
    x_sum_squares = sum(value * value for value in x_centered)
    y_sum_squares = sum(value * value for value in y_centered)
    denominator = math.sqrt(x_sum_squares * y_sum_squares)
    if denominator == 0:
        return None
    return numerator / denominator


def _is_better_lag(
    candidate: dict[str, float | int | None],
    current: dict[str, float | int | None] | None,
) -> bool:
    if current is None:
        return True

    candidate_rho = abs(float(candidate["rho"]))
    current_rho = abs(float(current["rho"]))
    if candidate_rho != current_rho:
        return candidate_rho > current_rho

    candidate_lag = int(candidate["lag_steps"])
    current_lag = int(current["lag_steps"])
    if abs(candidate_lag) != abs(current_lag):
        return abs(candidate_lag) < abs(current_lag)
    return candidate_lag > current_lag


def _infer_step_seconds(frame: Any) -> int:
    timestamps = _extract_timestamps(frame)
    previous: datetime | None = None
    for timestamp in timestamps:
        if not isinstance(timestamp, datetime):
            previous = None
            continue
        if previous is not None:
            delta_seconds = int((timestamp - previous).total_seconds())
            if delta_seconds > 0:
                return delta_seconds
        previous = timestamp
    return 1


def _extract_timestamps(frame: Any) -> list[Any]:
    if hasattr(frame, "columns") and "timestamp_utc" in frame.columns:
        return list(frame["timestamp_utc"])
    if isinstance(frame, Mapping) and "timestamp_utc" in frame:
        return list(frame["timestamp_utc"])
    if hasattr(frame, "index"):
        return list(frame.index)
    try:
        return [row["timestamp_utc"] for row in frame]
    except (KeyError, TypeError):
        return []
