from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import rankdata

from resonance.analysis.contracts import LagScanResult


def lagged_spearman(frame: Any, max_lag_steps: int, min_overlap: int = 30) -> LagScanResult:
    """Scan lagged Spearman associations between ``x`` and ``y``.

    Positive lags compare earlier X observations with later Y observations.
    The result reports association only; it does not imply causation.
    """

    if max_lag_steps < 0:
        raise ValueError("max_lag_steps must be non-negative")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")

    x_values = numeric_array(_extract_column(frame, "x"))
    y_values = numeric_array(_extract_column(frame, "y"))
    if len(x_values) != len(y_values):
        raise ValueError("x and y columns must have the same length")

    step_seconds = _infer_step_seconds(frame)
    scores: list[dict[str, float | int | None]] = []
    best_row: dict[str, float | int | None] | None = None

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        rho, overlap = spearman_at_lag(
            x_values,
            y_values,
            lag_steps,
            min_overlap=min_overlap,
        )
        row: dict[str, float | int | None] = {
            "lag_steps": lag_steps,
            "lag_seconds": lag_steps * step_seconds,
            "rho": rho,
            "overlap_count": overlap,
        }
        scores.append(row)
        if rho is not None and _is_better_lag(row, best_row):
            best_row = row

    if best_row is None:
        return LagScanResult(
            scores=tuple(scores),
            best_lag_steps=0,
            best_lag_seconds=0,
            best_rho=None,
        )

    return LagScanResult(
        scores=tuple(scores),
        best_lag_steps=int(best_row["lag_steps"]),
        best_lag_seconds=int(best_row["lag_seconds"]),
        best_rho=float(best_row["rho"]),
    )


def numeric_array(values: Sequence[Any]) -> np.ndarray:
    """Coerce values to a float array, representing invalid values as NaN."""

    output = np.empty(len(values), dtype="float64")
    for index, value in enumerate(values):
        output[index] = _coerce_finite_float(value)
    return output


def spearman_at_lag(
    x_values: Sequence[Any] | np.ndarray,
    y_values: Sequence[Any] | np.ndarray,
    lag_steps: int,
    *,
    min_overlap: int,
) -> tuple[float | None, int]:
    """Return Spearman rho and finite overlap at one lag."""

    x = x_values if isinstance(x_values, np.ndarray) else numeric_array(x_values)
    y = y_values if isinstance(y_values, np.ndarray) else numeric_array(y_values)
    if len(x) != len(y):
        raise ValueError("x and y columns must have the same length")
    if abs(lag_steps) >= len(x):
        return None, 0

    if lag_steps > 0:
        left = x[:-lag_steps]
        right = y[lag_steps:]
    elif lag_steps < 0:
        offset = -lag_steps
        left = x[offset:]
        right = y[:-offset]
    else:
        left = x
        right = y

    finite = np.isfinite(left) & np.isfinite(right)
    overlap = int(finite.sum())
    if overlap < min_overlap:
        return None, overlap
    rho = spearman_rho(left[finite], right[finite])
    return rho, overlap


def spearman_rho(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float | None:
    """Calculate Spearman rank correlation without emitting constant-input warnings."""

    x = np.asarray(left, dtype="float64")
    y = np.asarray(right, dtype="float64")
    if len(x) != len(y) or len(x) < 2:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    ranked_x = rankdata(x, method="average")
    ranked_y = rankdata(y, method="average")
    correlation = float(np.corrcoef(ranked_x, ranked_y)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _extract_column(frame: Any, column: str) -> list[Any]:
    if hasattr(frame, "columns") and column in frame.columns:
        return list(frame[column])
    if isinstance(frame, Mapping) and column in frame:
        return list(frame[column])
    try:
        return [row[column] for row in frame]
    except (KeyError, TypeError) as exc:
        raise ValueError("frame must contain x and y columns") from exc


def _coerce_finite_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if math.isfinite(numeric) else np.nan


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
    deltas: list[int] = []
    previous: datetime | None = None
    for timestamp in timestamps:
        if not isinstance(timestamp, datetime):
            previous = None
            continue
        if previous is not None:
            delta_seconds = int((timestamp - previous).total_seconds())
            if delta_seconds > 0:
                deltas.append(delta_seconds)
        previous = timestamp
    if not deltas:
        return 1
    return max(1, int(np.median(np.asarray(deltas, dtype="float64"))))


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


__all__ = [
    "lagged_spearman",
    "numeric_array",
    "spearman_at_lag",
    "spearman_rho",
]
