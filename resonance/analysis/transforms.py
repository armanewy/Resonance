from __future__ import annotations

from collections.abc import Callable
from datetime import timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


Transform = Callable[[pd.Series], pd.Series]


def raw(series: pd.Series) -> pd.Series:
    """Return numeric observations unchanged apart from UTC normalization."""
    return _validate_series(series).copy()


def first_difference(series: pd.Series) -> pd.Series:
    """Return first differences without filling the initial missing value."""
    return _validate_series(series).diff()


def rolling_robust_zscore(
    series: pd.Series,
    window: int = 24,
    min_periods: int | None = None,
) -> pd.Series:
    """Return a rolling median/MAD z-score using only current and past values."""
    if window < 2:
        raise ValueError("window must be at least 2")
    min_periods = min_periods if min_periods is not None else max(3, window // 2)
    if min_periods < 2 or min_periods > window:
        raise ValueError("min_periods must be between 2 and window")

    values = _validate_series(series)
    rolling = values.rolling(window=window, min_periods=min_periods)
    medians = rolling.median()
    mad = rolling.apply(_median_absolute_deviation, raw=True)
    scale = 1.4826 * mad
    return (values - medians) / scale.replace(0.0, np.nan)


def calendar_residual(
    series: pd.Series,
    cadence_seconds: int | None = None,
    min_history: int = 3,
) -> pd.Series:
    """Remove prior median behavior for matching UTC time-of-week slots."""
    if min_history < 1:
        raise ValueError("min_history must be at least 1")

    values = _validate_series(series)
    cadence = cadence_seconds if cadence_seconds is not None else _infer_cadence(values)
    if cadence <= 0:
        raise ValueError("cadence_seconds must be positive")

    slots = _time_of_week_slots(values.index, cadence)
    residuals = pd.Series(np.nan, index=values.index, dtype=float, name=values.name)
    history: dict[int, list[float]] = {}

    for timestamp, slot, value in zip(values.index, slots, values, strict=True):
        prior = history.setdefault(int(slot), [])
        if len(prior) >= min_history:
            residuals.loc[timestamp] = float(value) - float(np.median(prior))
        prior.append(float(value))

    if residuals.notna().sum() == 0:
        raise ValueError("insufficient history for calendar residual")
    return residuals


TRANSFORMS: dict[str, Callable[..., pd.Series]] = {
    "raw": raw,
    "first_difference": first_difference,
    "rolling_robust_zscore": rolling_robust_zscore,
    "calendar_residual": calendar_residual,
}


def apply_transform(name: str, series: pd.Series, **kwargs: object) -> pd.Series:
    try:
        transform = TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown transform: {name}") from exc
    return transform(series, **kwargs)


def _validate_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    if series.index.tz is None:
        raise ValueError("series timestamps must be timezone-aware")
    if not is_numeric_dtype(series):
        raise ValueError("series values must be numeric")

    cleaned = series.dropna().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    if cleaned.empty:
        raise ValueError("series has no numeric observations")
    return cleaned.astype(float).tz_convert(timezone.utc)


def _infer_cadence(series: pd.Series) -> int:
    if len(series) < 2:
        raise ValueError("insufficient observations to infer cadence")
    deltas = series.index.to_series().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        raise ValueError("insufficient timestamp variation to infer cadence")
    return max(1, int(round(float(deltas.median()))))


def _time_of_week_slots(index: pd.DatetimeIndex, cadence_seconds: int) -> np.ndarray:
    seconds = (
        index.dayofweek.to_numpy() * 86_400
        + index.hour.to_numpy() * 3_600
        + index.minute.to_numpy() * 60
        + index.second.to_numpy()
    )
    return seconds // cadence_seconds


def _median_absolute_deviation(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))
