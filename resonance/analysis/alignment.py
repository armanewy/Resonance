from __future__ import annotations

from datetime import timezone

import pandas as pd
from pandas.api.types import is_numeric_dtype

from resonance.analysis.contracts import AlignedPair


def align_series(
    x: pd.Series,
    y: pd.Series,
    cadence_seconds: int | None = None,
    min_points: int = 30,
) -> AlignedPair:
    """Align two timestamped numeric series to a shared UTC cadence."""
    if min_points < 2:
        raise ValueError("min_points must be at least 2")

    x_utc = _validate_series(x, "x")
    y_utc = _validate_series(y, "y")
    cadence = cadence_seconds if cadence_seconds is not None else _infer_cadence(x_utc, y_utc)
    if cadence <= 0:
        raise ValueError("cadence_seconds must be positive")

    x_binned = _bin_mean(x_utc, cadence)
    y_binned = _bin_mean(y_utc, cadence)
    x_window, y_window = _shared_expected_window(x_binned, y_binned, cadence)

    expected_count = len(x_window)
    if expected_count == 0:
        raise ValueError("insufficient overlapping bins for alignment")

    x_coverage = float(x_window.notna().sum() / expected_count)
    y_coverage = float(y_window.notna().sum() / expected_count)

    frame = pd.concat((x_window.rename("x"), y_window.rename("y")), axis=1)
    frame = frame.dropna(how="any")
    if len(frame) < min_points:
        raise ValueError(
            f"insufficient aligned observations: got {len(frame)}, need {min_points}"
        )
    _reject_constant(frame["x"], "x")
    _reject_constant(frame["y"], "y")

    return AlignedPair(
        x_metric=str(x.name or "x"),
        y_metric=str(y.name or "y"),
        cadence_seconds=int(cadence),
        frame=frame,
        x_coverage=x_coverage,
        y_coverage=y_coverage,
        start_utc=frame.index[0].to_pydatetime(),
        end_utc=frame.index[-1].to_pydatetime(),
    )


def _validate_series(series: pd.Series, label: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must have a DatetimeIndex")
    if series.index.tz is None:
        raise ValueError(f"{label} timestamps must be timezone-aware")
    if not is_numeric_dtype(series):
        raise ValueError(f"{label} values must be numeric")

    cleaned = series.dropna().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    if cleaned.empty:
        raise ValueError(f"{label} has no numeric observations")
    return cleaned.astype(float).tz_convert(timezone.utc)


def _infer_cadence(x: pd.Series, y: pd.Series) -> int:
    x_cadence = _median_cadence_seconds(x, "x")
    y_cadence = _median_cadence_seconds(y, "y")
    return max(x_cadence, y_cadence)


def _median_cadence_seconds(series: pd.Series, label: str) -> int:
    if len(series) < 2:
        raise ValueError(f"insufficient observations to infer {label} cadence")
    deltas = series.index.to_series().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        raise ValueError(f"insufficient timestamp variation to infer {label} cadence")
    return max(1, int(round(float(deltas.median()))))


def _bin_mean(series: pd.Series, cadence_seconds: int) -> pd.Series:
    rule = pd.to_timedelta(cadence_seconds, unit="s")
    return series.resample(rule, origin="epoch").mean()


def _shared_expected_window(
    x_binned: pd.Series,
    y_binned: pd.Series,
    cadence_seconds: int,
) -> tuple[pd.Series, pd.Series]:
    if x_binned.empty or y_binned.empty:
        raise ValueError("insufficient observations for alignment")

    start = max(x_binned.index.min(), y_binned.index.min())
    end = min(x_binned.index.max(), y_binned.index.max())
    if start > end:
        raise ValueError("series do not overlap in time")

    expected_index = pd.date_range(
        start=start,
        end=end,
        freq=pd.to_timedelta(cadence_seconds, unit="s"),
        tz=timezone.utc,
    )
    return x_binned.reindex(expected_index), y_binned.reindex(expected_index)


def _reject_constant(series: pd.Series, label: str) -> None:
    if series.nunique(dropna=True) < 2:
        raise ValueError(f"{label} aligned data is constant")
