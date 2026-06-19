from __future__ import annotations

import pandas as pd
import pytest

from resonance.analysis import (
    apply_transform,
    calendar_residual,
    first_difference,
    raw,
    rolling_robust_zscore,
)


def test_basic_transforms_preserve_utc_index_and_missing_first_difference() -> None:
    series = pd.Series(
        [1.0, 3.0, 6.0, 10.0],
        index=pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h"),
        name="metric",
    )

    assert raw(series).index.tz is not None
    differenced = first_difference(series)
    assert pd.isna(differenced.iloc[0])
    assert differenced.iloc[1:].tolist() == [2.0, 3.0, 4.0]
    assert apply_transform("raw", series).equals(raw(series))


def test_rolling_robust_zscore_uses_trailing_window() -> None:
    series = pd.Series(
        [10.0, 11.0, 12.0, 40.0],
        index=pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h"),
    )

    transformed = rolling_robust_zscore(series, window=3, min_periods=3)

    assert transformed.iloc[:2].isna().all()
    assert transformed.iloc[2] == pytest.approx(0.67449, abs=1e-5)
    assert transformed.iloc[3] > 10.0


def test_calendar_residual_removes_time_of_week_seasonality() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-05T00:00:00Z",
            "2026-01-05T01:00:00Z",
            "2026-01-12T00:00:00Z",
            "2026-01-12T01:00:00Z",
            "2026-01-19T00:00:00Z",
            "2026-01-19T01:00:00Z",
        ]
    )
    series = pd.Series([10.0, 20.0, 12.0, 18.0, 13.0, 17.0], index=index)

    residual = calendar_residual(series, cadence_seconds=3600, min_history=2)

    assert residual.iloc[:4].isna().all()
    assert residual.loc[pd.Timestamp("2026-01-19T00:00:00Z")] == pytest.approx(2.0)
    assert residual.loc[pd.Timestamp("2026-01-19T01:00:00Z")] == pytest.approx(-2.0)


def test_calendar_residual_requires_enough_history() -> None:
    series = pd.Series(
        [10.0, 12.0],
        index=pd.DatetimeIndex(["2026-01-05T00:00:00Z", "2026-01-12T00:00:00Z"]),
    )

    with pytest.raises(ValueError, match="insufficient history"):
        calendar_residual(series, cadence_seconds=3600, min_history=2)
