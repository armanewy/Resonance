from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from resonance.analysis import align_series


def test_align_series_infers_coarser_cadence_and_aggregates_means() -> None:
    x = pd.Series(
        range(12),
        index=pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="5min"),
        name="cpu_percent",
    )
    y = pd.Series(
        [100.0, 101.0, 103.0, 106.0],
        index=pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min"),
        name="temperature_2m",
    )

    aligned = align_series(x, y, min_points=4)

    assert aligned.cadence_seconds == 900
    assert aligned.x_metric == "cpu_percent"
    assert aligned.y_metric == "temperature_2m"
    assert aligned.frame["x"].tolist() == [1.0, 4.0, 7.0, 10.0]
    assert aligned.frame["y"].tolist() == [100.0, 101.0, 103.0, 106.0]
    assert aligned.x_coverage == 1.0
    assert aligned.y_coverage == 1.0


def test_align_series_keeps_missing_bins_and_never_forward_fills() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="5min")
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    y = pd.Series([10.0, 30.0, 50.0], index=index[[0, 2, 4]])

    aligned = align_series(x, y, cadence_seconds=300, min_points=3)

    assert aligned.frame.index.tolist() == index[[0, 2, 4]].tolist()
    assert aligned.frame["y"].tolist() == [10.0, 30.0, 50.0]
    assert aligned.x_coverage == 1.0
    assert aligned.y_coverage == 0.6


def test_align_series_converts_timezones_to_utc() -> None:
    eastern_index = pd.date_range(
        "2026-01-01 00:00",
        periods=3,
        freq="1h",
        tz="America/New_York",
    )
    utc_index = eastern_index.tz_convert("UTC")
    x = pd.Series([1.0, 2.0, 3.0], index=eastern_index)
    y = pd.Series([2.0, 3.0, 4.0], index=utc_index)

    aligned = align_series(x, y, cadence_seconds=3600, min_points=3)

    assert aligned.start_utc.tzinfo is timezone.utc
    assert aligned.end_utc.tzinfo is timezone.utc
    assert str(aligned.frame.index.tz) == "UTC"
    assert aligned.frame.index[0] == pd.Timestamp("2026-01-01T05:00:00Z")


def test_align_series_rejects_naive_timestamps() -> None:
    x = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2026-01-01 00:00", periods=3, freq="1h"),
    )
    y = pd.Series(
        [1.0, 3.0, 5.0],
        index=pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h"),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        align_series(x, y, cadence_seconds=3600, min_points=3)


def test_align_series_rejects_constant_aligned_data() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h")
    x = pd.Series([1.0, 1.0, 1.0], index=index)
    y = pd.Series([2.0, 3.0, 4.0], index=index)

    with pytest.raises(ValueError, match="constant"):
        align_series(x, y, cadence_seconds=3600, min_points=3)
