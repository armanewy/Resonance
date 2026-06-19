from __future__ import annotations

import random
import warnings
from datetime import datetime, timedelta, timezone

from resonance.analysis.correlation import lagged_spearman
from resonance.synthetic import generate_synthetic_series


def test_lagged_spearman_recovers_positive_known_lag() -> None:
    lag_steps = 4
    frame = _lagged_frame(lag_steps=lag_steps)

    result = lagged_spearman(frame, max_lag_steps=8, min_overlap=40)

    assert result.best_lag_steps == lag_steps
    assert result.best_rho is not None
    assert result.best_rho > 0.99


def test_lagged_spearman_recovers_negative_known_lag() -> None:
    lag_steps = -5
    frame = _lagged_frame(lag_steps=lag_steps)

    result = lagged_spearman(frame, max_lag_steps=8, min_overlap=40)

    assert result.best_lag_steps == lag_steps
    assert result.best_rho is not None
    assert result.best_rho > 0.99


def test_lagged_spearman_recovers_approximate_strong_lag_synthetic_data() -> None:
    dataset = generate_synthetic_series(
        "strong_lag",
        sample_interval_seconds=300,
        duration_hours=24,
        noise=0.15,
        seed=11,
    )
    true_lag_steps = dataset.samples[0].true_lag_seconds // dataset.metadata["sample_interval_seconds"]
    frame = {
        "timestamp_utc": [sample.timestamp_utc for sample in dataset.samples],
        "x": [sample.x for sample in dataset.samples],
        "y": [sample.y for sample in dataset.samples],
    }

    result = lagged_spearman(frame, max_lag_steps=8, min_overlap=30)

    assert abs(result.best_lag_steps - true_lag_steps) <= 1
    assert result.best_lag_seconds == result.best_lag_steps * dataset.metadata["sample_interval_seconds"]
    assert result.best_rho is not None
    assert result.best_rho > 0.9


def test_lagged_spearman_drops_missing_values_pairwise_per_lag() -> None:
    frame = _lagged_frame(lag_steps=3, count=90)
    frame["x"][10] = None
    frame["x"][11] = float("nan")
    frame["y"][33] = None
    frame["y"][34] = float("inf")

    result = lagged_spearman(frame, max_lag_steps=6, min_overlap=30)
    best_score = _score_for_lag(result.scores, 3)

    assert result.best_lag_steps == 3
    assert best_score["overlap_count"] == 83
    assert best_score["rho"] is not None


def test_lagged_spearman_rejects_constant_series_without_warning() -> None:
    frame = {
        "x": [1.0] * 50,
        "y": [float(index) for index in range(50)],
    }

    result = lagged_spearman(frame, max_lag_steps=3, min_overlap=30)

    assert result.best_lag_steps == 0
    assert result.best_lag_seconds == 0
    assert result.best_rho is None
    assert all(score["rho"] is None for score in result.scores)


def test_lagged_spearman_rejects_lags_with_insufficient_overlap() -> None:
    frame = _lagged_frame(lag_steps=2, count=12)

    result = lagged_spearman(frame, max_lag_steps=5, min_overlap=20)

    assert result.best_rho is None
    assert all(score["rho"] is None for score in result.scores)
    assert [_score_for_lag(result.scores, lag)["overlap_count"] for lag in (-5, 0, 5)] == [7, 12, 7]


def test_lagged_spearman_handles_empty_and_pathological_input_without_warnings() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        empty_result = lagged_spearman({"x": [], "y": []}, max_lag_steps=2, min_overlap=3)
        pathological_result = lagged_spearman(
            {"x": ["bad", None, float("nan")], "y": [1.0, float("inf"), "also bad"]},
            max_lag_steps=1,
            min_overlap=2,
        )

    assert captured == []
    assert empty_result.best_rho is None
    assert pathological_result.best_rho is None
    assert len(empty_result.scores) == 5
    assert len(pathological_result.scores) == 3


def test_lagged_spearman_positive_lag_means_x_precedes_y() -> None:
    frame = _lagged_frame(lag_steps=2, count=120)

    result = lagged_spearman(frame, max_lag_steps=2, min_overlap=40)
    positive_lag_score = _score_for_lag(result.scores, 2)
    negative_lag_score = _score_for_lag(result.scores, -2)

    assert result.best_lag_steps == 2
    assert positive_lag_score["rho"] is not None
    assert negative_lag_score["rho"] is not None
    assert positive_lag_score["rho"] > 0.99
    assert abs(positive_lag_score["rho"]) > abs(negative_lag_score["rho"])


def test_lagged_spearman_prefers_positive_lag_when_equal_scores_have_same_distance() -> None:
    frame = {
        "x": [0.0, 1.0, 0.0, -1.0] * 20,
        "y": [1.0, 0.0, -1.0, 0.0] * 20,
    }

    result = lagged_spearman(frame, max_lag_steps=1, min_overlap=30)

    assert result.best_lag_steps == 1
    assert abs(_score_for_lag(result.scores, 1)["rho"]) == abs(
        _score_for_lag(result.scores, -1)["rho"]
    )


def _lagged_frame(*, lag_steps: int, count: int = 120) -> dict[str, list[float | datetime | None]]:
    rng = random.Random(20260619)
    driver = [rng.gauss(0, 1) for _ in range(count)]
    x_values = [value + index * 0.0001 for index, value in enumerate(driver)]
    y_values = [rng.gauss(0, 1) for _ in range(count)]

    if lag_steps > 0:
        for index in range(lag_steps, count):
            y_values[index] = 10 * x_values[index - lag_steps] + rng.gauss(0, 0.001)
    elif lag_steps < 0:
        offset = -lag_steps
        for index in range(offset, count):
            x_values[index] = 10 * y_values[index - offset] + rng.gauss(0, 0.001)
    else:
        y_values = [10 * value + rng.gauss(0, 0.001) for value in x_values]

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "timestamp_utc": [start + timedelta(minutes=5 * index) for index in range(count)],
        "x": x_values,
        "y": y_values,
    }


def _score_for_lag(scores, lag_steps: int):
    for score in scores:
        if score["lag_steps"] == lag_steps:
            return score
    raise AssertionError(f"missing score for lag {lag_steps}")
