from __future__ import annotations

from resonance.analysis import (
    chronological_holdout_validation,
    max_lag_block_permutation_test,
    window_stability,
)
from resonance.synthetic import generate_synthetic_series


def test_strong_lag_validates_on_holdout_permutation_and_windows() -> None:
    dataset = generate_synthetic_series(
        "strong_lag",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.25,
        seed=11,
    )
    frame = _frame(dataset)

    holdout = chronological_holdout_validation(
        frame,
        candidate_lag_steps=range(0, 7),
        min_overlap=12,
    )
    p_value = max_lag_block_permutation_test(
        frame,
        candidate_lag_steps=range(0, 7),
        permutations=99,
        block_size=12,
        min_overlap=12,
        seed=7,
    )
    stability = window_stability(frame, lag_steps=3, window_count=4, min_overlap=12)

    assert holdout.holdout_rho is not None
    assert holdout.holdout_rho > 0.98
    assert holdout.holdout_overlap > 100
    assert p_value is not None
    assert p_value <= 0.05
    assert stability.sign_stability == 1.0
    assert all(score["rho"] is not None and score["rho"] > 0.98 for score in stability.window_scores)


def test_shared_seasonality_only_is_not_promoted_as_residual_relationship() -> None:
    dataset = generate_synthetic_series(
        "shared_seasonality_only",
        sample_interval_seconds=1800,
        duration_hours=168,
        noise=0.8,
        seed=42,
    )
    frame = _residual_time_of_day_frame(dataset)

    holdout = chronological_holdout_validation(
        frame,
        candidate_lag_steps=range(-6, 7),
        min_overlap=12,
    )
    p_value = max_lag_block_permutation_test(
        frame,
        candidate_lag_steps=range(-6, 7),
        permutations=99,
        block_size=12,
        min_overlap=12,
        seed=7,
    )

    assert holdout.holdout_rho is not None
    assert abs(holdout.holdout_rho) < 0.25
    assert p_value is not None
    assert p_value > 0.05


def test_single_shared_outlier_is_window_unstable() -> None:
    dataset = generate_synthetic_series(
        "single_shared_outlier",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.5,
        seed=21,
    )
    frame = _frame(dataset)

    stability = window_stability(frame, lag_steps=0, window_count=4, min_overlap=12)

    assert stability.sign_stability is not None
    assert stability.sign_stability <= 0.5
    # Spearman correlation is intentionally robust to the single extreme
    # point, so no window should look convincingly associated.
    assert max(
        abs(score["rho"])
        for score in stability.window_scores
        if score["rho"] is not None
    ) < 0.3
    assert min(abs(score["rho"]) for score in stability.window_scores if score["rho"] is not None) < 0.05


def test_relationship_break_fails_chronological_holdout() -> None:
    dataset = generate_synthetic_series(
        "relationship_break",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.25,
        seed=34,
    )
    frame = _frame(dataset)

    holdout = chronological_holdout_validation(
        frame,
        candidate_lag_steps=range(0, 7),
        min_overlap=12,
    )

    assert holdout.holdout_rho is not None
    assert abs(holdout.holdout_rho) < 0.25
    assert holdout.holdout_overlap > 100


def test_independent_autocorrelated_is_not_promoted_by_block_permutation() -> None:
    dataset = generate_synthetic_series(
        "independent_autocorrelated",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.7,
        seed=55,
    )
    frame = _frame(dataset)

    p_value = max_lag_block_permutation_test(
        frame,
        candidate_lag_steps=range(-6, 7),
        permutations=99,
        block_size=12,
        min_overlap=12,
        seed=7,
    )

    assert p_value is not None
    assert p_value > 0.05


def _frame(dataset):
    return [
        {"timestamp_utc": sample.timestamp_utc, "x": sample.x, "y": sample.y}
        for sample in dataset.samples
    ]


def _residual_time_of_day_frame(dataset):
    sample_interval_seconds = dataset.metadata["sample_interval_seconds"]
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]
    residual_x = _remove_time_of_day_means(x_values, sample_interval_seconds)
    residual_y = _remove_time_of_day_means(y_values, sample_interval_seconds)
    return [
        {"timestamp_utc": sample.timestamp_utc, "x": x, "y": y}
        for sample, x, y in zip(dataset.samples, residual_x, residual_y)
    ]


def _remove_time_of_day_means(values, sample_interval_seconds: int) -> list[float]:
    groups: dict[int, list[float]] = {}
    for index, value in enumerate(values):
        groups.setdefault((index * sample_interval_seconds) % 86_400, []).append(value)

    means = {offset: sum(group) / len(group) for offset, group in groups.items()}
    return [value - means[(index * sample_interval_seconds) % 86_400] for index, value in enumerate(values)]
