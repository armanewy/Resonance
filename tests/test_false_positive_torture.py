from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from resonance.analysis.correlation import lagged_spearman
from resonance.analysis.validation import (
    chronological_holdout_validation,
    max_lag_block_permutation_test,
    window_stability,
)


START_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROMOTION_RHO_FLOOR = 0.35
PROMOTION_P_VALUE = 0.05


@dataclass(frozen=True)
class Evidence:
    best_lag_steps: int
    best_rho: float | None
    holdout_rho: float | None
    holdout_overlap: int
    permutation_p_value: float | None
    sign_stability: float | None
    window_scores: tuple[dict[str, Any], ...]
    duplicate_or_derived: bool = False


def test_shared_daily_seasonality_with_independent_residuals_is_not_promotable() -> None:
    step_seconds = 1_800
    raw_frame = _shared_daily_seasonality(step_seconds=step_seconds)
    residual_frame = _period_residual_frame(raw_frame, period_seconds=86_400, step_seconds=step_seconds)

    raw = _evidence(raw_frame, max_lag_steps=12, min_overlap=24, block_size=24)
    residual = _evidence(residual_frame, max_lag_steps=12, min_overlap=24, block_size=24)

    assert raw.best_lag_steps == 0
    assert raw.best_rho is not None and raw.best_rho > 0.9
    assert raw.holdout_rho is not None and raw.holdout_rho > 0.9
    assert raw.permutation_p_value is not None and raw.permutation_p_value <= PROMOTION_P_VALUE
    assert residual.best_rho is not None and abs(residual.best_rho) < 0.25
    assert residual.holdout_rho is not None and abs(residual.holdout_rho) < 0.25
    assert residual.permutation_p_value is not None and 0.1 < residual.permutation_p_value < 0.8
    assert not _may_promote(residual)


def test_shared_weekly_seasonality_is_not_promotable_after_weekly_residual_check() -> None:
    step_seconds = 3_600
    raw_frame = _shared_weekly_seasonality(step_seconds=step_seconds)
    residual_frame = _period_residual_frame(
        raw_frame,
        period_seconds=7 * 86_400,
        step_seconds=step_seconds,
    )

    raw = _evidence(raw_frame, max_lag_steps=72, min_overlap=24, block_size=24)
    residual = _evidence(residual_frame, max_lag_steps=72, min_overlap=24, block_size=24)

    assert raw.best_lag_steps == 0
    assert raw.best_rho is not None and raw.best_rho > 0.9
    assert raw.holdout_rho is not None and raw.holdout_rho > 0.9
    assert raw.permutation_p_value is not None and raw.permutation_p_value <= PROMOTION_P_VALUE
    assert residual.best_rho is not None and abs(residual.best_rho) < 0.2
    assert residual.holdout_rho is not None and abs(residual.holdout_rho) < 0.2
    assert residual.permutation_p_value is not None and 0.1 < residual.permutation_p_value < 0.8
    assert not _may_promote(residual)


def test_one_enormous_simultaneous_outlier_is_not_promotable() -> None:
    frame = _one_shared_outlier()
    evidence = _evidence(frame, max_lag_steps=12, min_overlap=30, block_size=12)

    assert _pearson(_column(frame, "x"), _column(frame, "y")) > 0.9
    assert evidence.best_rho is not None and abs(evidence.best_rho) < 0.2
    assert evidence.holdout_rho is not None and abs(evidence.holdout_rho) < 0.25
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value > PROMOTION_P_VALUE
    assert not _may_promote(evidence)


def test_two_independent_random_walks_are_not_promotable_even_when_one_test_is_tempted() -> None:
    frame = _independent_random_walks()
    evidence = _evidence(frame, max_lag_steps=36, min_overlap=30, block_size=24)

    assert evidence.best_rho is not None and abs(evidence.best_rho) > 0.55
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value <= PROMOTION_P_VALUE
    assert evidence.holdout_rho is not None and abs(evidence.holdout_rho) < 0.25
    assert evidence.sign_stability is not None and evidence.sign_stability < 0.75
    assert not _may_promote(evidence)


def test_two_independent_autocorrelated_series_are_not_promotable() -> None:
    frame = _independent_autocorrelated_series()
    evidence = _evidence(frame, max_lag_steps=36, min_overlap=30, block_size=24)

    assert _lag_one_autocorrelation(_column(frame, "x")) > 0.85
    assert _lag_one_autocorrelation(_column(frame, "y")) > 0.8
    assert evidence.best_rho is not None and abs(evidence.best_rho) < 0.35
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value > 0.2
    assert not _may_promote(evidence)


def test_relationship_present_only_in_discovery_period_is_not_promotable() -> None:
    frame = _relationship_only_in_discovery()
    evidence = _evidence(frame, max_lag_steps=6, min_overlap=30, block_size=12)

    assert evidence.best_lag_steps == 0
    assert evidence.best_rho is not None and evidence.best_rho > 0.9
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value <= PROMOTION_P_VALUE
    assert evidence.holdout_rho is not None and abs(evidence.holdout_rho) < 0.25
    assert not _may_promote(evidence)


def test_sign_reversal_in_holdout_is_not_promotable() -> None:
    frame = _sign_reversal_in_holdout()
    evidence = _evidence(frame, max_lag_steps=6, min_overlap=30, block_size=12)

    assert evidence.best_lag_steps == 0
    assert evidence.best_rho is not None and evidence.best_rho > 0.9
    assert evidence.holdout_rho is not None and evidence.holdout_rho < -0.9
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value <= PROMOTION_P_VALUE
    assert evidence.sign_stability is not None and evidence.sign_stability >= 0.75
    assert not _may_promote(evidence)


def test_heavy_missing_independent_data_is_not_promotable() -> None:
    frame = _heavy_missing_independent_data()
    evidence = _evidence(frame, max_lag_steps=8, min_overlap=20, block_size=10)

    observed_x = sum(row["x"] is not None for row in frame)
    observed_y = sum(row["y"] is not None for row in frame)
    assert observed_x / len(frame) < 0.35
    assert observed_y / len(frame) < 0.35
    assert evidence.best_rho is not None and abs(evidence.best_rho) < 0.25
    assert evidence.holdout_rho is None
    assert evidence.holdout_overlap < 20
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value > 0.5
    assert not _may_promote(evidence)


def test_duplicate_or_derived_series_are_not_promotable_despite_strong_statistics() -> None:
    evidence = _evidence(_duplicate_derived_series(), max_lag_steps=6, min_overlap=30, block_size=12)
    evidence = replace(evidence, duplicate_or_derived=True)

    assert evidence.best_lag_steps == 0
    assert evidence.best_rho is not None and evidence.best_rho > 0.99
    assert evidence.holdout_rho is not None and evidence.holdout_rho > 0.99
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value <= PROMOTION_P_VALUE
    assert evidence.sign_stability == 1.0
    assert not _may_promote(evidence)


def test_best_lag_inflation_from_searching_many_lags_is_not_promotable() -> None:
    frame = _independent_noise_for_many_lag_search()
    evidence = _evidence(frame, max_lag_steps=80, min_overlap=25, block_size=10)

    assert abs(evidence.best_lag_steps) > 60
    assert evidence.best_rho is not None and 0.25 < abs(evidence.best_rho) < 0.45
    assert evidence.holdout_rho is None
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value > PROMOTION_P_VALUE
    assert evidence.sign_stability is None
    assert not _may_promote(evidence)


def test_real_strong_lag_hidden_beneath_seasonality_is_promotable_after_residual_check() -> None:
    step_seconds = 900
    raw_frame = _strong_lag_hidden_beneath_daily_seasonality(step_seconds=step_seconds)
    residual_frame = _period_residual_frame(raw_frame, period_seconds=86_400, step_seconds=step_seconds)

    raw = _evidence(raw_frame, max_lag_steps=12, min_overlap=30, block_size=24)
    residual = _evidence(residual_frame, max_lag_steps=12, min_overlap=30, block_size=24)

    assert raw.best_lag_steps == 0
    assert raw.best_rho is not None and raw.best_rho > 0.9
    assert residual.best_lag_steps == 4
    assert residual.best_rho is not None and 0.45 < residual.best_rho < 0.75
    assert residual.holdout_rho is not None and 0.45 < residual.holdout_rho < 0.75
    assert residual.permutation_p_value is not None and residual.permutation_p_value <= PROMOTION_P_VALUE
    assert residual.sign_stability == 1.0
    assert _may_promote(residual)


def test_real_moderate_relationship_repeated_across_episodes_is_promotable() -> None:
    frame = _moderate_repeated_episodes()
    evidence = _evidence(frame, max_lag_steps=6, min_overlap=30, block_size=12)

    assert evidence.best_lag_steps == 2
    assert evidence.best_rho is not None and 0.45 < evidence.best_rho < 0.75
    assert evidence.holdout_rho is not None and 0.4 < evidence.holdout_rho < 0.7
    assert evidence.permutation_p_value is not None and evidence.permutation_p_value <= PROMOTION_P_VALUE
    assert evidence.sign_stability == 1.0
    assert _may_promote(evidence)


def _evidence(
    frame: list[dict[str, Any]],
    *,
    max_lag_steps: int,
    min_overlap: int,
    block_size: int,
) -> Evidence:
    discovery_frame = frame[: round(len(frame) * 0.75)]
    lag = lagged_spearman(_mapping(discovery_frame), max_lag_steps=max_lag_steps, min_overlap=min_overlap)
    holdout = chronological_holdout_validation(
        frame,
        candidate_lag_steps=(lag.best_lag_steps,),
        holdout_fraction=0.25,
        min_overlap=min_overlap,
    )
    p_value = max_lag_block_permutation_test(
        frame,
        candidate_lag_steps=range(-max_lag_steps, max_lag_steps + 1),
        permutations=99,
        block_size=block_size,
        min_overlap=min_overlap,
        seed=7,
    )
    stability = window_stability(
        frame,
        lag_steps=lag.best_lag_steps,
        window_count=4,
        min_overlap=min_overlap,
    )

    return Evidence(
        best_lag_steps=lag.best_lag_steps,
        best_rho=lag.best_rho,
        holdout_rho=holdout.holdout_rho,
        holdout_overlap=holdout.holdout_overlap,
        permutation_p_value=p_value,
        sign_stability=stability.sign_stability,
        window_scores=stability.window_scores,
    )


def _may_promote(evidence: Evidence) -> bool:
    if evidence.duplicate_or_derived:
        return False
    if evidence.best_rho is None or evidence.holdout_rho is None:
        return False
    if abs(evidence.best_rho) < PROMOTION_RHO_FLOOR:
        return False
    if abs(evidence.holdout_rho) < PROMOTION_RHO_FLOOR:
        return False
    if evidence.permutation_p_value is None or evidence.permutation_p_value > PROMOTION_P_VALUE:
        return False
    if evidence.sign_stability is None or evidence.sign_stability < 0.75:
        return False
    return (evidence.best_rho >= 0) == (evidence.holdout_rho >= 0)


def _shared_daily_seasonality(*, step_seconds: int) -> list[dict[str, Any]]:
    rng = random.Random(1)
    sample_count = 14 * 24 * 2 + 1
    seasonal = _seasonal_cycle(sample_count, step_seconds, period_seconds=86_400, amplitude=10.0)
    x_values = [value + rng.gauss(0, 1.0) for value in seasonal]
    y_values = [0.9 * value + rng.gauss(0, 1.0) for value in seasonal]
    return _frame(x_values, y_values, step_seconds=step_seconds)


def _shared_weekly_seasonality(*, step_seconds: int) -> list[dict[str, Any]]:
    rng = random.Random(2)
    sample_count = 8 * 7 * 24 + 1
    seasonal = _seasonal_cycle(sample_count, step_seconds, period_seconds=7 * 86_400, amplitude=10.0)
    x_values = [value + rng.gauss(0, 1.0) for value in seasonal]
    y_values = [1.1 * value + rng.gauss(0, 1.0) for value in seasonal]
    return _frame(x_values, y_values, step_seconds=step_seconds)


def _one_shared_outlier() -> list[dict[str, Any]]:
    rng = random.Random(3)
    sample_count = 400
    x_values = [rng.gauss(0, 1.0) for _ in range(sample_count)]
    y_values = [rng.gauss(0, 1.0) for _ in range(sample_count)]
    event_index = sample_count // 2
    x_values[event_index] += 100.0
    y_values[event_index] += 100.0
    return _frame(x_values, y_values)


def _independent_random_walks() -> list[dict[str, Any]]:
    return _frame(
        _random_walk(400, random.Random(4)),
        _random_walk(400, random.Random(5)),
    )


def _independent_autocorrelated_series() -> list[dict[str, Any]]:
    return _frame(
        _ar1_values(400, random.Random(6), phi=0.95),
        _ar1_values(400, random.Random(7), phi=0.93),
    )


def _relationship_only_in_discovery() -> list[dict[str, Any]]:
    rng = random.Random(8)
    independent_rng = random.Random(9)
    sample_count = 400
    split_index = round(sample_count * 0.75)
    x_values = _ar1_values(sample_count, rng, phi=0.4)
    y_values = [
        1.2 * x_values[index] + rng.gauss(0, 0.2)
        if index < split_index
        else independent_rng.gauss(0, 1.0)
        for index in range(sample_count)
    ]
    return _frame(x_values, y_values)


def _sign_reversal_in_holdout() -> list[dict[str, Any]]:
    rng = random.Random(10)
    sample_count = 400
    split_index = round(sample_count * 0.75)
    x_values = _ar1_values(sample_count, rng, phi=0.4)
    y_values = [
        (1.2 if index < split_index else -1.2) * x_values[index] + rng.gauss(0, 0.2)
        for index in range(sample_count)
    ]
    return _frame(x_values, y_values)


def _heavy_missing_independent_data() -> list[dict[str, Any]]:
    sample_count = 500
    x_values: list[float | None] = _ar1_values(sample_count, random.Random(15), phi=0.4)
    y_values: list[float | None] = _ar1_values(sample_count, random.Random(16), phi=0.4)
    for index in range(sample_count):
        if index % 10 not in {0, 1, 2} or 200 <= index < 280:
            x_values[index] = None
        if index % 10 not in {0, 1, 2} or 320 <= index < 410:
            y_values[index] = None
    return _frame(x_values, y_values)


def _duplicate_derived_series() -> list[dict[str, Any]]:
    rng = random.Random(12)
    x_values = _ar1_values(300, rng, phi=0.4)
    y_values = [2.0 * value + 0.001 * rng.gauss(0, 1.0) for value in x_values]
    return _frame(x_values, y_values)


def _independent_noise_for_many_lag_search() -> list[dict[str, Any]]:
    rng = random.Random(20)
    sample_count = 180
    return _frame(
        [rng.gauss(0, 1.0) for _ in range(sample_count)],
        [rng.gauss(0, 1.0) for _ in range(sample_count)],
    )


def _strong_lag_hidden_beneath_daily_seasonality(*, step_seconds: int) -> list[dict[str, Any]]:
    rng = random.Random(13)
    sample_count = 14 * 24 * 4 + 1
    lag_steps = 4
    seasonal = _seasonal_cycle(sample_count, step_seconds, period_seconds=86_400, amplitude=10.0)
    driver = [rng.gauss(0, 1.0) + 0.3 * math.sin(index * 0.41) for index in range(sample_count)]
    x_values = [seasonal[index] + driver[index] + rng.gauss(0, 0.6) for index in range(sample_count)]
    y_values = [
        1.05 * seasonal[index]
        + (0.0 if index < lag_steps else 0.55 * driver[index - lag_steps])
        + rng.gauss(0, 0.6)
        for index in range(sample_count)
    ]
    return _frame(x_values, y_values, step_seconds=step_seconds)


def _moderate_repeated_episodes() -> list[dict[str, Any]]:
    rng = random.Random(14)
    sample_count = 640
    lag_steps = 2
    x_values = _ar1_values(sample_count, rng, phi=0.2)
    y_values = [rng.gauss(0, 1.0) for _ in range(sample_count)]
    for start, end in ((20, 120), (180, 280), (340, 440), (500, 600)):
        for index in range(start + lag_steps, end):
            y_values[index] = 0.9 * x_values[index - lag_steps] + rng.gauss(0, 0.7)
    return _frame(x_values, y_values)


def _frame(
    x_values: list[float | None],
    y_values: list[float | None],
    *,
    step_seconds: int = 300,
) -> list[dict[str, Any]]:
    assert len(x_values) == len(y_values)
    return [
        {
            "timestamp_utc": START_UTC + timedelta(seconds=index * step_seconds),
            "x": x_values[index],
            "y": y_values[index],
        }
        for index in range(len(x_values))
    ]


def _mapping(frame: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "timestamp_utc": [row["timestamp_utc"] for row in frame],
        "x": [row["x"] for row in frame],
        "y": [row["y"] for row in frame],
    }


def _period_residual_frame(
    frame: list[dict[str, Any]],
    *,
    period_seconds: int,
    step_seconds: int,
) -> list[dict[str, Any]]:
    x_residuals = _period_residuals(_column(frame, "x"), period_seconds=period_seconds, step_seconds=step_seconds)
    y_residuals = _period_residuals(_column(frame, "y"), period_seconds=period_seconds, step_seconds=step_seconds)
    return _frame(x_residuals, y_residuals, step_seconds=step_seconds)


def _period_residuals(
    values: list[float],
    *,
    period_seconds: int,
    step_seconds: int,
) -> list[float]:
    groups: dict[int, list[float]] = {}
    for index, value in enumerate(values):
        groups.setdefault((index * step_seconds) % period_seconds, []).append(value)
    means = {slot: sum(slot_values) / len(slot_values) for slot, slot_values in groups.items()}
    return [value - means[(index * step_seconds) % period_seconds] for index, value in enumerate(values)]


def _seasonal_cycle(
    sample_count: int,
    step_seconds: int,
    *,
    period_seconds: int,
    amplitude: float,
) -> list[float]:
    return [
        amplitude * math.sin(math.tau * ((index * step_seconds) % period_seconds) / period_seconds)
        + 0.25 * amplitude * math.cos(math.tau * ((index * step_seconds) % period_seconds) / period_seconds)
        for index in range(sample_count)
    ]


def _ar1_values(sample_count: int, rng: random.Random, *, phi: float) -> list[float]:
    value = rng.gauss(0, 1.0)
    values = []
    for _ in range(sample_count):
        value = phi * value + rng.gauss(0, 1.0)
        values.append(value)
    return values


def _random_walk(sample_count: int, rng: random.Random) -> list[float]:
    value = 0.0
    values = []
    for _ in range(sample_count):
        value += rng.gauss(0, 1.0)
        values.append(value)
    return values


def _column(frame: list[dict[str, Any]], column: str) -> list[float]:
    values = [row[column] for row in frame]
    if any(value is None for value in values):
        raise AssertionError(f"{column} contains missing values")
    return [float(value) for value in values]


def _pearson(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    return numerator / math.sqrt(left_variance * right_variance)


def _lag_one_autocorrelation(values: list[float]) -> float:
    return _pearson(values[:-1], values[1:])
