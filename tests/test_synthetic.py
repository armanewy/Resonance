from __future__ import annotations

import csv
import json
import math
import subprocess
import sys

import pytest

from resonance.synthetic import (
    CSV_COLUMNS,
    LAGGED_SCENARIOS,
    SCENARIO_DESCRIPTIONS,
    default_metadata_path,
    generate_synthetic_series,
)


@pytest.mark.parametrize("scenario", sorted(SCENARIO_DESCRIPTIONS))
def test_scenarios_are_deterministic_and_include_required_metadata(scenario: str) -> None:
    first = generate_synthetic_series(
        scenario,
        sample_interval_seconds=300,
        duration_hours=24,
        noise=0.5,
        seed=1234,
    )
    second = generate_synthetic_series(
        scenario,
        sample_interval_seconds=300,
        duration_hours=24,
        noise=0.5,
        seed=1234,
    )

    assert first == second
    assert first.metadata["columns"] == CSV_COLUMNS
    assert first.metadata["description"] == SCENARIO_DESCRIPTIONS[scenario]
    assert first.metadata["row_count"] == len(first.samples)
    assert {sample.scenario for sample in first.samples} == {scenario}

    if scenario in LAGGED_SCENARIOS:
        assert {sample.true_lag_seconds for sample in first.samples} == {900}
    else:
        assert {sample.true_lag_seconds for sample in first.samples} == {None}


def test_cli_writes_csv_and_metadata(tmp_path) -> None:
    output_path = tmp_path / "resonance-synthetic.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "resonance.synthetic",
            "--scenario",
            "strong_lag",
            "--output",
            str(output_path),
            "--duration-hours",
            "2",
            "--sample-interval-seconds",
            "300",
            "--noise",
            "0.2",
            "--seed",
            "99",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metadata_path = default_metadata_path(output_path)
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert metadata_path.exists()
    assert rows[0].keys() == set(CSV_COLUMNS)
    assert rows[0]["timestamp_utc"] == "2026-01-01T00:00:00Z"
    assert rows[0]["scenario"] == "strong_lag"
    assert rows[0]["true_lag_seconds"] == "900"
    assert metadata["scenario"] == "strong_lag"
    assert metadata["description"]
    assert metadata["row_count"] == len(rows)


def test_strong_lag_x_drives_future_y() -> None:
    dataset = generate_synthetic_series(
        "strong_lag",
        sample_interval_seconds=300,
        duration_hours=24,
        noise=0.15,
        seed=11,
    )
    lag_steps = _lag_steps(dataset)
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]

    lagged = _pearson(x_values[:-lag_steps], y_values[lag_steps:])
    simultaneous = _pearson(x_values[lag_steps:], y_values[lag_steps:])

    assert lagged > 0.98
    assert lagged - simultaneous > 0.55


def test_shared_seasonality_only_has_independent_residual_noise() -> None:
    sample_interval_seconds = 1800
    dataset = generate_synthetic_series(
        "shared_seasonality_only",
        sample_interval_seconds=sample_interval_seconds,
        duration_hours=168,
        noise=0.8,
        seed=42,
    )
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]
    residual_x = _remove_time_of_day_means(x_values, sample_interval_seconds)
    residual_y = _remove_time_of_day_means(y_values, sample_interval_seconds)

    assert _pearson(x_values, y_values) > 0.97
    assert abs(_pearson(residual_x, residual_y)) < 0.12


def test_single_shared_outlier_has_one_simultaneous_event() -> None:
    dataset = generate_synthetic_series(
        "single_shared_outlier",
        sample_interval_seconds=300,
        duration_hours=24,
        noise=0.5,
        seed=21,
    )
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]
    x_event = max(range(len(x_values)), key=lambda index: x_values[index])
    y_event = max(range(len(y_values)), key=lambda index: y_values[index])
    rest_x = [value for index, value in enumerate(x_values) if index != x_event]
    rest_y = [value for index, value in enumerate(y_values) if index != y_event]

    assert x_event == y_event == len(dataset.samples) // 2
    assert x_values[x_event] - max(rest_x) > 25
    assert y_values[y_event] - max(rest_y) > 30
    assert abs(_pearson(rest_x, rest_y)) < 0.2


def test_relationship_break_changes_from_related_to_independent() -> None:
    dataset = generate_synthetic_series(
        "relationship_break",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.25,
        seed=34,
    )
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]
    halfway = len(dataset.samples) // 2

    assert _pearson(x_values[:halfway], y_values[:halfway]) > 0.99
    assert abs(_pearson(x_values[halfway:], y_values[halfway:])) < 0.2


def test_independent_autocorrelated_series_have_internal_memory_only() -> None:
    dataset = generate_synthetic_series(
        "independent_autocorrelated",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.7,
        seed=55,
    )
    x_values = [sample.x for sample in dataset.samples]
    y_values = [sample.y for sample in dataset.samples]

    assert _lag_one_autocorrelation(x_values) > 0.8
    assert _lag_one_autocorrelation(y_values) > 0.75
    assert abs(_pearson(x_values, y_values)) < 0.2


def test_missing_data_keeps_lagged_relationship_across_available_points() -> None:
    dataset = generate_synthetic_series(
        "missing_data",
        sample_interval_seconds=300,
        duration_hours=48,
        noise=0.2,
        seed=89,
    )
    lag_steps = _lag_steps(dataset)
    samples = dataset.samples
    x_missing_runs = _missing_run_lengths([sample.x is None for sample in samples])
    y_missing_runs = _missing_run_lengths([sample.y is None for sample in samples])
    both_missing = [sample.x is None and sample.y is None for sample in samples]
    lagged_x = []
    observed_y = []

    for index in range(lag_steps, len(samples)):
        source_x = samples[index - lag_steps].x
        target_y = samples[index].y
        if source_x is not None and target_y is not None:
            lagged_x.append(source_x)
            observed_y.append(target_y)

    assert max(x_missing_runs) >= 2
    assert max(y_missing_runs) >= 2
    assert max(_missing_run_lengths(both_missing)) >= 2
    assert _pearson(lagged_x, observed_y) > 0.98
    assert dataset.metadata["missing_intervals"]


def _lag_steps(dataset) -> int:
    true_lag_seconds = dataset.samples[0].true_lag_seconds
    sample_interval_seconds = dataset.metadata["sample_interval_seconds"]
    assert true_lag_seconds is not None
    return true_lag_seconds // sample_interval_seconds


def _pearson(left, right) -> float:
    assert len(left) == len(right)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    return numerator / math.sqrt(left_variance * right_variance)


def _remove_time_of_day_means(values, sample_interval_seconds: int) -> list[float]:
    groups: dict[int, list[float]] = {}
    for index, value in enumerate(values):
        offset = (index * sample_interval_seconds) % 86_400
        groups.setdefault(offset, []).append(value)

    means = {offset: sum(group) / len(group) for offset, group in groups.items()}
    return [value - means[(index * sample_interval_seconds) % 86_400] for index, value in enumerate(values)]


def _lag_one_autocorrelation(values) -> float:
    return _pearson(values[:-1], values[1:])


def _missing_run_lengths(missing_flags: list[bool]) -> list[int]:
    run_lengths = []
    current = 0
    for missing in missing_flags:
        if missing:
            current += 1
        elif current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    return run_lengths or [0]
