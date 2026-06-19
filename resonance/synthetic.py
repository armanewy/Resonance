from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260619
DEFAULT_SAMPLE_INTERVAL_SECONDS = 300
DEFAULT_DURATION_HOURS = 48.0
DEFAULT_NOISE = 0.6
DEFAULT_LAG_SECONDS = 900
DEFAULT_START_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)

CSV_COLUMNS = ["timestamp_utc", "x", "y", "scenario", "true_lag_seconds"]
LAGGED_SCENARIOS = {"strong_lag", "missing_data"}

SCENARIO_DESCRIPTIONS = {
    "strong_lag": "X drives Y after a fixed positive lag plus independent observation noise.",
    "shared_seasonality_only": "X and Y share the same daily cycle while their residual noise remains independent.",
    "single_shared_outlier": "Mostly independent X and Y series contain one dramatic simultaneous event.",
    "relationship_break": "X and Y are strongly related in the first half, then Y becomes independent.",
    "independent_autocorrelated": "X and Y are independent series that each have strong lag-one autocorrelation.",
    "missing_data": "A genuine lagged X-to-Y relationship is interrupted by structured missing intervals.",
}


@dataclass(frozen=True)
class SyntheticSample:
    timestamp_utc: datetime
    x: float | None
    y: float | None
    scenario: str
    true_lag_seconds: int | None


@dataclass(frozen=True)
class SyntheticDataset:
    samples: list[SyntheticSample]
    metadata: dict[str, Any]


def generate_synthetic_series(
    scenario: str,
    *,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    duration_hours: float = DEFAULT_DURATION_HOURS,
    noise: float = DEFAULT_NOISE,
    seed: int = DEFAULT_SEED,
    start_timestamp_utc: datetime = DEFAULT_START_UTC,
) -> SyntheticDataset:
    """Generate deterministic synthetic X/Y time series for analysis tests."""
    _validate_inputs(scenario, sample_interval_seconds, duration_hours, noise)

    sample_count = int(duration_hours * 3600 // sample_interval_seconds) + 1
    rng = random.Random(seed)
    timestamps = [
        _ensure_utc(start_timestamp_utc) + timedelta(seconds=index * sample_interval_seconds)
        for index in range(sample_count)
    ]

    true_lag_seconds = _true_lag_seconds(sample_interval_seconds) if scenario in LAGGED_SCENARIOS else None
    lag_steps = true_lag_seconds // sample_interval_seconds if true_lag_seconds is not None else 0
    if scenario in LAGGED_SCENARIOS and sample_count <= lag_steps + 2:
        raise ValueError("duration must produce more samples than the lagged relationship requires")

    missing_intervals: list[dict[str, int | str]] = []
    if scenario == "strong_lag":
        x_values, y_values = _strong_lag(sample_count, sample_interval_seconds, lag_steps, rng, noise)
    elif scenario == "shared_seasonality_only":
        x_values, y_values = _shared_seasonality_only(sample_count, sample_interval_seconds, rng, noise)
    elif scenario == "single_shared_outlier":
        x_values, y_values = _single_shared_outlier(sample_count, rng, noise)
    elif scenario == "relationship_break":
        x_values, y_values = _relationship_break(sample_count, sample_interval_seconds, seed, rng, noise)
    elif scenario == "independent_autocorrelated":
        x_values, y_values = _independent_autocorrelated(sample_count, seed, noise)
    elif scenario == "missing_data":
        x_values, y_values, missing_intervals = _missing_data(
            sample_count,
            sample_interval_seconds,
            lag_steps,
            rng,
            noise,
        )
    else:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")

    samples = [
        SyntheticSample(
            timestamp_utc=timestamps[index],
            x=x_values[index],
            y=y_values[index],
            scenario=scenario,
            true_lag_seconds=true_lag_seconds,
        )
        for index in range(sample_count)
    ]

    metadata: dict[str, Any] = {
        "scenario": scenario,
        "description": SCENARIO_DESCRIPTIONS[scenario],
        "seed": seed,
        "sample_interval_seconds": sample_interval_seconds,
        "duration_hours": duration_hours,
        "noise": noise,
        "start_timestamp_utc": _format_timestamp(_ensure_utc(start_timestamp_utc)),
        "row_count": sample_count,
        "columns": CSV_COLUMNS,
        "true_lag_seconds": true_lag_seconds,
    }
    if missing_intervals:
        metadata["missing_intervals"] = missing_intervals

    return SyntheticDataset(samples=samples, metadata=metadata)


def write_dataset_csv(dataset: SyntheticDataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for sample in dataset.samples:
            writer.writerow(
                {
                    "timestamp_utc": _format_timestamp(sample.timestamp_utc),
                    "x": _format_optional_float(sample.x),
                    "y": _format_optional_float(sample.y),
                    "scenario": sample.scenario,
                    "true_lag_seconds": "" if sample.true_lag_seconds is None else sample.true_lag_seconds,
                }
            )


def write_metadata(dataset: SyntheticDataset, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(dataset.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_metadata_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".metadata.json")
    return output_path.with_name(f"{output_path.name}.metadata.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic Resonance time-series data.")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIO_DESCRIPTIONS))
    parser.add_argument("--output", required=True, type=Path, help="CSV output path.")
    parser.add_argument("--metadata-output", type=Path, help="Optional metadata JSON output path.")
    parser.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="Seconds between samples.",
    )
    parser.add_argument("--duration-hours", type=float, default=DEFAULT_DURATION_HOURS, help="Generated duration.")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE, help="Observation noise standard deviation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    dataset = generate_synthetic_series(
        args.scenario,
        sample_interval_seconds=args.sample_interval_seconds,
        duration_hours=args.duration_hours,
        noise=args.noise,
        seed=args.seed,
    )
    metadata_path = args.metadata_output or default_metadata_path(args.output)

    write_dataset_csv(dataset, args.output)
    write_metadata(dataset, metadata_path)
    print(f"Wrote {len(dataset.samples)} rows to {args.output}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


def _validate_inputs(scenario: str, sample_interval_seconds: int, duration_hours: float, noise: float) -> None:
    if scenario not in SCENARIO_DESCRIPTIONS:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")


def _true_lag_seconds(sample_interval_seconds: int) -> int:
    lag_steps = max(1, round(DEFAULT_LAG_SECONDS / sample_interval_seconds))
    return lag_steps * sample_interval_seconds


def _strong_lag(
    sample_count: int,
    sample_interval_seconds: int,
    lag_steps: int,
    rng: random.Random,
    noise: float,
) -> tuple[list[float], list[float]]:
    x_values = _driver_values(sample_count, sample_interval_seconds, rng, noise)
    y_values = []
    for index in range(sample_count):
        if index < lag_steps:
            y_values.append(rng.gauss(0, noise))
        else:
            y_values.append(1.2 * x_values[index - lag_steps] + rng.gauss(0, noise))
    return x_values, y_values


def _shared_seasonality_only(
    sample_count: int,
    sample_interval_seconds: int,
    rng: random.Random,
    noise: float,
) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []
    for index in range(sample_count):
        seasonal = _daily_cycle(index, sample_interval_seconds)
        x_values.append(seasonal + rng.gauss(0, noise))
        y_values.append(0.9 * seasonal + rng.gauss(0, noise))
    return x_values, y_values


def _single_shared_outlier(sample_count: int, rng: random.Random, noise: float) -> tuple[list[float], list[float]]:
    x_values = [rng.gauss(0, noise) for _ in range(sample_count)]
    y_values = [rng.gauss(0, noise) for _ in range(sample_count)]
    event_index = sample_count // 2
    x_values[event_index] += 30 + 10 * noise
    y_values[event_index] += 36 + 10 * noise
    return x_values, y_values


def _relationship_break(
    sample_count: int,
    sample_interval_seconds: int,
    seed: int,
    rng: random.Random,
    noise: float,
) -> tuple[list[float], list[float]]:
    x_values = _driver_values(sample_count, sample_interval_seconds, rng, noise)
    independent_rng = random.Random(seed + 404)
    independent_y = _ar1_values(sample_count, independent_rng, max(noise, 0.2), phi=0.55)
    halfway = sample_count // 2
    y_values = []
    for index in range(sample_count):
        if index < halfway:
            y_values.append(1.1 * x_values[index] + rng.gauss(0, noise))
        else:
            y_values.append(independent_y[index])
    return x_values, y_values


def _independent_autocorrelated(sample_count: int, seed: int, noise: float) -> tuple[list[float], list[float]]:
    innovation_scale = max(noise, 0.2)
    x_values = _ar1_values(sample_count, random.Random(seed + 101), innovation_scale, phi=0.88)
    y_values = _ar1_values(sample_count, random.Random(seed + 202), innovation_scale, phi=0.83)
    return x_values, y_values


def _missing_data(
    sample_count: int,
    sample_interval_seconds: int,
    lag_steps: int,
    rng: random.Random,
    noise: float,
) -> tuple[list[float | None], list[float | None], list[dict[str, int | str]]]:
    x_values, y_values = _strong_lag(sample_count, sample_interval_seconds, lag_steps, rng, noise)
    missing_intervals = [
        {"series": "x", "start_index": int(sample_count * 0.20), "end_index_exclusive": int(sample_count * 0.28)},
        {"series": "y", "start_index": int(sample_count * 0.52), "end_index_exclusive": int(sample_count * 0.60)},
        {"series": "both", "start_index": int(sample_count * 0.76), "end_index_exclusive": int(sample_count * 0.82)},
    ]

    for interval in missing_intervals:
        start_index = max(lag_steps + 1, int(interval["start_index"]))
        end_index = min(sample_count, max(start_index + 2, int(interval["end_index_exclusive"])))
        interval["start_index"] = start_index
        interval["end_index_exclusive"] = end_index
        series = interval["series"]
        for index in range(start_index, end_index):
            if series in {"x", "both"}:
                x_values[index] = None
            if series in {"y", "both"}:
                y_values[index] = None

    return x_values, y_values, missing_intervals


def _driver_values(
    sample_count: int,
    sample_interval_seconds: int,
    rng: random.Random,
    noise: float,
) -> list[float]:
    values = []
    for index in range(sample_count):
        elapsed_hours = index * sample_interval_seconds / 3600
        local_step = 4.0 if (index // 11) % 5 == 2 else 0.0
        local_step -= 2.5 if (index // 17) % 4 == 1 else 0.0
        values.append(
            5.0 * math.sin(index * 0.73)
            + 2.5 * math.sin(index * 0.17 + 0.4)
            + 0.03 * elapsed_hours
            + local_step
            + rng.gauss(0, noise)
        )
    return values


def _daily_cycle(index: int, sample_interval_seconds: int) -> float:
    phase = (index * sample_interval_seconds % 86_400) / 86_400
    return 9.0 * math.sin(math.tau * phase) + 2.0 * math.cos(math.tau * phase)


def _ar1_values(sample_count: int, rng: random.Random, innovation_scale: float, *, phi: float) -> list[float]:
    value = rng.gauss(0, innovation_scale)
    values = []
    for _ in range(sample_count):
        value = phi * value + rng.gauss(0, innovation_scale)
        values.append(value)
    return values


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _ensure_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
