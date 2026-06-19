from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from resonance.analyze_pair import analyze_pair
from resonance.storage import Measurement, ensure_database, insert_measurements
from resonance.synthetic import SCENARIO_DESCRIPTIONS, generate_synthetic_series


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
X_METRIC = "tcp_latency_ms"
Y_METRIC = "cpu_percent"


def test_analyze_pair_recovers_strong_lag_after_first_difference(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=24, noise=0.15, seed=11)

    report = analyze_pair(
        db_path,
        x_metric=X_METRIC,
        y_metric=Y_METRIC,
        hours=24,
        transform_name="first_difference",
        max_lag_minutes=60,
        now=NOW,
    )

    assert report["status"] == "ok"
    assert abs(report["lag"]["best_lag_seconds"] - 900) <= report["aligned"]["cadence_seconds"]
    assert report["lag"]["best_rho"] > 0.8
    assert report["validation"]["holdout_overlap"] > 30
    assert "does not establish causation" in report["causation_warning"]


@pytest.mark.parametrize(
    "scenario",
    sorted(set(SCENARIO_DESCRIPTIONS) - {"strong_lag", "missing_data"}),
)
def test_analyze_pair_runs_null_and_adversarial_scenarios_without_crashing(tmp_path, scenario) -> None:
    db_path = tmp_path / f"{scenario}.db"
    _write_scenario_db(db_path, scenario, duration_hours=48, noise=0.6, seed=42)

    report = analyze_pair(
        db_path,
        x_metric=X_METRIC,
        y_metric=Y_METRIC,
        hours=48,
        transform_name="raw",
        max_lag_minutes=60,
        now=NOW,
    )

    assert report["status"] == "ok"
    assert report["lag"]["score_count"] > 0
    assert "permutation_p_value" in report["validation"]
    assert "does not establish causation" in report["causation_warning"]


def test_analyze_pair_reports_insufficient_data_without_persisting_findings(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=1, noise=0.15, seed=11)

    report = analyze_pair(
        db_path,
        x_metric=X_METRIC,
        y_metric="missing_metric",
        hours=24,
        transform_name="raw",
        max_lag_minutes=60,
        now=NOW,
    )

    assert report["status"] == "insufficient_data"
    assert "missing_metric" in report["reason"]
    conn = ensure_database(db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
    finally:
        conn.close()
    assert tables == {"collector_errors", "correlation_findings", "events", "measurements"}


def test_analyze_pair_cli_emits_json(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=24, noise=0.15, seed=11)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "resonance.analyze_pair",
            "--x",
            X_METRIC,
            "--y",
            Y_METRIC,
            "--hours",
            "24",
            "--transform",
            "first_difference",
            "--max-lag-minutes",
            "60",
            "--database",
            str(db_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ok"
    assert report["x_metric"] == X_METRIC
    assert report["y_metric"] == Y_METRIC
    assert abs(report["lag"]["best_lag_seconds"] - 900) <= report["aligned"]["cadence_seconds"]


def _write_scenario_db(
    db_path,
    scenario: str,
    *,
    duration_hours: float,
    noise: float,
    seed: int,
) -> None:
    dataset = generate_synthetic_series(
        scenario,
        sample_interval_seconds=300,
        duration_hours=duration_hours,
        noise=noise,
        seed=seed,
        start_timestamp_utc=NOW - timedelta(hours=duration_hours),
    )
    measurements = []
    for sample in dataset.samples:
        if sample.x is not None:
            measurements.append(
                Measurement(sample.timestamp_utc, X_METRIC, sample.x, "ms", "synthetic")
            )
        if sample.y is not None:
            measurements.append(
                Measurement(sample.timestamp_utc, Y_METRIC, sample.y, "percent", "synthetic")
            )

    conn = ensure_database(db_path)
    try:
        insert_measurements(conn, measurements)
    finally:
        conn.close()
