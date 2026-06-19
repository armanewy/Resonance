from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone

from resonance.analysis.scanner import ScannerOptions, scan_correlations
from resonance.storage import (
    Measurement,
    ensure_database,
    fetch_correlation_findings,
    insert_measurements,
)
from resonance.synthetic import generate_synthetic_series


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
X_METRIC = "tcp_latency_ms"
Y_METRIC = "cpu_percent"


def test_scan_dry_run_promotes_without_writing(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=48, noise=0.05, seed=11)

    findings = scan_correlations(
        db_path,
        hours=48,
        dry_run=True,
        now=NOW,
        options=ScannerOptions(permutations=99),
    )

    assert len(findings) == 1
    assert findings[0].transform == "first_difference"
    assert abs(findings[0].lag_seconds) == 900
    assert findings[0].evidence["selected_on"] == "first_70_percent"
    assert findings[0].evidence["validated_on"] == "last_30_percent"

    conn = ensure_database(db_path)
    try:
        assert fetch_correlation_findings(conn) == []
    finally:
        conn.close()


def test_scan_persists_repeated_runs_as_one_finding(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=48, noise=0.05, seed=11)

    options = ScannerOptions(permutations=99)
    first = scan_correlations(db_path, hours=48, dry_run=False, now=NOW, options=options)
    second = scan_correlations(db_path, hours=48, dry_run=False, now=NOW, options=options)

    conn = ensure_database(db_path)
    try:
        rows = fetch_correlation_findings(conn)
    finally:
        conn.close()

    assert len(first) == 1
    assert len(second) == 1
    assert len(rows) == 1
    assert rows[0]["x_metric"] == "cpu_percent"
    assert rows[0]["y_metric"] == "tcp_latency_ms"
    assert rows[0]["status"] == "active"


def test_scan_multiple_testing_correction_blocks_borderline_scan(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_scenario_db(db_path, "strong_lag", duration_hours=48, noise=0.05, seed=11)
    _add_independent_metric(db_path, count=577)

    findings = scan_correlations(
        db_path,
        hours=48,
        dry_run=True,
        now=NOW,
        options=ScannerOptions(permutations=19, max_corrected_q=0.10),
    )

    assert findings == ()


def test_scan_cli_is_silent_when_nothing_passes(tmp_path) -> None:
    missing_db = tmp_path / "missing.db"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "resonance.scan",
            "--hours",
            "168",
            "--dry-run",
            "--database",
            str(missing_db),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


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
            measurements.append(Measurement(sample.timestamp_utc, X_METRIC, sample.x, "ms", "synthetic"))
        if sample.y is not None:
            measurements.append(Measurement(sample.timestamp_utc, Y_METRIC, sample.y, "percent", "synthetic"))

    conn = ensure_database(db_path)
    try:
        insert_measurements(conn, measurements)
    finally:
        conn.close()


def _add_independent_metric(db_path, *, count: int) -> None:
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                Measurement(
                    NOW - timedelta(hours=48) + timedelta(minutes=5 * index),
                    "ambient_noise_level",
                    ((index * 17) % 29) + (0.1 if index % 2 else 0.0),
                    "level",
                    "synthetic",
                )
                for index in range(count)
            ],
        )
    finally:
        conn.close()
