from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from resonance.analysis.scanner import ScannerOptions, _adjust_p_values, _select_scanner_candidate_pairs, scan_correlations
from resonance.public_sources.eia_grid import SOURCE_ID, ensure_eia_registry
from resonance.storage import (
    Measurement,
    PublicObservation,
    SeriesRecord,
    ensure_database,
    fetch_correlation_findings,
    insert_measurements,
    insert_public_observations,
    upsert_series_record,
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


def test_scan_dry_run_can_return_public_legacy_pair_without_writing(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_public_legacy_lag_db(db_path, hours=72)

    findings = scan_correlations(
        db_path,
        hours=72,
        dry_run=True,
        now=NOW,
        options=_public_scanner_options(),
    )

    conn = ensure_database(db_path)
    try:
        stored = fetch_correlation_findings(conn)
    finally:
        conn.close()

    assert len(findings) == 1
    assert {findings[0].x_metric, findings[0].y_metric} == {
        "eia_grid_monitor:ISNE:system_load",
        "weather_temperature_c",
    }
    assert findings[0].evidence["dry_run_only"] is True
    assert findings[0].evidence["pair_compatibility"]["geography"] == "local_to_regional_context"
    assert stored == []


def test_scan_non_dry_run_keeps_public_series_out_of_persistence_path(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_public_legacy_lag_db(db_path, hours=72)

    findings = scan_correlations(
        db_path,
        hours=72,
        dry_run=False,
        now=NOW,
        options=_public_scanner_options(),
    )

    conn = ensure_database(db_path)
    try:
        stored = fetch_correlation_findings(conn)
    finally:
        conn.close()

    assert findings == ()
    assert stored == []


def test_unified_scanner_rejects_incompatible_cadence_geography_and_lineage(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    start = NOW - timedelta(hours=6)
    conn = ensure_database(db_path)
    try:
        ensure_eia_registry(conn)
        upsert_series_record(
            conn,
            SeriesRecord(
                series_id="eia_grid_monitor:NYIS:system_load",
                source_id=SOURCE_ID,
                metric_name="system_load",
                display_name="NYISO system load",
                unit="MWh",
                cadence_seconds=3600,
                aggregation="hourly",
                geography_type="balancing_authority",
                geography_id="NYIS",
                timezone="America/New_York",
                timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
                parent_series_id=None,
                lineage_id="eia_grid_monitor:NYIS:system_load",
                quality_tier="official",
            ),
        )
        insert_public_observations(
            conn,
            [
                _public_observation("eia_grid_monitor:ISNE:system_load", start + timedelta(hours=hour), 100.0 + hour)
                for hour in range(6)
            ]
            + [
                _public_observation("eia_grid_monitor:NYIS:system_load", start + timedelta(hours=hour), 200.0 + hour)
                for hour in range(6)
            ]
            + [
                _public_observation("eia_grid_monitor:ISNE:generation_natural_gas", start + timedelta(hours=hour), 50.0 + hour)
                for hour in range(6)
            ]
            + [
                _public_observation("eia_grid_monitor:ISNE:generation_wind", start + timedelta(hours=hour), 10.0 + hour)
                for hour in range(6)
            ],
        )
        insert_measurements(
            conn,
            [
                Measurement(start + timedelta(seconds=1000 * index), "odd_cadence_metric", float(index), "units", "synthetic")
                for index in range(20)
            ],
        )
    finally:
        conn.close()

    selection = _select_scanner_candidate_pairs(
        db_path,
        start,
        NOW,
        include_public=True,
        options={
            "min_observations": 4,
            "min_coverage": 0.1,
            "min_aligned_bins": 2,
        },
    )
    reasons = {frozenset(rejection.metrics): rejection.reason for rejection in selection.rejections}

    assert reasons[frozenset(("eia_grid_monitor:ISNE:system_load", "odd_cadence_metric"))] == "incompatible_cadence"
    assert reasons[frozenset(("eia_grid_monitor:ISNE:system_load", "eia_grid_monitor:NYIS:system_load"))] == "incompatible_geography"
    assert reasons[frozenset(("eia_grid_monitor:ISNE:generation_natural_gas", "eia_grid_monitor:ISNE:generation_wind"))] == "shared_lineage"
    rejected_pairs = {frozenset(rejection.metrics) for rejection in selection.rejections}
    selected_pairs = {frozenset((pair.x_metric, pair.y_metric)) for pair in selection.pairs}
    assert not (rejected_pairs & selected_pairs)


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


def test_default_by_adjustment_is_more_conservative_than_bh() -> None:
    p_values = [0.01, 0.02, 0.04]

    bh = _adjust_p_values(p_values, total_tests=3, method="bh")
    by = _adjust_p_values(p_values, total_tests=3, method="by")

    assert all(by_value >= bh_value for by_value, bh_value in zip(by, bh, strict=True))
    assert any(by_value > bh_value for by_value, bh_value in zip(by, bh, strict=True))


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


def _write_public_legacy_lag_db(db_path, *, hours: int) -> None:
    start = NOW - timedelta(hours=hours - 1)
    public = []
    weather = []
    for index in range(hours):
        timestamp = start + timedelta(hours=index)
        signal = math.sin(index / 3.0) + 0.25 * math.cos(index / 5.0)
        lagged = math.sin((index - 1) / 3.0) + 0.25 * math.cos((index - 1) / 5.0)
        public.append(
            _public_observation(
                "eia_grid_monitor:ISNE:system_load",
                timestamp,
                signal,
                revision=f"public-{index}",
            )
        )
        weather.append(Measurement(timestamp, "weather_temperature_c", lagged, "C", "open-meteo"))

    conn = ensure_database(db_path)
    try:
        ensure_eia_registry(conn)
        insert_public_observations(conn, public)
        insert_measurements(conn, weather)
    finally:
        conn.close()


def _public_observation(
    series_id: str,
    timestamp: datetime,
    value: float,
    *,
    revision: str | None = None,
) -> PublicObservation:
    return PublicObservation(
        series_id=series_id,
        valid_start_utc=timestamp,
        valid_end_utc=timestamp + timedelta(hours=1),
        observed_at_utc=timestamp + timedelta(hours=1),
        ingested_at_utc=timestamp + timedelta(days=1),
        value=value,
        quality="reported",
        source_revision=revision or f"{series_id}:{timestamp:%Y%m%dT%H}",
        source_observation_key=f"{series_id}:{timestamp:%Y%m%dT%H}",
    )


def _public_scanner_options() -> ScannerOptions:
    return ScannerOptions(
        min_aligned_observations=24,
        min_coverage=0.5,
        discovery_fraction=0.7,
        min_discovery_abs_rho=0.4,
        max_corrected_q=1.0,
        min_holdout_abs_rho=0.2,
        min_sign_stability=0.5,
        max_lag_seconds=7200,
        min_overlap=8,
        window_count=2,
        permutations=19,
        calendar_min_history=999,
        multiple_testing_method="bh",
    )
