from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.analysis.service import ValidationOptions, analyze_metric_pair, list_analyzable_metrics
from resonance.analysis.scanner import scan_correlations
from resonance.public_series import align_public_series_with_measurement, fetch_series, list_series
from resonance.public_sources.eia_grid import SOURCE_ID, ensure_eia_registry
from resonance.storage import (
    Measurement,
    PublicObservation,
    SeriesRecord,
    ensure_database,
    insert_measurements,
    insert_public_observations,
    upsert_series_record,
)


START = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc)


def test_registered_series_identity_is_stable_and_geographic() -> None:
    conn = ensure_database(":memory:")
    try:
        ensure_eia_registry(conn)
        first = list_series(conn, source_id=SOURCE_ID, geography_id="ISNE")
        ensure_eia_registry(conn)
        second = list_series(conn, source_id=SOURCE_ID, geography_id="ISNE")
    finally:
        conn.close()

    assert [series.series_id for series in first] == [series.series_id for series in second]
    assert {series.geography_type for series in first} == {"balancing_authority"}
    assert {series.geography_id for series in first} == {"ISNE"}
    assert "eia_grid_monitor:ISNE:system_load" in {series.series_id for series in first}


def test_geography_and_lineage_distinguish_public_series() -> None:
    conn = ensure_database(":memory:")
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
        system_load = list_series(conn, source_id=SOURCE_ID, geography_type="balancing_authority")
        mix = list_series(conn, source_id=SOURCE_ID, geography_id="ISNE")
    finally:
        conn.close()

    system_load_ids = {series.series_id for series in system_load if series.metric_name == "system_load"}
    assert system_load_ids == {"eia_grid_monitor:ISNE:system_load", "eia_grid_monitor:NYIS:system_load"}
    mix_by_id = {series.series_id: series for series in mix}
    assert mix_by_id["eia_grid_monitor:ISNE:generation_natural_gas"].lineage_id == "eia_grid_monitor:ISNE:generation_mix"
    assert mix_by_id["eia_grid_monitor:ISNE:generation_wind"].lineage_id == "eia_grid_monitor:ISNE:generation_mix"
    assert mix_by_id["eia_grid_monitor:ISNE:generation_natural_gas"].series_id != mix_by_id["eia_grid_monitor:ISNE:generation_wind"].series_id


def test_public_observations_deduplicate_and_latest_revision_wins() -> None:
    conn = ensure_database(":memory:")
    try:
        ensure_eia_registry(conn)
        base = _observation(value=100.0, revision="rev-1", ingested=START + timedelta(hours=2))
        duplicate = _observation(value=100.0, revision="rev-1", ingested=START + timedelta(hours=3))
        revised = _observation(value=125.0, revision="rev-2", ingested=START + timedelta(hours=4))

        assert insert_public_observations(conn, [base]) == 1
        assert insert_public_observations(conn, [duplicate]) == 0
        assert insert_public_observations(conn, [revised]) == 1
        rows = fetch_series(conn, "eia_grid_monitor:ISNE:system_load", START, START + timedelta(hours=1))
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].value == 125.0
    assert rows[0].source_revision == "rev-2"


def test_existing_measurements_are_mapped_to_registered_series() -> None:
    conn = ensure_database(":memory:")
    try:
        insert_measurements(
            conn,
            [
                Measurement(START, "cpu_percent", 20.0, "percent", "personal"),
                Measurement(START, "weather_temperature_c", 18.0, "C", "open-meteo"),
            ],
        )
        rows = conn.execute(
            "SELECT source, metric, series_id FROM measurement_series_map ORDER BY source, metric"
        ).fetchall()
        series = list_series(conn)
    finally:
        conn.close()

    assert {(row["source"], row["metric"], row["series_id"]) for row in rows} == {
        ("open-meteo", "weather_temperature_c", "measurement:open-meteo:weather_temperature_c"),
        ("personal", "cpu_percent", "measurement:personal:cpu_percent"),
    }
    assert "measurement:personal:cpu_percent" in {item.series_id for item in series}


def test_public_series_aligns_with_existing_weather_measurement_and_pair_analysis(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    conn = ensure_database(db_path)
    try:
        ensure_eia_registry(conn)
        public = [
            _observation(value=100.0 + hour, revision=f"rev-{hour}", ingested=START + timedelta(days=1), hour=hour)
            for hour in range(12)
        ]
        weather = [
            Measurement(START + timedelta(hours=hour), "weather_temperature_c", 10.0 + hour, "C", "open-meteo")
            for hour in range(12)
        ]
        insert_public_observations(conn, public)
        insert_measurements(conn, weather)
        aligned = align_public_series_with_measurement(
            conn,
            public_series_id="eia_grid_monitor:ISNE:system_load",
            metric="weather_temperature_c",
            start_utc=START,
            end_utc=START + timedelta(hours=11),
            min_points=4,
        )
    finally:
        conn.close()

    assert len(aligned.aligned_pair.frame) == 12
    assert aligned.public_series.display_name == "ISO New England system load"

    metrics = list_analyzable_metrics(db_path, START, START + timedelta(hours=11))
    labels = {metric.display_name for metric in metrics}
    assert "ISO New England system load [ISNE]" in labels

    analysis = analyze_metric_pair(
        db_path,
        "eia_grid_monitor:ISNE:system_load",
        "weather_temperature_c",
        START,
        START + timedelta(hours=11),
        "raw",
        max_lag_steps=1,
        validation_options=ValidationOptions(min_aligned_points=4, min_overlap=3, permutations=9, permutation_seed=7),
    )
    assert analysis.x_metric_summary.series_id == "eia_grid_monitor:ISNE:system_load"
    assert analysis.y_metric_summary.metric == "weather_temperature_c"


def test_scanner_does_not_collapse_public_geographies_into_legacy_metrics(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
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
                _observation(value=100.0 + hour, revision=f"isne-{hour}", ingested=START + timedelta(days=1), hour=hour)
                for hour in range(4)
            ]
            + [
                PublicObservation(
                    series_id="eia_grid_monitor:NYIS:system_load",
                    valid_start_utc=START + timedelta(hours=hour),
                    valid_end_utc=START + timedelta(hours=hour + 1),
                    observed_at_utc=START + timedelta(hours=hour + 1),
                    ingested_at_utc=START + timedelta(days=1),
                    value=200.0 + hour,
                    quality="reported",
                    source_revision=f"nyis-{hour}",
                    source_observation_key=f"region-data:NYIS:D:2026-06-19T{hour:02d}",
                )
                for hour in range(4)
            ],
        )
    finally:
        conn.close()

    assert scan_correlations(db_path, hours=4, dry_run=True, now=START + timedelta(hours=4)) == ()


def _observation(
    *,
    value: float,
    revision: str,
    ingested: datetime,
    hour: int = 0,
) -> PublicObservation:
    valid_start = START + timedelta(hours=hour)
    return PublicObservation(
        series_id="eia_grid_monitor:ISNE:system_load",
        valid_start_utc=valid_start,
        valid_end_utc=valid_start + timedelta(hours=1),
        observed_at_utc=valid_start + timedelta(hours=1),
        ingested_at_utc=ingested,
        value=value,
        quality="reported",
        source_revision=revision,
        source_observation_key=f"region-data:ISNE:D:{valid_start:%Y-%m-%dT%H}",
    )
