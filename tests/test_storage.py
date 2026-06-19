from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.storage import Measurement, fetch_measurements, init_db, insert_measurements, sample_counts_by_metric


def test_sqlite_measurement_insertion_and_querying(sqlite_conn) -> None:
    now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    inserted = insert_measurements(
        sqlite_conn,
        [
            Measurement(now, "cpu_percent", 42.0, "percent", "personal"),
            Measurement(now + timedelta(seconds=30), "memory_percent", 55.0, "percent", "personal"),
        ],
    )

    rows = fetch_measurements(sqlite_conn, now - timedelta(seconds=1), now + timedelta(minutes=1))
    counts = sample_counts_by_metric(sqlite_conn, now - timedelta(seconds=1), now + timedelta(minutes=1))

    assert inserted == 2
    assert [row["metric"] for row in rows] == ["cpu_percent", "memory_percent"]
    assert {(row["metric"], row["sample_count"]) for row in counts} == {("cpu_percent", 1), ("memory_percent", 1)}


def test_duplicate_weather_prevention(sqlite_conn) -> None:
    now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    measurement = Measurement(now, "weather_temperature_c", 22.0, "C", "open-meteo")

    first = insert_measurements(sqlite_conn, [measurement])
    second = insert_measurements(sqlite_conn, [measurement])

    rows = fetch_measurements(sqlite_conn, now - timedelta(seconds=1), now + timedelta(seconds=1))
    assert first == 1
    assert second == 0
    assert len(rows) == 1

