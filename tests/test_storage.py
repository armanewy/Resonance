from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from resonance.storage import (
    EventMarker,
    Measurement,
    fetch_event_markers,
    fetch_measurements,
    init_db,
    insert_event_marker,
    insert_measurements,
    sample_counts_by_metric,
)


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


def test_event_marker_insertion_and_recent_query(sqlite_conn) -> None:
    first = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    second = first + timedelta(minutes=5)

    first_id = insert_event_marker(
        sqlite_conn,
        EventMarker(first, " internet felt bad ", " DNS lookup stalled ", first + timedelta(seconds=1)),
    )
    second_id = insert_event_marker(sqlite_conn, EventMarker(second, "started a download"))

    rows = fetch_event_markers(sqlite_conn, 1)
    all_rows = fetch_event_markers(sqlite_conn, None)

    assert first_id > 0
    assert second_id > first_id
    assert len(rows) == 1
    assert rows[0]["label"] == "started a download"
    assert rows[0]["note"] == ""
    assert rows[0]["timestamp_utc"] == "2026-06-19T12:05:00Z"
    assert rows[0]["created_at_utc"] == "2026-06-19T12:05:00Z"
    assert [row["label"] for row in all_rows] == ["started a download", "internet felt bad"]
    assert all_rows[1]["note"] == "DNS lookup stalled"


def test_event_marker_requires_label(sqlite_conn) -> None:
    now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="label is required"):
        insert_event_marker(sqlite_conn, EventMarker(now, " "))

