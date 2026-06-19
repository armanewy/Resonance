from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from resonance.audit import audit_database, main
from resonance.storage import (
    CollectorError,
    Measurement,
    init_db,
    insert_collector_error,
    insert_measurements,
)
from resonance.time_utils import to_utc_iso


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_empty_database_reports_cleanly(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    conn.close()

    report = audit_database(db_path, hours=1, now=NOW)

    assert report["database_exists"] is True
    assert report["total_measurements"] == 0
    assert report["total_collector_errors"] == 0
    assert report["metrics"] == []
    assert report["stale_metrics"] == []
    assert report["metrics_with_less_than_80_percent_coverage"] == []


def test_missing_database_is_not_created(tmp_path) -> None:
    db_path = tmp_path / "missing.db"

    report = audit_database(db_path, hours=1, now=NOW)

    assert report["database_exists"] is False
    assert report["total_measurements"] == 0
    assert not db_path.exists()


def test_audit_reports_gaps_duplicates_stale_data_and_event_metrics(tmp_path) -> None:
    db_path = tmp_path / "audit.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    start = NOW - timedelta(hours=1)

    insert_measurements(
        conn,
        [
            Measurement(start, "cpu_percent", 10.0, "percent", "personal"),
            Measurement(start + timedelta(minutes=10), "cpu_percent", 20.0, "percent", "personal"),
            Measurement(start + timedelta(minutes=20), "cpu_percent", 30.0, "percent", "personal"),
            Measurement(start + timedelta(minutes=20), "cpu_percent", 40.0, "percent", "demo"),
            Measurement(NOW, "cpu_percent", 50.0, "percent", "personal"),
            Measurement(start, "memory_percent", 60.0, "percent", "personal"),
            Measurement(start + timedelta(minutes=10), "memory_percent", 61.0, "percent", "personal"),
            Measurement(start + timedelta(minutes=20), "memory_percent", 62.0, "percent", "personal"),
            Measurement(start, "tcp_success", 0.0, "boolean", "personal"),
            Measurement(start + timedelta(minutes=30), "tcp_success", 1.0, "boolean", "personal"),
            Measurement(NOW, "tcp_success", 1.0, "boolean", "personal"),
        ],
    )
    conn.execute(
        """
        INSERT INTO measurements (timestamp_utc, metric, value, unit, source, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (to_utc_iso(start + timedelta(minutes=30)), "cpu_percent", "bad", "percent", "personal", "{}"),
    )
    conn.commit()
    insert_collector_error(
        conn,
        CollectorError(start + timedelta(minutes=5), "personal", "network_failed", "example"),
    )
    insert_collector_error(
        conn,
        CollectorError(start - timedelta(minutes=5), "personal", "outside_window", "ignored"),
    )
    conn.close()

    report = audit_database(db_path, hours=1, now=NOW)
    metrics = {metric["metric"]: metric for metric in report["metrics"]}

    assert report["total_measurements"] == 12
    assert report["total_collector_errors"] == 1
    assert report["stale_metrics"] == ["memory_percent"]
    assert report["metrics_with_less_than_80_percent_coverage"] == [
        "cpu_percent",
        "memory_percent",
    ]

    cpu = metrics["cpu_percent"]
    assert cpu["sample_count"] == 6
    assert cpu["distinct_timestamp_count"] == 5
    assert cpu["earliest_timestamp_utc"] == "2026-06-19T11:00:00Z"
    assert cpu["latest_timestamp_utc"] == "2026-06-19T12:00:00Z"
    assert cpu["latest_sample_age_seconds"] == 0
    assert cpu["median_sampling_interval_seconds"] == 600
    assert cpu["expected_sample_count"] == 7
    assert cpu["coverage_percentage"] == pytest.approx(71.428571)
    assert cpu["longest_gap_seconds"] == 1800
    assert cpu["duplicate_timestamp_count"] == 1
    assert cpu["null_or_non_numeric_count"] == 1
    assert cpu["minimum"] == 10
    assert cpu["median"] == 30
    assert cpu["maximum"] == 50
    assert cpu["standard_deviation"] == pytest.approx(14.142136)
    assert cpu["source_values"] == ["demo", "personal"]

    memory = metrics["memory_percent"]
    assert memory["is_stale"] is True
    assert memory["latest_sample_age_seconds"] == 2400
    assert memory["coverage_percentage"] == pytest.approx(42.857143)

    tcp = metrics["tcp_success"]
    assert tcp["minimum"] is None
    assert tcp["median"] is None
    assert tcp["standard_deviation"] is None
    assert tcp["value_distribution"] == {"0": 1, "1": 2}
    assert tcp["coverage_percentage"] == 100


def test_json_cli_output_is_machine_readable(tmp_path, capsys) -> None:
    db_path = tmp_path / "audit.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    insert_measurements(conn, [Measurement(NOW, "cpu_percent", 42.0, "percent", "personal")])
    conn.close()

    exit_code = main(["--database", str(db_path), "--hours", "1", "--json"], now=NOW)

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert parsed["audit_interval"]["end_utc"] == "2026-06-19T12:00:00Z"
    assert parsed["metrics"][0]["metric"] == "cpu_percent"
