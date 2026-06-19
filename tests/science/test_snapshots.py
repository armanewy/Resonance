from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from resonance.science.snapshots import (
    BlindEvaluatorCapability,
    EmptySnapshotError,
    InsufficientSnapshotDataError,
    create_blind_evaluator_capability,
    create_snapshot,
    load_blind_view,
    load_exploration_view,
)
from resonance.storage import Measurement, init_db, insert_measurements


START = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


def test_snapshot_hash_is_stable_for_identical_rows_and_config(tmp_path: Path) -> None:
    db_path = _create_db(tmp_path, _measurements(16))
    artifact_root = tmp_path / "artifacts"

    first = create_snapshot(
        db_path=db_path,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=0,
        artifact_root=artifact_root,
    )
    second = create_snapshot(
        db_path=db_path,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=0,
        artifact_root=artifact_root,
    )

    assert second["snapshot_id"] == first["snapshot_id"]
    assert second["artifacts"]["snapshot"]["sha256"] == first["artifacts"]["snapshot"]["sha256"]
    assert second == first


def test_splits_are_chronological_with_configured_embargo(tmp_path: Path) -> None:
    db_path = _create_db(tmp_path, _measurements(16))
    manifest = create_snapshot(
        db_path=db_path,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=3600,
        artifact_root=tmp_path / "artifacts",
    )

    assert manifest["row_counts"] == {
        "total": 16,
        "exploration": 7,
        "tuning": 2,
        "blind": 3,
        "embargoed": 4,
    }
    partitions = manifest["split_boundaries"]["partitions"]
    assert partitions["exploration"]["end_utc"] == "2026-06-01T06:00:00Z"
    assert partitions["tuning"] == {
        "start_utc": "2026-06-01T09:00:00Z",
        "end_utc": "2026-06-01T10:00:00Z",
        "row_count": 2,
    }
    assert partitions["blind"]["start_utc"] == "2026-06-01T13:00:00Z"
    for boundary in manifest["split_boundaries"]["boundaries"]:
        assert boundary["embargo_start_utc"] < boundary["boundary_utc"] < boundary["embargo_end_utc"]


def test_missing_observations_are_not_forward_filled(tmp_path: Path) -> None:
    rows = [
        Measurement(START, "cpu_percent", 10.0, "percent", "test"),
        Measurement(START, "tcp_latency_ms", 20.0, "ms", "test"),
        Measurement(START + timedelta(hours=1), "cpu_percent", 11.0, "percent", "test"),
        Measurement(START + timedelta(hours=2), "tcp_latency_ms", 22.0, "ms", "test"),
        Measurement(START + timedelta(hours=3), "cpu_percent", 13.0, "percent", "test"),
        Measurement(START + timedelta(hours=4), "tcp_latency_ms", 24.0, "ms", "test"),
        Measurement(START + timedelta(hours=5), "cpu_percent", 15.0, "percent", "test"),
        Measurement(START + timedelta(hours=6), "tcp_latency_ms", 26.0, "ms", "test"),
    ]
    db_path = _create_db(tmp_path, rows)
    manifest = create_snapshot(
        db_path=db_path,
        hours=24,
        metrics=["cpu_percent", "tcp_latency_ms"],
        max_lag_seconds=0,
        artifact_root=tmp_path / "artifacts",
    )
    snapshot = _read_gzip_artifact(tmp_path / "artifacts", manifest["artifacts"]["snapshot"])

    by_timestamp = {row["timestamp_utc"]: row["metrics"] for row in snapshot["rows"]}
    assert "tcp_latency_ms" not in by_timestamp["2026-06-01T01:00:00Z"]
    assert "cpu_percent" not in by_timestamp["2026-06-01T02:00:00Z"]
    assert manifest["coverage"]["tcp_latency_ms"]["missing_timestamp_count"] == 3


def test_exploration_loader_does_not_return_blind_values(tmp_path: Path) -> None:
    db_path = _create_db(tmp_path, _measurements(8, start_value=100.0))
    artifact_root = tmp_path / "artifacts"
    manifest = create_snapshot(
        db_path=db_path,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=0,
        artifact_root=artifact_root,
    )

    exploration = load_exploration_view(manifest["snapshot_id"], artifact_root=artifact_root)
    serialized_exploration = json.dumps(exploration, sort_keys=True)

    assert exploration["partition"] == "exploration"
    assert "blind" not in serialized_exploration
    assert "106.0" not in serialized_exploration
    assert "107.0" not in serialized_exploration
    with pytest.raises(PermissionError):
        load_blind_view(
            manifest["snapshot_id"],
            BlindEvaluatorCapability("not-the-token"),
            artifact_root=artifact_root,
        )

    blind = load_blind_view(
        manifest["snapshot_id"],
        create_blind_evaluator_capability(),
        artifact_root=artifact_root,
    )
    assert blind["partition"] == "blind"
    assert blind["rows"][-1]["metrics"]["cpu_percent"][0]["value"] == 107.0


def test_changed_input_data_changes_snapshot_id(tmp_path: Path) -> None:
    first_db = _create_db(tmp_path / "first", _measurements(8))
    changed_rows = _measurements(8)
    changed_rows[-1] = Measurement(
        changed_rows[-1].timestamp_utc,
        changed_rows[-1].metric,
        999.0,
        changed_rows[-1].unit,
        changed_rows[-1].source,
    )
    second_db = _create_db(tmp_path / "second", changed_rows)

    first = create_snapshot(
        db_path=first_db,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=0,
        artifact_root=tmp_path / "artifacts",
    )
    second = create_snapshot(
        db_path=second_db,
        hours=24,
        metrics=["cpu_percent"],
        max_lag_seconds=0,
        artifact_root=tmp_path / "artifacts",
    )

    assert second["snapshot_id"] != first["snapshot_id"]


def test_empty_and_insufficient_data_fail_closed(tmp_path: Path) -> None:
    empty_db = _create_db(tmp_path / "empty", [])
    with pytest.raises(EmptySnapshotError):
        create_snapshot(
            db_path=empty_db,
            hours=24,
            metrics=["cpu_percent"],
            max_lag_seconds=0,
            artifact_root=tmp_path / "artifacts-empty",
        )

    small_db = _create_db(tmp_path / "small", _measurements(3))
    with pytest.raises(InsufficientSnapshotDataError):
        create_snapshot(
            db_path=small_db,
            hours=24,
            metrics=["cpu_percent"],
            max_lag_seconds=0,
            artifact_root=tmp_path / "artifacts-small",
        )

    embargoed_db = _create_db(tmp_path / "embargoed", _measurements(8))
    with pytest.raises(InsufficientSnapshotDataError):
        create_snapshot(
            db_path=embargoed_db,
            hours=24,
            metrics=["cpu_percent"],
            max_lag_seconds=7200,
            artifact_root=tmp_path / "artifacts-embargoed",
        )


def _measurements(count: int, start_value: float = 1.0) -> list[Measurement]:
    return [
        Measurement(
            START + timedelta(hours=index),
            "cpu_percent",
            start_value + index,
            "percent",
            "test",
        )
        for index in range(count)
    ]


def _create_db(tmp_path: Path, measurements: list[Measurement]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "resonance.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    insert_measurements(conn, measurements)
    conn.close()
    return db_path


def _read_gzip_artifact(root: Path, artifact: dict[str, str]) -> dict:
    content = (root / artifact["path"]).read_bytes()
    return json.loads(gzip.decompress(content).decode("utf-8"))
