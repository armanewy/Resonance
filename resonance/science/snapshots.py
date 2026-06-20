from __future__ import annotations

import gzip
import hashlib
import io
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event
from resonance.storage import DEFAULT_DB_PATH
from resonance.time_utils import parse_utc, to_utc_iso, utc_now


DEFAULT_ARTIFACT_ROOT = Path("data/science/artifacts")
SNAPSHOT_SCHEMA_VERSION = 1
_BLIND_TOKEN_VALUE = "resonance-internal-blind-evaluator"


class SnapshotError(ValueError):
    """Base class for snapshot creation failures."""


class EmptySnapshotError(SnapshotError):
    """Raised when no selected measurements are available."""


class InsufficientSnapshotDataError(SnapshotError):
    """Raised when chronological splits cannot satisfy the seal."""


@dataclass(frozen=True)
class BlindEvaluatorCapability:
    _token: str


def create_blind_evaluator_capability() -> BlindEvaluatorCapability:
    return BlindEvaluatorCapability(_BLIND_TOKEN_VALUE)


def create_snapshot(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    hours: int,
    metrics: Sequence[str],
    max_lag_seconds: int,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_metrics = _normalize_metrics(metrics)
    if hours <= 0:
        raise SnapshotError("hours must be positive")
    if max_lag_seconds < 0:
        raise SnapshotError("max_lag_seconds must be non-negative")

    rows = _read_snapshot_rows(db_path, hours, selected_metrics)
    split = _split_rows(rows, max_lag_seconds)
    config = {
        "hours": int(hours),
        "max_lag_seconds": int(max_lag_seconds),
        "metrics": selected_metrics,
    }
    time_range = {
        "start_utc": rows[0]["timestamp_utc"],
        "end_utc": rows[-1]["timestamp_utc"],
    }
    metric_catalog = _metric_catalog(rows, selected_metrics)
    snapshot_payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "config": config,
        "time_range_utc": time_range,
        "metric_catalog": metric_catalog,
        "rows": rows,
    }
    snapshot_bytes = _canonical_gzip_json(snapshot_payload)
    snapshot_hash = _sha256(snapshot_bytes)
    snapshot_id = snapshot_hash

    root = Path(artifact_root)
    index_path = _snapshot_index_path(root, snapshot_id)
    if index_path.exists():
        return _load_manifest(snapshot_id, root)

    artifacts: dict[str, dict[str, str]] = {}
    artifacts["snapshot"] = _store_artifact(root, snapshot_hash, "json.gz", snapshot_bytes)

    for name in ("exploration", "tuning", "blind"):
        partition_payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "partition": name,
            "rows": split["partitions"][name],
        }
        partition_bytes = _canonical_gzip_json(partition_payload)
        partition_hash = _sha256(partition_bytes)
        artifacts[name] = _store_artifact(root, partition_hash, "json.gz", partition_bytes)

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "source_database_path": _display_db_path(db_path),
        "selected_metrics": selected_metrics,
        "requested_hours": int(hours),
        "max_lag_seconds": int(max_lag_seconds),
        "embargo_seconds": int(max_lag_seconds),
        "time_range_utc": time_range,
        "row_counts": {
            "total": len(rows),
            "exploration": len(split["partitions"]["exploration"]),
            "tuning": len(split["partitions"]["tuning"]),
            "blind": len(split["partitions"]["blind"]),
            "embargoed": len(split["embargoed_indices"]),
        },
        "metric_catalog": metric_catalog,
        "coverage": _coverage(rows, selected_metrics),
        "cadence": _cadence(rows, selected_metrics),
        "split_boundaries": split["boundaries"],
        "artifacts": artifacts,
        "created_at_utc": to_utc_iso(utc_now()),
        "git_commit": _git_commit(),
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_hash = _sha256(manifest_bytes)
    manifest_ref = _store_artifact(root, manifest_hash, "json", manifest_bytes)
    _write_snapshot_index(root, snapshot_id, manifest_ref)
    if ledger_path is not None:
        append_snapshot_created_event(manifest, artifact_root=root, ledger_path=ledger_path)
    return manifest


def append_snapshot_created_event(
    manifest: dict[str, Any],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Append a ledger entry that records a created snapshot and its artifacts."""

    root = Path(artifact_root)
    artifacts = manifest["artifacts"]
    return append_event(
        "snapshot_created",
        {
            "snapshot_id": manifest["snapshot_id"],
            "artifact_root": str(root.resolve()),
            "time_range_utc": manifest["time_range_utc"],
            "split_boundaries": manifest["split_boundaries"],
            "row_counts": manifest["row_counts"],
            "metric_catalog": manifest["metric_catalog"],
            "artifacts": artifacts,
        },
        artifact_hashes={name: artifact["sha256"] for name, artifact in artifacts.items()},
        code_commit=manifest.get("git_commit"),
        ledger_path=ledger_path,
    )


def load_snapshot_manifest(
    snapshot_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    return _load_manifest(snapshot_id, Path(artifact_root))


def snapshot_summary(
    snapshot_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=artifact_root)
    blind_artifact_path = _resolve_artifact_path(
        Path(artifact_root),
        manifest["artifacts"]["blind"]["path"],
    )
    return {
        "snapshot_id": manifest["snapshot_id"],
        "git_commit": manifest["git_commit"],
        "created_at_utc": manifest["created_at_utc"],
        "time_range_utc": manifest["time_range_utc"],
        "split_ranges": manifest["split_boundaries"]["partitions"],
        "metric_catalog": manifest["metric_catalog"],
        "blind_data": {
            "available": blind_artifact_path.exists(),
            "row_count": manifest["row_counts"]["blind"],
            "values_exposed": False,
        },
    }


def verify_snapshot_artifacts(
    snapshot_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> tuple[str, ...]:
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=artifact_root)
    root = Path(artifact_root)
    errors: list[str] = []
    for name, artifact in manifest["artifacts"].items():
        path = _resolve_artifact_path(root, artifact["path"])
        if not path.exists():
            errors.append(f"{name}: missing artifact {path}")
            continue
        if _sha256(path.read_bytes()) != artifact["sha256"]:
            errors.append(f"{name}: hash mismatch for {path}")
    return tuple(errors)


def load_exploration_view(
    snapshot_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    manifest = _load_manifest(snapshot_id, Path(artifact_root))
    exploration = _read_json_gz_artifact(Path(artifact_root), manifest["artifacts"]["exploration"])
    return {
        "snapshot_id": snapshot_id,
        "partition": "exploration",
        "rows": exploration["rows"],
        "metadata": {
            "selected_metrics": manifest["selected_metrics"],
            "time_range_utc": {
                "exploration": manifest["split_boundaries"]["partitions"]["exploration"],
                "tuning": manifest["split_boundaries"]["partitions"]["tuning"],
            },
            "row_counts": {
                "exploration": manifest["row_counts"]["exploration"],
                "tuning": manifest["row_counts"]["tuning"],
            },
            "embargo_seconds": manifest["embargo_seconds"],
        },
    }


def load_blind_view(
    snapshot_id: str,
    capability: BlindEvaluatorCapability,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    if not isinstance(capability, BlindEvaluatorCapability) or capability._token != _BLIND_TOKEN_VALUE:
        raise PermissionError("blind snapshot access requires an evaluator capability")
    manifest = _load_manifest(snapshot_id, Path(artifact_root))
    blind = _read_json_gz_artifact(Path(artifact_root), manifest["artifacts"]["blind"])
    return {
        "snapshot_id": snapshot_id,
        "partition": "blind",
        "rows": blind["rows"],
        "metadata": {
            "selected_metrics": manifest["selected_metrics"],
            "time_range_utc": manifest["split_boundaries"]["partitions"]["blind"],
            "row_count": manifest["row_counts"]["blind"],
            "embargo_seconds": manifest["embargo_seconds"],
        },
    }


def parse_metric_csv(value: str) -> list[str]:
    return _normalize_metrics(value.split(","))


def _read_snapshot_rows(
    db_path: str | Path,
    hours: int,
    selected_metrics: Sequence[str],
) -> list[dict[str, Any]]:
    db = Path(db_path)
    if str(db_path) != ":memory:" and not db.exists():
        raise EmptySnapshotError(f"database not found: {db}")
    conn = _connect_read_only(db_path)
    try:
        placeholders = ",".join("?" for _ in selected_metrics)
        raw_rows = list(
            conn.execute(
                f"""
                SELECT id, timestamp_utc, metric, value, unit, source, metadata_json
                FROM measurements
                WHERE metric IN ({placeholders})
                """,
                list(selected_metrics),
            )
        )
    finally:
        conn.close()
    if not raw_rows:
        raise EmptySnapshotError("no measurements found for selected metrics")

    observations = [_observation_from_row(row) for row in raw_rows]
    end = max(parse_utc(row["timestamp_utc"]) for row in observations)
    start = end - timedelta(hours=hours)
    observations = [
        row for row in observations if start <= parse_utc(row["timestamp_utc"]) <= end
    ]
    if not observations:
        raise EmptySnapshotError("no measurements found in requested time range")

    observations.sort(
        key=lambda row: (
            row["timestamp_utc"],
            row["metric"],
            row["source"],
            row["unit"],
            row["value"],
            row["id"],
        )
    )
    grouped: dict[str, dict[str, Any]] = {}
    for observation in observations:
        timestamp = observation["timestamp_utc"]
        group = grouped.setdefault(timestamp, {"timestamp_utc": timestamp, "metrics": {}})
        metric_observations = group["metrics"].setdefault(observation["metric"], [])
        metric_observations.append(
            {
                "value": observation["value"],
                "unit": observation["unit"],
                "source": observation["source"],
                "metadata": observation["metadata"],
            }
        )
    return [grouped[timestamp] for timestamp in sorted(grouped)]


def _connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        uri_path = Path(db_path).resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _observation_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": int(row["id"]),
        "timestamp_utc": to_utc_iso(parse_utc(row["timestamp_utc"])),
        "metric": row["metric"],
        "value": float(row["value"]),
        "unit": row["unit"],
        "source": row["source"],
        "metadata": metadata,
    }


def _split_rows(rows: Sequence[dict[str, Any]], max_lag_seconds: int) -> dict[str, Any]:
    if len(rows) < 4:
        raise InsufficientSnapshotDataError("at least four timestamp rows are required")
    first_boundary_index = len(rows) // 2
    second_boundary_index = (len(rows) * 3) // 4
    if (
        first_boundary_index <= 0
        or second_boundary_index <= first_boundary_index
        or second_boundary_index >= len(rows)
    ):
        raise InsufficientSnapshotDataError("not enough rows to create sealed splits")

    timestamps = [parse_utc(row["timestamp_utc"]) for row in rows]
    raw_boundaries = [
        ("exploration_tuning", first_boundary_index),
        ("tuning_blind", second_boundary_index),
    ]
    boundary_details = []
    embargoed_indices: set[int] = set()
    for name, index in raw_boundaries:
        boundary_at = _midpoint(timestamps[index - 1], timestamps[index])
        window_start = boundary_at - timedelta(seconds=max_lag_seconds)
        window_end = boundary_at + timedelta(seconds=max_lag_seconds)
        for row_index, timestamp in enumerate(timestamps):
            if window_start <= timestamp <= window_end:
                embargoed_indices.add(row_index)
        boundary_details.append(
            {
                "name": name,
                "raw_index": index,
                "boundary_utc": to_utc_iso(boundary_at),
                "left_timestamp_utc": rows[index - 1]["timestamp_utc"],
                "right_timestamp_utc": rows[index]["timestamp_utc"],
                "embargo_start_utc": to_utc_iso(window_start),
                "embargo_end_utc": to_utc_iso(window_end),
            }
        )

    partitions = {"exploration": [], "tuning": [], "blind": []}
    for index, row in enumerate(rows):
        if index in embargoed_indices:
            continue
        if index < first_boundary_index:
            partitions["exploration"].append(row)
        elif index < second_boundary_index:
            partitions["tuning"].append(row)
        else:
            partitions["blind"].append(row)

    if any(not partitions[name] for name in partitions):
        raise InsufficientSnapshotDataError(
            "embargo leaves too little data for one or more partitions"
        )

    return {
        "partitions": partitions,
        "embargoed_indices": sorted(embargoed_indices),
        "boundaries": {
            "strategy": "chronological_50_25_25",
            "embargo_seconds": int(max_lag_seconds),
            "boundaries": boundary_details,
            "partitions": {
                name: _partition_range(partition) for name, partition in partitions.items()
            },
        },
    }


def _midpoint(left: datetime, right: datetime) -> datetime:
    return left + (right - left) / 2


def _partition_range(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"start_utc": None, "end_utc": None, "row_count": 0}
    return {
        "start_utc": rows[0]["timestamp_utc"],
        "end_utc": rows[-1]["timestamp_utc"],
        "row_count": len(rows),
    }


def _coverage(rows: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    total_timestamps = len(rows)
    result: dict[str, Any] = {}
    for metric in metrics:
        present_rows = [row for row in rows if metric in row["metrics"]]
        sample_count = sum(len(row["metrics"][metric]) for row in present_rows)
        result[metric] = {
            "sample_count": sample_count,
            "observed_timestamp_count": len(present_rows),
            "missing_timestamp_count": total_timestamps - len(present_rows),
            "coverage_fraction": (
                round(len(present_rows) / total_timestamps, 6) if total_timestamps else 0.0
            ),
            "first_timestamp_utc": present_rows[0]["timestamp_utc"] if present_rows else None,
            "last_timestamp_utc": present_rows[-1]["timestamp_utc"] if present_rows else None,
        }
    return result


def _cadence(rows: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    result = {"all_timestamps": _cadence_for_timestamps(row["timestamp_utc"] for row in rows)}
    for metric in metrics:
        result[metric] = _cadence_for_timestamps(
            row["timestamp_utc"] for row in rows if metric in row["metrics"]
        )
    return result


def _cadence_for_timestamps(timestamps: Iterable[str]) -> dict[str, Any]:
    parsed = [parse_utc(timestamp) for timestamp in timestamps]
    if len(parsed) < 2:
        return {"median_seconds": None, "min_seconds": None, "max_seconds": None}
    deltas = [
        int((right - left).total_seconds())
        for left, right in zip(parsed, parsed[1:], strict=False)
    ]
    return {
        "median_seconds": float(median(deltas)),
        "min_seconds": min(deltas),
        "max_seconds": max(deltas),
    }


def _metric_catalog(rows: Sequence[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    coverage = _coverage(rows, metrics)
    cadence = _cadence(rows, metrics)
    catalog_metrics: list[dict[str, Any]] = []
    for metric in metrics:
        units = sorted(
            {
                observation["unit"]
                for row in rows
                for observation in row["metrics"].get(metric, [])
            }
        )
        sources = sorted(
            {
                observation["source"]
                for row in rows
                for observation in row["metrics"].get(metric, [])
            }
        )
        catalog_metrics.append(
            {
                "name": metric,
                "units": units,
                "sources": sources,
                "coverage": coverage[metric],
                "cadence": cadence[metric],
            }
        )
    catalog = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "metric_names": list(metrics),
        "metrics": catalog_metrics,
    }
    return {**catalog, "catalog_id": _sha256(_canonical_json(catalog))}


def _store_artifact(root: Path, digest: str, extension: str, content: bytes) -> dict[str, str]:
    path = _artifact_path(root, digest, extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise SnapshotError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": _relative_artifact_path(digest, extension), "format": extension}


def _artifact_path(root: Path, digest: str, extension: str) -> Path:
    _validate_hash(digest, "artifact digest")
    if not extension or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in extension.lower()):
        raise SnapshotError("invalid artifact extension")
    return root / "sha256" / digest[:2] / f"{digest}.{extension}"


def _relative_artifact_path(digest: str, extension: str) -> str:
    return f"sha256/{digest[:2]}/{digest}.{extension}"


def _snapshot_index_path(root: Path, snapshot_id: str) -> Path:
    _validate_hash(snapshot_id, "snapshot id")
    return root / "snapshots" / f"{snapshot_id}.json"


def _write_snapshot_index(root: Path, snapshot_id: str, manifest_ref: dict[str, str]) -> None:
    index_path = _snapshot_index_path(root, snapshot_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_payload = _canonical_json({"snapshot_id": snapshot_id, "manifest": manifest_ref})
    if index_path.exists():
        if index_path.read_bytes() != index_payload:
            raise SnapshotError(f"snapshot index already exists: {index_path}")
        return
    index_path.write_bytes(index_payload)


def _load_manifest(snapshot_id: str, root: Path) -> dict[str, Any]:
    _validate_hash(snapshot_id, "snapshot id")
    index_path = _snapshot_index_path(root, snapshot_id)
    if not index_path.exists():
        raise FileNotFoundError(f"snapshot index not found: {snapshot_id}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SnapshotError(f"invalid snapshot index for {snapshot_id}: {exc}") from exc
    if index.get("snapshot_id") != snapshot_id:
        raise SnapshotError("snapshot index identity does not match requested snapshot")
    manifest_ref = index.get("manifest")
    if not isinstance(manifest_ref, dict):
        raise SnapshotError("snapshot index lacks a manifest reference")
    _validate_hash(str(manifest_ref.get("sha256", "")), "manifest digest")
    manifest_path = _resolve_artifact_path(root, str(manifest_ref.get("path", "")))
    content = manifest_path.read_bytes()
    if _sha256(content) != manifest_ref["sha256"]:
        raise SnapshotError(f"manifest hash mismatch for snapshot {snapshot_id}")
    try:
        manifest = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid manifest for snapshot {snapshot_id}: {exc}") from exc
    if manifest.get("snapshot_id") != snapshot_id:
        raise SnapshotError("manifest identity does not match requested snapshot")
    return manifest


def _read_json_gz_artifact(root: Path, artifact: dict[str, str]) -> dict[str, Any]:
    _validate_hash(str(artifact.get("sha256", "")), "artifact digest")
    path = _resolve_artifact_path(root, str(artifact.get("path", "")))
    content = path.read_bytes()
    if _sha256(content) != artifact["sha256"]:
        raise SnapshotError(f"artifact hash mismatch for {path}")
    try:
        return json.loads(gzip.decompress(content).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid compressed artifact {path}: {exc}") from exc


def _resolve_artifact_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise SnapshotError("artifact path is empty")
    resolved_root = root.resolve()
    path = (resolved_root / relative_path).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise SnapshotError("artifact path escapes artifact root")
    return path


def _validate_hash(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SnapshotError(f"{label} must be a 64-character lowercase hex digest")


def _canonical_gzip_json(payload: dict[str, Any]) -> bytes:
    raw = _canonical_json(payload)
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
        gz.write(raw)
    return output.getvalue()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalize_metrics(metrics: Sequence[str]) -> list[str]:
    selected = sorted({metric.strip() for metric in metrics if metric.strip()})
    if not selected:
        raise SnapshotError("at least one metric is required")
    return selected


def _display_db_path(db_path: str | Path) -> str:
    if str(db_path) == ":memory:":
        return ":memory:"
    return str(Path(db_path).resolve())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None
