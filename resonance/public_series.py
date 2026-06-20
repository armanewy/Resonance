from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from resonance.analysis.alignment import align_series
from resonance.analysis.contracts import AlignedPair
from resonance.storage import SeriesRecord
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso


@dataclass(frozen=True)
class PublicObservationRow:
    series_id: str
    valid_start_utc: datetime
    valid_end_utc: datetime
    observed_at_utc: datetime
    ingested_at_utc: datetime
    value: float
    quality: str
    source_revision: str
    source_observation_key: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PublicAlignmentResult:
    aligned_pair: AlignedPair
    public_series: SeriesRecord
    measurement_metric: str
    measurement_source: str | None
    public_provenance: tuple[Mapping[str, Any], ...]


def list_series(
    conn: sqlite3.Connection,
    *,
    source_id: str | None = None,
    geography_type: str | None = None,
    geography_id: str | None = None,
    cadence_seconds: int | None = None,
) -> tuple[SeriesRecord, ...]:
    clauses = []
    params: list[Any] = []
    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if geography_type is not None:
        clauses.append("geography_type = ?")
        params.append(geography_type)
    if geography_id is not None:
        clauses.append("geography_id = ?")
        params.append(geography_id)
    if cadence_seconds is not None:
        clauses.append("cadence_seconds = ?")
        params.append(int(cadence_seconds))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT series_id, source_id, metric_name, display_name, unit, cadence_seconds,
               aggregation, geography_type, geography_id, timezone, timestamp_semantics,
               parent_series_id, lineage_id, quality_tier, metadata_json
        FROM series_registry
        {where}
        ORDER BY source_id, geography_type, geography_id, display_name, series_id
        """,
        params,
    ).fetchall()
    return tuple(_series_from_row(row) for row in rows)


def get_series(conn: sqlite3.Connection, series_id: str) -> SeriesRecord | None:
    row = conn.execute(
        """
        SELECT series_id, source_id, metric_name, display_name, unit, cadence_seconds,
               aggregation, geography_type, geography_id, timezone, timestamp_semantics,
               parent_series_id, lineage_id, quality_tier, metadata_json
        FROM series_registry
        WHERE series_id = ?
        """,
        (series_id,),
    ).fetchone()
    return _series_from_row(row) if row else None


def is_registered_series(conn: sqlite3.Connection, series_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM series_registry WHERE series_id = ? LIMIT 1",
        (series_id,),
    ).fetchone()
    return row is not None


def fetch_series(
    conn: sqlite3.Connection,
    series_id: str,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[PublicObservationRow, ...]:
    start = ensure_utc(start_utc).replace(microsecond=0)
    end = ensure_utc(end_utc).replace(microsecond=0)
    rows = conn.execute(
        """
        SELECT o.series_id, o.valid_start_utc, o.valid_end_utc, o.observed_at_utc,
               o.ingested_at_utc, o.value, o.quality, o.source_revision,
               o.source_observation_key, o.raw_archive_sha256, o.metadata_json,
               s.source_id, s.metric_name, s.display_name, s.unit, s.geography_type,
               s.geography_id, s.lineage_id, a.path AS raw_archive_path,
               a.request_url AS raw_request_url, a.retrieved_at_utc AS raw_retrieved_at_utc
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        LEFT JOIN public_raw_archives a ON a.sha256 = o.raw_archive_sha256
        WHERE o.series_id = ?
          AND o.valid_start_utc >= ?
          AND o.valid_start_utc <= ?
          AND NOT EXISTS (
              SELECT 1
              FROM public_observations newer
              WHERE newer.series_id = o.series_id
                AND newer.source_observation_key = o.source_observation_key
                AND (
                    newer.ingested_at_utc > o.ingested_at_utc
                    OR (
                        newer.ingested_at_utc = o.ingested_at_utc
                        AND newer.observation_id > o.observation_id
                    )
                )
          )
        ORDER BY o.valid_start_utc ASC, o.source_observation_key ASC
        """,
        (series_id, to_utc_iso(start), to_utc_iso(end)),
    ).fetchall()
    return tuple(_observation_from_row(row) for row in rows)


def fetch_series_frame(
    conn: sqlite3.Connection,
    series_id: str,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.Series:
    rows = fetch_series(conn, series_id, start_utc, end_utc)
    if not rows:
        return pd.Series(dtype=float, name=series_id)
    return pd.Series(
        [row.value for row in rows],
        index=pd.DatetimeIndex([row.valid_start_utc for row in rows]),
        name=series_id,
        dtype=float,
    )


def align_public_series_with_measurement(
    conn: sqlite3.Connection,
    *,
    public_series_id: str,
    metric: str,
    start_utc: datetime,
    end_utc: datetime,
    measurement_source: str | None = None,
    cadence_seconds: int | None = None,
    min_points: int = 2,
) -> PublicAlignmentResult:
    public_series = get_series(conn, public_series_id)
    if public_series is None:
        raise ValueError(f"unknown public series: {public_series_id}")
    public_rows = fetch_series(conn, public_series_id, start_utc, end_utc)
    public_frame = pd.Series(
        [row.value for row in public_rows],
        index=pd.DatetimeIndex([row.valid_start_utc for row in public_rows]),
        name=public_series_id,
        dtype=float,
    )
    measurement_frame = _measurement_frame(
        conn,
        metric=metric,
        start_utc=start_utc,
        end_utc=end_utc,
        source=measurement_source,
    )
    if public_frame.empty:
        raise ValueError(f"no observations for public series: {public_series_id}")
    if measurement_frame.empty:
        raise ValueError(f"no measurements for metric: {metric}")
    return PublicAlignmentResult(
        aligned_pair=align_series(
            public_frame,
            measurement_frame,
            cadence_seconds=cadence_seconds,
            min_points=min_points,
        ),
        public_series=public_series,
        measurement_metric=metric,
        measurement_source=measurement_source,
        public_provenance=tuple(row.provenance for row in public_rows),
    )


def open_read_only(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _measurement_frame(
    conn: sqlite3.Connection,
    *,
    metric: str,
    start_utc: datetime,
    end_utc: datetime,
    source: str | None,
) -> pd.Series:
    params: list[Any] = [to_utc_iso(start_utc), to_utc_iso(end_utc), metric]
    source_clause = ""
    if source is not None:
        source_clause = "AND source = ?"
        params.append(source)
    rows = conn.execute(
        f"""
        SELECT timestamp_utc, value
        FROM measurements
        WHERE timestamp_utc >= ?
          AND timestamp_utc <= ?
          AND metric = ?
          {source_clause}
        ORDER BY timestamp_utc ASC, id ASC
        """,
        params,
    ).fetchall()
    timestamps: list[datetime] = []
    values: list[float] = []
    for row in rows:
        try:
            timestamps.append(parse_utc(str(row["timestamp_utc"])))
            values.append(float(row["value"]))
        except (TypeError, ValueError):
            continue
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name=metric, dtype=float)


def _series_from_row(row: sqlite3.Row) -> SeriesRecord:
    return SeriesRecord(
        series_id=str(row["series_id"]),
        source_id=str(row["source_id"]),
        metric_name=str(row["metric_name"]),
        display_name=str(row["display_name"]),
        unit=str(row["unit"]),
        cadence_seconds=int(row["cadence_seconds"]),
        aggregation=str(row["aggregation"]),
        geography_type=str(row["geography_type"]),
        geography_id=str(row["geography_id"]),
        timezone=str(row["timezone"]),
        timestamp_semantics=str(row["timestamp_semantics"]),
        parent_series_id=row["parent_series_id"],
        lineage_id=str(row["lineage_id"]),
        quality_tier=str(row["quality_tier"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _observation_from_row(row: sqlite3.Row) -> PublicObservationRow:
    provenance = {
        "source_id": row["source_id"],
        "series_id": row["series_id"],
        "metric_name": row["metric_name"],
        "display_name": row["display_name"],
        "unit": row["unit"],
        "geography_type": row["geography_type"],
        "geography_id": row["geography_id"],
        "lineage_id": row["lineage_id"],
        "raw_archive_sha256": row["raw_archive_sha256"],
        "raw_archive_path": row["raw_archive_path"],
        "raw_request_url": row["raw_request_url"],
        "raw_retrieved_at_utc": row["raw_retrieved_at_utc"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }
    return PublicObservationRow(
        series_id=str(row["series_id"]),
        valid_start_utc=parse_utc(str(row["valid_start_utc"])),
        valid_end_utc=parse_utc(str(row["valid_end_utc"])),
        observed_at_utc=parse_utc(str(row["observed_at_utc"])),
        ingested_at_utc=parse_utc(str(row["ingested_at_utc"])),
        value=float(row["value"]),
        quality=str(row["quality"]),
        source_revision=str(row["source_revision"]),
        source_observation_key=str(row["source_observation_key"]),
        provenance=provenance,
    )
