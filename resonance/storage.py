from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from resonance.time_utils import parse_utc, to_utc_iso


DEFAULT_DB_PATH = Path("data/resonance.db")


@dataclass(frozen=True)
class Measurement:
    timestamp_utc: datetime
    metric: str
    value: float
    unit: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CollectorError:
    timestamp_utc: datetime
    collector: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EventMarker:
    timestamp_utc: datetime
    label: str
    note: str = ""
    created_at_utc: datetime | None = None


@dataclass(frozen=True)
class CorrelationFinding:
    x_metric: str
    y_metric: str
    transform: str
    lag_seconds: int
    discovery_rho: float
    holdout_rho: float
    corrected_q: float
    stability: float
    overlap_count: int
    first_seen_utc: datetime
    last_verified_utc: datetime
    status: str
    evidence: dict


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            source TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collector_errors (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            collector TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            label TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS correlation_findings (
            finding_id INTEGER PRIMARY KEY,
            x_metric TEXT NOT NULL,
            y_metric TEXT NOT NULL,
            transform TEXT NOT NULL,
            lag_seconds INTEGER NOT NULL,
            discovery_rho REAL NOT NULL,
            holdout_rho REAL NOT NULL,
            corrected_q REAL NOT NULL,
            stability REAL NOT NULL,
            overlap_count INTEGER NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_verified_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurements_metric_timestamp
        ON measurements(metric, timestamp_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_measurements_source_timestamp
        ON measurements(source, timestamp_utc)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_measurements_open_meteo_unique
        ON measurements(source, metric, timestamp_utc)
        WHERE source = 'open-meteo'
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_collector_errors_timestamp
        ON collector_errors(timestamp_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp_utc)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_correlation_findings_identity
        ON correlation_findings(x_metric, y_metric, transform)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_correlation_findings_status
        ON correlation_findings(status, last_verified_utc)
        """
    )
    conn.commit()


def ensure_database(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = connect(db_path)
    init_db(conn)
    return conn


def insert_measurements(conn: sqlite3.Connection, measurements: Iterable[Measurement]) -> int:
    inserted = 0
    for measurement in measurements:
        metadata_json = json.dumps(measurement.metadata, sort_keys=True, separators=(",", ":"))
        try:
            conn.execute(
                """
                INSERT INTO measurements (
                    timestamp_utc, metric, value, unit, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    to_utc_iso(measurement.timestamp_utc),
                    measurement.metric,
                    float(measurement.value),
                    measurement.unit,
                    measurement.source,
                    metadata_json,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    return inserted


def insert_collector_error(conn: sqlite3.Connection, error: CollectorError) -> None:
    conn.execute(
        """
        INSERT INTO collector_errors (timestamp_utc, collector, error_type, message)
        VALUES (?, ?, ?, ?)
        """,
        (to_utc_iso(error.timestamp_utc), error.collector, error.error_type, error.message),
    )
    conn.commit()


def insert_collector_errors(conn: sqlite3.Connection, errors: Iterable[CollectorError]) -> int:
    count = 0
    for error in errors:
        conn.execute(
            """
            INSERT INTO collector_errors (timestamp_utc, collector, error_type, message)
            VALUES (?, ?, ?, ?)
            """,
            (to_utc_iso(error.timestamp_utc), error.collector, error.error_type, error.message),
        )
        count += 1
    conn.commit()
    return count


def insert_event_marker(conn: sqlite3.Connection, event: EventMarker) -> int:
    label = event.label.strip()
    if not label:
        raise ValueError("Event label is required.")
    note = event.note.strip()
    created_at_utc = event.created_at_utc or event.timestamp_utc
    cursor = conn.execute(
        """
        INSERT INTO events (timestamp_utc, label, note, created_at_utc)
        VALUES (?, ?, ?, ?)
        """,
        (to_utc_iso(event.timestamp_utc), label, note, to_utc_iso(created_at_utc)),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_event_markers(conn: sqlite3.Connection, limit: int | None = 20) -> list[sqlite3.Row]:
    if limit is None:
        return list(
            conn.execute(
                """
                SELECT id, timestamp_utc, label, note, created_at_utc
                FROM events
                ORDER BY timestamp_utc DESC, id DESC
                """
            )
        )
    return list(
        conn.execute(
            """
            SELECT id, timestamp_utc, label, note, created_at_utc
            FROM events
            ORDER BY timestamp_utc DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
    )


def upsert_correlation_findings(
    conn: sqlite3.Connection,
    findings: Iterable[CorrelationFinding],
) -> int:
    count = 0
    for finding in findings:
        evidence_json = json.dumps(finding.evidence, sort_keys=True, separators=(",", ":"))
        conn.execute(
            """
            INSERT INTO correlation_findings (
                x_metric,
                y_metric,
                transform,
                lag_seconds,
                discovery_rho,
                holdout_rho,
                corrected_q,
                stability,
                overlap_count,
                first_seen_utc,
                last_verified_utc,
                status,
                evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(x_metric, y_metric, transform) DO UPDATE SET
                lag_seconds = excluded.lag_seconds,
                discovery_rho = excluded.discovery_rho,
                holdout_rho = excluded.holdout_rho,
                corrected_q = excluded.corrected_q,
                stability = excluded.stability,
                overlap_count = excluded.overlap_count,
                last_verified_utc = excluded.last_verified_utc,
                status = excluded.status,
                evidence_json = excluded.evidence_json
            """,
            (
                finding.x_metric,
                finding.y_metric,
                finding.transform,
                int(finding.lag_seconds),
                float(finding.discovery_rho),
                float(finding.holdout_rho),
                float(finding.corrected_q),
                float(finding.stability),
                int(finding.overlap_count),
                to_utc_iso(finding.first_seen_utc),
                to_utc_iso(finding.last_verified_utc),
                finding.status,
                evidence_json,
            ),
        )
        count += 1
    conn.commit()
    return count


def fetch_correlation_findings(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
) -> list[sqlite3.Row]:
    if status is None:
        return list(
            conn.execute(
                """
                SELECT finding_id, x_metric, y_metric, transform, lag_seconds,
                       discovery_rho, holdout_rho, corrected_q, stability,
                       overlap_count, first_seen_utc, last_verified_utc,
                       status, evidence_json
                FROM correlation_findings
                ORDER BY corrected_q ASC, ABS(holdout_rho) DESC, finding_id ASC
                """
            )
        )
    return list(
        conn.execute(
            """
            SELECT finding_id, x_metric, y_metric, transform, lag_seconds,
                   discovery_rho, holdout_rho, corrected_q, stability,
                   overlap_count, first_seen_utc, last_verified_utc,
                   status, evidence_json
            FROM correlation_findings
            WHERE status = ?
            ORDER BY corrected_q ASC, ABS(holdout_rho) DESC, finding_id ASC
            """,
            (status,),
        )
    )


def correlation_finding_from_row(row: sqlite3.Row) -> CorrelationFinding:
    return CorrelationFinding(
        x_metric=row["x_metric"],
        y_metric=row["y_metric"],
        transform=row["transform"],
        lag_seconds=int(row["lag_seconds"]),
        discovery_rho=float(row["discovery_rho"]),
        holdout_rho=float(row["holdout_rho"]),
        corrected_q=float(row["corrected_q"]),
        stability=float(row["stability"]),
        overlap_count=int(row["overlap_count"]),
        first_seen_utc=parse_utc(row["first_seen_utc"]),
        last_verified_utc=parse_utc(row["last_verified_utc"]),
        status=row["status"],
        evidence=json.loads(row["evidence_json"] or "{}"),
    )


def fetch_measurements(
    conn: sqlite3.Connection,
    start_utc: datetime,
    end_utc: datetime,
    metrics: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    params: list[str] = [to_utc_iso(start_utc), to_utc_iso(end_utc)]
    metric_filter = ""
    if metrics:
        placeholders = ",".join("?" for _ in metrics)
        metric_filter = f" AND metric IN ({placeholders})"
        params.extend(metrics)
    return list(
        conn.execute(
            f"""
            SELECT id, timestamp_utc, metric, value, unit, source, metadata_json
            FROM measurements
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?{metric_filter}
            ORDER BY timestamp_utc ASC, metric ASC
            """,
            params,
        )
    )


def latest_timestamp_by_source(conn: sqlite3.Connection, source: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(timestamp_utc) AS timestamp_utc FROM measurements WHERE source = ?",
        (source,),
    ).fetchone()
    return row["timestamp_utc"] if row and row["timestamp_utc"] else None


def latest_measurement_by_metric(conn: sqlite3.Connection, metric: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT timestamp_utc, metric, value, unit, source, metadata_json
        FROM measurements
        WHERE metric = ?
        ORDER BY timestamp_utc DESC, id DESC
        LIMIT 1
        """,
        (metric,),
    ).fetchone()


def sample_counts_by_metric(
    conn: sqlite3.Connection, start_utc: datetime, end_utc: datetime
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT metric, source, COUNT(*) AS sample_count
            FROM measurements
            WHERE timestamp_utc >= ? AND timestamp_utc <= ?
            GROUP BY metric, source
            ORDER BY metric, source
            """,
            (to_utc_iso(start_utc), to_utc_iso(end_utc)),
        )
    )


def recent_errors(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT timestamp_utc, collector, error_type, message
            FROM collector_errors
            ORDER BY timestamp_utc DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
    )


def delete_measurements_by_source(conn: sqlite3.Connection, source: str) -> int:
    cursor = conn.execute("DELETE FROM measurements WHERE source = ?", (source,))
    conn.commit()
    return int(cursor.rowcount)

