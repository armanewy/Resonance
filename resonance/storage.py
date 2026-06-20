from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class PublicSource:
    source_id: str
    display_name: str
    publisher: str
    documentation_reference: str
    license_summary: str
    authentication_type: str
    default_polling_cadence_seconds: int
    quality_tier: str
    enabled: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesRecord:
    series_id: str
    source_id: str
    metric_name: str
    display_name: str
    unit: str
    cadence_seconds: int
    aggregation: str
    geography_type: str
    geography_id: str
    timezone: str
    timestamp_semantics: str
    parent_series_id: str | None
    lineage_id: str
    quality_tier: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicObservation:
    series_id: str
    valid_start_utc: datetime
    valid_end_utc: datetime
    observed_at_utc: datetime
    ingested_at_utc: datetime
    value: float
    quality: str
    source_revision: str
    source_observation_key: str
    raw_archive_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicRawArchive:
    sha256: str
    source_id: str
    retrieved_at_utc: datetime
    request_url: str
    status_code: int | None
    path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicFetchEvent:
    source_id: str
    retrieved_at_utc: datetime
    request_url: str
    status_code: int | None
    content_sha256: str
    route: str
    page_offset: int
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicCollectionState:
    source_id: str
    route: str
    last_successful_poll_utc: datetime | None = None
    newest_complete_valid_period_utc: datetime | None = None
    earliest_unresolved_gap_utc: datetime | None = None
    latest_error_utc: datetime | None = None
    latest_error: str = ""
    consecutive_failure_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
        CREATE TABLE IF NOT EXISTS public_sources (
            source_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            publisher TEXT NOT NULL,
            documentation_reference TEXT NOT NULL,
            license_summary TEXT NOT NULL,
            authentication_type TEXT NOT NULL,
            default_polling_cadence_seconds INTEGER NOT NULL,
            quality_tier TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS series_registry (
            series_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            unit TEXT NOT NULL,
            cadence_seconds INTEGER NOT NULL,
            aggregation TEXT NOT NULL,
            geography_type TEXT NOT NULL,
            geography_id TEXT NOT NULL,
            timezone TEXT NOT NULL,
            timestamp_semantics TEXT NOT NULL,
            parent_series_id TEXT,
            lineage_id TEXT NOT NULL,
            quality_tier TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(source_id) REFERENCES public_sources(source_id),
            FOREIGN KEY(parent_series_id) REFERENCES series_registry(series_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS measurement_series_map (
            source TEXT NOT NULL,
            metric TEXT NOT NULL,
            series_id TEXT NOT NULL,
            PRIMARY KEY(source, metric),
            FOREIGN KEY(series_id) REFERENCES series_registry(series_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public_raw_archives (
            sha256 TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            retrieved_at_utc TEXT NOT NULL,
            request_url TEXT NOT NULL,
            status_code INTEGER,
            path TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(source_id) REFERENCES public_sources(source_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public_fetch_events (
            fetch_id INTEGER PRIMARY KEY,
            source_id TEXT NOT NULL,
            retrieved_at_utc TEXT NOT NULL,
            request_url TEXT NOT NULL,
            status_code INTEGER,
            content_sha256 TEXT NOT NULL,
            route TEXT NOT NULL,
            page_offset INTEGER NOT NULL,
            request_metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(source_id) REFERENCES public_sources(source_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public_observations (
            observation_id INTEGER PRIMARY KEY,
            series_id TEXT NOT NULL,
            valid_start_utc TEXT NOT NULL,
            valid_end_utc TEXT NOT NULL,
            observed_at_utc TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            value REAL NOT NULL,
            quality TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            source_observation_key TEXT NOT NULL,
            raw_archive_sha256 TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(series_id) REFERENCES series_registry(series_id),
            FOREIGN KEY(raw_archive_sha256) REFERENCES public_raw_archives(sha256),
            UNIQUE(series_id, source_observation_key, source_revision)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public_collection_state (
            source_id TEXT NOT NULL,
            route TEXT NOT NULL,
            last_successful_poll_utc TEXT,
            newest_complete_valid_period_utc TEXT,
            earliest_unresolved_gap_utc TEXT,
            latest_error_utc TEXT,
            latest_error TEXT NOT NULL DEFAULT '',
            consecutive_failure_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(source_id, route),
            FOREIGN KEY(source_id) REFERENCES public_sources(source_id)
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_series_registry_source_geography
        ON series_registry(source_id, geography_type, geography_id, cadence_seconds)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_public_observations_series_valid
        ON public_observations(series_id, valid_start_utc, valid_end_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_public_observations_key_ingested
        ON public_observations(series_id, source_observation_key, ingested_at_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_public_raw_archives_source_time
        ON public_raw_archives(source_id, retrieved_at_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_public_fetch_events_source_time
        ON public_fetch_events(source_id, retrieved_at_utc)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_public_collection_state_source
        ON public_collection_state(source_id, route)
        """
    )
    _ensure_existing_measurement_series(conn)
    conn.commit()


def ensure_database(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = connect(db_path)
    init_db(conn)
    return conn


def insert_measurements(conn: sqlite3.Connection, measurements: Iterable[Measurement]) -> int:
    pending = tuple(measurements)
    _ensure_measurement_series(conn, pending)
    inserted = 0
    for measurement in pending:
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


def upsert_public_source(conn: sqlite3.Connection, source: PublicSource) -> None:
    metadata_json = _json_dumps(source.metadata)
    conn.execute(
        """
        INSERT INTO public_sources (
            source_id,
            display_name,
            publisher,
            documentation_reference,
            license_summary,
            authentication_type,
            default_polling_cadence_seconds,
            quality_tier,
            enabled,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            display_name = excluded.display_name,
            publisher = excluded.publisher,
            documentation_reference = excluded.documentation_reference,
            license_summary = excluded.license_summary,
            authentication_type = excluded.authentication_type,
            default_polling_cadence_seconds = excluded.default_polling_cadence_seconds,
            quality_tier = excluded.quality_tier,
            enabled = excluded.enabled,
            metadata_json = excluded.metadata_json
        """,
        (
            source.source_id,
            source.display_name,
            source.publisher,
            source.documentation_reference,
            source.license_summary,
            source.authentication_type,
            int(source.default_polling_cadence_seconds),
            source.quality_tier,
            1 if source.enabled else 0,
            metadata_json,
        ),
    )


def upsert_series_record(conn: sqlite3.Connection, series: SeriesRecord) -> None:
    metadata_json = _json_dumps(series.metadata)
    conn.execute(
        """
        INSERT INTO series_registry (
            series_id,
            source_id,
            metric_name,
            display_name,
            unit,
            cadence_seconds,
            aggregation,
            geography_type,
            geography_id,
            timezone,
            timestamp_semantics,
            parent_series_id,
            lineage_id,
            quality_tier,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(series_id) DO UPDATE SET
            source_id = excluded.source_id,
            metric_name = excluded.metric_name,
            display_name = excluded.display_name,
            unit = excluded.unit,
            cadence_seconds = excluded.cadence_seconds,
            aggregation = excluded.aggregation,
            geography_type = excluded.geography_type,
            geography_id = excluded.geography_id,
            timezone = excluded.timezone,
            timestamp_semantics = excluded.timestamp_semantics,
            parent_series_id = excluded.parent_series_id,
            lineage_id = excluded.lineage_id,
            quality_tier = excluded.quality_tier,
            metadata_json = excluded.metadata_json
        """,
        (
            series.series_id,
            series.source_id,
            series.metric_name,
            series.display_name,
            series.unit,
            int(series.cadence_seconds),
            series.aggregation,
            series.geography_type,
            series.geography_id,
            series.timezone,
            series.timestamp_semantics,
            series.parent_series_id,
            series.lineage_id,
            series.quality_tier,
            metadata_json,
        ),
    )


def map_measurement_series(conn: sqlite3.Connection, *, source: str, metric: str, series_id: str) -> None:
    conn.execute(
        """
        INSERT INTO measurement_series_map (source, metric, series_id)
        VALUES (?, ?, ?)
        ON CONFLICT(source, metric) DO UPDATE SET series_id = excluded.series_id
        """,
        (source, metric, series_id),
    )


def record_public_raw_archive(conn: sqlite3.Connection, archive: PublicRawArchive) -> None:
    conn.execute(
        """
        INSERT INTO public_raw_archives (
            sha256, source_id, retrieved_at_utc, request_url, status_code, path, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sha256) DO NOTHING
        """,
        (
            archive.sha256,
            archive.source_id,
            to_utc_iso(archive.retrieved_at_utc),
            archive.request_url,
            archive.status_code,
            archive.path,
            _json_dumps(archive.metadata),
        ),
    )


def record_public_fetch_event(conn: sqlite3.Connection, event: PublicFetchEvent) -> int:
    cursor = conn.execute(
        """
        INSERT INTO public_fetch_events (
            source_id,
            retrieved_at_utc,
            request_url,
            status_code,
            content_sha256,
            route,
            page_offset,
            request_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.source_id,
            to_utc_iso(event.retrieved_at_utc),
            event.request_url,
            event.status_code,
            event.content_sha256,
            event.route,
            int(event.page_offset),
            _json_dumps(event.request_metadata),
        ),
    )
    return int(cursor.lastrowid)


def upsert_public_collection_state(conn: sqlite3.Connection, state: PublicCollectionState) -> None:
    conn.execute(
        """
        INSERT INTO public_collection_state (
            source_id,
            route,
            last_successful_poll_utc,
            newest_complete_valid_period_utc,
            earliest_unresolved_gap_utc,
            latest_error_utc,
            latest_error,
            consecutive_failure_count,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, route) DO UPDATE SET
            last_successful_poll_utc = excluded.last_successful_poll_utc,
            newest_complete_valid_period_utc = excluded.newest_complete_valid_period_utc,
            earliest_unresolved_gap_utc = excluded.earliest_unresolved_gap_utc,
            latest_error_utc = excluded.latest_error_utc,
            latest_error = excluded.latest_error,
            consecutive_failure_count = excluded.consecutive_failure_count,
            metadata_json = excluded.metadata_json
        """,
        (
            state.source_id,
            state.route,
            to_utc_iso(state.last_successful_poll_utc) if state.last_successful_poll_utc else None,
            to_utc_iso(state.newest_complete_valid_period_utc) if state.newest_complete_valid_period_utc else None,
            to_utc_iso(state.earliest_unresolved_gap_utc) if state.earliest_unresolved_gap_utc else None,
            to_utc_iso(state.latest_error_utc) if state.latest_error_utc else None,
            state.latest_error,
            int(state.consecutive_failure_count),
            _json_dumps(state.metadata),
        ),
    )


def fetch_public_collection_state(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    route: str,
) -> PublicCollectionState | None:
    row = conn.execute(
        """
        SELECT source_id, route, last_successful_poll_utc, newest_complete_valid_period_utc,
               earliest_unresolved_gap_utc, latest_error_utc, latest_error,
               consecutive_failure_count, metadata_json
        FROM public_collection_state
        WHERE source_id = ? AND route = ?
        """,
        (source_id, route),
    ).fetchone()
    if row is None:
        return None
    return PublicCollectionState(
        source_id=row["source_id"],
        route=row["route"],
        last_successful_poll_utc=parse_utc(row["last_successful_poll_utc"]) if row["last_successful_poll_utc"] else None,
        newest_complete_valid_period_utc=parse_utc(row["newest_complete_valid_period_utc"]) if row["newest_complete_valid_period_utc"] else None,
        earliest_unresolved_gap_utc=parse_utc(row["earliest_unresolved_gap_utc"]) if row["earliest_unresolved_gap_utc"] else None,
        latest_error_utc=parse_utc(row["latest_error_utc"]) if row["latest_error_utc"] else None,
        latest_error=row["latest_error"] or "",
        consecutive_failure_count=int(row["consecutive_failure_count"] or 0),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def insert_public_observations(
    conn: sqlite3.Connection,
    observations: Iterable[PublicObservation],
) -> int:
    inserted = 0
    for observation in observations:
        try:
            conn.execute(
                """
                INSERT INTO public_observations (
                    series_id,
                    valid_start_utc,
                    valid_end_utc,
                    observed_at_utc,
                    ingested_at_utc,
                    value,
                    quality,
                    source_revision,
                    source_observation_key,
                    raw_archive_sha256,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.series_id,
                    to_utc_iso(observation.valid_start_utc),
                    to_utc_iso(observation.valid_end_utc),
                    to_utc_iso(observation.observed_at_utc),
                    to_utc_iso(observation.ingested_at_utc),
                    float(observation.value),
                    observation.quality,
                    observation.source_revision,
                    observation.source_observation_key,
                    observation.raw_archive_sha256,
                    _json_dumps(observation.metadata),
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


def _ensure_existing_measurement_series(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT source, metric, unit
        FROM measurements
        GROUP BY source, metric
        ORDER BY source, metric
        """
    ).fetchall()
    _ensure_measurement_series(
        conn,
        (
            Measurement(
                timestamp_utc=parse_utc("1970-01-01T00:00:00Z"),
                metric=str(row["metric"]),
                value=0.0,
                unit=str(row["unit"]),
                source=str(row["source"]),
            )
            for row in rows
        ),
    )


def _ensure_measurement_series(
    conn: sqlite3.Connection,
    measurements: Iterable[Measurement],
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for measurement in measurements:
        key = (measurement.source, measurement.metric, measurement.unit)
        if key in seen:
            continue
        seen.add(key)
        source = _measurement_source_record(measurement.source)
        series = _measurement_series_record(
            source=measurement.source,
            metric=measurement.metric,
            unit=measurement.unit,
        )
        upsert_public_source(conn, source)
        upsert_series_record(conn, series)
        map_measurement_series(conn, source=measurement.source, metric=measurement.metric, series_id=series.series_id)


def _measurement_source_record(source: str) -> PublicSource:
    source_id = _safe_identity(source)
    if source == "personal":
        return PublicSource(
            source_id=source_id,
            display_name="Local personal collector",
            publisher="Resonance local collector",
            documentation_reference="AGENTS.md#Repository Layout",
            license_summary="Private local measurements; no public license.",
            authentication_type="none",
            default_polling_cadence_seconds=30,
            quality_tier="local",
            enabled=True,
        )
    if source == "open-meteo":
        return PublicSource(
            source_id=source_id,
            display_name="Open-Meteo local weather",
            publisher="Open-Meteo",
            documentation_reference="https://open-meteo.com/en/docs",
            license_summary="Open-Meteo public API terms; see upstream documentation.",
            authentication_type="none",
            default_polling_cadence_seconds=900,
            quality_tier="public_reference",
            enabled=True,
        )
    return PublicSource(
        source_id=source_id,
        display_name=source,
        publisher=source,
        documentation_reference="local measurement compatibility mapping",
        license_summary="Unknown local measurement provenance.",
        authentication_type="unknown",
        default_polling_cadence_seconds=0,
        quality_tier="compatibility",
        enabled=True,
    )


def _measurement_series_record(*, source: str, metric: str, unit: str) -> SeriesRecord:
    source_id = _safe_identity(source)
    series_id = f"measurement:{source_id}:{_safe_identity(metric)}"
    geography_type = "device" if source == "personal" else "configured_location"
    geography_id = "local_machine" if source == "personal" else "configured_location"
    cadence_seconds = 900 if source == "open-meteo" else 30 if source == "personal" else 0
    return SeriesRecord(
        series_id=series_id,
        source_id=source_id,
        metric_name=metric,
        display_name=metric,
        unit=unit,
        cadence_seconds=cadence_seconds,
        aggregation="instantaneous",
        geography_type=geography_type,
        geography_id=geography_id,
        timezone="configured",
        timestamp_semantics="measurement_timestamp",
        parent_series_id=None,
        lineage_id=series_id,
        quality_tier="local" if source == "personal" else "public_reference" if source == "open-meteo" else "compatibility",
        metadata={"compatibility_measurement_source": source},
    )


def _safe_identity(value: str) -> str:
    cleaned = []
    for char in value.strip().lower():
        if char.isalnum() or char in ("_", "-", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
