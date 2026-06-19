from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from resonance.storage import CorrelationFinding
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


LIFECYCLE_STATUSES = ("new", "verified", "strengthened", "weakened", "broken", "expired")
DEFAULT_LAG_TOLERANCE_SECONDS = 300
DEFAULT_COEFFICIENT_EPSILON = 0.05
DEFAULT_BROKEN_AFTER_FAILURES = 2
DEFAULT_EXPIRE_AFTER = timedelta(hours=24)


@dataclass(frozen=True)
class FindingIdentity:
    x_metric: str
    y_metric: str
    transform: str
    lag_seconds: int | None = None


@dataclass(frozen=True)
class LifecycleOptions:
    lag_tolerance_seconds: int = DEFAULT_LAG_TOLERANCE_SECONDS
    coefficient_epsilon: float = DEFAULT_COEFFICIENT_EPSILON
    broken_after_failures: int = DEFAULT_BROKEN_AFTER_FAILURES
    expire_after: timedelta = DEFAULT_EXPIRE_AFTER


@dataclass(frozen=True)
class LifecycleEvent:
    relationship_key: str
    status: str
    event_utc: datetime
    scan_utc: datetime
    x_metric: str
    y_metric: str
    transform: str
    lag_seconds: int
    previous_status: str | None = None
    failure_count: int = 0
    missing_since_utc: datetime | None = None
    discovery_rho: float | None = None
    holdout_rho: float | None = None
    corrected_q: float | None = None
    stability: float | None = None
    overlap_count: int | None = None
    details: dict = field(default_factory=dict)
    event_id: int | None = None


@dataclass(frozen=True)
class _LifecycleState:
    relationship_key: str
    status: str
    event_utc: datetime
    x_metric: str
    y_metric: str
    transform: str
    lag_seconds: int
    failure_count: int
    missing_since_utc: datetime | None
    discovery_rho: float | None
    holdout_rho: float | None
    corrected_q: float | None
    stability: float | None
    overlap_count: int | None


def update_finding_lifecycle(
    conn: sqlite3.Connection,
    current_findings: Iterable[CorrelationFinding],
    *,
    previous_findings: Iterable[CorrelationFinding] = (),
    insufficient_data: Iterable[FindingIdentity | CorrelationFinding] = (),
    scan_utc: datetime | None = None,
    options: LifecycleOptions | None = None,
) -> tuple[LifecycleEvent, ...]:
    """Classify scanner findings and append lifecycle events.

    ``current_findings`` should be the findings promoted by the latest scan.
    ``previous_findings`` is optional context for callers that have scanner
    output but no lifecycle history yet. Relationships omitted from the current
    scan are treated as validation failures unless they are listed in
    ``insufficient_data``.
    """

    resolved_options = options or LifecycleOptions()
    _validate_options(resolved_options)
    scan_time = ensure_utc(scan_utc or utc_now()).replace(microsecond=0)
    current = tuple(current_findings)
    missing = tuple(_identity_from(item) for item in insufficient_data)

    ensure_lifecycle_schema(conn)
    states = _latest_states(conn)
    states.extend(_supplemental_states(previous_findings, states, resolved_options))

    matched_keys: set[str] = set()
    events: list[LifecycleEvent] = []

    for finding in current:
        state = _match_state(finding, states, matched_keys, resolved_options)
        if state is None:
            event = _event_from_finding(
                finding,
                relationship_key=_relationship_key(finding),
                status="new",
                previous_status=None,
                scan_utc=scan_time,
                details={"reason": "first_seen"},
            )
        else:
            matched_keys.add(state.relationship_key)
            status, details = _current_status(state, finding, resolved_options)
            event = _event_from_finding(
                finding,
                relationship_key=state.relationship_key,
                status=status,
                previous_status=state.status,
                scan_utc=scan_time,
                failure_count=0,
                details=details,
            )
        events.append(event)

    for state in states:
        if state.relationship_key in matched_keys:
            continue
        if state.status in {"broken", "expired"}:
            continue
        if _matches_any_identity(state, missing, resolved_options):
            events.append(_insufficient_data_event(state, scan_time, resolved_options))
        else:
            events.append(_validation_failure_event(state, scan_time, resolved_options))

    persisted = tuple(_insert_lifecycle_event(conn, event) for event in events)
    conn.commit()
    return persisted


def ensure_lifecycle_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS finding_lifecycle_events (
            event_id INTEGER PRIMARY KEY,
            relationship_key TEXT NOT NULL,
            event_utc TEXT NOT NULL,
            scan_utc TEXT NOT NULL,
            x_metric TEXT NOT NULL,
            y_metric TEXT NOT NULL,
            transform TEXT NOT NULL,
            lag_seconds INTEGER NOT NULL,
            previous_status TEXT,
            status TEXT NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            missing_since_utc TEXT,
            discovery_rho REAL,
            holdout_rho REAL,
            corrected_q REAL,
            stability REAL,
            overlap_count INTEGER,
            details_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_finding_lifecycle_events_relationship
        ON finding_lifecycle_events(relationship_key, event_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_finding_lifecycle_events_status
        ON finding_lifecycle_events(status, event_utc)
        """
    )


def fetch_lifecycle_events(
    conn: sqlite3.Connection,
    *,
    relationship_key: str | None = None,
) -> tuple[LifecycleEvent, ...]:
    ensure_lifecycle_schema(conn)
    if relationship_key is None:
        rows = conn.execute(
            """
            SELECT *
            FROM finding_lifecycle_events
            ORDER BY event_id ASC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM finding_lifecycle_events
            WHERE relationship_key = ?
            ORDER BY event_id ASC
            """,
            (relationship_key,),
        ).fetchall()
    return tuple(_event_from_row(row) for row in rows)


def _validate_options(options: LifecycleOptions) -> None:
    if options.lag_tolerance_seconds < 0:
        raise ValueError("lag_tolerance_seconds must be non-negative")
    if options.coefficient_epsilon < 0:
        raise ValueError("coefficient_epsilon must be non-negative")
    if options.broken_after_failures < 1:
        raise ValueError("broken_after_failures must be at least 1")
    if options.expire_after.total_seconds() < 0:
        raise ValueError("expire_after must be non-negative")


def _latest_states(conn: sqlite3.Connection) -> list[_LifecycleState]:
    rows = conn.execute(
        """
        SELECT *
        FROM finding_lifecycle_events
        ORDER BY event_id ASC
        """
    ).fetchall()
    latest: dict[str, _LifecycleState] = {}
    for row in rows:
        latest[row["relationship_key"]] = _state_from_row(row)
    return list(latest.values())


def _supplemental_states(
    previous_findings: Iterable[CorrelationFinding],
    states: Sequence[_LifecycleState],
    options: LifecycleOptions,
) -> list[_LifecycleState]:
    supplemental: list[_LifecycleState] = []
    known = list(states)
    for finding in previous_findings:
        if _match_state(finding, known, set(), options) is not None:
            continue
        state = _state_from_finding(finding)
        supplemental.append(state)
        known.append(state)
    return supplemental


def _match_state(
    finding: CorrelationFinding,
    states: Sequence[_LifecycleState],
    matched_keys: set[str],
    options: LifecycleOptions,
) -> _LifecycleState | None:
    candidates = [
        state
        for state in states
        if state.relationship_key not in matched_keys
        and state.x_metric == finding.x_metric
        and state.y_metric == finding.y_metric
        and state.transform == finding.transform
        and abs(state.lag_seconds - int(finding.lag_seconds)) <= options.lag_tolerance_seconds
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda state: abs(state.lag_seconds - int(finding.lag_seconds)))


def _current_status(
    state: _LifecycleState,
    finding: CorrelationFinding,
    options: LifecycleOptions,
) -> tuple[str, dict]:
    details = {
        "reason": "matched_current_scan",
        "previous_lag_seconds": state.lag_seconds,
        "lag_delta_seconds": int(finding.lag_seconds) - state.lag_seconds,
    }
    if state.status in {"broken", "expired"}:
        details["recovered_from"] = state.status
        return "verified", details

    previous_score = abs(float(state.holdout_rho or 0.0))
    current_score = abs(float(finding.holdout_rho))
    coefficient_delta = current_score - previous_score
    details["coefficient_delta"] = round(coefficient_delta, 12)
    if coefficient_delta > options.coefficient_epsilon:
        return "strengthened", details
    if coefficient_delta < -options.coefficient_epsilon:
        return "weakened", details
    return "verified", details


def _insufficient_data_event(
    state: _LifecycleState,
    scan_utc: datetime,
    options: LifecycleOptions,
) -> LifecycleEvent:
    missing_since = state.missing_since_utc or scan_utc
    elapsed = scan_utc - missing_since
    status = "expired" if elapsed >= options.expire_after else "weakened"
    return _event_from_state(
        state,
        status=status,
        previous_status=state.status,
        scan_utc=scan_utc,
        failure_count=state.failure_count,
        missing_since_utc=missing_since,
        details={
            "reason": "insufficient_data",
            "missing_seconds": int(elapsed.total_seconds()),
        },
    )


def _validation_failure_event(
    state: _LifecycleState,
    scan_utc: datetime,
    options: LifecycleOptions,
) -> LifecycleEvent:
    failure_count = state.failure_count + 1
    status = "broken" if failure_count >= options.broken_after_failures else "weakened"
    return _event_from_state(
        state,
        status=status,
        previous_status=state.status,
        scan_utc=scan_utc,
        failure_count=failure_count,
        missing_since_utc=None,
        details={
            "reason": "validation_failed",
            "broken_after_failures": options.broken_after_failures,
        },
    )


def _event_from_finding(
    finding: CorrelationFinding,
    *,
    relationship_key: str,
    status: str,
    previous_status: str | None,
    scan_utc: datetime,
    failure_count: int = 0,
    missing_since_utc: datetime | None = None,
    details: dict | None = None,
) -> LifecycleEvent:
    return LifecycleEvent(
        relationship_key=relationship_key,
        status=status,
        previous_status=previous_status,
        event_utc=scan_utc,
        scan_utc=scan_utc,
        x_metric=finding.x_metric,
        y_metric=finding.y_metric,
        transform=finding.transform,
        lag_seconds=int(finding.lag_seconds),
        failure_count=failure_count,
        missing_since_utc=missing_since_utc,
        discovery_rho=float(finding.discovery_rho),
        holdout_rho=float(finding.holdout_rho),
        corrected_q=float(finding.corrected_q),
        stability=float(finding.stability),
        overlap_count=int(finding.overlap_count),
        details=details or {},
    )


def _event_from_state(
    state: _LifecycleState,
    *,
    status: str,
    previous_status: str | None,
    scan_utc: datetime,
    failure_count: int,
    missing_since_utc: datetime | None,
    details: dict,
) -> LifecycleEvent:
    return LifecycleEvent(
        relationship_key=state.relationship_key,
        status=status,
        previous_status=previous_status,
        event_utc=scan_utc,
        scan_utc=scan_utc,
        x_metric=state.x_metric,
        y_metric=state.y_metric,
        transform=state.transform,
        lag_seconds=state.lag_seconds,
        failure_count=failure_count,
        missing_since_utc=missing_since_utc,
        discovery_rho=state.discovery_rho,
        holdout_rho=state.holdout_rho,
        corrected_q=state.corrected_q,
        stability=state.stability,
        overlap_count=state.overlap_count,
        details=details,
    )


def _insert_lifecycle_event(conn: sqlite3.Connection, event: LifecycleEvent) -> LifecycleEvent:
    if event.status not in LIFECYCLE_STATUSES:
        raise ValueError(f"unsupported lifecycle status: {event.status}")
    cursor = conn.execute(
        """
        INSERT INTO finding_lifecycle_events (
            relationship_key,
            event_utc,
            scan_utc,
            x_metric,
            y_metric,
            transform,
            lag_seconds,
            previous_status,
            status,
            failure_count,
            missing_since_utc,
            discovery_rho,
            holdout_rho,
            corrected_q,
            stability,
            overlap_count,
            details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.relationship_key,
            to_utc_iso(event.event_utc),
            to_utc_iso(event.scan_utc),
            event.x_metric,
            event.y_metric,
            event.transform,
            int(event.lag_seconds),
            event.previous_status,
            event.status,
            int(event.failure_count),
            to_utc_iso(event.missing_since_utc) if event.missing_since_utc else None,
            event.discovery_rho,
            event.holdout_rho,
            event.corrected_q,
            event.stability,
            event.overlap_count,
            json.dumps(event.details, sort_keys=True, separators=(",", ":")),
        ),
    )
    return LifecycleEvent(**{**event.__dict__, "event_id": int(cursor.lastrowid)})


def _state_from_finding(finding: CorrelationFinding) -> _LifecycleState:
    return _LifecycleState(
        relationship_key=_relationship_key(finding),
        status="verified",
        event_utc=ensure_utc(finding.last_verified_utc),
        x_metric=finding.x_metric,
        y_metric=finding.y_metric,
        transform=finding.transform,
        lag_seconds=int(finding.lag_seconds),
        failure_count=0,
        missing_since_utc=None,
        discovery_rho=float(finding.discovery_rho),
        holdout_rho=float(finding.holdout_rho),
        corrected_q=float(finding.corrected_q),
        stability=float(finding.stability),
        overlap_count=int(finding.overlap_count),
    )


def _state_from_row(row: sqlite3.Row) -> _LifecycleState:
    return _LifecycleState(
        relationship_key=row["relationship_key"],
        status=row["status"],
        event_utc=parse_utc(row["event_utc"]),
        x_metric=row["x_metric"],
        y_metric=row["y_metric"],
        transform=row["transform"],
        lag_seconds=int(row["lag_seconds"]),
        failure_count=int(row["failure_count"]),
        missing_since_utc=parse_utc(row["missing_since_utc"]) if row["missing_since_utc"] else None,
        discovery_rho=_optional_float(row["discovery_rho"]),
        holdout_rho=_optional_float(row["holdout_rho"]),
        corrected_q=_optional_float(row["corrected_q"]),
        stability=_optional_float(row["stability"]),
        overlap_count=int(row["overlap_count"]) if row["overlap_count"] is not None else None,
    )


def _event_from_row(row: sqlite3.Row) -> LifecycleEvent:
    return LifecycleEvent(
        event_id=int(row["event_id"]),
        relationship_key=row["relationship_key"],
        status=row["status"],
        previous_status=row["previous_status"],
        event_utc=parse_utc(row["event_utc"]),
        scan_utc=parse_utc(row["scan_utc"]),
        x_metric=row["x_metric"],
        y_metric=row["y_metric"],
        transform=row["transform"],
        lag_seconds=int(row["lag_seconds"]),
        failure_count=int(row["failure_count"]),
        missing_since_utc=parse_utc(row["missing_since_utc"]) if row["missing_since_utc"] else None,
        discovery_rho=_optional_float(row["discovery_rho"]),
        holdout_rho=_optional_float(row["holdout_rho"]),
        corrected_q=_optional_float(row["corrected_q"]),
        stability=_optional_float(row["stability"]),
        overlap_count=int(row["overlap_count"]) if row["overlap_count"] is not None else None,
        details=json.loads(row["details_json"]),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _identity_from(item: FindingIdentity | CorrelationFinding) -> FindingIdentity:
    if isinstance(item, FindingIdentity):
        return item
    return FindingIdentity(
        x_metric=item.x_metric,
        y_metric=item.y_metric,
        transform=item.transform,
        lag_seconds=int(item.lag_seconds),
    )


def _matches_any_identity(
    state: _LifecycleState,
    identities: Sequence[FindingIdentity],
    options: LifecycleOptions,
) -> bool:
    return any(_matches_identity(state, identity, options) for identity in identities)


def _matches_identity(
    state: _LifecycleState,
    identity: FindingIdentity,
    options: LifecycleOptions,
) -> bool:
    if state.x_metric != identity.x_metric:
        return False
    if state.y_metric != identity.y_metric:
        return False
    if state.transform != identity.transform:
        return False
    if identity.lag_seconds is None:
        return True
    return abs(state.lag_seconds - identity.lag_seconds) <= options.lag_tolerance_seconds


def _relationship_key(finding: CorrelationFinding) -> str:
    return "|".join(
        (
            finding.x_metric,
            finding.y_metric,
            finding.transform,
            str(int(finding.lag_seconds)),
        )
    )


__all__ = [
    "FindingIdentity",
    "LifecycleEvent",
    "LifecycleOptions",
    "ensure_lifecycle_schema",
    "fetch_lifecycle_events",
    "update_finding_lifecycle",
]
