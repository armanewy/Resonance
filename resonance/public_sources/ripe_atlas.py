from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from resonance.config import LocationConfig, RipeAtlasPublicSourceConfig, load_config
from resonance.storage import (
    CollectorError,
    PublicCollectionState,
    PublicFetchEvent,
    PublicObservation,
    PublicRawArchive,
    PublicSource,
    SeriesRecord,
    ensure_database,
    fetch_public_collection_state,
    insert_collector_error,
    insert_public_observations,
    record_public_fetch_event,
    record_public_raw_archive,
    upsert_public_collection_state,
    upsert_public_source,
    upsert_series_record,
)
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


SOURCE_ID = "ripe_atlas_ipv4_ping"
STATE_ROUTE = "regional_ipv4_ping"
API_BASE_URL = "https://atlas.ripe.net/api/v2"
DEFAULT_RAW_ROOT = Path("data/public/raw")
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_RETRIES = 2
AGGREGATION_VERSION = "ripe-atlas-ipv4-ping-v1"
MIN_COMPOSITE_RESPONDING_PROBES = 5
MIN_COMPOSITE_UNIQUE_ASNS = 3
MIN_COMPOSITE_TARGETS = 2
UNKNOWN_ASN = -1

TARGETS = {
    1001: {"code": "k_root", "display": "k-root", "target": "k.root-servers.net"},
    1004: {"code": "f_root", "display": "f-root", "target": "f.root-servers.net"},
    1009: {"code": "a_root", "display": "a-root", "target": "a.root-servers.net"},
}

COMPOSITE_SERIES = {
    "median_rtt_ms": ("Regional IPv4 median RTT", "ms", "median_rtt_ms"),
    "p90_rtt_ms": ("Regional IPv4 p90 RTT", "ms", "p90_rtt_ms"),
    "packet_loss_fraction": ("Regional IPv4 packet loss", "fraction", "packet_loss_fraction"),
    "responding_probe_fraction": ("RIPE Atlas responding probe fraction", "fraction", "responding_probe_fraction"),
    "responding_probe_count": ("RIPE Atlas responding probe count", "count", "responding_probe_count"),
    "unique_responding_asn_count": ("RIPE Atlas unique responding ASN count", "count", "unique_asn_count"),
    "target_coverage_fraction": ("RIPE Atlas target coverage fraction", "fraction", "target_coverage_fraction"),
}


class RipeAtlasError(RuntimeError):
    """Raised when RIPE Atlas collection cannot complete safely."""


@dataclass(frozen=True)
class RipeProbe:
    probe_id: int
    asn_v4: int | None
    latitude: float
    longitude: float
    distance_km: float
    is_anchor: bool
    status: int | None
    is_public: bool
    has_ipv4: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeCohort:
    cohort_id: int
    source_id: str
    center_latitude: float
    center_longitude: float
    selected_radius_km: int
    created_at_utc: datetime
    effective_start_utc: datetime
    effective_end_utc: datetime | None
    selection_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeCohortMember:
    cohort_id: int
    probe_id: int
    asn_v4: int | None
    latitude: float
    longitude: float
    distance_km: float
    is_anchor: bool
    effective_start_utc: datetime
    effective_end_utc: datetime | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RipeHttpFetchResult:
    payload: Any
    raw_bytes: bytes
    request_url: str
    status_code: int | None
    retrieved_at_utc: datetime
    route: str
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedPingResult:
    measurement_id: int
    probe_id: int
    timestamp_utc: datetime
    address_family: int
    target_address: str
    firmware: str
    packets_sent: int
    packets_received: int
    packet_loss_fraction: float
    min_rtt_ms: float | None
    avg_rtt_ms: float | None
    max_rtt_ms: float | None
    parse_status: str
    source_identity: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RipeIngestResult:
    source_id: str
    cohort_id: int | None
    selected_radius_km: int | None
    selected_probe_count: int
    unique_asn_count: int
    start_utc: datetime
    end_utc: datetime
    measurement_count: int
    request_count: int
    raw_result_count: int
    valid_parsed_result_count: int
    finalized_aggregate_bin_count: int
    inserted_observations: int
    duplicate_observations: int
    skipped_low_quality_bin_count: int
    unresolved_gaps: tuple[str, ...]
    newest_finalized_bin_utc: datetime | None


ProbeFetcher = Callable[[float, float, int], Sequence[RipeHttpFetchResult]]
ResultFetcher = Callable[[int, datetime, datetime, Sequence[int]], RipeHttpFetchResult]


def ensure_ripe_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ripe_probe_cohorts (
            cohort_id INTEGER PRIMARY KEY,
            source_id TEXT NOT NULL,
            center_latitude REAL NOT NULL,
            center_longitude REAL NOT NULL,
            selected_radius_km INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            effective_start_utc TEXT NOT NULL,
            effective_end_utc TEXT,
            selection_version TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ripe_probe_cohort_members (
            cohort_id INTEGER NOT NULL,
            probe_id INTEGER NOT NULL,
            asn_v4 INTEGER,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            distance_km REAL NOT NULL,
            is_anchor INTEGER NOT NULL,
            effective_start_utc TEXT NOT NULL,
            effective_end_utc TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(cohort_id, probe_id),
            FOREIGN KEY(cohort_id) REFERENCES ripe_probe_cohorts(cohort_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ripe_probe_cohorts_active
        ON ripe_probe_cohorts(source_id, effective_end_utc, created_at_utc)
        """
    )
    conn.commit()


def ripe_source_record(*, enabled: bool = False) -> PublicSource:
    return PublicSource(
        source_id=SOURCE_ID,
        display_name="RIPE Atlas regional IPv4 Internet health",
        publisher="RIPE NCC",
        documentation_reference="https://atlas.ripe.net/docs/apis/rest-api-reference/",
        license_summary="Public RIPE Atlas API data; review RIPE NCC terms and Atlas documentation.",
        authentication_type="optional_api_key_env:RIPE_ATLAS_API_KEY",
        default_polling_cadence_seconds=900,
        quality_tier="public_reference",
        enabled=enabled,
        metadata={"api": "RIPE Atlas API v2", "measurements": TARGETS},
    )


def ensure_ripe_registry(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    enabled: bool = False,
) -> None:
    ensure_ripe_schema(conn)
    upsert_public_source(conn, ripe_source_record(enabled=enabled))
    for series in _series_records(config=config, location=location):
        upsert_series_record(conn, series)
    conn.commit()


def backfill_regional_ipv4_health(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    start_utc: datetime,
    end_utc: datetime,
    raw_root: Path = DEFAULT_RAW_ROOT,
    probe_fetcher: ProbeFetcher | None = None,
    result_fetcher: ResultFetcher | None = None,
    now: datetime | None = None,
) -> RipeIngestResult:
    start = _floor_time(start_utc, config.aggregation_seconds)
    end = _floor_time(end_utc, config.aggregation_seconds)
    if start >= end:
        raise RipeAtlasError("start_utc must be before end_utc")
    ensure_ripe_registry(conn, config=config, location=location, enabled=config.enabled)
    current = ensure_utc(now or utc_now()).replace(microsecond=0)
    cohort, members, discovery_pages = select_probe_cohort(
        conn,
        config=config,
        location=location,
        effective_start_utc=start,
        now=current,
        force_refresh=False,
        probe_fetcher=probe_fetcher,
    )
    for page in discovery_pages:
        _archive_fetch(conn, page, raw_root=raw_root, api_key=os.environ.get("RIPE_ATLAS_API_KEY"))
    return _collect_results(
        conn,
        config=config,
        location=location,
        cohort=cohort,
        members=members,
        start_utc=start,
        end_utc=end,
        raw_root=raw_root,
        result_fetcher=result_fetcher,
        now=current,
    )


def poll_regional_ipv4_health(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    raw_root: Path = DEFAULT_RAW_ROOT,
    probe_fetcher: ProbeFetcher | None = None,
    result_fetcher: ResultFetcher | None = None,
    now: datetime | None = None,
) -> RipeIngestResult:
    current = ensure_utc(now or utc_now()).replace(microsecond=0)
    finalized_end = _floor_time(current - timedelta(seconds=config.finalization_delay_seconds), config.aggregation_seconds)
    ensure_ripe_registry(conn, config=config, location=location, enabled=config.enabled)
    state = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=STATE_ROUTE)
    active_before_refresh = _active_cohort(conn)
    force_refresh = _needs_cohort_refresh(conn, config=config, now=current)
    if state is None or state.newest_complete_valid_period_utc is None:
        start = finalized_end - timedelta(hours=config.initial_backfill_hours)
    else:
        normal_start = state.newest_complete_valid_period_utc - timedelta(hours=config.normal_lookback_hours)
        gap = _earliest_aggregate_gap(conn, config=config, horizon_start=normal_start, horizon_end=finalized_end)
        start = min(normal_start, gap) if gap else normal_start
    start = _floor_time(start, config.aggregation_seconds)
    if force_refresh and active_before_refresh is not None and state is not None:
        members = _cohort_members(conn, active_before_refresh.cohort_id)
        result = _collect_results(
            conn,
            config=config,
            location=location,
            cohort=active_before_refresh,
            members=members,
            start_utc=start,
            end_utc=finalized_end,
            raw_root=raw_root,
            result_fetcher=result_fetcher,
            now=current,
        )
        new_cohort, _new_members, discovery_pages = select_probe_cohort(
            conn,
            config=config,
            location=location,
            effective_start_utc=current,
            now=current,
            force_refresh=True,
            probe_fetcher=probe_fetcher,
        )
        for page in discovery_pages:
            _archive_fetch(conn, page, raw_root=raw_root, api_key=os.environ.get("RIPE_ATLAS_API_KEY"))
        _record_probe_refresh_state(conn, cohort=new_cohort, now=current)
        conn.commit()
        return result
    cohort, members, discovery_pages = select_probe_cohort(
        conn,
        config=config,
        location=location,
        effective_start_utc=start if state is None else current,
        now=current,
        force_refresh=force_refresh,
        probe_fetcher=probe_fetcher,
    )
    for page in discovery_pages:
        _archive_fetch(conn, page, raw_root=raw_root, api_key=os.environ.get("RIPE_ATLAS_API_KEY"))
    return _collect_results(
        conn,
        config=config,
        location=location,
        cohort=cohort,
        members=members,
        start_utc=start,
        end_utc=finalized_end,
        raw_root=raw_root,
        result_fetcher=result_fetcher,
        now=current,
    )


def _record_probe_refresh_state(conn, *, cohort: ProbeCohort, now: datetime) -> None:
    previous = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=STATE_ROUTE)
    metadata = dict(previous.metadata if previous else {})
    metadata.update(
        {
            "active_cohort_id": cohort.cohort_id,
            "selected_radius_km": cohort.selected_radius_km,
            "last_successful_probe_discovery_utc": to_utc_iso(now),
        }
    )
    upsert_public_collection_state(
        conn,
        PublicCollectionState(
            source_id=SOURCE_ID,
            route=STATE_ROUTE,
            last_successful_poll_utc=previous.last_successful_poll_utc if previous else None,
            newest_complete_valid_period_utc=previous.newest_complete_valid_period_utc if previous else None,
            earliest_unresolved_gap_utc=previous.earliest_unresolved_gap_utc if previous else None,
            latest_error_utc=previous.latest_error_utc if previous else None,
            latest_error=previous.latest_error if previous else "",
            consecutive_failure_count=previous.consecutive_failure_count if previous else 0,
            metadata=metadata,
        ),
    )


def select_probe_cohort(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    effective_start_utc: datetime,
    now: datetime | None = None,
    force_refresh: bool = False,
    probe_fetcher: ProbeFetcher | None = None,
) -> tuple[ProbeCohort, tuple[ProbeCohortMember, ...], tuple[RipeHttpFetchResult, ...]]:
    ensure_ripe_schema(conn)
    current = ensure_utc(now or utc_now()).replace(microsecond=0)
    active = _active_cohort(conn)
    if active and not force_refresh and current - active.created_at_utc < timedelta(hours=config.cohort_refresh_hours):
        members = _cohort_members(conn, active.cohort_id)
        if len(members) >= config.minimum_probe_count:
            return active, members, ()
    fetcher = probe_fetcher or RipeAtlasClient().fetch_probe_pages
    retained = _eligible_retained_members(_cohort_members(conn, active.cohort_id) if active else ())
    discovery_pages: list[RipeHttpFetchResult] = []
    selected_probes: tuple[RipeProbe, ...] = ()
    selected_radius = config.initial_radius_km
    for radius in _radius_steps(config.initial_radius_km, config.maximum_radius_km):
        pages = tuple(fetcher(location.latitude, location.longitude, radius))
        discovery_pages.extend(pages)
        probes = _parse_probe_pages(pages, location=location)
        selected_probes = _select_probes(
            probes,
            retained=retained,
            desired_count=config.desired_probe_count,
            minimum_count=config.minimum_probe_count,
            maximum_per_asn=config.maximum_probes_per_asn,
            maximum_anchors=config.maximum_anchor_count,
        )
        selected_radius = radius
        if len(selected_probes) >= config.minimum_probe_count:
            break
    if len(selected_probes) < config.minimum_probe_count:
        raise RipeAtlasError(
            f"insufficient eligible RIPE Atlas probes: selected {len(selected_probes)} of {config.minimum_probe_count}"
        )
    if active:
        _close_cohort(conn, active.cohort_id, current)
    cohort_id = _insert_cohort(
        conn,
        source_id=SOURCE_ID,
        location=location,
        radius_km=selected_radius,
        created_at_utc=current,
        effective_start_utc=effective_start_utc,
        metadata={
            "selection_version": "asn-diverse-nearest-v1",
            "desired_probe_count": config.desired_probe_count,
            "minimum_probe_count": config.minimum_probe_count,
            "maximum_probes_per_asn": config.maximum_probes_per_asn,
            "maximum_anchor_count": config.maximum_anchor_count,
            "unknown_asn_count": sum(1 for probe in selected_probes if probe.asn_v4 is None),
            "radius_steps": list(_radius_steps(config.initial_radius_km, config.maximum_radius_km)),
        },
    )
    members = tuple(
        _member_from_probe(
            probe,
            cohort_id=cohort_id,
            effective_start_utc=effective_start_utc,
        )
        for probe in selected_probes
    )
    _insert_members(conn, members)
    conn.commit()
    return _active_cohort(conn), members, tuple(discovery_pages)


def parse_ping_results(payload: Any) -> tuple[ParsedPingResult, ...]:
    rows = payload if isinstance(payload, list) else payload.get("results", []) if isinstance(payload, Mapping) else []
    parsed: list[ParsedPingResult] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        try:
            parsed.append(_parse_ping_result(row, index=index))
        except RipeAtlasError:
            continue
    return tuple(parsed)


def aggregate_ping_results(
    results: Sequence[ParsedPingResult],
    *,
    config: RipeAtlasPublicSourceConfig,
    cohort: ProbeCohort,
    members: Sequence[ProbeCohortMember],
    ingested_at_utc: datetime,
) -> tuple[PublicObservation, ...]:
    member_by_probe = {member.probe_id: member for member in members}
    bins: dict[datetime, list[ParsedPingResult]] = {}
    for result in results:
        member = member_by_probe.get(result.probe_id)
        if member is None or not _member_effective_for(member, result.timestamp_utc):
            continue
        bin_start = _floor_time(result.timestamp_utc, config.aggregation_seconds)
        bins.setdefault(bin_start, []).append(result)

    observations: list[PublicObservation] = []
    for bin_start, bin_results in sorted(bins.items()):
        bin_end = bin_start + timedelta(seconds=config.aggregation_seconds)
        eligible_members = [member for member in members if _member_effective_for(member, bin_start)]
        aggregates = _aggregate_bin(bin_results, eligible_members=eligible_members, measurement_ids=config.measurement_ids)
        observations.extend(
            _observations_from_aggregate(
                aggregates,
                bin_start=bin_start,
                bin_end=bin_end,
                cohort=cohort,
                ingested_at_utc=ingested_at_utc,
            )
        )
    return tuple(observations)


def status_payload(conn, *, config: RipeAtlasPublicSourceConfig, location: LocationConfig, now: datetime | None = None) -> dict[str, Any]:
    ensure_ripe_registry(conn, config=config, location=location, enabled=config.enabled)
    current = ensure_utc(now or utc_now())
    state = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=STATE_ROUTE)
    active = _active_cohort(conn)
    members = _cohort_members(conn, active.cohort_id) if active else ()
    newest = state.newest_complete_valid_period_utc if state else None
    metadata = state.metadata if state else {}
    return {
        "source_id": SOURCE_ID,
        "source_name": "RIPE Atlas regional IPv4 Internet health",
        "enabled": bool(config.enabled),
        "active_cohort_id": active.cohort_id if active else None,
        "selected_radius_km": active.selected_radius_km if active else None,
        "active_probe_count": len(members),
        "unique_asn_count": len({member.asn_v4 for member in members if member.asn_v4 is not None}),
        "latest_cohort_refresh": to_utc_iso(active.created_at_utc) if active else None,
        "latest_result_fetch": to_utc_iso(state.last_successful_poll_utc) if state and state.last_successful_poll_utc else None,
        "latest_finalized_aggregate": to_utc_iso(newest) if newest else None,
        "lag_hours": round((current - newest).total_seconds() / 3600, 2) if newest else None,
        "unresolved_gap_count": int(metadata.get("unresolved_gap_count", 0)),
        "latest_error": state.latest_error if state else "",
        "consecutive_failure_count": state.consecutive_failure_count if state else 0,
        "unknown_asn_count": int(metadata.get("unknown_asn_count", 0)),
    }


def cohort_rows(conn) -> list[dict[str, Any]]:
    active = _active_cohort(conn)
    if active is None:
        return []
    return [
        {
            "probe_id": member.probe_id,
            "asn": member.asn_v4 if member.asn_v4 is not None else "unknown",
            "distance_km": round(member.distance_km, 2),
            "anchor": "yes" if member.is_anchor else "no",
            "effective_start": to_utc_iso(member.effective_start_utc),
            "effective_end": to_utc_iso(member.effective_end_utc) if member.effective_end_utc else "",
            "current": "yes" if member.effective_end_utc is None else "no",
        }
        for member in _cohort_members(conn, active.cohort_id)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect RIPE Atlas regional IPv4 Internet-health data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "probes"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--database", default="data/resonance.db")
        sub.add_argument("--config", default="config.toml")
    for name in ("backfill", "poll"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--database", default="data/resonance.db")
        sub.add_argument("--config", default="config.toml")
        sub.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    backfill = subparsers.choices["backfill"]
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end", required=True)
    args = parser.parse_args(argv)

    app_config = load_config(args.config)
    config = app_config.public_sources.ripe_atlas
    location = app_config.location
    conn = ensure_database(args.database)
    try:
        if args.command == "status":
            print(json.dumps(status_payload(conn, config=config, location=location), indent=2, sort_keys=True))
            return 0
        if args.command == "probes":
            ensure_ripe_registry(conn, config=config, location=location, enabled=config.enabled)
            print(json.dumps(cohort_rows(conn), indent=2, sort_keys=True))
            return 0
        try:
            if args.command == "backfill":
                result = backfill_regional_ipv4_health(
                    conn,
                    config=config,
                    location=location,
                    start_utc=parse_utc(args.start),
                    end_utc=parse_utc(args.end),
                    raw_root=Path(args.raw_root),
                )
            else:
                result = poll_regional_ipv4_health(
                    conn,
                    config=config,
                    location=location,
                    raw_root=Path(args.raw_root),
                )
        except Exception as exc:
            _record_failure(conn, exc, utc_now())
            insert_collector_error(conn, CollectorError(utc_now(), SOURCE_ID, exc.__class__.__name__, _safe_error_message(exc)))
            parser.exit(2, f"RIPE Atlas collection failed: {_safe_error_message(exc)}\n")
        print(json.dumps(_result_dict(result), indent=2, sort_keys=True))
        return 0
    finally:
        conn.close()


class RipeAtlasClient:
    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, retries: int = DEFAULT_RETRIES) -> None:
        if timeout_seconds <= 0:
            raise RipeAtlasError("timeout_seconds must be positive")
        if retries < 0:
            raise RipeAtlasError("retries must be non-negative")
        self._api_key = api_key or os.environ.get("RIPE_ATLAS_API_KEY", "")
        self._timeout_seconds = timeout_seconds
        self._retries = retries

    def fetch_probe_pages(self, latitude: float, longitude: float, radius_km: int) -> tuple[RipeHttpFetchResult, ...]:
        url = f"{API_BASE_URL}/probes/"
        params = {
            "radius": f"{latitude},{longitude}:{radius_km}",
            "status": "1",
            "is_public": "true",
            "page_size": "500",
        }
        pages = []
        next_url: str | None = url
        while next_url:
            result = self._get_json(next_url, params=params if next_url == url else None, route="probes", metadata={"radius_km": radius_km})
            pages.append(result)
            next_url = result.payload.get("next") if isinstance(result.payload, Mapping) else None
            params = None
        return tuple(pages)

    def fetch_results(self, measurement_id: int, start_utc: datetime, stop_utc: datetime, probe_ids: Sequence[int]) -> RipeHttpFetchResult:
        url = f"{API_BASE_URL}/measurements/{measurement_id}/results/"
        params = {
            "start": str(int(ensure_utc(start_utc).timestamp())),
            "stop": str(int(ensure_utc(stop_utc).timestamp())),
            "probe_ids": ",".join(str(probe_id) for probe_id in probe_ids),
            "public_only": "true",
        }
        return self._get_json(
            url,
            params=params,
            route=f"results:{measurement_id}",
            metadata={
                "measurement_id": measurement_id,
                "start": to_utc_iso(start_utc),
                "stop": to_utc_iso(stop_utc),
                "probe_ids_sha256": _probe_ids_hash(probe_ids),
            },
        )

    def _get_json(self, url: str, *, params: Mapping[str, str] | None, route: str, metadata: Mapping[str, Any]) -> RipeHttpFetchResult:
        headers = {"Authorization": f"Key {self._api_key}"} if self._api_key else {}
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds, headers=headers) as client:
                    response = client.get(url, params=params)
                if response.status_code == 429 and attempt < self._retries:
                    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                    time.sleep(retry_after if retry_after is not None else 0.25 * (2**attempt))
                    continue
                if response.status_code >= 500 and attempt < self._retries:
                    time.sleep(0.25 * (2**attempt))
                    continue
                response.raise_for_status()
                return RipeHttpFetchResult(
                    payload=response.json(),
                    raw_bytes=response.content,
                    request_url=str(response.url),
                    status_code=response.status_code,
                    retrieved_at_utc=utc_now(),
                    route=route,
                    request_metadata={**dict(metadata), "attempt": attempt + 1, "authenticated_read": bool(self._api_key)},
                )
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self._retries:
                    time.sleep(0.25 * (2**attempt))
                    continue
        raise RipeAtlasError(f"RIPE Atlas request failed for {route}: {_safe_error_message(last_error)}") from last_error


def _series_records(*, config: RipeAtlasPublicSourceConfig, location: LocationConfig) -> tuple[SeriesRecord, ...]:
    records: list[SeriesRecord] = []
    geography_id = f"{location.latitude:.4f},{location.longitude:.4f}:regional"
    common = {
        "source_id": SOURCE_ID,
        "cadence_seconds": config.aggregation_seconds,
        "aggregation": "fixed_utc_bin",
        "geography_type": "regional_radius",
        "geography_id": geography_id,
        "timezone": "UTC",
        "timestamp_semantics": "fixed UTC aggregation bin start; observed_at is newest contributing RIPE result timestamp",
        "parent_series_id": None,
        "lineage_id": f"{SOURCE_ID}:regional_health",
        "quality_tier": "derived",
    }
    for metric, (display, unit, field_name) in COMPOSITE_SERIES.items():
        records.append(
            SeriesRecord(
                series_id=f"{SOURCE_ID}:regional:{metric}",
                metric_name=metric,
                display_name=display,
                unit=unit,
                metadata={
                    "analysis_eligible": metric in {"median_rtt_ms", "p90_rtt_ms", "packet_loss_fraction"},
                    "diagnostic": metric not in {"median_rtt_ms", "p90_rtt_ms", "packet_loss_fraction"},
                    "field": field_name,
                    "measurement_ids": list(config.measurement_ids),
                    "center_latitude": location.latitude,
                    "center_longitude": location.longitude,
                    "quality_formula": "(responding_probe_fraction + min(unique_asn_count/3, 1) + target_coverage_fraction) / 3",
                },
                **common,
            )
        )
    for measurement_id in config.measurement_ids:
        target = TARGETS.get(measurement_id, {"code": f"measurement_{measurement_id}", "display": str(measurement_id), "target": str(measurement_id)})
        for metric, unit, field_name in (
            ("median_rtt_ms", "ms", "median_rtt_ms"),
            ("packet_loss_fraction", "fraction", "packet_loss_fraction"),
        ):
            records.append(
                SeriesRecord(
                    series_id=f"{SOURCE_ID}:target:{target['code']}:{metric}",
                    metric_name=f"{target['code']}_{metric}",
                    display_name=f"RIPE Atlas {target['display']} {metric.replace('_', ' ')}",
                    unit=unit,
                    metadata={
                        "analysis_eligible": False,
                        "diagnostic": True,
                        "measurement_id": measurement_id,
                        "target": target["target"],
                        "field": field_name,
                    },
                    **common,
                )
            )
    return tuple(records)


def _collect_results(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    cohort: ProbeCohort,
    members: Sequence[ProbeCohortMember],
    start_utc: datetime,
    end_utc: datetime,
    raw_root: Path,
    result_fetcher: ResultFetcher | None,
    now: datetime,
) -> RipeIngestResult:
    if start_utc >= end_utc:
        return _empty_result(cohort=cohort, members=members, start_utc=start_utc, end_utc=end_utc)
    fetcher = result_fetcher or RipeAtlasClient().fetch_results
    requests = _planned_result_requests(
        config=config,
        start_utc=start_utc,
        end_utc=end_utc,
        probe_ids=[member.probe_id for member in members],
    )
    if len(requests) > config.maximum_requests_per_poll:
        raise RipeAtlasError(f"RIPE Atlas request budget exceeded: {len(requests)} > {config.maximum_requests_per_poll}")
    parsed: list[ParsedPingResult] = []
    raw_result_count = 0
    for request in requests:
        measurement_id, chunk_start, chunk_stop, probe_ids = request
        fetch_result = fetcher(measurement_id, chunk_start, chunk_stop, probe_ids)
        metadata = {
            **dict(fetch_result.request_metadata),
            "measurement_id": measurement_id,
            "target_name": TARGETS.get(measurement_id, {}).get("target", str(measurement_id)),
            "start": to_utc_iso(chunk_start),
            "stop": to_utc_iso(chunk_stop),
            "probe_ids_sha256": _probe_ids_hash(probe_ids),
            "cohort_id": cohort.cohort_id,
        }
        fetch_result = RipeHttpFetchResult(
            payload=fetch_result.payload,
            raw_bytes=fetch_result.raw_bytes,
            request_url=fetch_result.request_url,
            status_code=fetch_result.status_code,
            retrieved_at_utc=fetch_result.retrieved_at_utc,
            route=f"results:{measurement_id}",
            request_metadata=metadata,
        )
        _archive_fetch(conn, fetch_result, raw_root=raw_root, api_key=os.environ.get("RIPE_ATLAS_API_KEY"))
        rows = fetch_result.payload if isinstance(fetch_result.payload, list) else []
        raw_result_count += len(rows)
        parsed.extend(parse_ping_results(rows))
    ingested_at = ensure_utc(now).replace(microsecond=0)
    observations = aggregate_ping_results(parsed, config=config, cohort=cohort, members=members, ingested_at_utc=ingested_at)
    inserted = insert_public_observations(conn, observations)
    duplicate = len(observations) - inserted
    newest = _newest_finalized_bin(conn)
    gaps = _aggregate_gaps(conn, config=config, start_utc=start_utc, end_utc=end_utc)
    skipped = max(0, _expected_bin_count(start_utc, end_utc, config.aggregation_seconds) - len({obs.valid_start_utc for obs in observations if _is_composite_core(obs.series_id)}))
    _record_success_state(
        conn,
        cohort=cohort,
        members=members,
        now=ingested_at,
        newest=newest,
        gaps=gaps,
        unknown_asn_count=sum(1 for member in members if member.asn_v4 is None),
    )
    conn.commit()
    return RipeIngestResult(
        source_id=SOURCE_ID,
        cohort_id=cohort.cohort_id,
        selected_radius_km=cohort.selected_radius_km,
        selected_probe_count=len(members),
        unique_asn_count=len({member.asn_v4 for member in members if member.asn_v4 is not None}),
        start_utc=start_utc,
        end_utc=end_utc,
        measurement_count=len(config.measurement_ids),
        request_count=len(requests),
        raw_result_count=raw_result_count,
        valid_parsed_result_count=len(parsed),
        finalized_aggregate_bin_count=len({obs.valid_start_utc for obs in observations}),
        inserted_observations=inserted,
        duplicate_observations=duplicate,
        skipped_low_quality_bin_count=skipped,
        unresolved_gaps=tuple(to_utc_iso(gap) for gap in gaps),
        newest_finalized_bin_utc=newest,
    )


def _planned_result_requests(
    *,
    config: RipeAtlasPublicSourceConfig,
    start_utc: datetime,
    end_utc: datetime,
    probe_ids: Sequence[int],
) -> tuple[tuple[int, datetime, datetime, tuple[int, ...]], ...]:
    requests = []
    for measurement_id in config.measurement_ids:
        chunk_start = start_utc
        while chunk_start < end_utc:
            chunk_stop = min(end_utc, chunk_start + timedelta(hours=config.result_chunk_hours))
            for batch in _batches(tuple(probe_ids), config.maximum_probe_batch_size):
                requests.append((measurement_id, chunk_start, chunk_stop, batch))
            chunk_start = chunk_stop
    return tuple(requests)


def _parse_probe_pages(pages: Sequence[RipeHttpFetchResult], *, location: LocationConfig) -> tuple[RipeProbe, ...]:
    probes: list[RipeProbe] = []
    for page in pages:
        payload = page.payload
        rows = payload.get("results", []) if isinstance(payload, Mapping) else []
        for row in rows:
            probe = _probe_from_row(row, location=location)
            if probe is not None:
                probes.append(probe)
    by_id = {probe.probe_id: probe for probe in probes}
    return tuple(sorted(by_id.values(), key=lambda probe: (probe.distance_km, probe.probe_id)))


def _probe_from_row(row: Any, *, location: LocationConfig) -> RipeProbe | None:
    if not isinstance(row, Mapping):
        return None
    try:
        probe_id = int(row.get("id"))
    except (TypeError, ValueError):
        return None
    coordinates = _probe_coordinates(row)
    if coordinates is None:
        return None
    latitude, longitude = coordinates
    status = _probe_status(row.get("status"))
    is_public = bool(row.get("is_public", row.get("public", False)))
    asn_v4 = _optional_int(row.get("asn_v4"))
    is_anchor = bool(row.get("is_anchor") or row.get("anchor"))
    has_ipv4 = bool(row.get("address_v4") or asn_v4 is not None or row.get("prefix_v4"))
    return RipeProbe(
        probe_id=probe_id,
        asn_v4=asn_v4,
        latitude=latitude,
        longitude=longitude,
        distance_km=_haversine_km(location.latitude, location.longitude, latitude, longitude),
        is_anchor=is_anchor,
        status=status,
        is_public=is_public,
        has_ipv4=has_ipv4,
        metadata={"raw": _probe_metadata(row)},
    )


def _probe_coordinates(row: Mapping[str, Any]) -> tuple[float, float] | None:
    latitude = _optional_float(row.get("latitude"))
    longitude = _optional_float(row.get("longitude"))
    if latitude is not None and longitude is not None:
        return latitude, longitude
    geometry = row.get("geometry")
    if isinstance(geometry, Mapping):
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes, bytearray)) and len(coordinates) >= 2:
            longitude = _optional_float(coordinates[0])
            latitude = _optional_float(coordinates[1])
            if latitude is not None and longitude is not None:
                return latitude, longitude
    return None


def _select_probes(
    probes: Sequence[RipeProbe],
    *,
    retained: Sequence[RipeProbe],
    desired_count: int,
    minimum_count: int,
    maximum_per_asn: int,
    maximum_anchors: int,
) -> tuple[RipeProbe, ...]:
    eligible = {
        probe.probe_id: probe
        for probe in probes
        if probe.status == 1 and probe.is_public and probe.has_ipv4 and _valid_coordinates(probe.latitude, probe.longitude)
    }
    retained_valid = [eligible[probe.probe_id] for probe in retained if probe.probe_id in eligible]
    selected: list[RipeProbe] = []
    counts: dict[int, int] = {}
    anchor_count = 0

    def try_add(probe: RipeProbe, *, allow_unknown: bool) -> bool:
        nonlocal anchor_count
        asn_key = probe.asn_v4 if probe.asn_v4 is not None else UNKNOWN_ASN
        if probe.asn_v4 is None and not allow_unknown:
            return False
        if counts.get(asn_key, 0) >= maximum_per_asn:
            return False
        if probe.is_anchor and anchor_count >= maximum_anchors:
            return False
        if probe.probe_id in {item.probe_id for item in selected}:
            return False
        selected.append(probe)
        counts[asn_key] = counts.get(asn_key, 0) + 1
        if probe.is_anchor:
            anchor_count += 1
        return True

    for probe in sorted(retained_valid, key=lambda item: (item.distance_km, item.probe_id)):
        try_add(probe, allow_unknown=False)
    for probe in sorted(eligible.values(), key=lambda item: (item.distance_km, item.probe_id)):
        if len(selected) >= desired_count:
            break
        try_add(probe, allow_unknown=False)
    if len(selected) < minimum_count:
        for probe in sorted(eligible.values(), key=lambda item: (item.distance_km, item.probe_id)):
            if len(selected) >= minimum_count:
                break
            try_add(probe, allow_unknown=True)
    return tuple(selected[:desired_count])


def _parse_ping_result(row: Mapping[str, Any], *, index: int) -> ParsedPingResult:
    measurement_id = _required_int(row, "msm_id")
    probe_id = _required_int(row, "prb_id")
    timestamp = _parse_result_timestamp(row.get("timestamp"))
    address_family = int(row.get("af") or 4)
    if address_family != 4:
        raise RipeAtlasError("non-IPv4 result ignored")
    packet_rtts: list[float] = []
    packet_count = 0
    warnings: list[str] = []
    result_entries = row.get("result")
    if isinstance(result_entries, list):
        packet_count = len(result_entries)
        for entry in result_entries:
            if isinstance(entry, Mapping) and "rtt" in entry:
                rtt = _optional_float(entry.get("rtt"))
                if rtt is None or rtt < 0:
                    warnings.append("invalid_packet_rtt")
                    continue
                packet_rtts.append(rtt)
            else:
                warnings.append("packet_timeout_or_error")
    top_avg = _optional_nonnegative_float(row.get("avg"))
    top_min = _optional_nonnegative_float(row.get("min"))
    top_max = _optional_nonnegative_float(row.get("max"))
    avg = top_avg if top_avg is not None else (statistics.fmean(packet_rtts) if packet_rtts else None)
    min_rtt = top_min if top_min is not None else (min(packet_rtts) if packet_rtts else None)
    max_rtt = top_max if top_max is not None else (max(packet_rtts) if packet_rtts else None)
    sent = _optional_int(row.get("sent")) or packet_count or (_optional_int(row.get("packets_sent")) or 0)
    received = _optional_int(row.get("rcvd")) or _optional_int(row.get("packets_received")) or len(packet_rtts)
    if sent <= 0 and avg is not None:
        sent = max(received, len(packet_rtts), 1)
    if received <= 0 and avg is not None:
        received = max(len(packet_rtts), 1)
    if sent <= 0:
        sent = max(received, len(packet_rtts))
    if received > sent:
        sent = received
    if avg is None:
        warnings.append("no_valid_rtt")
    return ParsedPingResult(
        measurement_id=measurement_id,
        probe_id=probe_id,
        timestamp_utc=timestamp,
        address_family=address_family,
        target_address=str(row.get("dst_addr") or row.get("dst_name") or ""),
        firmware=str(row.get("fw") or row.get("lts") or ""),
        packets_sent=int(sent),
        packets_received=int(received),
        packet_loss_fraction=1.0 - (float(received) / float(sent)) if sent else 1.0,
        min_rtt_ms=min_rtt,
        avg_rtt_ms=avg,
        max_rtt_ms=max_rtt,
        parse_status="ok" if avg is not None else "loss_only",
        source_identity=f"{measurement_id}:{probe_id}:{int(timestamp.timestamp())}:{index}:{_result_digest(row)}",
        metadata={"warnings": warnings, "format": row.get("type") or "ping"},
    )


def _aggregate_bin(
    results: Sequence[ParsedPingResult],
    *,
    eligible_members: Sequence[ProbeCohortMember],
    measurement_ids: Sequence[int],
) -> Mapping[str, Any]:
    member_by_probe = {member.probe_id: member for member in eligible_members}
    probe_target: dict[tuple[int, int], list[ParsedPingResult]] = {}
    for result in results:
        probe_target.setdefault((result.probe_id, result.measurement_id), []).append(result)
    per_probe_target = {}
    for key, values in probe_target.items():
        rtts = [value.avg_rtt_ms for value in values if value.avg_rtt_ms is not None]
        sent = sum(value.packets_sent for value in values)
        received = sum(value.packets_received for value in values)
        per_probe_target[key] = {
            "median_rtt_ms": statistics.median(rtts) if rtts else None,
            "packets_sent": sent,
            "packets_received": received,
            "source_identities": sorted(value.source_identity for value in values),
            "source_timestamps": [value.timestamp_utc for value in values],
        }
    target_aggregates = {}
    responding_probe_ids: set[int] = set()
    responding_asns: set[int] = set()
    for measurement_id in measurement_ids:
        probe_values = {
            probe_id: value
            for (probe_id, target_id), value in per_probe_target.items()
            if target_id == measurement_id
        }
        rtts = [value["median_rtt_ms"] for value in probe_values.values() if value["median_rtt_ms"] is not None]
        sent = sum(int(value["packets_sent"]) for value in probe_values.values())
        received = sum(int(value["packets_received"]) for value in probe_values.values())
        for probe_id, value in probe_values.items():
            if int(value["packets_received"]) > 0:
                responding_probe_ids.add(probe_id)
                member = member_by_probe.get(probe_id)
                if member and member.asn_v4 is not None:
                    responding_asns.add(member.asn_v4)
        target_aggregates[measurement_id] = {
            "median_rtt_ms": statistics.median(rtts) if rtts else None,
            "p90_rtt_ms": _percentile(rtts, 0.90) if rtts else None,
            "packet_loss_fraction": 1.0 - (received / sent) if sent else None,
            "packets_sent": sent,
            "packets_received": received,
            "responding_probe_count": sum(1 for value in probe_values.values() if int(value["packets_received"]) > 0),
            "source_identities": sorted(identity for value in probe_values.values() for identity in value["source_identities"]),
            "source_timestamps": [timestamp for value in probe_values.values() for timestamp in value["source_timestamps"]],
        }
    target_values = [value for value in target_aggregates.values() if value["median_rtt_ms"] is not None]
    target_coverage_fraction = len(target_values) / len(measurement_ids) if measurement_ids else 0.0
    eligible_count = len(eligible_members)
    responding_fraction = len(responding_probe_ids) / eligible_count if eligible_count else 0.0
    quality = min(1.0, (responding_fraction + min(len(responding_asns) / MIN_COMPOSITE_UNIQUE_ASNS, 1.0) + target_coverage_fraction) / 3.0)
    composite_allowed = (
        len(responding_probe_ids) >= MIN_COMPOSITE_RESPONDING_PROBES
        and len(responding_asns) >= MIN_COMPOSITE_UNIQUE_ASNS
        and len(target_values) >= MIN_COMPOSITE_TARGETS
    )
    all_sent = sum(int(value["packets_sent"]) for value in target_aggregates.values())
    all_received = sum(int(value["packets_received"]) for value in target_aggregates.values())
    return {
        "target_aggregates": target_aggregates,
        "composite_allowed": composite_allowed,
        "median_rtt_ms": statistics.median([value["median_rtt_ms"] for value in target_values]) if composite_allowed else None,
        "p90_rtt_ms": statistics.median([value["p90_rtt_ms"] for value in target_values if value["p90_rtt_ms"] is not None]) if composite_allowed else None,
        "packet_loss_fraction": 1.0 - (all_received / all_sent) if composite_allowed and all_sent else None,
        "responding_probe_fraction": responding_fraction,
        "responding_probe_count": len(responding_probe_ids),
        "eligible_probe_count": eligible_count,
        "unique_asn_count": len(responding_asns),
        "target_coverage_fraction": target_coverage_fraction,
        "quality_score": quality,
        "source_identities": sorted(identity for value in target_aggregates.values() for identity in value["source_identities"]),
        "newest_result_utc": max((timestamp for value in target_aggregates.values() for timestamp in value["source_timestamps"]), default=None),
    }


def _observations_from_aggregate(
    aggregate: Mapping[str, Any],
    *,
    bin_start: datetime,
    bin_end: datetime,
    cohort: ProbeCohort,
    ingested_at_utc: datetime,
) -> tuple[PublicObservation, ...]:
    observations: list[PublicObservation] = []
    observed_at = aggregate.get("newest_result_utc") or bin_end
    for metric, (_display, _unit, field_name) in COMPOSITE_SERIES.items():
        value = aggregate.get(field_name)
        if value is None:
            continue
        series_id = f"{SOURCE_ID}:regional:{metric}"
        observations.append(
            _aggregate_observation(
                series_id,
                value=float(value),
                bin_start=bin_start,
                bin_end=bin_end,
                observed_at=observed_at,
                ingested_at_utc=ingested_at_utc,
                cohort=cohort,
                source_identities=aggregate["source_identities"],
                metadata={
                    "cohort_id": cohort.cohort_id,
                    "quality_score": aggregate["quality_score"],
                    "eligible_probe_count": aggregate["eligible_probe_count"],
                    "responding_probe_count": aggregate["responding_probe_count"],
                    "unique_asn_count": aggregate["unique_asn_count"],
                    "target_coverage_fraction": aggregate["target_coverage_fraction"],
                    "aggregation_version": AGGREGATION_VERSION,
                },
            )
        )
    for measurement_id, target_aggregate in aggregate["target_aggregates"].items():
        target = TARGETS.get(measurement_id, {"code": f"measurement_{measurement_id}"})
        for metric in ("median_rtt_ms", "packet_loss_fraction"):
            value = target_aggregate.get(metric)
            if value is None:
                continue
            series_id = f"{SOURCE_ID}:target:{target['code']}:{metric}"
            observations.append(
                _aggregate_observation(
                    series_id,
                    value=float(value),
                    bin_start=bin_start,
                    bin_end=bin_end,
                    observed_at=observed_at,
                    ingested_at_utc=ingested_at_utc,
                    cohort=cohort,
                    source_identities=target_aggregate["source_identities"],
                    metadata={"cohort_id": cohort.cohort_id, "measurement_id": measurement_id, "aggregation_version": AGGREGATION_VERSION},
                )
            )
    return tuple(observations)


def _aggregate_observation(
    series_id: str,
    *,
    value: float,
    bin_start: datetime,
    bin_end: datetime,
    observed_at: datetime,
    ingested_at_utc: datetime,
    cohort: ProbeCohort,
    source_identities: Sequence[str],
    metadata: Mapping[str, Any],
) -> PublicObservation:
    key = f"{series_id}:{to_utc_iso(bin_start)}"
    revision = _revision_hash(
        {
            "algorithm": AGGREGATION_VERSION,
            "cohort_id": cohort.cohort_id,
            "series_id": series_id,
            "bin_start": to_utc_iso(bin_start),
            "contributors": sorted(source_identities),
            "value": round(float(value), 12),
        }
    )
    return PublicObservation(
        series_id=series_id,
        valid_start_utc=bin_start,
        valid_end_utc=bin_end,
        observed_at_utc=observed_at,
        ingested_at_utc=ingested_at_utc,
        value=float(value),
        quality="derived",
        source_revision=revision,
        source_observation_key=key,
        metadata=dict(metadata),
    )


def _archive_fetch(conn, result: RipeHttpFetchResult, *, raw_root: Path, api_key: str | None) -> None:
    raw = result.raw_bytes
    digest = hashlib.sha256(raw).hexdigest()
    retrieved = ensure_utc(result.retrieved_at_utc).replace(microsecond=0)
    directory = raw_root / SOURCE_ID / f"{retrieved.year:04d}" / f"{retrieved.month:02d}" / f"{retrieved.day:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        path.write_bytes(raw)
    safe_url = _redact_api_key(result.request_url, api_key)
    archive = PublicRawArchive(
        sha256=digest,
        source_id=SOURCE_ID,
        retrieved_at_utc=retrieved,
        request_url=safe_url,
        status_code=result.status_code,
        path=str(path),
        metadata=_redact_mapping({"route": result.route, **dict(result.request_metadata)}, api_key),
    )
    record_public_raw_archive(conn, archive)
    record_public_fetch_event(
        conn,
        PublicFetchEvent(
            source_id=SOURCE_ID,
            retrieved_at_utc=retrieved,
            request_url=safe_url,
            status_code=result.status_code,
            content_sha256=digest,
            route=result.route,
            page_offset=0,
            request_metadata=_redact_mapping(dict(result.request_metadata), api_key),
        ),
    )


def _active_cohort(conn) -> ProbeCohort | None:
    ensure_ripe_schema(conn)
    row = conn.execute(
        """
        SELECT *
        FROM ripe_probe_cohorts
        WHERE source_id = ? AND effective_end_utc IS NULL
        ORDER BY cohort_id DESC
        LIMIT 1
        """,
        (SOURCE_ID,),
    ).fetchone()
    return _cohort_from_row(row) if row else None


def _cohort_members(conn, cohort_id: int) -> tuple[ProbeCohortMember, ...]:
    rows = conn.execute(
        """
        SELECT *
        FROM ripe_probe_cohort_members
        WHERE cohort_id = ?
        ORDER BY distance_km ASC, probe_id ASC
        """,
        (cohort_id,),
    ).fetchall()
    return tuple(_member_from_row(row) for row in rows)


def _insert_cohort(
    conn,
    *,
    source_id: str,
    location: LocationConfig,
    radius_km: int,
    created_at_utc: datetime,
    effective_start_utc: datetime,
    metadata: Mapping[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO ripe_probe_cohorts (
            source_id, center_latitude, center_longitude, selected_radius_km,
            created_at_utc, effective_start_utc, effective_end_utc,
            selection_version, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            source_id,
            location.latitude,
            location.longitude,
            int(radius_km),
            to_utc_iso(created_at_utc),
            to_utc_iso(effective_start_utc),
            "asn-diverse-nearest-v1",
            _json_dumps(metadata),
        ),
    )
    return int(cursor.lastrowid)


def _insert_members(conn, members: Sequence[ProbeCohortMember]) -> None:
    for member in members:
        conn.execute(
            """
            INSERT INTO ripe_probe_cohort_members (
                cohort_id, probe_id, asn_v4, latitude, longitude, distance_km,
                is_anchor, effective_start_utc, effective_end_utc, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                member.cohort_id,
                member.probe_id,
                member.asn_v4,
                member.latitude,
                member.longitude,
                member.distance_km,
                1 if member.is_anchor else 0,
                to_utc_iso(member.effective_start_utc),
                to_utc_iso(member.effective_end_utc) if member.effective_end_utc else None,
                _json_dumps(member.metadata),
            ),
        )


def _close_cohort(conn, cohort_id: int, end_utc: datetime) -> None:
    conn.execute(
        "UPDATE ripe_probe_cohorts SET effective_end_utc = ? WHERE cohort_id = ?",
        (to_utc_iso(end_utc), cohort_id),
    )
    conn.execute(
        "UPDATE ripe_probe_cohort_members SET effective_end_utc = ? WHERE cohort_id = ? AND effective_end_utc IS NULL",
        (to_utc_iso(end_utc), cohort_id),
    )


def _record_success_state(
    conn,
    *,
    cohort: ProbeCohort,
    members: Sequence[ProbeCohortMember],
    now: datetime,
    newest: datetime | None,
    gaps: Sequence[datetime],
    unknown_asn_count: int,
) -> None:
    upsert_public_collection_state(
        conn,
        PublicCollectionState(
            source_id=SOURCE_ID,
            route=STATE_ROUTE,
            last_successful_poll_utc=now,
            newest_complete_valid_period_utc=newest,
            earliest_unresolved_gap_utc=gaps[0] if gaps else None,
            latest_error_utc=None,
            latest_error="",
            consecutive_failure_count=0,
            metadata={
                "active_cohort_id": cohort.cohort_id,
                "selected_radius_km": cohort.selected_radius_km,
                "selected_probe_count": len(members),
                "selected_unique_asn_count": len({member.asn_v4 for member in members if member.asn_v4 is not None}),
                "unresolved_gap_count": len(gaps),
                "unknown_asn_count": unknown_asn_count,
            },
        ),
    )


def _record_failure(conn, exc: BaseException, now: datetime) -> None:
    previous = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=STATE_ROUTE)
    upsert_public_collection_state(
        conn,
        PublicCollectionState(
            source_id=SOURCE_ID,
            route=STATE_ROUTE,
            last_successful_poll_utc=previous.last_successful_poll_utc if previous else None,
            newest_complete_valid_period_utc=previous.newest_complete_valid_period_utc if previous else None,
            earliest_unresolved_gap_utc=previous.earliest_unresolved_gap_utc if previous else None,
            latest_error_utc=ensure_utc(now),
            latest_error=_safe_error_message(exc),
            consecutive_failure_count=(previous.consecutive_failure_count if previous else 0) + 1,
            metadata=previous.metadata if previous else {},
        ),
    )
    conn.commit()


def _newest_finalized_bin(conn) -> datetime | None:
    row = conn.execute(
        """
        SELECT MAX(valid_start_utc) AS newest
        FROM public_observations
        WHERE series_id = ?
        """,
        (f"{SOURCE_ID}:regional:median_rtt_ms",),
    ).fetchone()
    return parse_utc(row["newest"]) if row and row["newest"] else None


def _aggregate_gaps(conn, *, config: RipeAtlasPublicSourceConfig, start_utc: datetime, end_utc: datetime) -> tuple[datetime, ...]:
    rows = conn.execute(
        """
        SELECT DISTINCT valid_start_utc
        FROM public_observations
        WHERE series_id = ?
          AND valid_start_utc >= ?
          AND valid_start_utc < ?
        """,
        (f"{SOURCE_ID}:regional:median_rtt_ms", to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchall()
    observed = {parse_utc(row["valid_start_utc"]) for row in rows}
    gaps = []
    cursor = start_utc
    while cursor < end_utc:
        if cursor not in observed:
            gaps.append(cursor)
        cursor += timedelta(seconds=config.aggregation_seconds)
    return tuple(gaps)


def _earliest_aggregate_gap(conn, *, config: RipeAtlasPublicSourceConfig, horizon_start: datetime, horizon_end: datetime) -> datetime | None:
    gaps = _aggregate_gaps(conn, config=config, start_utc=horizon_start, end_utc=horizon_end)
    return gaps[0] if gaps else None


def _needs_cohort_refresh(conn, *, config: RipeAtlasPublicSourceConfig, now: datetime) -> bool:
    active = _active_cohort(conn)
    if active is None:
        return True
    return ensure_utc(now) - active.created_at_utc >= timedelta(hours=config.cohort_refresh_hours)


def _result_dict(result: RipeIngestResult) -> dict[str, Any]:
    return {
        "source_id": result.source_id,
        "cohort_id": result.cohort_id,
        "radius": result.selected_radius_km,
        "selected_probe_count": result.selected_probe_count,
        "unique_asn_count": result.unique_asn_count,
        "requested_range": {"start_utc": to_utc_iso(result.start_utc), "end_utc": to_utc_iso(result.end_utc)},
        "measurement_count": result.measurement_count,
        "request_count": result.request_count,
        "raw_result_count": result.raw_result_count,
        "valid_parsed_result_count": result.valid_parsed_result_count,
        "finalized_aggregate_bin_count": result.finalized_aggregate_bin_count,
        "inserted_revision_count": result.inserted_observations,
        "duplicate_count": result.duplicate_observations,
        "skipped_low_quality_bin_count": result.skipped_low_quality_bin_count,
        "unresolved_gaps": list(result.unresolved_gaps),
        "newest_finalized_bin": to_utc_iso(result.newest_finalized_bin_utc) if result.newest_finalized_bin_utc else None,
    }


def _cohort_from_row(row) -> ProbeCohort:
    return ProbeCohort(
        cohort_id=int(row["cohort_id"]),
        source_id=row["source_id"],
        center_latitude=float(row["center_latitude"]),
        center_longitude=float(row["center_longitude"]),
        selected_radius_km=int(row["selected_radius_km"]),
        created_at_utc=parse_utc(row["created_at_utc"]),
        effective_start_utc=parse_utc(row["effective_start_utc"]),
        effective_end_utc=parse_utc(row["effective_end_utc"]) if row["effective_end_utc"] else None,
        selection_version=row["selection_version"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _member_from_row(row) -> ProbeCohortMember:
    return ProbeCohortMember(
        cohort_id=int(row["cohort_id"]),
        probe_id=int(row["probe_id"]),
        asn_v4=int(row["asn_v4"]) if row["asn_v4"] is not None else None,
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        distance_km=float(row["distance_km"]),
        is_anchor=bool(row["is_anchor"]),
        effective_start_utc=parse_utc(row["effective_start_utc"]),
        effective_end_utc=parse_utc(row["effective_end_utc"]) if row["effective_end_utc"] else None,
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _member_from_probe(probe: RipeProbe, *, cohort_id: int, effective_start_utc: datetime) -> ProbeCohortMember:
    return ProbeCohortMember(
        cohort_id=cohort_id,
        probe_id=probe.probe_id,
        asn_v4=probe.asn_v4,
        latitude=probe.latitude,
        longitude=probe.longitude,
        distance_km=probe.distance_km,
        is_anchor=probe.is_anchor,
        effective_start_utc=effective_start_utc,
        effective_end_utc=None,
        metadata=probe.metadata,
    )


def _eligible_retained_members(members: Sequence[ProbeCohortMember]) -> tuple[RipeProbe, ...]:
    return tuple(
        RipeProbe(
            probe_id=member.probe_id,
            asn_v4=member.asn_v4,
            latitude=member.latitude,
            longitude=member.longitude,
            distance_km=member.distance_km,
            is_anchor=member.is_anchor,
            status=1,
            is_public=True,
            has_ipv4=True,
            metadata=member.metadata,
        )
        for member in members
        if member.effective_end_utc is None
    )


def _radius_steps(initial: int, maximum: int) -> tuple[int, ...]:
    if initial > maximum:
        return ()
    step = max(25, min(100, (maximum - initial) // 4 or 50))
    values = list(range(initial, maximum + 1, step))
    if values[-1] != maximum:
        values.append(maximum)
    return tuple(dict.fromkeys(values))


def _probe_status(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _optional_int(value.get("id"))
    return _optional_int(value)


def _probe_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "id": row.get("id"),
        "asn_v4": row.get("asn_v4"),
        "status": row.get("status"),
        "is_public": row.get("is_public", row.get("public")),
        "is_anchor": row.get("is_anchor", row.get("anchor")),
        "address_v4": row.get("address_v4"),
    }


def _member_effective_for(member: ProbeCohortMember, timestamp: datetime) -> bool:
    value = ensure_utc(timestamp)
    return member.effective_start_utc <= value and (member.effective_end_utc is None or value < member.effective_end_utc)


def _batches(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index : index + size])


def _floor_time(value: datetime, cadence_seconds: int) -> datetime:
    utc = ensure_utc(value)
    epoch = int(utc.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % cadence_seconds), tz=timezone.utc)


def _parse_result_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        if value < 0:
            raise RipeAtlasError("malformed negative result timestamp")
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0)
    try:
        return parse_utc(str(value))
    except ValueError as exc:
        raise RipeAtlasError("malformed result timestamp") from exc


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = _optional_int(row.get(key))
    if value is None:
        raise RipeAtlasError(f"RIPE Atlas result missing {key}")
    return value


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_nonnegative_float(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    if numeric < 0:
        raise RipeAtlasError("negative RTT is invalid")
    return numeric


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _expected_bin_count(start_utc: datetime, end_utc: datetime, cadence_seconds: int) -> int:
    if start_utc >= end_utc:
        return 0
    return int((end_utc - start_utc).total_seconds()) // cadence_seconds


def _is_composite_core(series_id: str) -> bool:
    return series_id in {
        f"{SOURCE_ID}:regional:median_rtt_ms",
        f"{SOURCE_ID}:regional:p90_rtt_ms",
        f"{SOURCE_ID}:regional:packet_loss_fraction",
    }


def _valid_coordinates(latitude: float, longitude: float) -> bool:
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _result_digest(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(row), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _revision_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _probe_ids_hash(probe_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(probe_id) for probe_id in sorted(probe_ids)).encode("ascii")).hexdigest()


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, min(30.0, float(value)))
    except ValueError:
        return None


def _redact_api_key(text: str, api_key: str | None = None) -> str:
    redacted = text
    for value in (api_key, os.environ.get("RIPE_ATLAS_API_KEY")):
        if value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _redact_mapping(value: Mapping[str, Any], api_key: str | None) -> Mapping[str, Any]:
    return json.loads(_redact_api_key(json.dumps(dict(value), default=str), api_key))


def _safe_error_message(exc: BaseException | None) -> str:
    return _redact_api_key(str(exc))


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)


def _empty_result(*, cohort: ProbeCohort | None, members: Sequence[ProbeCohortMember], start_utc: datetime, end_utc: datetime) -> RipeIngestResult:
    return RipeIngestResult(
        source_id=SOURCE_ID,
        cohort_id=cohort.cohort_id if cohort else None,
        selected_radius_km=cohort.selected_radius_km if cohort else None,
        selected_probe_count=len(members),
        unique_asn_count=len({member.asn_v4 for member in members if member.asn_v4 is not None}),
        start_utc=start_utc,
        end_utc=end_utc,
        measurement_count=0,
        request_count=0,
        raw_result_count=0,
        valid_parsed_result_count=0,
        finalized_aggregate_bin_count=0,
        inserted_observations=0,
        duplicate_observations=0,
        skipped_low_quality_bin_count=0,
        unresolved_gaps=(),
        newest_finalized_bin_utc=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
