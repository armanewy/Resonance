from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from resonance.config import EiaGridPublicSourceConfig, load_config
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


SOURCE_ID = "eia_grid_monitor"
BALANCING_AUTHORITY_ID = "ISNE"
BALANCING_AUTHORITY_NAME = "ISO New England"
REGION_ROUTE = "electricity/rto/region-data"
FUEL_ROUTE = "electricity/rto/fuel-type-data"
ROUTES = (REGION_ROUTE, FUEL_ROUTE)
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRIES = 2
DEFAULT_PAGE_LENGTH = 5000
DEFAULT_MAX_PAGES = 100
DEFAULT_RAW_ROOT = Path("data/public/raw")
EIA_BASE_URL = "https://api.eia.gov/v2"

REGION_ALLOWED_TYPES = ("D", "DF", "TI")
FUEL_ALLOWED_TYPES = ("NG", "WND")


class EiaGridError(RuntimeError):
    """Raised when the EIA grid connector cannot fetch or normalize data."""


@dataclass(frozen=True)
class EiaPageFetchResult:
    payload: Mapping[str, Any]
    raw_bytes: bytes
    request_url: str
    status_code: int | None
    retrieved_at_utc: datetime
    route: str
    page_offset: int = 0
    total: int = -1
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


EiaFetchResult = EiaPageFetchResult


@dataclass(frozen=True)
class EiaIngestResult:
    source_id: str
    inserted_observations: int
    duplicate_observations: int
    parsed_observations: int
    raw_row_count: int
    page_count: int
    raw_archives: tuple[PublicRawArchive, ...]
    start_utc: datetime
    end_utc: datetime
    unresolved_gaps: tuple[str, ...]
    newest_valid_period_utc: datetime | None


PageFetchCallable = Callable[..., EiaPageFetchResult]


SERIES: tuple[SeriesRecord, ...] = (
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:system_load",
        source_id=SOURCE_ID,
        metric_name="system_load",
        display_name="ISO New England system load",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
        parent_series_id=None,
        lineage_id="eia_grid_monitor:ISNE:system_load",
        quality_tier="official",
        metadata={"eia_route": REGION_ROUTE, "eia_type": "D", "eia_type_name": "Demand"},
    ),
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:demand_forecast",
        source_id=SOURCE_ID,
        metric_name="demand_forecast",
        display_name="ISO New England demand forecast",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
        parent_series_id=None,
        lineage_id="eia_grid_monitor:ISNE:demand_forecast",
        quality_tier="official",
        metadata={"eia_route": REGION_ROUTE, "eia_type": "DF", "eia_type_name": "Demand Forecast"},
    ),
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:forecast_error",
        source_id=SOURCE_ID,
        metric_name="forecast_error",
        display_name="ISO New England demand forecast error",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly_difference",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="Derived as system_load minus demand_forecast for the same EIA hourly UTC period",
        parent_series_id="eia_grid_monitor:ISNE:demand_forecast",
        lineage_id="eia_grid_monitor:ISNE:forecast_error",
        quality_tier="derived",
        metadata={"derived_from": ("D", "DF"), "formula": "system_load - demand_forecast"},
    ),
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:net_interchange",
        source_id=SOURCE_ID,
        metric_name="net_interchange",
        display_name="ISO New England net interchange",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
        parent_series_id=None,
        lineage_id="eia_grid_monitor:ISNE:net_interchange",
        quality_tier="official",
        metadata={"eia_route": REGION_ROUTE, "eia_type": "TI", "eia_type_name": "Total Interchange"},
    ),
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:generation_natural_gas",
        source_id=SOURCE_ID,
        metric_name="generation_natural_gas",
        display_name="ISO New England natural gas generation",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
        parent_series_id=None,
        lineage_id="eia_grid_monitor:ISNE:generation_mix",
        quality_tier="official",
        metadata={"eia_route": FUEL_ROUTE, "eia_fuel_type": "NG", "eia_fuel_type_name": "Natural Gas"},
    ),
    SeriesRecord(
        series_id="eia_grid_monitor:ISNE:generation_wind",
        source_id=SOURCE_ID,
        metric_name="generation_wind",
        display_name="ISO New England wind generation",
        unit="MWh",
        cadence_seconds=3600,
        aggregation="hourly",
        geography_type="balancing_authority",
        geography_id=BALANCING_AUTHORITY_ID,
        timezone="America/New_York",
        timestamp_semantics="EIA hourly UTC period treated as valid hour starting at period",
        parent_series_id=None,
        lineage_id="eia_grid_monitor:ISNE:generation_mix",
        quality_tier="official",
        metadata={"eia_route": FUEL_ROUTE, "eia_fuel_type": "WND", "eia_fuel_type_name": "Wind"},
    ),
)

REGION_TYPE_TO_SERIES = {
    "D": "eia_grid_monitor:ISNE:system_load",
    "DF": "eia_grid_monitor:ISNE:demand_forecast",
    "TI": "eia_grid_monitor:ISNE:net_interchange",
}
FUEL_TYPE_TO_SERIES = {
    "NG": "eia_grid_monitor:ISNE:generation_natural_gas",
    "WND": "eia_grid_monitor:ISNE:generation_wind",
}
ROUTE_SERIES = {
    REGION_ROUTE: ("eia_grid_monitor:ISNE:system_load", "eia_grid_monitor:ISNE:demand_forecast"),
    FUEL_ROUTE: ("eia_grid_monitor:ISNE:generation_natural_gas", "eia_grid_monitor:ISNE:generation_wind"),
}


def eia_source_record(*, enabled: bool = False) -> PublicSource:
    return PublicSource(
        source_id=SOURCE_ID,
        display_name="EIA Hourly Electric Grid Monitor",
        publisher="U.S. Energy Information Administration",
        documentation_reference="https://www.eia.gov/opendata/browser/electricity/rto/region-data",
        license_summary="Official U.S. government public data; review EIA copyright and reuse policies.",
        authentication_type="api_key_env:EIA_API_KEY",
        default_polling_cadence_seconds=3600,
        quality_tier="official",
        enabled=enabled,
        metadata={
            "product": "Form EIA-930 / Hourly Electric Grid Monitor",
            "balancing_authority": BALANCING_AUTHORITY_ID,
        },
    )


def ensure_eia_registry(conn, *, enabled: bool = False) -> None:
    upsert_public_source(conn, eia_source_record(enabled=enabled))
    for series in SERIES:
        upsert_series_record(conn, series)
    conn.commit()


def backfill_new_england_grid(
    conn,
    *,
    start_utc: datetime,
    end_utc: datetime,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    raw_root: Path = DEFAULT_RAW_ROOT,
    page_fetcher: PageFetchCallable | None = None,
    fetcher: PageFetchCallable | None = None,
    now: datetime | None = None,
    page_length: int = DEFAULT_PAGE_LENGTH,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> EiaIngestResult:
    start = _floor_hour(start_utc)
    end = _floor_hour(end_utc)
    if start > end:
        raise EiaGridError("start_utc must be before or equal to end_utc")
    resolved_api_key = api_key or os.environ.get("EIA_API_KEY", "")
    ensure_eia_registry(conn, enabled=bool(resolved_api_key))
    page_fetch = _page_fetcher(page_fetcher or fetcher, resolved_api_key, timeout_seconds, retries)
    return _ingest_routes(
        conn,
        route_ranges={route: (start, end) for route in ROUTES},
        page_fetcher=page_fetch,
        api_key=resolved_api_key,
        raw_root=raw_root,
        now=now,
        page_length=page_length,
        max_pages=max_pages,
    )


def poll_new_england_grid(
    conn,
    *,
    config: EiaGridPublicSourceConfig | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    raw_root: Path = DEFAULT_RAW_ROOT,
    page_fetcher: PageFetchCallable | None = None,
    fetcher: PageFetchCallable | None = None,
    now: datetime | None = None,
    page_length: int = DEFAULT_PAGE_LENGTH,
    max_pages: int = DEFAULT_MAX_PAGES,
    lookback_hours: int | None = None,
) -> EiaIngestResult:
    settings = config or EiaGridPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=3600,
        initial_backfill_hours=720,
        normal_lookback_hours=lookback_hours or 72,
        maximum_gap_repair_hours=2160,
    )
    current = _floor_hour(now or utc_now())
    resolved_api_key = api_key or os.environ.get("EIA_API_KEY", "")
    ensure_eia_registry(conn, enabled=settings.enabled and bool(resolved_api_key))
    page_fetch = _page_fetcher(page_fetcher or fetcher, resolved_api_key, timeout_seconds, retries)
    route_ranges = {
        route: _route_poll_range(conn, route, settings=settings, now=current)
        for route in ROUTES
    }
    return _ingest_routes(
        conn,
        route_ranges=route_ranges,
        page_fetcher=page_fetch,
        api_key=resolved_api_key,
        raw_root=raw_root,
        now=current,
        page_length=page_length,
        max_pages=max_pages,
    )


def parse_eia_observations(
    payload: Mapping[str, Any],
    *,
    route: str,
    ingested_at_utc: datetime,
    raw_archive_sha256: str | None,
) -> tuple[PublicObservation, ...]:
    rows = _payload_rows(payload)
    direct: list[PublicObservation] = []
    demand_by_period: dict[str, float] = {}
    forecast_by_period: dict[str, float] = {}
    for row in rows:
        respondent = _row_value(row, "respondent", "balancing_authority", "ba")
        if respondent and respondent != BALANCING_AUTHORITY_ID:
            continue
        if route == REGION_ROUTE:
            code = _row_value(row, "type", "type-code", "series")
            series_id = REGION_TYPE_TO_SERIES.get(code or "")
        elif route == FUEL_ROUTE:
            code = _row_value(row, "fueltype", "fuel-type", "type", "type-code")
            series_id = FUEL_TYPE_TO_SERIES.get(code or "")
        else:
            continue
        if not series_id or not code:
            continue
        value = _numeric_value(row)
        if value is None:
            continue
        period = _required_period(row)
        if route == REGION_ROUTE and code == "D":
            demand_by_period[period] = value
        if route == REGION_ROUTE and code == "DF":
            forecast_by_period[period] = value
        direct.append(
            _observation_from_row(
                row,
                route=route,
                code=code,
                series_id=series_id,
                value=value,
                ingested_at_utc=ingested_at_utc,
                quality="reported",
                raw_archive_sha256=raw_archive_sha256,
            )
        )
    derived = [
        _derived_forecast_error(period, demand_by_period[period] - forecast_by_period[period], ingested_at_utc, raw_archive_sha256)
        for period in sorted(set(demand_by_period).intersection(forecast_by_period))
    ]
    return tuple((*direct, *derived))


def archive_raw_response(
    *,
    source_id: str,
    raw_bytes: bytes,
    request_url: str,
    status_code: int | None,
    retrieved_at_utc: datetime,
    raw_root: Path = DEFAULT_RAW_ROOT,
    metadata: Mapping[str, Any] | None = None,
) -> PublicRawArchive:
    retrieved = ensure_utc(retrieved_at_utc).replace(microsecond=0)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    directory = raw_root / source_id / f"{retrieved.year:04d}" / f"{retrieved.month:02d}" / f"{retrieved.day:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if not path.exists():
        path.write_bytes(raw_bytes)
    return PublicRawArchive(
        sha256=digest,
        source_id=source_id,
        retrieved_at_utc=retrieved,
        request_url=request_url,
        status_code=status_code,
        path=str(path),
        metadata=metadata or {},
    )


class EiaGridClient:
    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if not api_key:
            raise EiaGridError("EIA_API_KEY is required for live EIA grid collection")
        if timeout_seconds <= 0:
            raise EiaGridError("timeout_seconds must be positive")
        if retries < 0:
            raise EiaGridError("retries must be non-negative")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._retries = retries

    def fetch_page(
        self,
        route: str,
        start_utc: datetime,
        end_utc: datetime,
        offset: int,
        length: int,
    ) -> EiaPageFetchResult:
        url = f"{EIA_BASE_URL}/{route}/data/"
        facets = _route_facets(route)
        params = _request_params(self._api_key, route, start_utc, end_utc, offset, length)
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.get(url, params=params)
                if response.status_code >= 500 and attempt < self._retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                response.raise_for_status()
                payload = response.json()
                total = _payload_total(payload)
                return EiaPageFetchResult(
                    payload=payload,
                    raw_bytes=response.content,
                    request_url=str(response.url),
                    status_code=response.status_code,
                    retrieved_at_utc=utc_now(),
                    route=route,
                    page_offset=offset,
                    total=total,
                    request_metadata={
                        "page_offset": offset,
                        "length": length,
                        "total": total,
                        "route": route,
                        "requested_facets": facets,
                    },
                )
            except (httpx.HTTPError, json.JSONDecodeError, EiaGridError) as exc:
                last_error = exc
                if attempt < self._retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break
        raise EiaGridError(f"EIA request failed for {route}: {_redact_api_key(str(last_error), self._api_key)}") from last_error


def status_payload(conn, *, config: EiaGridPublicSourceConfig | None = None, now: datetime | None = None) -> dict[str, Any]:
    from resonance.public_health import eia_source_health

    settings = config or load_config().public_sources.eia_grid
    ensure_eia_registry(conn, enabled=settings.enabled and bool(os.environ.get("EIA_API_KEY")))
    return eia_source_health(
        conn,
        config=settings,
        now_utc=ensure_utc(now or utc_now()),
        credential_available=bool(os.environ.get("EIA_API_KEY")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect EIA New England hourly grid data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--database", default="data/resonance.db")
    status.add_argument("--config", default="config.toml")
    for name in ("backfill", "poll"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--database", default="data/resonance.db")
        sub.add_argument("--config", default="config.toml")
        sub.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
        sub.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
        sub.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
        sub.add_argument("--page-length", type=int, default=DEFAULT_PAGE_LENGTH)
        sub.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    backfill = subparsers.choices["backfill"]
    backfill.add_argument("--start", required=True, help="UTC ISO start, e.g. 2026-06-19T00:00:00Z")
    backfill.add_argument("--end", required=True, help="UTC ISO end, e.g. 2026-06-20T00:00:00Z")
    args = parser.parse_args(argv)

    conn = ensure_database(args.database)
    try:
        config = load_config(args.config).public_sources.eia_grid
        api_key = os.environ.get("EIA_API_KEY", "")
        if args.command == "status":
            print(json.dumps(status_payload(conn, config=config), indent=2, sort_keys=True))
            return 0
        try:
            if args.command == "backfill":
                result = backfill_new_england_grid(
                    conn,
                    start_utc=parse_utc(args.start),
                    end_utc=parse_utc(args.end),
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                    raw_root=Path(args.raw_root),
                    page_length=args.page_length,
                    max_pages=args.max_pages,
                )
            else:
                result = poll_new_england_grid(
                    conn,
                    config=config,
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                    raw_root=Path(args.raw_root),
                    page_length=args.page_length,
                    max_pages=args.max_pages,
                )
        except Exception as exc:
            _record_route_failure(conn, ROUTES, exc, utc_now())
            insert_collector_error(
                conn,
                CollectorError(utc_now(), SOURCE_ID, exc.__class__.__name__, _safe_error_message(exc, api_key)),
            )
            parser.exit(2, f"EIA grid collection failed: {_safe_error_message(exc, api_key)}\n")
        print(json.dumps(_result_dict(result), indent=2, sort_keys=True))
    finally:
        conn.close()
    return 0


def _ingest_routes(
    conn,
    *,
    route_ranges: Mapping[str, tuple[datetime, datetime]],
    page_fetcher: PageFetchCallable,
    api_key: str,
    raw_root: Path,
    now: datetime | None,
    page_length: int,
    max_pages: int,
) -> EiaIngestResult:
    ingested_at = ensure_utc(now or utc_now()).replace(microsecond=0)
    archives: list[PublicRawArchive] = []
    observations: list[PublicObservation] = []
    raw_row_count = 0
    page_count = 0
    for route, (start, end) in route_ranges.items():
        pages = _fetch_complete_route(
            page_fetcher,
            route=route,
            start_utc=start,
            end_utc=end,
            page_length=page_length,
            max_pages=max_pages,
        )
        page_count += len(pages)
        for page in pages:
            rows = _payload_rows(page.payload)
            raw_row_count += len(rows)
            archive = archive_raw_response(
                source_id=SOURCE_ID,
                raw_bytes=page.raw_bytes,
                request_url=_redact_api_key(page.request_url, api_key),
                status_code=page.status_code,
                retrieved_at_utc=page.retrieved_at_utc,
                raw_root=raw_root,
                metadata={
                    "route": page.route,
                    "page_offset": page.page_offset,
                    "total": page.total,
                    "requested_facets": _route_facets(page.route),
                },
            )
            record_public_raw_archive(conn, archive)
            record_public_fetch_event(
                conn,
                PublicFetchEvent(
                    source_id=SOURCE_ID,
                    retrieved_at_utc=page.retrieved_at_utc,
                    request_url=_redact_api_key(page.request_url, api_key),
                    status_code=page.status_code,
                    content_sha256=archive.sha256,
                    route=page.route,
                    page_offset=page.page_offset,
                    request_metadata={
                        **dict(page.request_metadata),
                        "requested_facets": _route_facets(page.route),
                    },
                ),
            )
            archives.append(archive)
            observations.extend(
                parse_eia_observations(
                    page.payload,
                    route=page.route,
                    ingested_at_utc=ingested_at,
                    raw_archive_sha256=archive.sha256,
                )
            )
    inserted = insert_public_observations(conn, observations)
    duplicate_count = len(observations) - inserted
    unresolved: list[str] = []
    newest_values: list[datetime] = []
    for route in route_ranges:
        newest = _newest_valid_period(conn, route)
        gap = _earliest_gap(conn, route, horizon_start=route_ranges[route][0], horizon_end=newest) if newest else None
        if newest:
            newest_values.append(newest)
        if gap:
            unresolved.append(f"{route}:{to_utc_iso(gap)}")
        upsert_public_collection_state(
            conn,
            PublicCollectionState(
                source_id=SOURCE_ID,
                route=route,
                last_successful_poll_utc=ingested_at,
                newest_complete_valid_period_utc=newest,
                earliest_unresolved_gap_utc=gap,
                latest_error_utc=None,
                latest_error="",
                consecutive_failure_count=0,
                metadata={"last_requested_start_utc": to_utc_iso(route_ranges[route][0]), "last_requested_end_utc": to_utc_iso(route_ranges[route][1])},
            ),
        )
    conn.commit()
    requested_start = min(start for start, _end in route_ranges.values())
    requested_end = max(end for _start, end in route_ranges.values())
    return EiaIngestResult(
        source_id=SOURCE_ID,
        inserted_observations=inserted,
        duplicate_observations=duplicate_count,
        parsed_observations=len(observations),
        raw_row_count=raw_row_count,
        page_count=page_count,
        raw_archives=tuple(archives),
        start_utc=requested_start,
        end_utc=requested_end,
        unresolved_gaps=tuple(unresolved),
        newest_valid_period_utc=max(newest_values) if newest_values else None,
    )


def _fetch_complete_route(
    page_fetcher: PageFetchCallable,
    *,
    route: str,
    start_utc: datetime,
    end_utc: datetime,
    page_length: int,
    max_pages: int,
) -> tuple[EiaPageFetchResult, ...]:
    if page_length <= 0:
        raise EiaGridError("page_length must be positive")
    if max_pages <= 0:
        raise EiaGridError("max_pages must be positive")
    pages: list[EiaPageFetchResult] = []
    seen_hashes: set[str] = set()
    expected_total: int | None = None
    received = 0
    offset = 0
    for _page_index in range(max_pages):
        page = _invoke_page_fetcher(page_fetcher, route, start_utc, end_utc, offset, page_length)
        rows = _payload_rows(page.payload)
        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise EiaGridError(f"EIA pagination total changed for {route}: {expected_total} -> {page.total}")
        page_digest = hashlib.sha256(page.raw_bytes).hexdigest()
        if rows and page_digest in seen_hashes:
            raise EiaGridError(f"EIA pagination repeated a page for {route} at offset {offset}")
        seen_hashes.add(page_digest)
        pages.append(page)
        received += len(rows)
        if received >= expected_total:
            if received != expected_total:
                raise EiaGridError(f"EIA pagination received {received} rows for expected total {expected_total} on {route}")
            return tuple(pages)
        if not rows:
            raise EiaGridError(f"EIA pagination ended early for {route}: received {received} of {expected_total}")
        offset += len(rows)
    raise EiaGridError(f"EIA pagination exceeded maximum page limit for {route}")


def _page_fetcher(
    supplied: PageFetchCallable | None,
    api_key: str,
    timeout_seconds: float,
    retries: int,
) -> PageFetchCallable:
    if supplied is not None:
        return supplied
    return EiaGridClient(api_key=api_key, timeout_seconds=timeout_seconds, retries=retries).fetch_page


def _invoke_page_fetcher(
    fetcher: PageFetchCallable,
    route: str,
    start: datetime,
    end: datetime,
    offset: int,
    length: int,
) -> EiaPageFetchResult:
    parameter_count = len(inspect.signature(fetcher).parameters)
    if parameter_count <= 3:
        result = fetcher(route, start, end)
    else:
        result = fetcher(route, start, end, offset, length)
    if not isinstance(result, EiaPageFetchResult):
        raise EiaGridError("EIA page fetcher returned an invalid result")
    if result.total < 0:
        return EiaPageFetchResult(
            payload=result.payload,
            raw_bytes=result.raw_bytes,
            request_url=result.request_url,
            status_code=result.status_code,
            retrieved_at_utc=result.retrieved_at_utc,
            route=result.route,
            page_offset=offset,
            total=len(_payload_rows(result.payload)),
            request_metadata={
                **dict(result.request_metadata),
                "page_offset": offset,
                "length": length,
                "total": len(_payload_rows(result.payload)),
                "route": route,
                "requested_facets": _route_facets(route),
            },
        )
    return result


def _route_poll_range(
    conn,
    route: str,
    *,
    settings: EiaGridPublicSourceConfig,
    now: datetime,
) -> tuple[datetime, datetime]:
    end = _floor_hour(now)
    state = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=route)
    horizon_start = end - timedelta(hours=settings.maximum_gap_repair_hours)
    if state is None or state.newest_complete_valid_period_utc is None:
        start = end - timedelta(hours=settings.initial_backfill_hours)
    else:
        newest = state.newest_complete_valid_period_utc
        start = newest - timedelta(hours=settings.normal_lookback_hours)
        gap = state.earliest_unresolved_gap_utc or _earliest_gap(conn, route, horizon_start=horizon_start, horizon_end=newest)
        if gap is not None:
            start = min(start, gap)
    return max(horizon_start, _floor_hour(start)), end


def _record_route_failure(conn, routes: Sequence[str], exc: BaseException, now: datetime) -> None:
    safe = _safe_error_message(exc, os.environ.get("EIA_API_KEY"))
    for route in routes:
        previous = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=route)
        upsert_public_collection_state(
            conn,
            PublicCollectionState(
                source_id=SOURCE_ID,
                route=route,
                last_successful_poll_utc=previous.last_successful_poll_utc if previous else None,
                newest_complete_valid_period_utc=previous.newest_complete_valid_period_utc if previous else None,
                earliest_unresolved_gap_utc=previous.earliest_unresolved_gap_utc if previous else None,
                latest_error_utc=ensure_utc(now),
                latest_error=safe,
                consecutive_failure_count=(previous.consecutive_failure_count if previous else 0) + 1,
                metadata=previous.metadata if previous else {},
            ),
        )
    conn.commit()


def _newest_valid_period(conn, route: str) -> datetime | None:
    placeholders = ",".join("?" for _ in ROUTE_SERIES[route])
    row = conn.execute(
        f"""
        SELECT MAX(valid_start_utc) AS newest
        FROM public_observations
        WHERE series_id IN ({placeholders})
        """,
        ROUTE_SERIES[route],
    ).fetchone()
    return parse_utc(row["newest"]) if row and row["newest"] else None


def _earliest_gap(
    conn,
    route: str,
    *,
    horizon_start: datetime,
    horizon_end: datetime | None,
) -> datetime | None:
    if horizon_end is None:
        return None
    start = _floor_hour(horizon_start)
    end = _floor_hour(horizon_end)
    if start > end:
        return None
    placeholders = ",".join("?" for _ in ROUTE_SERIES[route])
    rows = conn.execute(
        f"""
        SELECT DISTINCT valid_start_utc
        FROM public_observations
        WHERE series_id IN ({placeholders})
          AND valid_start_utc >= ?
          AND valid_start_utc <= ?
        """,
        (*ROUTE_SERIES[route], to_utc_iso(start), to_utc_iso(end)),
    ).fetchall()
    observed = {parse_utc(row["valid_start_utc"]) for row in rows}
    if not observed:
        return None
    first_observed = min(observed)
    cursor = max(start, first_observed)
    while cursor <= end:
        if cursor not in observed:
            return cursor
        cursor += timedelta(hours=1)
    return None


def _payload_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    response = payload.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("data"), list):
        return [row for row in response["data"] if isinstance(row, Mapping)]
    if isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, Mapping)]
    raise EiaGridError("EIA payload does not contain response.data")


def _payload_total(payload: Mapping[str, Any]) -> int:
    response = payload.get("response")
    total = response.get("total") if isinstance(response, Mapping) else payload.get("total")
    if total is None:
        raise EiaGridError("EIA payload does not contain response.total")
    try:
        parsed = int(total)
    except (TypeError, ValueError) as exc:
        raise EiaGridError("EIA response.total is not an integer") from exc
    if parsed < 0:
        raise EiaGridError("EIA response.total must be non-negative")
    return parsed


def _observation_from_row(
    row: Mapping[str, Any],
    *,
    route: str,
    code: str,
    series_id: str,
    value: float,
    ingested_at_utc: datetime,
    quality: str,
    raw_archive_sha256: str | None,
) -> PublicObservation:
    period = _required_period(row)
    valid_start = _parse_eia_period(period)
    valid_end = valid_start + timedelta(hours=1)
    key = f"{route}:{BALANCING_AUTHORITY_ID}:{code}:{period}"
    return PublicObservation(
        series_id=series_id,
        valid_start_utc=valid_start,
        valid_end_utc=valid_end,
        observed_at_utc=valid_end,
        ingested_at_utc=ensure_utc(ingested_at_utc),
        value=value,
        quality=quality,
        source_revision=_revision_hash({"route": route, "code": code, "period": period, "value": value, "row": dict(row)}),
        source_observation_key=key,
        raw_archive_sha256=raw_archive_sha256,
        metadata={"eia_period": period, "eia_code": code, "eia_route": route},
    )


def _derived_forecast_error(
    period: str,
    value: float,
    ingested_at_utc: datetime,
    raw_archive_sha256: str | None,
) -> PublicObservation:
    valid_start = _parse_eia_period(period)
    valid_end = valid_start + timedelta(hours=1)
    key = f"derived:forecast_error:{BALANCING_AUTHORITY_ID}:{period}"
    return PublicObservation(
        series_id="eia_grid_monitor:ISNE:forecast_error",
        valid_start_utc=valid_start,
        valid_end_utc=valid_end,
        observed_at_utc=valid_end,
        ingested_at_utc=ensure_utc(ingested_at_utc),
        value=value,
        quality="derived",
        source_revision=_revision_hash({"period": period, "value": value, "formula": "D-DF"}),
        source_observation_key=key,
        raw_archive_sha256=raw_archive_sha256,
        metadata={"formula": "system_load - demand_forecast", "eia_period": period},
    )


def _request_params(
    api_key: str,
    route: str,
    start_utc: datetime,
    end_utc: datetime,
    offset: int,
    length: int,
) -> list[tuple[str, str]]:
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", BALANCING_AUTHORITY_ID),
        ("start", _eia_period(ensure_utc(start_utc))),
        ("end", _eia_period(ensure_utc(end_utc))),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("sort[1][column]", "respondent"),
        ("sort[1][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(length)),
    ]
    for key, values in _route_facets(route).items():
        for value in values:
            params.append((f"facets[{key}][]", value))
    return params


def _route_facets(route: str) -> dict[str, tuple[str, ...]]:
    if route == REGION_ROUTE:
        return {"type": REGION_ALLOWED_TYPES}
    if route == FUEL_ROUTE:
        return {"fueltype": FUEL_ALLOWED_TYPES}
    raise EiaGridError(f"unsupported EIA route: {route}")


def _required_period(row: Mapping[str, Any]) -> str:
    period = _row_value(row, "period", "time", "utc_time")
    if not period:
        raise EiaGridError("EIA row missing period")
    return period


def _row_value(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _numeric_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("value")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_eia_period(period: str) -> datetime:
    normalized = period.strip()
    if normalized.endswith("Z") or "+" in normalized[10:] or "-" in normalized[10:]:
        return ensure_utc(datetime.fromisoformat(normalized.replace("Z", "+00:00"))).replace(minute=0, second=0, microsecond=0)
    return datetime.strptime(normalized[:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def _eia_period(value: datetime) -> str:
    return ensure_utc(value).strftime("%Y-%m-%dT%H")


def _floor_hour(value: datetime) -> datetime:
    return ensure_utc(value).replace(minute=0, second=0, microsecond=0)


def _revision_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _redact_api_key(url: str, api_key: str | None = None) -> str:
    redacted = url
    for value in (api_key, os.environ.get("EIA_API_KEY")):
        if value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _safe_error_message(exc: BaseException, api_key: str | None) -> str:
    return _redact_api_key(str(exc), api_key)


def _result_dict(result: EiaIngestResult) -> dict[str, Any]:
    return {
        "source_id": result.source_id,
        "requested_range": {
            "start_utc": to_utc_iso(result.start_utc),
            "end_utc": to_utc_iso(result.end_utc),
        },
        "page_count": result.page_count,
        "raw_row_count": result.raw_row_count,
        "relevant_parsed_row_count": result.parsed_observations,
        "inserted_revision_count": result.inserted_observations,
        "duplicate_count": result.duplicate_observations,
        "unresolved_gaps": list(result.unresolved_gaps),
        "newest_valid_period": to_utc_iso(result.newest_valid_period_utc) if result.newest_valid_period_utc else None,
        "raw_archives": [
            {
                "sha256": archive.sha256,
                "path": archive.path,
                "retrieved_at_utc": to_utc_iso(archive.retrieved_at_utc),
            }
            for archive in result.raw_archives
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
