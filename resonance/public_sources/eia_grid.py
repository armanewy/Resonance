from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from resonance.storage import (
    CollectorError,
    PublicObservation,
    PublicRawArchive,
    PublicSource,
    SeriesRecord,
    ensure_database,
    insert_collector_error,
    insert_public_observations,
    record_public_raw_archive,
    upsert_public_source,
    upsert_series_record,
)
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


SOURCE_ID = "eia_grid_monitor"
BALANCING_AUTHORITY_ID = "ISNE"
BALANCING_AUTHORITY_NAME = "ISO New England"
REGION_ROUTE = "electricity/rto/region-data"
FUEL_ROUTE = "electricity/rto/fuel-type-data"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRIES = 2
DEFAULT_RAW_ROOT = Path("data/public/raw")
EIA_BASE_URL = "https://api.eia.gov/v2"


class EiaGridError(RuntimeError):
    """Raised when the EIA grid connector cannot fetch or normalize data."""


@dataclass(frozen=True)
class EiaFetchResult:
    payload: Mapping[str, Any]
    raw_bytes: bytes
    request_url: str
    status_code: int | None
    retrieved_at_utc: datetime
    route: str


@dataclass(frozen=True)
class EiaIngestResult:
    source_id: str
    inserted_observations: int
    parsed_observations: int
    raw_archives: tuple[PublicRawArchive, ...]
    start_utc: datetime
    end_utc: datetime


FetchCallable = Callable[[str, datetime, datetime], EiaFetchResult]


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
    fetcher: FetchCallable | None = None,
    now: datetime | None = None,
) -> EiaIngestResult:
    start = ensure_utc(start_utc).replace(minute=0, second=0, microsecond=0)
    end = ensure_utc(end_utc).replace(minute=0, second=0, microsecond=0)
    if start > end:
        raise EiaGridError("start_utc must be before or equal to end_utc")
    ensure_eia_registry(conn, enabled=bool(api_key))
    resolved_api_key = api_key or os.environ.get("EIA_API_KEY", "")
    resolved_fetcher = fetcher or EiaGridClient(
        api_key=resolved_api_key,
        timeout_seconds=timeout_seconds,
        retries=retries,
    ).fetch
    ingested_at = ensure_utc(now or utc_now()).replace(microsecond=0)
    fetch_results = (
        resolved_fetcher(REGION_ROUTE, start, end),
        resolved_fetcher(FUEL_ROUTE, start, end),
    )
    archives: list[PublicRawArchive] = []
    observations: list[PublicObservation] = []
    for result in fetch_results:
        archive = archive_raw_response(
            source_id=SOURCE_ID,
            raw_bytes=result.raw_bytes,
            request_url=_redact_api_key(result.request_url, resolved_api_key),
            status_code=result.status_code,
            retrieved_at_utc=result.retrieved_at_utc,
            raw_root=raw_root,
            metadata={"route": result.route},
        )
        record_public_raw_archive(conn, archive)
        archives.append(archive)
        observations.extend(
            parse_eia_observations(
                result.payload,
                route=result.route,
                ingested_at_utc=ingested_at,
                raw_archive_sha256=archive.sha256,
            )
        )
    inserted = insert_public_observations(conn, observations)
    return EiaIngestResult(
        source_id=SOURCE_ID,
        inserted_observations=inserted,
        parsed_observations=len(observations),
        raw_archives=tuple(archives),
        start_utc=start,
        end_utc=end,
    )


def poll_new_england_grid(
    conn,
    *,
    api_key: str | None = None,
    lookback_hours: int = 48,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    raw_root: Path = DEFAULT_RAW_ROOT,
    fetcher: FetchCallable | None = None,
    now: datetime | None = None,
) -> EiaIngestResult:
    end = ensure_utc(now or utc_now()).replace(minute=0, second=0, microsecond=0)
    latest = _latest_eia_valid_start(conn)
    if latest is None:
        start = end - timedelta(hours=lookback_hours)
    else:
        start = max(latest - timedelta(hours=max(1, min(lookback_hours, 24))), end - timedelta(hours=lookback_hours))
    return backfill_new_england_grid(
        conn,
        start_utc=start,
        end_utc=end,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        retries=retries,
        raw_root=raw_root,
        fetcher=fetcher,
        now=now,
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

    def fetch(self, route: str, start_utc: datetime, end_utc: datetime) -> EiaFetchResult:
        url = f"{EIA_BASE_URL}/{route}/data/"
        params = _request_params(self._api_key, start_utc, end_utc)
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.get(url, params=params)
                if response.status_code >= 500 and attempt < self._retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                response.raise_for_status()
                return EiaFetchResult(
                    payload=response.json(),
                    raw_bytes=response.content,
                    request_url=str(response.url),
                    status_code=response.status_code,
                    retrieved_at_utc=utc_now(),
                    route=route,
                )
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self._retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break
        raise EiaGridError(f"EIA request failed for {route}: {last_error}") from last_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect EIA New England hourly grid data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("backfill", "poll"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--database", default="data/resonance.db")
        sub.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
        sub.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
        sub.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    backfill = subparsers.choices["backfill"]
    backfill.add_argument("--start", required=True, help="UTC ISO start, e.g. 2026-06-19T00:00:00Z")
    backfill.add_argument("--end", required=True, help="UTC ISO end, e.g. 2026-06-20T00:00:00Z")
    poll = subparsers.choices["poll"]
    poll.add_argument("--lookback-hours", type=int, default=48)
    args = parser.parse_args(argv)

    conn = ensure_database(args.database)
    try:
        api_key = os.environ.get("EIA_API_KEY", "")
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
                )
            else:
                result = poll_new_england_grid(
                    conn,
                    api_key=api_key,
                    lookback_hours=args.lookback_hours,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                    raw_root=Path(args.raw_root),
                )
        except Exception as exc:
            insert_collector_error(
                conn,
                CollectorError(utc_now(), SOURCE_ID, exc.__class__.__name__, str(exc)),
            )
            parser.exit(2, f"EIA grid collection failed: {exc}\n")
        print(json.dumps(_result_dict(result), indent=2, sort_keys=True))
    finally:
        conn.close()
    return 0


def _payload_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    response = payload.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("data"), list):
        return [row for row in response["data"] if isinstance(row, Mapping)]
    if isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, Mapping)]
    raise EiaGridError("EIA payload does not contain response.data")


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


def _latest_eia_valid_start(conn) -> datetime | None:
    row = conn.execute(
        """
        SELECT MAX(o.valid_start_utc) AS valid_start_utc
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE s.source_id = ?
        """,
        (SOURCE_ID,),
    ).fetchone()
    if row is None or not row["valid_start_utc"]:
        return None
    return parse_utc(row["valid_start_utc"])


def _request_params(api_key: str, start_utc: datetime, end_utc: datetime) -> list[tuple[str, str]]:
    return [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", BALANCING_AUTHORITY_ID),
        ("start", _eia_period(ensure_utc(start_utc))),
        ("end", _eia_period(ensure_utc(end_utc))),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", "0"),
        ("length", "5000"),
    ]


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


def _revision_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _redact_api_key(url: str, api_key: str | None = None) -> str:
    redacted = url
    for value in (api_key, os.environ.get("EIA_API_KEY")):
        if value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


def _result_dict(result: EiaIngestResult) -> dict[str, Any]:
    return {
        "source_id": result.source_id,
        "inserted_observations": result.inserted_observations,
        "parsed_observations": result.parsed_observations,
        "start_utc": to_utc_iso(result.start_utc),
        "end_utc": to_utc_iso(result.end_utc),
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
