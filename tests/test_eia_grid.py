from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from resonance.public_series import fetch_series
from resonance.public_sources.eia_grid import (
    FUEL_ALLOWED_TYPES,
    FUEL_ROUTE,
    REGION_ROUTE,
    REGION_ALLOWED_TYPES,
    EiaFetchResult,
    EiaGridError,
    EiaPageFetchResult,
    SOURCE_ID,
    backfill_new_england_grid,
    main,
    parse_eia_observations,
    poll_new_england_grid,
    _payload_total,
)
from resonance.storage import ensure_database


START = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 6, 19, 1, 0, tzinfo=timezone.utc)


def test_eia_fixture_parser_normalizes_region_and_fuel_rows() -> None:
    region = _fixture("eia_region_data_isne.json")
    fuel = _fixture("eia_fuel_type_isne.json")

    region_observations = parse_eia_observations(
        region,
        route=REGION_ROUTE,
        ingested_at_utc=START + timedelta(days=1),
        raw_archive_sha256="region-sha",
    )
    fuel_observations = parse_eia_observations(
        fuel,
        route=FUEL_ROUTE,
        ingested_at_utc=START + timedelta(days=1),
        raw_archive_sha256="fuel-sha",
    )

    by_series = {}
    for observation in (*region_observations, *fuel_observations):
        by_series.setdefault(observation.series_id, []).append(observation)

    assert len(by_series["eia_grid_monitor:ISNE:system_load"]) == 2
    assert len(by_series["eia_grid_monitor:ISNE:demand_forecast"]) == 2
    assert len(by_series["eia_grid_monitor:ISNE:forecast_error"]) == 2
    assert len(by_series["eia_grid_monitor:ISNE:net_interchange"]) == 1
    assert len(by_series["eia_grid_monitor:ISNE:generation_natural_gas"]) == 2
    assert len(by_series["eia_grid_monitor:ISNE:generation_wind"]) == 2
    assert by_series["eia_grid_monitor:ISNE:forecast_error"][0].value == 150.0


def test_eia_backfill_archives_raw_payloads_and_deduplicates(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    try:
        fetcher = _fixture_fetcher()
        first = backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=END,
            raw_root=tmp_path / "raw",
            fetcher=fetcher,
            now=START + timedelta(days=1),
        )
        second = backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=END,
            raw_root=tmp_path / "raw",
            fetcher=fetcher,
            now=START + timedelta(days=1, minutes=1),
        )
        system_load = fetch_series(conn, "eia_grid_monitor:ISNE:system_load", START, END)
    finally:
        conn.close()

    assert first.parsed_observations == 11
    assert first.inserted_observations == 11
    assert second.inserted_observations == 0
    assert all(Path(archive.path).exists() for archive in first.raw_archives)
    assert [row.value for row in system_load] == [14200.0, 13900.0]


def test_eia_fetch_events_are_append_only_for_deduplicated_raw_content(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    try:
        fetcher = _fixture_fetcher()
        backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=END,
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            fetcher=fetcher,
            now=START + timedelta(days=1),
        )
        backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=END,
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            fetcher=fetcher,
            now=START + timedelta(days=1, minutes=1),
        )
        archive_count = conn.execute("SELECT COUNT(*) AS count FROM public_raw_archives").fetchone()["count"]
        fetch_events = conn.execute(
            "SELECT request_url, content_sha256 FROM public_fetch_events ORDER BY fetch_id"
        ).fetchall()
    finally:
        conn.close()

    assert archive_count == 2
    assert len(fetch_events) == 4
    assert all("SECRET" not in row["request_url"] for row in fetch_events)
    assert all("REDACTED" in row["request_url"] for row in fetch_events)
    assert len({row["content_sha256"] for row in fetch_events}) == 2


def test_eia_incremental_poll_uses_recent_overlap(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    calls: list[tuple[str, datetime, datetime]] = []
    try:
        backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=END,
            raw_root=tmp_path / "raw",
            fetcher=_fixture_fetcher(calls),
            now=START + timedelta(days=1),
        )
        calls.clear()
        poll_new_england_grid(
            conn,
            raw_root=tmp_path / "raw",
            fetcher=_fixture_fetcher(calls),
            lookback_hours=48,
            now=START + timedelta(days=2),
        )
    finally:
        conn.close()

    assert {call[0] for call in calls} == {REGION_ROUTE, FUEL_ROUTE}
    assert all(call[1] == END - timedelta(hours=48) for call in calls)


def test_eia_complete_pagination_fetches_short_final_page(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    calls: list[tuple[str, int, int]] = []
    region_rows = [_region_row(START + timedelta(hours=hour), "D", 10_000 + hour) for hour in range(5001)]
    pages = {
        REGION_ROUTE: {
            0: _payload(region_rows[:5000], total=5001),
            5000: _payload(region_rows[5000:], total=5001),
        },
        FUEL_ROUTE: {0: _payload([], total=0)},
    }
    try:
        result = backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=START + timedelta(hours=5000),
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            page_fetcher=_paged_fetcher(pages, calls),
            now=START + timedelta(days=300),
        )
        event_rows = conn.execute(
            "SELECT route, page_offset, request_metadata_json FROM public_fetch_events ORDER BY fetch_id"
        ).fetchall()
    finally:
        conn.close()

    assert result.page_count == 3
    assert result.raw_row_count == 5001
    assert result.parsed_observations == 5001
    assert result.inserted_observations == 5001
    assert calls == [
        (REGION_ROUTE, 0, 5000),
        (REGION_ROUTE, 5000, 5000),
        (FUEL_ROUTE, 0, 5000),
    ]
    assert [(row["route"], row["page_offset"]) for row in event_rows] == [
        (REGION_ROUTE, 0),
        (REGION_ROUTE, 5000),
        (FUEL_ROUTE, 0),
    ]


def test_eia_pagination_rejects_repeated_pages(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    rows = [_region_row(START + timedelta(hours=hour), "D", 10_000 + hour) for hour in range(2)]
    repeated = _payload(rows, total=3)
    pages = {
        REGION_ROUTE: {0: repeated, 2: repeated},
        FUEL_ROUTE: {0: _payload([], total=0)},
    }
    try:
        with pytest.raises(EiaGridError, match="repeated a page"):
            backfill_new_england_grid(
                conn,
                start_utc=START,
                end_utc=START + timedelta(hours=2),
                api_key="SECRET",
                raw_root=tmp_path / "raw",
                page_fetcher=_paged_fetcher(pages),
                now=START + timedelta(days=1),
                page_length=2,
            )
    finally:
        conn.close()


def test_eia_pagination_rejects_changing_total(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    pages = {
        REGION_ROUTE: {
            0: _payload([_region_row(START, "D", 10_000), _region_row(START + timedelta(hours=1), "D", 10_001)], total=3),
            2: _payload([_region_row(START + timedelta(hours=2), "D", 10_002)], total=4),
        },
        FUEL_ROUTE: {0: _payload([], total=0)},
    }
    try:
        with pytest.raises(EiaGridError, match="total changed"):
            backfill_new_england_grid(
                conn,
                start_utc=START,
                end_utc=START + timedelta(hours=2),
                api_key="SECRET",
                raw_root=tmp_path / "raw",
                page_fetcher=_paged_fetcher(pages),
                now=START + timedelta(days=1),
                page_length=2,
            )
    finally:
        conn.close()


def test_eia_payload_requires_total() -> None:
    with pytest.raises(EiaGridError, match="response.total"):
        _payload_total({"response": {"data": []}})


def test_eia_parser_ignores_irrelevant_rows() -> None:
    region_payload = _payload(
        [
            _region_row(START, "D", 10_000),
            _region_row(START, "DF", 9_900),
            _region_row(START, "TI", -400),
            _region_row(START, "NG", 123),
            _region_row(START, "D", 99_999, respondent="NYIS"),
        ],
        total=5,
    )
    fuel_payload = _payload(
        [
            _fuel_row(START, "NG", 5_000),
            _fuel_row(START, "WND", 900),
            _fuel_row(START, "COL", 100),
            _fuel_row(START, "NG", 1_000, respondent="NYIS"),
        ],
        total=4,
    )

    region = parse_eia_observations(
        region_payload,
        route=REGION_ROUTE,
        ingested_at_utc=START + timedelta(days=1),
        raw_archive_sha256="region-sha",
    )
    fuel = parse_eia_observations(
        fuel_payload,
        route=FUEL_ROUTE,
        ingested_at_utc=START + timedelta(days=1),
        raw_archive_sha256="fuel-sha",
    )

    assert {row.metadata["eia_code"] for row in region if row.quality == "reported"} == set(REGION_ALLOWED_TYPES)
    assert {row.metadata["eia_code"] for row in fuel} == set(FUEL_ALLOWED_TYPES)
    assert len(region) == 4
    assert len(fuel) == 2


def test_eia_poll_plans_initial_backfill_then_route_specific_gap_repair(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    calls: list[tuple[str, datetime, datetime]] = []
    config = _eia_config(initial_backfill_hours=12, normal_lookback_hours=3, maximum_gap_repair_hours=48)
    try:
        poll_new_england_grid(
            conn,
            config=config,
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            fetcher=_fixture_fetcher(calls),
            now=START + timedelta(hours=24),
        )
        first_calls = tuple(calls)
        calls.clear()
        poll_new_england_grid(
            conn,
            config=config,
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            fetcher=_fixture_fetcher(calls),
            now=START + timedelta(hours=30),
        )
    finally:
        conn.close()

    assert all(call[1] == START + timedelta(hours=12) for call in first_calls)
    assert all(call[2] == START + timedelta(hours=24) for call in first_calls)
    assert all(call[1] == END - timedelta(hours=3) for call in calls)
    assert all(call[2] == START + timedelta(hours=30) for call in calls)


def test_eia_poll_repairs_middle_gap_without_conflating_routes(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    calls: list[tuple[str, datetime, datetime]] = []
    config = _eia_config(initial_backfill_hours=6, normal_lookback_hours=2, maximum_gap_repair_hours=48)
    try:
        backfill_new_england_grid(
            conn,
            start_utc=START,
            end_utc=START + timedelta(hours=4),
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            page_fetcher=_route_specific_gap_fetcher(),
            now=START + timedelta(days=1),
        )
        poll_new_england_grid(
            conn,
            config=config,
            api_key="SECRET",
            raw_root=tmp_path / "raw",
            fetcher=_fixture_fetcher(calls),
            now=START + timedelta(hours=10),
        )
    finally:
        conn.close()

    starts = {route: start for route, start, _end in calls}
    assert starts[REGION_ROUTE] == START + timedelta(hours=2)
    assert starts[FUEL_ROUTE] == START + timedelta(hours=2)


def test_eia_cli_records_collector_error_when_credentials_are_missing(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "resonance.db"
    monkeypatch.delenv("EIA_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "backfill",
                "--database",
                str(db_path),
                "--start",
                "2026-06-19T00:00:00Z",
                "--end",
                "2026-06-19T01:00:00Z",
            ]
        )

    assert exc.value.code == 2
    conn = ensure_database(db_path)
    try:
        rows = conn.execute("SELECT collector, error_type, message FROM collector_errors").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["collector"] == "eia_grid_monitor"
    assert rows[0]["error_type"] == "EiaGridError"
    assert "EIA_API_KEY" in rows[0]["message"]


def test_eia_cli_status_does_not_require_api_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    code = main(["status", "--database", str(tmp_path / "resonance.db")])
    output = capsys.readouterr().out

    assert code == 0
    assert "credential_available" in output
    assert "EIA_API_KEY" not in output


def _fixture(name: str) -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8"))


def _fixture_fetcher(calls: list[tuple[str, datetime, datetime]] | None = None):
    def fetch(route: str, start: datetime, end: datetime) -> EiaFetchResult:
        if calls is not None:
            calls.append((route, start, end))
        payload = _fixture("eia_region_data_isne.json" if route == REGION_ROUTE else "eia_fuel_type_isne.json")
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return EiaFetchResult(
            payload=payload,
            raw_bytes=raw,
            request_url=f"https://api.eia.gov/v2/{route}/data/?api_key=SECRET",
            status_code=200,
            retrieved_at_utc=START + timedelta(days=1),
            route=route,
        )

    return fetch


def _payload(rows: list[dict], *, total: int) -> dict:
    return {"response": {"total": total, "data": rows}}


def _region_row(period: datetime, code: str, value: float, *, respondent: str = "ISNE") -> dict:
    return {
        "period": period.strftime("%Y-%m-%dT%H"),
        "respondent": respondent,
        "respondent-name": "ISO New England" if respondent == "ISNE" else "Other BA",
        "type": code,
        "type-name": code,
        "value": str(value),
        "value-units": "megawatthours",
    }


def _fuel_row(period: datetime, code: str, value: float, *, respondent: str = "ISNE") -> dict:
    return {
        "period": period.strftime("%Y-%m-%dT%H"),
        "respondent": respondent,
        "respondent-name": "ISO New England" if respondent == "ISNE" else "Other BA",
        "fueltype": code,
        "fueltype-name": code,
        "value": str(value),
        "value-units": "megawatthours",
    }


def _paged_fetcher(
    pages: dict[str, dict[int, dict]],
    calls: list[tuple[str, int, int]] | None = None,
):
    def fetch(route: str, start: datetime, end: datetime, offset: int, length: int) -> EiaPageFetchResult:
        if calls is not None:
            calls.append((route, offset, length))
        payload = pages[route][offset]
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return EiaPageFetchResult(
            payload=payload,
            raw_bytes=raw,
            request_url=f"https://api.eia.gov/v2/{route}/data/?api_key=SECRET&offset={offset}&length={length}",
            status_code=200,
            retrieved_at_utc=START + timedelta(days=1),
            route=route,
            page_offset=offset,
            total=int(payload["response"]["total"]),
            request_metadata={"route": route, "page_offset": offset, "length": length},
        )

    return fetch


def _route_specific_gap_fetcher():
    def fetch(route: str, start: datetime, end: datetime, offset: int, length: int) -> EiaPageFetchResult:
        if route == REGION_ROUTE:
            rows = [
                _region_row(START, "D", 10_000),
                _region_row(START + timedelta(hours=1), "D", 10_001),
                _region_row(START + timedelta(hours=3), "D", 10_003),
                _region_row(START + timedelta(hours=4), "D", 10_004),
            ]
        else:
            rows = [
                _fuel_row(START, "NG", 5_000),
                _fuel_row(START + timedelta(hours=1), "NG", 5_001),
                _fuel_row(START + timedelta(hours=3), "NG", 5_003),
                _fuel_row(START + timedelta(hours=4), "NG", 5_004),
            ]
        payload = _payload(rows, total=len(rows))
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return EiaPageFetchResult(
            payload=payload,
            raw_bytes=raw,
            request_url=f"https://api.eia.gov/v2/{route}/data/?api_key=SECRET&offset={offset}&length={length}",
            status_code=200,
            retrieved_at_utc=START + timedelta(days=1),
            route=route,
            page_offset=offset,
            total=len(rows),
            request_metadata={"route": route, "page_offset": offset, "length": length},
        )

    return fetch


def _eia_config(
    *,
    initial_backfill_hours: int = 720,
    normal_lookback_hours: int = 72,
    maximum_gap_repair_hours: int = 2160,
):
    from resonance.config import EiaGridPublicSourceConfig

    return EiaGridPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=3600,
        initial_backfill_hours=initial_backfill_hours,
        normal_lookback_hours=normal_lookback_hours,
        maximum_gap_repair_hours=maximum_gap_repair_hours,
    )
