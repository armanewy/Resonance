from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from resonance.public_series import fetch_series
from resonance.public_sources.eia_grid import (
    FUEL_ROUTE,
    REGION_ROUTE,
    EiaFetchResult,
    backfill_new_england_grid,
    main,
    parse_eia_observations,
    poll_new_england_grid,
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
    assert all(call[1] == START for call in calls)


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
