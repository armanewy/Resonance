from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from resonance.config import EiaGridPublicSourceConfig, LocationConfig, RipeAtlasPublicSourceConfig
from resonance.public_collector import main, run_once, run_ripe_once
from resonance.public_sources.eia_grid import ROUTES, SOURCE_ID
from resonance.public_sources.ripe_atlas import SOURCE_ID as RIPE_SOURCE_ID
from resonance.storage import ensure_database


VALID_CONFIG = """
[location]
name = "Framingham, Massachusetts"
latitude = 42.2793
longitude = -71.4162
timezone = "America/New_York"

[collection]
personal_interval_seconds = 30
weather_interval_seconds = 900
tcp_test_host = "1.1.1.1"
tcp_test_port = 443
dns_test_hostname = "example.com"
router_host = ""
"""


@dataclass(frozen=True)
class _PollResult:
    inserted_observations: int


def test_public_collector_disabled_exits_cleanly(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    Path("config.toml").write_text(
        VALID_CONFIG
        + """
[public_sources.eia_grid]
enabled = false
""",
        encoding="utf-8",
    )

    assert main(stop_requested=Event()) == 0

    conn = ensure_database("data/resonance.db")
    try:
        source = conn.execute(
            "SELECT enabled FROM public_sources WHERE source_id = ?",
            (SOURCE_ID,),
        ).fetchone()
        ripe_source = conn.execute(
            "SELECT enabled FROM public_sources WHERE source_id = ?",
            (RIPE_SOURCE_ID,),
        ).fetchone()
    finally:
        conn.close()
    assert source["enabled"] == 0
    assert ripe_source["enabled"] == 0


def test_public_collector_missing_credentials_records_health_without_polling(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    called = False

    def poller(*_args, **_kwargs):
        nonlocal called
        called = True
        return _PollResult(inserted_observations=0)

    ok = run_once(
        _eia_config(),
        database_path=tmp_path / "resonance.db",
        raw_root=tmp_path / "raw",
        poller=poller,
    )

    conn = ensure_database(tmp_path / "resonance.db")
    try:
        states = conn.execute(
            "SELECT route, latest_error, consecutive_failure_count FROM public_collection_state ORDER BY route"
        ).fetchall()
        errors = conn.execute("SELECT collector, error_type, message FROM collector_errors").fetchall()
    finally:
        conn.close()

    assert ok is False
    assert called is False
    assert {row["route"] for row in states} == set(ROUTES)
    assert all(row["consecutive_failure_count"] == 1 for row in states)
    assert all("EIA_API_KEY" in row["latest_error"] for row in states)
    assert len(errors) == 1
    assert errors[0]["collector"] == SOURCE_ID
    assert errors[0]["error_type"] == "missing_credentials"


def test_public_collector_success_passes_env_key_without_persisting_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "SECRET")
    seen: dict[str, object] = {}

    def poller(conn, **kwargs):
        seen["api_key"] = kwargs["api_key"]
        seen["raw_root"] = kwargs["raw_root"]
        return _PollResult(inserted_observations=3)

    ok = run_once(
        _eia_config(),
        database_path=tmp_path / "resonance.db",
        raw_root=tmp_path / "raw",
        poller=poller,
    )

    conn = ensure_database(tmp_path / "resonance.db")
    try:
        source = conn.execute(
            "SELECT enabled FROM public_sources WHERE source_id = ?",
            (SOURCE_ID,),
        ).fetchone()
        persisted = "\n".join(
            str(value)
            for table in ("public_collection_state", "collector_errors", "public_sources")
            for row in conn.execute(f"SELECT * FROM {table}").fetchall()
            for value in tuple(row)
        )
    finally:
        conn.close()

    assert ok is True
    assert seen["api_key"] == "SECRET"
    assert seen["raw_root"] == tmp_path / "raw"
    assert source["enabled"] == 1
    assert "SECRET" not in persisted


def test_public_collector_redacts_api_key_from_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "SECRET")

    def poller(*_args, **_kwargs):
        raise RuntimeError("upstream rejected SECRET")

    ok = run_once(
        _eia_config(),
        database_path=tmp_path / "resonance.db",
        raw_root=tmp_path / "raw",
        poller=poller,
    )

    conn = ensure_database(tmp_path / "resonance.db")
    try:
        state_text = "\n".join(
            row["latest_error"]
            for row in conn.execute("SELECT latest_error FROM public_collection_state").fetchall()
        )
        error_text = "\n".join(row["message"] for row in conn.execute("SELECT message FROM collector_errors").fetchall())
    finally:
        conn.close()

    assert ok is False
    assert "SECRET" not in state_text
    assert "SECRET" not in error_text
    assert "REDACTED" in state_text
    assert "REDACTED" in error_text


def test_ripe_public_collector_runs_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RIPE_ATLAS_API_KEY", raising=False)
    seen: dict[str, object] = {}

    def poller(conn, **kwargs):
        seen["location"] = kwargs["location"]
        seen["raw_root"] = kwargs["raw_root"]
        return _PollResult(inserted_observations=4)

    ok = run_ripe_once(
        _ripe_config(),
        location_config=_location(),
        database_path=tmp_path / "resonance.db",
        raw_root=tmp_path / "raw",
        poller=poller,
    )

    conn = ensure_database(tmp_path / "resonance.db")
    try:
        source = conn.execute(
            "SELECT enabled FROM public_sources WHERE source_id = ?",
            (RIPE_SOURCE_ID,),
        ).fetchone()
        errors = conn.execute("SELECT collector, error_type, message FROM collector_errors").fetchall()
    finally:
        conn.close()

    assert ok is True
    assert seen["location"] == _location()
    assert seen["raw_root"] == tmp_path / "raw"
    assert source["enabled"] == 1
    assert errors == []


def test_ripe_public_collector_redacts_optional_api_key_from_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RIPE_ATLAS_API_KEY", "RIPE_SECRET")

    def poller(*_args, **_kwargs):
        raise RuntimeError("upstream rejected RIPE_SECRET")

    ok = run_ripe_once(
        _ripe_config(),
        location_config=_location(),
        database_path=tmp_path / "resonance.db",
        raw_root=tmp_path / "raw",
        poller=poller,
    )

    conn = ensure_database(tmp_path / "resonance.db")
    try:
        state_text = "\n".join(row["latest_error"] for row in conn.execute("SELECT latest_error FROM public_collection_state").fetchall())
        error_text = "\n".join(row["message"] for row in conn.execute("SELECT message FROM collector_errors").fetchall())
    finally:
        conn.close()

    assert ok is False
    assert "RIPE_SECRET" not in state_text
    assert "RIPE_SECRET" not in error_text
    assert "REDACTED" in state_text
    assert "REDACTED" in error_text


def _eia_config() -> EiaGridPublicSourceConfig:
    return EiaGridPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=3600,
        initial_backfill_hours=720,
        normal_lookback_hours=72,
        maximum_gap_repair_hours=2160,
    )


def _ripe_config() -> RipeAtlasPublicSourceConfig:
    return RipeAtlasPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=900,
        initial_backfill_hours=168,
        normal_lookback_hours=6,
        aggregation_seconds=900,
        finalization_delay_seconds=600,
        initial_radius_km=150,
        maximum_radius_km=500,
        desired_probe_count=24,
        minimum_probe_count=8,
        maximum_probes_per_asn=2,
        maximum_anchor_count=4,
        cohort_refresh_hours=24,
        result_chunk_hours=6,
        maximum_probe_batch_size=50,
        maximum_requests_per_poll=200,
        measurement_ids=(1001, 1004, 1009),
    )


def _location() -> LocationConfig:
    return LocationConfig("Framingham, Massachusetts", 42.2793, -71.4162, "America/New_York")
