from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from resonance.config import EiaGridPublicSourceConfig
from resonance.public_collector import main, run_once
from resonance.public_sources.eia_grid import ROUTES, SOURCE_ID
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
    finally:
        conn.close()
    assert source["enabled"] == 0


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


def _eia_config() -> EiaGridPublicSourceConfig:
    return EiaGridPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=3600,
        initial_backfill_hours=720,
        normal_lookback_hours=72,
        maximum_gap_repair_hours=2160,
    )
