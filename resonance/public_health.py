from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from resonance.config import EiaGridPublicSourceConfig
from resonance.public_sources.eia_grid import SOURCE_ID
from resonance.config import LocationConfig, RipeAtlasPublicSourceConfig
from resonance.public_sources.ripe_atlas import status_payload as ripe_status_payload
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso


def eia_source_health(
    conn,
    *,
    config: EiaGridPublicSourceConfig,
    now_utc: datetime,
    credential_available: bool,
) -> dict[str, Any]:
    now = ensure_utc(now_utc)
    source = conn.execute(
        """
        SELECT display_name, enabled
        FROM public_sources
        WHERE source_id = ?
        """,
        (SOURCE_ID,),
    ).fetchone()
    latest_fetch = conn.execute(
        """
        SELECT MAX(retrieved_at_utc) AS retrieved_at_utc
        FROM public_fetch_events
        WHERE source_id = ?
        """,
        (SOURCE_ID,),
    ).fetchone()
    latest_valid = conn.execute(
        """
        SELECT MAX(o.valid_start_utc) AS valid_start_utc
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE s.source_id = ?
        """,
        (SOURCE_ID,),
    ).fetchone()
    rows_24h = conn.execute(
        """
        SELECT COUNT(*) AS row_count
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE s.source_id = ?
          AND o.ingested_at_utc >= ?
        """,
        (SOURCE_ID, to_utc_iso(now - timedelta(hours=24))),
    ).fetchone()
    states = conn.execute(
        """
        SELECT route, earliest_unresolved_gap_utc, latest_error, latest_error_utc,
               consecutive_failure_count
        FROM public_collection_state
        WHERE source_id = ?
        ORDER BY route
        """,
        (SOURCE_ID,),
    ).fetchall()
    last_error = ""
    last_error_utc = None
    consecutive_failure_count = 0
    unresolved_gaps = 0
    for state in states:
        if state["earliest_unresolved_gap_utc"]:
            unresolved_gaps += 1
        consecutive_failure_count = max(consecutive_failure_count, int(state["consecutive_failure_count"] or 0))
        if state["latest_error"] and (
            last_error_utc is None
            or (state["latest_error_utc"] and state["latest_error_utc"] > last_error_utc)
        ):
            last_error = state["latest_error"]
            last_error_utc = state["latest_error_utc"]
    latest_valid_utc = _optional_parse(latest_valid["valid_start_utc"] if latest_valid else None)
    return {
        "source_id": SOURCE_ID,
        "source_name": source["display_name"] if source else "EIA Hourly Electric Grid Monitor",
        "enabled": bool(config.enabled),
        "registry_enabled": bool(source["enabled"]) if source else False,
        "credential_available": bool(credential_available),
        "latest_successful_fetch": latest_fetch["retrieved_at_utc"] if latest_fetch and latest_fetch["retrieved_at_utc"] else None,
        "latest_valid_observation": to_utc_iso(latest_valid_utc) if latest_valid_utc else None,
        "lag_hours": round((now - latest_valid_utc).total_seconds() / 3600, 2) if latest_valid_utc else None,
        "unresolved_gap_count": unresolved_gaps,
        "rows_collected_last_24h": int(rows_24h["row_count"] or 0) if rows_24h else 0,
        "last_error": last_error,
        "last_error_utc": last_error_utc,
        "consecutive_failure_count": consecutive_failure_count,
    }


def eia_source_health_rows(
    conn,
    *,
    config: EiaGridPublicSourceConfig,
    now_utc: datetime,
    credential_available: bool,
) -> list[dict[str, Any]]:
    health = eia_source_health(
        conn,
        config=config,
        now_utc=now_utc,
        credential_available=credential_available,
    )
    return [
        {
            "source": health["source_name"],
            "enabled": "yes" if health["enabled"] else "no",
            "credential": "available" if health["credential_available"] else "missing",
            "latest_fetch": health["latest_successful_fetch"] or "none",
            "latest_valid": health["latest_valid_observation"] or "none",
            "lag_hours": health["lag_hours"] if health["lag_hours"] is not None else "n/a",
            "unresolved_gaps": health["unresolved_gap_count"],
            "rows_24h": health["rows_collected_last_24h"],
            "last_error": health["last_error"],
            "failures": health["consecutive_failure_count"],
        }
    ]


def ripe_source_health_rows(
    conn,
    *,
    config: RipeAtlasPublicSourceConfig,
    location: LocationConfig,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    health = ripe_status_payload(conn, config=config, location=location, now=now_utc)
    return [
        {
            "source": health["source_name"],
            "enabled": "yes" if health["enabled"] else "no",
            "active_cohort": health["active_cohort_id"] or "none",
            "probes": health["active_probe_count"],
            "unique_asns": health["unique_asn_count"],
            "radius_km": health["selected_radius_km"] or "n/a",
            "latest_fetch": health["latest_result_fetch"] or "none",
            "latest_finalized": health["latest_finalized_aggregate"] or "none",
            "lag_hours": health["lag_hours"] if health["lag_hours"] is not None else "n/a",
            "unresolved_gaps": health["unresolved_gap_count"],
            "last_error": health["latest_error"],
            "failures": health["consecutive_failure_count"],
        }
    ]


def _optional_parse(value: str | None) -> datetime | None:
    if not value:
        return None
    return parse_utc(value)
