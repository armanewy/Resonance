from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.config import EiaGridPublicSourceConfig
from resonance.public_health import eia_source_health_rows
from resonance.public_sources.eia_grid import REGION_ROUTE, SOURCE_ID, ensure_eia_registry
from resonance.storage import (
    PublicCollectionState,
    PublicFetchEvent,
    PublicObservation,
    ensure_database,
    insert_public_observations,
    record_public_fetch_event,
    upsert_public_collection_state,
)


START = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc)


def test_eia_source_health_rows_summarize_fetches_observations_and_gaps() -> None:
    conn = ensure_database(":memory:")
    try:
        ensure_eia_registry(conn, enabled=True)
        insert_public_observations(
            conn,
            [
                PublicObservation(
                    series_id="eia_grid_monitor:ISNE:system_load",
                    valid_start_utc=START,
                    valid_end_utc=START + timedelta(hours=1),
                    observed_at_utc=START + timedelta(hours=1),
                    ingested_at_utc=START + timedelta(hours=2),
                    value=100.0,
                    quality="reported",
                    source_revision="rev-1",
                    source_observation_key="region-data:ISNE:D:2026-06-19T00",
                )
            ],
        )
        record_public_fetch_event(
            conn,
            PublicFetchEvent(
                source_id=SOURCE_ID,
                retrieved_at_utc=START + timedelta(hours=2),
                request_url="https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key=REDACTED",
                status_code=200,
                content_sha256="sha",
                route=REGION_ROUTE,
                page_offset=0,
                request_metadata={"route": REGION_ROUTE},
            ),
        )
        upsert_public_collection_state(
            conn,
            PublicCollectionState(
                source_id=SOURCE_ID,
                route=REGION_ROUTE,
                earliest_unresolved_gap_utc=START + timedelta(hours=1),
                latest_error="short final page",
                consecutive_failure_count=2,
            ),
        )
        conn.commit()
        rows = eia_source_health_rows(
            conn,
            config=_eia_config(),
            now_utc=START + timedelta(hours=3),
            credential_available=True,
        )
    finally:
        conn.close()

    assert rows == [
        {
            "source": "EIA Hourly Electric Grid Monitor",
            "enabled": "yes",
            "credential": "available",
            "latest_fetch": "2026-06-19T02:00:00Z",
            "latest_valid": "2026-06-19T00:00:00Z",
            "lag_hours": 3.0,
            "unresolved_gaps": 1,
            "rows_24h": 1,
            "last_error": "short final page",
            "failures": 2,
        }
    ]


def _eia_config() -> EiaGridPublicSourceConfig:
    return EiaGridPublicSourceConfig(
        enabled=True,
        poll_interval_seconds=3600,
        initial_backfill_hours=720,
        normal_lookback_hours=72,
        maximum_gap_repair_hours=2160,
    )
