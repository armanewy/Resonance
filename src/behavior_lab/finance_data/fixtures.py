from __future__ import annotations

from behavior_lab.finance_data.contracts import (
    AdjustmentBasis,
    DailyBar,
    EconomicRelease,
    MarketCalendar,
    MarketSessionStatus,
    ObservationMetadata,
    Quote,
    source_artifact_hash,
)
from behavior_lab.finance_data.store import FinanceDataStore, session_event_time


def fixture_metadata(
    *,
    instrument_id: str,
    source_id: str = "fixture_vendor",
    event_time: str,
    available_at: str,
    ingested_at: str | None = None,
    timezone: str = "America/New_York",
    currency: str = "USD",
    unit: str = "USD",
    adjustment_basis: str = AdjustmentBasis.RAW.value,
    revision_id: str = "rev_1",
    artifact_seed: str = "fixture",
) -> ObservationMetadata:
    return ObservationMetadata(
        instrument_id=instrument_id,
        source_id=source_id,
        event_time=event_time,
        available_at=available_at,
        ingested_at=ingested_at or available_at,
        timezone=timezone,
        currency=currency,
        unit=unit,
        adjustment_basis=adjustment_basis,
        revision_id=revision_id,
        source_artifact_hash=source_artifact_hash({"artifact_seed": artifact_seed, "revision_id": revision_id}),
    )


def one_sided_quote_fixture() -> Quote:
    return Quote(
        metadata=fixture_metadata(
            instrument_id="ABC",
            event_time="2026-01-02T14:30:00+00:00",
            available_at="2026-01-02T14:30:01+00:00",
            unit="USD/share",
            artifact_seed="one-sided-quote",
        ),
        bid_price=10.0,
        bid_size=100.0,
        bid_is_executable=True,
    )


def closed_market_store_fixture() -> FinanceDataStore:
    calendar = MarketCalendar(
        metadata=fixture_metadata(
            instrument_id="XNYS",
            event_time=session_event_time("2026-01-01"),
            available_at="2025-12-01T00:00:00+00:00",
            unit="calendar_session",
            artifact_seed="closed-calendar",
        ),
        calendar_id="XNYS",
        session_date="2026-01-01",
        status=MarketSessionStatus.CLOSED.value,
        reason="new_years_day",
    )
    prior_bar = DailyBar(
        metadata=fixture_metadata(
            instrument_id="ABC",
            event_time=session_event_time("2025-12-31"),
            available_at="2025-12-31T21:00:00+00:00",
            unit="USD/share",
            artifact_seed="prior-bar",
        ),
        session_date="2025-12-31",
        open_price=99.0,
        high_price=101.0,
        low_price=98.0,
        close_price=100.0,
        volume=1000.0,
    )
    return FinanceDataStore([calendar, prior_bar])


def adversarial_revision_release_fixture() -> list[EconomicRelease]:
    first = EconomicRelease(
        metadata=fixture_metadata(
            instrument_id="ECON:PAYROLLS",
            source_id="macro_fixture",
            event_time="2026-02-06T13:30:00+00:00",
            available_at="2026-02-06T13:30:00+00:00",
            timezone="America/New_York",
            unit="thousands_of_jobs",
            adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
            revision_id="advance",
            artifact_seed="payrolls-advance",
        ),
        series_id="PAYROLLS",
        period_start="2026-01-01",
        period_end="2026-01-31",
        value=150.0,
        release_stage="advance",
        revision_group_id="PAYROLLS-2026-01",
        vintage_id="2026-02-06",
    )
    revised = EconomicRelease(
        metadata=fixture_metadata(
            instrument_id="ECON:PAYROLLS",
            source_id="macro_fixture",
            event_time="2026-03-06T13:30:00+00:00",
            available_at="2026-03-06T13:30:00+00:00",
            timezone="America/New_York",
            unit="thousands_of_jobs",
            adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
            revision_id="second",
            artifact_seed="payrolls-second",
        ),
        series_id="PAYROLLS",
        period_start="2026-01-01",
        period_end="2026-01-31",
        value=50.0,
        release_stage="second",
        revision_group_id="PAYROLLS-2026-01",
        vintage_id="2026-03-06",
    )
    return [first, revised]
