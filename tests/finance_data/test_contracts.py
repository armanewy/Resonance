from __future__ import annotations

import _bootstrap  # noqa: F401
import pytest

from behavior_lab.finance_data import (
    AdjustmentBasis,
    AdjustedTotalReturnBar,
    BookLevel,
    CashRiskFreeBenchmark,
    CorporateAction,
    DailyBar,
    Distribution,
    EconomicRelease,
    FinanceDataError,
    MarketCalendar,
    MarketSessionStatus,
    MarketValueKind,
    OrderBookSnapshot,
    Quote,
    RevisionRecord,
    SettlementEvent,
    Trade,
    VintageSnapshot,
    observation_hash,
    observation_kind,
)
from behavior_lab.finance_data.fixtures import fixture_metadata, one_sided_quote_fixture
from behavior_lab.finance_data.store import session_event_time


def test_supported_observations_preserve_required_metadata_and_market_value_semantics() -> None:
    observations = [
        Quote(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time="2026-01-02T14:30:00+00:00",
                available_at="2026-01-02T14:30:01+00:00",
                unit="USD/share",
                artifact_seed="quote",
            ),
            bid_price=10.0,
            ask_price=10.1,
            bid_size=100.0,
            ask_size=200.0,
            indicative_price=10.05,
            bid_is_executable=True,
            ask_is_executable=True,
        ),
        Trade(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time="2026-01-02T14:31:00+00:00",
                available_at="2026-01-02T14:31:01+00:00",
                unit="USD/share",
                artifact_seed="trade",
            ),
            trade_price=10.08,
            quantity=50.0,
            trade_id="t-1",
        ),
        DailyBar(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time=session_event_time("2026-01-02"),
                available_at="2026-01-02T21:00:00+00:00",
                unit="USD/share",
                artifact_seed="daily",
            ),
            session_date="2026-01-02",
            open_price=10.0,
            high_price=10.5,
            low_price=9.9,
            close_price=10.2,
            volume=1000.0,
        ),
        AdjustedTotalReturnBar(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time=session_event_time("2026-01-02"),
                available_at="2026-01-02T21:00:00+00:00",
                unit="total_return_index",
                adjustment_basis=AdjustmentBasis.TOTAL_RETURN_ADJUSTED.value,
                artifact_seed="adjusted",
            ),
            session_date="2026-01-02",
            open_value=100.0,
            high_value=101.0,
            low_value=99.0,
            close_value=100.5,
            total_return_factor=1.005,
            corporate_action_knowledge_at="2026-01-02T21:00:00+00:00",
            adjustment_source_revision_ids=["ca-rev-1"],
        ),
        OrderBookSnapshot(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time="2026-01-02T14:30:00+00:00",
                available_at="2026-01-02T14:30:00+00:00",
                unit="USD/share",
                artifact_seed="book",
            ),
            bids=[],
            asks=[BookLevel(price=10.1, size=100.0)],
        ),
        MarketCalendar(
            metadata=fixture_metadata(
                instrument_id="XNYS",
                event_time=session_event_time("2026-01-02"),
                available_at="2025-12-01T00:00:00+00:00",
                unit="calendar_session",
                artifact_seed="calendar",
            ),
            calendar_id="XNYS",
            session_date="2026-01-02",
            status=MarketSessionStatus.OPEN.value,
            open_time="2026-01-02T14:30:00+00:00",
            close_time="2026-01-02T21:00:00+00:00",
        ),
        CorporateAction(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time="2026-01-10T00:00:00+00:00",
                available_at="2026-01-03T13:00:00+00:00",
                unit="split_ratio",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="corporate-action",
            ),
            action_id="split-1",
            action_type="split",
            effective_date="2026-01-10",
            announcement_time="2026-01-03T13:00:00+00:00",
            terms={"old_shares": 1, "new_shares": 2},
        ),
        Distribution(
            metadata=fixture_metadata(
                instrument_id="ABC",
                event_time="2026-01-15T00:00:00+00:00",
                available_at="2026-01-04T13:00:00+00:00",
                unit="USD/share",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="distribution",
            ),
            distribution_id="div-1",
            distribution_type="cash_dividend",
            ex_date="2026-01-15",
            record_date="2026-01-16",
            payable_date="2026-01-30",
            amount=0.25,
        ),
        SettlementEvent(
            metadata=fixture_metadata(
                instrument_id="ABC-FUT",
                event_time="2026-03-20T20:00:00+00:00",
                available_at="2026-03-20T20:05:00+00:00",
                unit="USD/contract",
                artifact_seed="settlement",
            ),
            settlement_id="settle-1",
            settlement_date="2026-03-20",
            settlement_value=101.25,
        ),
        EconomicRelease(
            metadata=fixture_metadata(
                instrument_id="ECON:CPI",
                source_id="macro_fixture",
                event_time="2026-02-12T13:30:00+00:00",
                available_at="2026-02-12T13:30:00+00:00",
                unit="index",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="release",
            ),
            series_id="CPI",
            period_start="2026-01-01",
            period_end="2026-01-31",
            value=305.1,
            release_stage="initial",
            revision_group_id="CPI-2026-01",
            vintage_id="2026-02-12",
        ),
        RevisionRecord(
            metadata=fixture_metadata(
                instrument_id="ECON:CPI",
                source_id="macro_fixture",
                event_time="2026-03-12T13:30:00+00:00",
                available_at="2026-03-12T13:30:00+00:00",
                unit="revision",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="revision",
            ),
            observed_entity_id="CPI",
            revision_group_id="CPI-2026-01",
            supersedes_revision_id="initial",
            new_revision_id="second",
            revision_reason="scheduled benchmark update",
            changed_fields=["value"],
        ),
        VintageSnapshot(
            metadata=fixture_metadata(
                instrument_id="ECON:CPI",
                source_id="macro_fixture",
                event_time="2026-03-12T13:30:00+00:00",
                available_at="2026-03-12T13:30:00+00:00",
                unit="index",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="vintage",
            ),
            vintage_id="2026-03-12",
            revision_group_id="CPI",
            as_of_date="2026-03-12",
            observations={"2026-01": 305.2},
            source_release_ids=["cpi-release-2"],
        ),
        CashRiskFreeBenchmark(
            metadata=fixture_metadata(
                instrument_id="USD-SOFR",
                source_id="rates_fixture",
                event_time="2026-01-02T13:00:00+00:00",
                available_at="2026-01-02T13:00:00+00:00",
                unit="annualized_rate",
                adjustment_basis=AdjustmentBasis.AS_REPORTED.value,
                artifact_seed="sofr",
            ),
            benchmark_id="SOFR",
            fixing_date="2026-01-02",
            rate=0.0425,
            tenor="overnight",
            day_count_convention="ACT/360",
            compounding="simple",
        ),
    ]

    required_metadata = {
        "instrument_id",
        "source_id",
        "event_time",
        "available_at",
        "ingested_at",
        "timezone",
        "currency",
        "unit",
        "adjustment_basis",
        "revision_id",
        "source_artifact_hash",
    }
    assert {observation_kind(observation) for observation in observations} == {
        "quote",
        "trade",
        "daily_bar",
        "adjusted_total_return_bar",
        "order_book_snapshot",
        "market_calendar",
        "corporate_action",
        "distribution",
        "settlement_event",
        "economic_release",
        "revision_record",
        "vintage_snapshot",
        "cash_risk_free_benchmark",
    }
    for observation in observations:
        payload = observation.to_dict()
        assert required_metadata <= payload["metadata"].keys()
        assert observation_hash(observation)

    quote = observations[0]
    assert isinstance(quote, Quote)
    assert set(quote.value_kinds) == {
        MarketValueKind.INDICATIVE.value,
        MarketValueKind.EXECUTABLE_BID.value,
        MarketValueKind.EXECUTABLE_ASK.value,
    }
    trade = observations[1]
    settlement = observations[8]
    assert isinstance(trade, Trade)
    assert isinstance(settlement, SettlementEvent)
    assert trade.value_kind == MarketValueKind.LAST_TRADED.value
    assert settlement.value_kind == MarketValueKind.SETTLEMENT.value


def test_one_sided_markets_are_preserved_not_rejected_or_filled() -> None:
    quote = one_sided_quote_fixture()

    assert quote.bid_price == 10.0
    assert quote.ask_price is None
    assert quote.value_kinds == [MarketValueKind.EXECUTABLE_BID.value]

    book = OrderBookSnapshot(
        metadata=fixture_metadata(
            instrument_id="ABC",
            event_time="2026-01-02T14:30:00+00:00",
            available_at="2026-01-02T14:30:00+00:00",
            unit="USD/share",
            artifact_seed="one-sided-book",
        ),
        bids=[],
        asks=[BookLevel(price=10.1, size=100.0)],
    )
    assert book.bids == []
    assert book.asks[0].price == 10.1


def test_adjusted_current_corporate_action_knowledge_must_be_marked() -> None:
    metadata = fixture_metadata(
        instrument_id="ABC",
        event_time=session_event_time("2026-01-02"),
        available_at="2026-01-02T21:00:00+00:00",
        unit="total_return_index",
        adjustment_basis=AdjustmentBasis.TOTAL_RETURN_ADJUSTED.value,
        artifact_seed="current-action",
    )

    with pytest.raises(FinanceDataError, match="current_knowledge_disclosure"):
        AdjustedTotalReturnBar(
            metadata=metadata,
            session_date="2026-01-02",
            open_value=None,
            high_value=None,
            low_value=None,
            close_value=100.0,
            total_return_factor=1.0,
            corporate_action_knowledge_at="2026-06-01T12:00:00+00:00",
            uses_current_corporate_action_knowledge=True,
        )

    bar = AdjustedTotalReturnBar(
        metadata=metadata,
        session_date="2026-01-02",
        open_value=None,
        high_value=None,
        low_value=None,
        close_value=100.0,
        total_return_factor=1.0,
        corporate_action_knowledge_at="2026-06-01T12:00:00+00:00",
        uses_current_corporate_action_knowledge=True,
        current_knowledge_disclosure="derived after the later split became known",
    )
    assert bar.uses_current_corporate_action_knowledge is True
    assert "later split" in (bar.current_knowledge_disclosure or "")
