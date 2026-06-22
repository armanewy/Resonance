from __future__ import annotations

import _bootstrap  # noqa: F401

from behavior_lab.finance_data import AdjustmentBasis, AdjustedTotalReturnBar, AsOfQuery, FinanceDataStore
from behavior_lab.finance_data.fixtures import (
    adversarial_revision_release_fixture,
    closed_market_store_fixture,
    fixture_metadata,
)
from behavior_lab.finance_data.store import session_event_time


def test_as_of_queries_do_not_look_through_adjusted_total_return_data() -> None:
    bar = AdjustedTotalReturnBar(
        metadata=fixture_metadata(
            instrument_id="ABC",
            event_time=session_event_time("2026-01-02"),
            available_at="2026-01-02T21:00:00+00:00",
            ingested_at="2026-06-01T12:00:00+00:00",
            unit="total_return_index",
            adjustment_basis=AdjustmentBasis.TOTAL_RETURN_ADJUSTED.value,
            artifact_seed="late-adjusted-bar",
        ),
        session_date="2026-01-02",
        open_value=100.0,
        high_value=101.0,
        low_value=99.0,
        close_value=100.5,
        total_return_factor=1.005,
        corporate_action_knowledge_at="2026-02-01T13:00:00+00:00",
        adjustment_source_revision_ids=["split-announcement-rev-1"],
    )
    store = FinanceDataStore([bar])

    before_action_was_known = store.query(
        kind="adjusted_total_return_bar",
        instrument_id="ABC",
        event_time=session_event_time("2026-01-02"),
        as_of="2026-01-15T12:00:00+00:00",
        require_ingested=False,
    )
    after_action_was_known = store.query(
        kind="adjusted_total_return_bar",
        instrument_id="ABC",
        event_time=session_event_time("2026-01-02"),
        as_of="2026-02-02T12:00:00+00:00",
        require_ingested=False,
    )
    before_local_ingestion = store.query(
        kind="adjusted_total_return_bar",
        instrument_id="ABC",
        event_time=session_event_time("2026-01-02"),
        as_of="2026-02-02T12:00:00+00:00",
    )

    assert before_action_was_known == []
    assert after_action_was_known == [bar]
    assert before_local_ingestion == []


def test_closed_market_periods_are_not_forward_filled_from_prior_daily_bars() -> None:
    store = closed_market_store_fixture()
    result = store.sample(
        AsOfQuery(
            kind="daily_bar",
            instrument_id="ABC",
            event_time=session_event_time("2026-01-01"),
            as_of="2026-01-01T20:00:00+00:00",
            calendar_id="XNYS",
        )
    )

    assert result.found is False
    assert result.observations == []
    assert result.missing_reason == "market_closed_no_forward_fill"


def test_adversarial_economic_revisions_return_latest_vintage_only_as_of_sample_time() -> None:
    store = FinanceDataStore(adversarial_revision_release_fixture())

    february_sample = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-02-10T12:00:00+00:00",
    )
    march_sample = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-03-10T12:00:00+00:00",
    )
    all_march_vintages = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-03-10T12:00:00+00:00",
        revision_policy="all_available",
    )

    assert [release.value for release in february_sample] == [150.0]
    assert [release.vintage_id for release in february_sample] == ["2026-02-06"]
    assert [release.value for release in march_sample] == [50.0]
    assert [release.vintage_id for release in march_sample] == ["2026-03-06"]
    assert [release.value for release in all_march_vintages] == [150.0, 50.0]
