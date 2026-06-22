from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import tempfile

from behavior_lab.labs.etf_risk.commands import backfill, paper_cycle, report
from behavior_lab.labs.etf_risk.engine import (
    ACTION_IDS,
    BASELINE_STRATEGY_IDS,
    ETFRiskError,
    ETFRiskConfig,
    ETFRiskLab,
    allocation_weights,
    financial_decision_contract,
)
from behavior_lab.labs.etf_risk.market_data import (
    AdjustedPrice,
    AssetSpec,
    DataAuthorization,
    InMemoryMarketDataProvider,
    MarketCalendar,
    MarketDataError,
    default_universe,
)
from behavior_lab.money.ledger import MoneyLedger


def test_provider_uses_availability_time_and_blocks_backward_correction_leaks() -> None:
    provider, sessions = _provider(session_count=70, corrected_session_index=10)
    universe = default_universe()
    corrected_session = sessions[10]

    same_day = provider.history([universe.asset_for_role("us_equity").asset_id], f"{corrected_session}T21:30:00+00:00")
    same_day_price = [price for price in same_day if price.market_date == corrected_session][0]
    assert same_day_price.revision_id == "original"
    assert same_day_price.adjusted_close != 777.0

    after_correction = provider.history([universe.asset_for_role("us_equity").asset_id], f"{sessions[11]}T13:00:00+00:00")
    corrected_price = [price for price in after_correction if price.market_date == corrected_session][0]
    assert corrected_price.revision_id == "correction_1"
    assert corrected_price.corrected_from == "original"
    assert corrected_price.adjusted_close == 777.0


def test_universe_rejects_forbidden_exposures() -> None:
    with pytest_raises(MarketDataError):
        AssetSpec("SINGLE_NAME", "us_equity", "single_stock proxy is not allowed")


def test_backfill_writes_every_baseline_decision_to_money_ledger() -> None:
    provider, _sessions = _provider(session_count=85)
    config = ETFRiskConfig(min_history_trading_days=30)
    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = str(Path(tmp) / "money.jsonl")
        result = backfill(provider, ledger_path=ledger_path, config=config)
        ledger = MoneyLedger(ledger_path)
        entries = ledger.latest_entries()

        assert result["walk_forward_only"] is True
        assert result["decision_cadence"] == "weekly"
        assert result["decision_count"] == len(result["ledger_records"])
        assert len(entries) == result["decision_count"]
        assert set(result["strategy_ids"]) == set(BASELINE_STRATEGY_IDS)
        assert ledger.verify() is True

        for entry in entries:
            assert entry.designation == "paper"
            assert entry.evidence_state == "paper_decision"
            assert entry.selected_action in ACTION_IDS
            assert entry.no_action_alternative == "cash"
            assert entry.provenance["no_broker_order_api"] is True
            assert entry.provenance["no_real_trading"] is True
            assert entry.provenance["no_individual_stocks_options_leverage_shorts_intraday_or_hft"] is True
            assert entry.provenance["price_snapshot"]
            for price in entry.provenance["price_snapshot"].values():
                assert price["availability_time"] <= entry.data_cutoff
                assert "adjustment_policy" in price["adjustment"]

        metrics = result["metrics"]
        assert metrics["walk_forward_only"] is True
        assert set(metrics["baseline_strategy_ids"]) == set(BASELINE_STRATEGY_IDS)
        assert "calibration" in metrics["strategies"]["target_only_autoregression"]
        assert "no_action_comparison" in metrics["strategies"]["target_only_autoregression"]
        assert "regime_period_concentration" in metrics["strategies"]["target_only_autoregression"]
        assert set(metrics["parameter_neighborhood_sensitivity"]) == {"0.9", "1.0", "1.1"}
        assert result["real_money_eligibility"]["eligible"] is False


def test_paper_cycle_and_report_are_callable_without_shared_cli_wiring() -> None:
    provider, sessions = _provider(session_count=90)
    config = ETFRiskConfig(min_history_trading_days=30)
    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = str(Path(tmp) / "money.jsonl")
        cycle = paper_cycle(
            provider,
            ledger_path=ledger_path,
            config=config,
            decision_cutoff=f"{sessions[-1]}T21:30:00+00:00",
        )
        generated = report(provider, ledger_path=ledger_path, config=config)

        assert cycle["paper_only"] is True
        assert cycle["decision"]["action_id"] in ACTION_IDS
        assert cycle["real_money_eligibility"]["eligible"] is False
        assert generated["paper_only"] is True
        assert generated["ledger_verified"] is True
        assert generated["money_summary"]["designation"] == "paper"
        assert generated["money_summary"]["decision_count"] == 1
        assert generated["real_money_eligibility"]["eligible"] is False
        assert "behavior_lab.labs.etf_risk.commands.paper_cycle" in " ".join(generated["integration_hooks_needed"])


def test_contract_and_allocations_are_paper_long_only_and_unlevered() -> None:
    provider, sessions = _provider(session_count=75)
    config = ETFRiskConfig(min_history_trading_days=30)
    lab = ETFRiskLab(provider, config)
    context = lab.decision_context(sessions[35])
    contract = financial_decision_contract(context, config)

    assert contract.domain == "etf_risk"
    assert contract.paper_only is True
    assert contract.notification_threshold["enabled"] is False
    assert {action.action_id for action in contract.automatic_evaluation_actions()} == set(ACTION_IDS)
    for action_id in ACTION_IDS:
        weights = allocation_weights(config.universe, action_id)
        assert all(0.0 <= weight <= 1.0 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_decision_context_rejects_incomplete_decision_date_price_snapshot() -> None:
    provider, sessions = _provider(
        session_count=70,
        omitted_price=(35, "GOLD"),
    )
    decision_date = sessions[35]
    lab = ETFRiskLab(provider, ETFRiskConfig(min_history_trading_days=30))

    with pytest_raises(ETFRiskError):
        lab.decision_context(decision_date, decision_cutoff=f"{decision_date}T21:30:00+00:00")


def _provider(
    session_count: int,
    corrected_session_index: int | None = None,
    omitted_price: tuple[int, str] | None = None,
) -> tuple[InMemoryMarketDataProvider, list[str]]:
    universe = default_universe()
    sessions = _sessions(date(2026, 1, 2), session_count)
    calendar = MarketCalendar("XNYS_TEST", tuple(sessions))
    prices = []
    levels = {
        "US_EQUITY": 100.0,
        "INTL_EQUITY": 80.0,
        "TREASURY_BOND": 50.0,
        "IG_CREDIT": 40.0,
        "GOLD": 70.0,
        "BROAD_COMMODITIES": 60.0,
        "CASH_EQUIVALENT": 1.0,
    }
    for index, session in enumerate(sessions):
        for asset in universe.assets:
            levels[asset.asset_id] *= 1.0 + _daily_return(asset.role, index)
            if omitted_price == (index, asset.asset_id):
                continue
            prices.append(
                AdjustedPrice(
                    asset_id=asset.asset_id,
                    market_date=session,
                    close=round(levels[asset.asset_id], 6),
                    adjusted_close=round(levels[asset.asset_id], 6),
                    event_time=f"{session}T21:00:00+00:00",
                    availability_time=f"{session}T21:05:00+00:00",
                    calendar_id=calendar.calendar_id,
                    source="unit_fixture",
                    adjustment={"adjustment_policy": "split_distribution_preserving_total_return"},
                )
            )
    if corrected_session_index is not None:
        corrected_session = sessions[corrected_session_index]
        prices.append(
            AdjustedPrice(
                asset_id="US_EQUITY",
                market_date=corrected_session,
                close=777.0,
                adjusted_close=777.0,
                event_time=f"{corrected_session}T21:00:00+00:00",
                availability_time=f"{sessions[corrected_session_index + 1]}T12:00:00+00:00",
                calendar_id=calendar.calendar_id,
                source="unit_fixture",
                revision_id="correction_1",
                corrected_from="original",
                adjustment={"adjustment_policy": "split_distribution_preserving_total_return", "correction": True},
            )
        )
    return (
        InMemoryMarketDataProvider(
            prices=prices,
            calendar=calendar,
            authorization=DataAuthorization(
                provider_id="authorized_fixture",
                authorized=True,
                permission_scope="offline_unit_test_adjusted_prices",
                as_of="2026-01-01T00:00:00+00:00",
                restrictions=("paper_only",),
            ),
        ),
        sessions,
    )


def _sessions(start: date, count: int) -> list[str]:
    sessions = []
    current = start
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


def _daily_return(role: str, index: int) -> float:
    cycle = ((index % 11) - 5) / 10_000
    if role == "us_equity":
        return 0.0010 + cycle
    if role == "international_equity":
        return 0.0007 + cycle * 1.1
    if role == "treasury_bond":
        return 0.0002 - cycle * 0.3
    if role == "investment_grade_credit":
        return 0.0003 - cycle * 0.1
    if role == "gold":
        return 0.0001 + cycle * 0.6
    if role == "broad_commodities":
        return 0.0002 + cycle * 0.8
    return 0.00005


class pytest_raises:
    def __init__(self, expected: type[BaseException]) -> None:
        self.expected = expected

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: object) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.expected.__name__}")
        if not issubclass(exc_type, self.expected):
            return False
        return True
