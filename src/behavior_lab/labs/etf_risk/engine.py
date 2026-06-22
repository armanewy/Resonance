from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from math import sqrt
from statistics import mean, stdev
from typing import Any

from behavior_lab.core import parse_time, stable_hash, to_jsonable
from behavior_lab.money.contracts import Action, FinancialDecisionContract
from behavior_lab.money.ledger import MoneyLedger, MoneyLedgerEntry

from behavior_lab.labs.etf_risk.market_data import (
    AdjustedPrice,
    AuthorizedMarketDataProvider,
    Universe,
    default_universe,
)


LAB_VERSION = "finance_wave2c_etf_risk_v1"
ACTION_IDS = ("cash", "low_exposure", "normal_exposure")
BASELINE_STRATEGY_IDS = (
    "buy_and_hold",
    "fixed_allocation",
    "simple_momentum",
    "volatility_scaling",
    "target_only_autoregression",
    "cash",
)
PRIMARY_STRATEGY_ID = "target_only_autoregression"


class ETFRiskError(ValueError):
    pass


@dataclass(frozen=True)
class ETFRiskConfig:
    universe: Universe = field(default_factory=default_universe)
    horizon_trading_days: int = 20
    min_history_trading_days: int = 45
    probability_lookback_windows: int = 80
    decision_cadence: str = "weekly"
    paper_notional: float = 100_000.0
    transaction_cost_bps: float = 5.0
    uncertainty_penalty_fraction: float = 0.02
    volatility_low_threshold: float = 0.18
    volatility_cash_threshold: float = 0.28
    drawdown_low_threshold: float = 0.25
    drawdown_cash_threshold: float = 0.40
    equity_underperform_cash_threshold: float = 0.45
    prospective_required_days: int = 183
    primary_strategy_id: str = PRIMARY_STRATEGY_ID
    contract_version: str = "etf-risk-v1"

    def __post_init__(self) -> None:
        if self.horizon_trading_days <= 0:
            raise ETFRiskError("horizon_trading_days must be positive")
        if self.min_history_trading_days < self.horizon_trading_days:
            raise ETFRiskError("min_history_trading_days must cover the horizon")
        if self.decision_cadence != "weekly":
            raise ETFRiskError("Wave 2C only supports weekly decision cadence")
        if float(self.paper_notional) <= 0.0:
            raise ETFRiskError("paper_notional must be positive")
        if float(self.transaction_cost_bps) < 0.0:
            raise ETFRiskError("transaction_cost_bps may not be negative")
        if self.primary_strategy_id not in BASELINE_STRATEGY_IDS:
            raise ETFRiskError("primary_strategy_id must be one of the baseline strategies")

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe.to_dict(),
            "horizon_trading_days": self.horizon_trading_days,
            "min_history_trading_days": self.min_history_trading_days,
            "probability_lookback_windows": self.probability_lookback_windows,
            "decision_cadence": self.decision_cadence,
            "paper_notional": self.paper_notional,
            "transaction_cost_bps": self.transaction_cost_bps,
            "uncertainty_penalty_fraction": self.uncertainty_penalty_fraction,
            "volatility_low_threshold": self.volatility_low_threshold,
            "volatility_cash_threshold": self.volatility_cash_threshold,
            "drawdown_low_threshold": self.drawdown_low_threshold,
            "drawdown_cash_threshold": self.drawdown_cash_threshold,
            "equity_underperform_cash_threshold": self.equity_underperform_cash_threshold,
            "prospective_required_days": self.prospective_required_days,
            "primary_strategy_id": self.primary_strategy_id,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class TargetForecast:
    next_20d_realized_volatility: float
    probability_5pct_drawdown: float
    probability_equities_underperform_cash: float
    expected_20d_return: float
    trailing_equity_cash_spread_20d: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DecisionContext:
    decision_date: str
    decision_timestamp: str
    data_cutoff: str
    price_snapshot: dict[str, Any]
    forecast: TargetForecast
    features: dict[str, Any]
    authorization: dict[str, Any]
    calendar: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class StrategyDecision:
    strategy_id: str
    action_id: str
    weights: dict[str, float]
    turnover: float
    transaction_cost: float
    expected_gross_value: float
    uncertainty_adjustment: float
    conservative_expected_net_value: float

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class ETFRiskLab:
    def __init__(self, provider: AuthorizedMarketDataProvider, config: ETFRiskConfig | None = None) -> None:
        self.provider = provider
        self.config = config or ETFRiskConfig()
        self.provider.data_authorization()

    def walk_forward(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        ledger_path: str | None = None,
        write_ledger: bool = False,
        strategy_ids: tuple[str, ...] = BASELINE_STRATEGY_IDS,
    ) -> dict[str, Any]:
        _validate_strategy_ids(strategy_ids)
        calendar = self.provider.market_calendar()
        sessions = calendar.weekly_decision_sessions(
            min_history_sessions=self.config.min_history_trading_days,
            horizon_sessions=self.config.horizon_trading_days,
            start=start,
            end=end,
        )
        latest_history = self.provider.history(self.config.universe.asset_ids, self.provider.latest_cutoff())
        full_table = _price_table(latest_history, self.config.universe.asset_ids)
        ledger = MoneyLedger(ledger_path) if write_ledger and ledger_path else None
        previous_weights = {strategy_id: allocation_weights(self.config.universe, "cash") for strategy_id in strategy_ids}
        rows = []
        ledger_records = []
        for decision_date in sessions:
            context = self.decision_context(decision_date)
            forward_rows = _forward_rows(full_table, decision_date, self.config.horizon_trading_days)
            for strategy_id in strategy_ids:
                decision = self.strategy_decision(strategy_id, context, previous_weights[strategy_id])
                realized = _realized_outcome(
                    full_rows=forward_rows,
                    universe=self.config.universe,
                    weights=decision.weights,
                    config=self.config,
                    transaction_cost=decision.transaction_cost,
                )
                row = {
                    "strategy_id": strategy_id,
                    "decision_date": decision_date,
                    "context": context.to_dict(),
                    "decision": decision.to_dict(),
                    "realized": realized,
                }
                rows.append(row)
                previous_weights[strategy_id] = decision.weights
                if ledger is not None:
                    record = ledger.append_entry(self.money_ledger_entry(context, decision, mode="backfill"))
                    ledger_records.append({"decision_id": record["payload"]["decision_id"], "record_hash": record["record_hash"]})
        return {
            "lab_version": LAB_VERSION,
            "walk_forward_only": True,
            "decision_cadence": self.config.decision_cadence,
            "strategy_ids": list(strategy_ids),
            "decision_count": len(rows),
            "decisions": rows,
            "metrics": evaluate_walk_forward(rows, self.config),
            "ledger_records": ledger_records,
            "real_money_eligibility": real_money_eligibility([], self.config),
        }

    def paper_cycle(
        self,
        *,
        ledger_path: str,
        decision_cutoff: str | None = None,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        strategy = strategy_id or self.config.primary_strategy_id
        _validate_strategy_ids((strategy,))
        cutoff = decision_cutoff or self.provider.latest_cutoff()
        latest = self.provider.latest_prices(self.config.universe.asset_ids, cutoff)
        decision_date = max(price.market_date for price in latest.values())
        context = self.decision_context(decision_date, decision_cutoff=cutoff)
        previous_weights = _latest_strategy_weights(ledger_path, strategy, self.config.universe)
        decision = self.strategy_decision(strategy, context, previous_weights)
        ledger = MoneyLedger(ledger_path)
        record = ledger.append_entry(self.money_ledger_entry(context, decision, mode="paper_cycle"))
        return {
            "lab_version": LAB_VERSION,
            "paper_only": True,
            "decision": decision.to_dict(),
            "context": context.to_dict(),
            "ledger_record": {"decision_id": record["payload"]["decision_id"], "record_hash": record["record_hash"]},
            "real_money_eligibility": real_money_eligibility(ledger.latest_entries(), self.config),
        }

    def decision_context(self, decision_date: str, *, decision_cutoff: str | None = None) -> DecisionContext:
        calendar = self.provider.market_calendar()
        if not calendar.contains(decision_date):
            raise ETFRiskError(f"decision_date is not a market session: {decision_date}")
        cutoff = decision_cutoff or _session_cutoff(decision_date)
        history = self.provider.history(self.config.universe.asset_ids, cutoff)
        table = _price_table(history, self.config.universe.asset_ids)
        if not table or table[-1]["market_date"] > decision_date:
            table = [row for row in table if row["market_date"] <= decision_date]
        if len(table) < self.config.min_history_trading_days:
            raise ETFRiskError("insufficient authorized history before decision cutoff")
        latest_prices = _latest_prices_for_date(history, self.config.universe.asset_ids, table[-1]["market_date"])
        forecast, features = _forecast_targets(table, self.config)
        return DecisionContext(
            decision_date=decision_date,
            decision_timestamp=cutoff,
            data_cutoff=cutoff,
            price_snapshot={
                asset_id: {
                    "market_date": price.market_date,
                    "adjusted_close": price.adjusted_close,
                    "close": price.close,
                    "event_time": price.event_time,
                    "availability_time": price.availability_time,
                    "revision_id": price.revision_id,
                    "corrected_from": price.corrected_from,
                    "adjustment": dict(price.adjustment),
                }
                for asset_id, price in latest_prices.items()
            },
            forecast=forecast,
            features=features,
            authorization=self.provider.data_authorization().to_dict(),
            calendar=calendar.to_dict(),
        )

    def strategy_decision(
        self,
        strategy_id: str,
        context: DecisionContext,
        previous_weights: dict[str, float] | None = None,
    ) -> StrategyDecision:
        _validate_strategy_ids((strategy_id,))
        action_id = _select_action(strategy_id, context.forecast, self.config)
        weights = allocation_weights(self.config.universe, action_id)
        previous = previous_weights or allocation_weights(self.config.universe, "cash")
        turnover = _turnover(previous, weights)
        transaction_cost = _money(self.config.paper_notional * turnover * self.config.transaction_cost_bps / 10_000.0)
        expected_gross_value = _money(context.forecast.expected_20d_return * self.config.paper_notional)
        uncertainty = _money(
            abs(context.forecast.next_20d_realized_volatility)
            * self.config.paper_notional
            * self.config.uncertainty_penalty_fraction
        )
        conservative = _money(expected_gross_value - transaction_cost - uncertainty)
        return StrategyDecision(
            strategy_id=strategy_id,
            action_id=action_id,
            weights=weights,
            turnover=turnover,
            transaction_cost=transaction_cost,
            expected_gross_value=expected_gross_value,
            uncertainty_adjustment=uncertainty,
            conservative_expected_net_value=conservative,
        )

    def money_ledger_entry(self, context: DecisionContext, decision: StrategyDecision, *, mode: str) -> MoneyLedgerEntry:
        contract = financial_decision_contract(context, self.config)
        selected_cash_weight = decision.weights[self.config.universe.asset_for_role("cash_equivalent").asset_id]
        maximum_possible_loss = _money(self.config.paper_notional * (1.0 - selected_cash_weight))
        return MoneyLedgerEntry(
            decision_id=f"etf_risk_{mode}_{decision.strategy_id}_{context.decision_date}",
            contract_hash=contract.contract_hash(),
            decision_timestamp=context.decision_timestamp,
            data_cutoff=context.data_cutoff,
            target={
                "name": "broad_etf_20d_risk_allocation",
                "horizon_trading_days": self.config.horizon_trading_days,
                "decision_date": context.decision_date,
                "forecasts": context.forecast.to_dict(),
                "paper_only": True,
            },
            action_alternatives=list(ACTION_IDS),
            selected_action=decision.action_id,
            no_action_alternative="cash",
            capital_required=_money(self.config.paper_notional * (1.0 - selected_cash_weight)),
            maximum_possible_loss=maximum_possible_loss,
            expected_gross_value=decision.expected_gross_value,
            uncertainty_adjustment=decision.uncertainty_adjustment,
            fees=0.0,
            slippage=decision.transaction_cost,
            shipping=0.0,
            taxes_or_tax_assumption_reference=0.0,
            holding_costs=0.0,
            return_refund_allowance=0.0,
            research_api_cost=0.0,
            conservative_expected_net_value=decision.conservative_expected_net_value,
            decision_deadline=_deadline_after(context.decision_timestamp),
            feature_program_hash=stable_hash(
                {
                    "lab_version": LAB_VERSION,
                    "strategy_id": decision.strategy_id,
                    "config": self.config.to_dict(),
                }
            ),
            evidence_state="paper_decision",
            designation="paper",
            economic_event_key=f"etf_risk_{context.decision_date}_{decision.strategy_id}",
            provenance={
                "source_id": context.authorization["provider_id"],
                "strategy_id": decision.strategy_id,
                "mode": mode,
                "paper_only": True,
                "decision_cadence": self.config.decision_cadence,
                "actions_are_allocations_only": True,
                "no_broker_order_api": True,
                "no_real_trading": True,
                "no_individual_stocks_options_leverage_shorts_intraday_or_hft": True,
                "price_snapshot": context.price_snapshot,
                "weights": decision.weights,
                "transaction_cost_assumption_bps": self.config.transaction_cost_bps,
            },
            artifact_hashes={
                "price_snapshot_hash": stable_hash(context.price_snapshot),
                "forecast_hash": stable_hash(context.forecast.to_dict()),
                "contract_hash": contract.contract_hash(),
            },
            assumption_versions={
                "lab": LAB_VERSION,
                "adjusted_price_semantics": "split_distribution_preserving_v1",
                "transaction_costs": "turnover_bps_v1",
                "prospective_requirement": "six_months_paper_decisions_v1",
            },
        )


def financial_decision_contract(context: DecisionContext, config: ETFRiskConfig | None = None) -> FinancialDecisionContract:
    cfg = config or ETFRiskConfig()
    cash_asset_id = cfg.universe.asset_for_role("cash_equivalent").asset_id
    return FinancialDecisionContract(
        contract_id=f"etf_risk_20d_{context.decision_date}",
        domain="etf_risk",
        target={
            "name": "broad_etf_20d_risk_allocation",
            "targets": [
                "next_20_trading_day_realized_volatility",
                "probability_5pct_drawdown",
                "probability_equities_underperform_cash_20d",
            ],
            "universe": cfg.universe.to_dict(),
        },
        decision_horizon=f"{cfg.horizon_trading_days} trading days",
        decision_deadline=_deadline_after(context.decision_timestamp),
        available_actions=[
            Action(
                action_id=action_id,
                action_type="paper_broad_etf_allocation",
                parameters={"weights": allocation_weights(cfg.universe, action_id)},
                capital_required=_money(cfg.paper_notional * (1.0 - allocation_weights(cfg.universe, action_id)[cash_asset_id])),
                maximum_possible_loss=_money(cfg.paper_notional * (1.0 - allocation_weights(cfg.universe, action_id)[cash_asset_id])),
                fixed_costs=0.0,
                variable_costs={"transaction_cost_bps": cfg.transaction_cost_bps},
                constraints={
                    "paper_only": True,
                    "long_only": True,
                    "no_leverage": True,
                    "no_options": True,
                    "no_shorts": True,
                    "no_individual_stocks": True,
                    "no_broker_order_api": True,
                },
                action_mode="reactive",
                reversible=True,
            )
            for action_id in ACTION_IDS
        ],
        no_action_id="cash",
        payoff_specification={
            "paper_metric": "20_trading_day_net_return_after_assumed_turnover_costs",
            "risk_targets": [
                "realized_volatility",
                "5pct_drawdown_event",
                "equity_underperforms_cash_event",
            ],
        },
        cost_policy={
            "transaction_cost_bps": cfg.transaction_cost_bps,
            "slippage_is_turnover_based": True,
            "material_cost_fields": ["slippage"],
        },
        risk_policy={
            "paper_only": True,
            "long_only": True,
            "no_leverage_options_shorts_intraday_hft": True,
            "six_months_prospective_paper_required_before_real_money_eligibility": True,
        },
        liquidity_policy={"allowed_universe": "broad_etf_or_cash_equivalent_only"},
        resolution_source={"source": "authorized_market_data_provider", "provider_id": context.authorization["provider_id"]},
        data_cutoff_policy={
            "event_time_preserved": True,
            "availability_time_preserved": True,
            "no_revised_value_leaks_backward": True,
            "exact_prices_recorded_at_cutoff": True,
            "data_cutoff": context.data_cutoff,
        },
        prospective_requirement={"minimum_days": cfg.prospective_required_days, "paper_only_until_satisfied": True},
        notification_threshold={"enabled": False},
        paper_only=True,
        contract_version=cfg.contract_version,
    )


def allocation_weights(universe: Universe, action_id: str) -> dict[str, float]:
    role_weights = {
        "cash": {
            "cash_equivalent": 1.0,
        },
        "low_exposure": {
            "us_equity": 0.20,
            "international_equity": 0.10,
            "treasury_bond": 0.30,
            "investment_grade_credit": 0.15,
            "gold": 0.05,
            "broad_commodities": 0.05,
            "cash_equivalent": 0.15,
        },
        "normal_exposure": {
            "us_equity": 0.45,
            "international_equity": 0.20,
            "treasury_bond": 0.15,
            "investment_grade_credit": 0.07,
            "gold": 0.05,
            "broad_commodities": 0.05,
            "cash_equivalent": 0.03,
        },
    }
    if action_id not in role_weights:
        raise ETFRiskError(f"unknown action_id: {action_id}")
    weights = {asset.asset_id: 0.0 for asset in universe.assets}
    for role, weight in role_weights[action_id].items():
        weights[universe.asset_for_role(role).asset_id] = float(weight)
    _assert_long_only_fully_invested(weights)
    return weights


def evaluate_walk_forward(rows: list[dict[str, Any]], config: ETFRiskConfig | None = None) -> dict[str, Any]:
    cfg = config or ETFRiskConfig()
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy_id"], []).append(row)
    metrics = {
        "walk_forward_only": True,
        "decision_cadence": cfg.decision_cadence,
        "transaction_cost_assumption_bps": cfg.transaction_cost_bps,
        "strategies": {},
        "baseline_strategy_ids": list(BASELINE_STRATEGY_IDS),
        "parameter_neighborhood_sensitivity": parameter_neighborhood_sensitivity(rows, cfg),
    }
    for strategy_id, strategy_rows in sorted(by_strategy.items()):
        net_returns = [float(row["realized"]["net_return"]) for row in strategy_rows]
        no_action_returns = [float(row["realized"]["no_action_return"]) for row in strategy_rows]
        cumulative = _cumulative_return_curve(net_returns)
        annualized_return = _annualized_return(net_returns)
        annualized_vol = _annualized_volatility(net_returns)
        metrics["strategies"][strategy_id] = {
            "decision_count": len(strategy_rows),
            "total_turnover": _money(sum(float(row["decision"]["turnover"]) for row in strategy_rows)),
            "average_turnover": _ratio(sum(float(row["decision"]["turnover"]) for row in strategy_rows), len(strategy_rows)),
            "total_transaction_cost": _money(sum(float(row["decision"]["transaction_cost"]) for row in strategy_rows)),
            "cumulative_net_return": _round6(cumulative[-1] - 1.0 if cumulative else 0.0),
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "risk_adjusted_return": _ratio(annualized_return, annualized_vol),
            "maximum_drawdown": _maximum_drawdown(cumulative),
            "no_action_comparison": {
                "average_net_return_minus_cash": _round6(mean([net - cash for net, cash in zip(net_returns, no_action_returns)]) if net_returns else 0.0),
                "outperformed_cash_frequency": _ratio(sum(1 for net, cash in zip(net_returns, no_action_returns) if net > cash), len(net_returns)),
            },
            "calibration": {
                "probability_5pct_drawdown": _binary_calibration(
                    strategy_rows,
                    forecast_key="probability_5pct_drawdown",
                    outcome_key="drawdown_5pct_event",
                ),
                "probability_equities_underperform_cash": _binary_calibration(
                    strategy_rows,
                    forecast_key="probability_equities_underperform_cash",
                    outcome_key="equities_underperform_cash_event",
                ),
                "realized_volatility": _volatility_error(strategy_rows),
            },
            "regime_period_concentration": _regime_period_concentration(strategy_rows),
        }
    return metrics


def real_money_eligibility(entries: list[Any], config: ETFRiskConfig | None = None) -> dict[str, Any]:
    cfg = config or ETFRiskConfig()
    materialized = [entry.to_dict() if hasattr(entry, "to_dict") else dict(entry) for entry in entries]
    paper_cycle_dates = sorted(
        {
            parse_time(entry["decision_timestamp"]).date()
            for entry in materialized
            if entry.get("designation") == "paper"
            and (entry.get("provenance") or {}).get("mode") == "paper_cycle"
            and entry.get("evidence_state") == "paper_decision"
        }
    )
    observed_days = (paper_cycle_dates[-1] - paper_cycle_dates[0]).days if len(paper_cycle_dates) >= 2 else 0
    return {
        "eligible": False,
        "paper_cycle_span_days": observed_days,
        "required_days": cfg.prospective_required_days,
        "paper_decision_count": len(paper_cycle_dates),
        "blocking_reasons": [
            "real-money eligibility is blocked in this wave",
            "six months of prospective paper decisions are required before any later eligibility review",
        ],
    }


def parameter_neighborhood_sensitivity(rows: list[dict[str, Any]], config: ETFRiskConfig | None = None) -> dict[str, Any]:
    cfg = config or ETFRiskConfig()
    target_rows = [row for row in rows if row["strategy_id"] == PRIMARY_STRATEGY_ID]
    output = {}
    for multiplier in (0.9, 1.0, 1.1):
        perturbed = ETFRiskConfig(
            universe=cfg.universe,
            horizon_trading_days=cfg.horizon_trading_days,
            min_history_trading_days=cfg.min_history_trading_days,
            probability_lookback_windows=cfg.probability_lookback_windows,
            paper_notional=cfg.paper_notional,
            transaction_cost_bps=cfg.transaction_cost_bps,
            uncertainty_penalty_fraction=cfg.uncertainty_penalty_fraction,
            volatility_low_threshold=cfg.volatility_low_threshold * multiplier,
            volatility_cash_threshold=cfg.volatility_cash_threshold * multiplier,
            drawdown_low_threshold=cfg.drawdown_low_threshold * multiplier,
            drawdown_cash_threshold=cfg.drawdown_cash_threshold * multiplier,
            equity_underperform_cash_threshold=cfg.equity_underperform_cash_threshold * multiplier,
            prospective_required_days=cfg.prospective_required_days,
            primary_strategy_id=cfg.primary_strategy_id,
            contract_version=cfg.contract_version,
        )
        action_changes = 0
        returns = []
        for row in target_rows:
            forecast = TargetForecast(**row["context"]["forecast"])
            action = _select_action(PRIMARY_STRATEGY_ID, forecast, perturbed)
            if action != row["decision"]["action_id"]:
                action_changes += 1
            returns.append(row["realized"]["returns_by_action"][action])
        output[str(multiplier)] = {
            "action_change_frequency": _ratio(action_changes, len(target_rows)),
            "average_20d_return": _round6(mean(returns) if returns else 0.0),
        }
    return output


def _forecast_targets(table: list[dict[str, Any]], config: ETFRiskConfig) -> tuple[TargetForecast, dict[str, Any]]:
    normal_weights = allocation_weights(config.universe, "normal_exposure")
    equity_weights = _equity_weights(config.universe)
    cash_weights = allocation_weights(config.universe, "cash")
    normal_returns = _portfolio_daily_returns(table, normal_weights)
    equity_returns = _portfolio_daily_returns(table, equity_weights)
    cash_returns = _portfolio_daily_returns(table, cash_weights)
    trailing_normal = normal_returns[-config.horizon_trading_days :]
    trailing_equity = equity_returns[-config.horizon_trading_days :]
    trailing_cash = cash_returns[-config.horizon_trading_days :]
    volatility = _annualized_volatility(trailing_normal)
    windows = _prior_windows(normal_returns, config.horizon_trading_days, config.probability_lookback_windows)
    equity_windows = _prior_windows(equity_returns, config.horizon_trading_days, config.probability_lookback_windows)
    cash_windows = _prior_windows(cash_returns, config.horizon_trading_days, config.probability_lookback_windows)
    drawdown_events = [_return_drawdown(window) <= -0.05 for window in windows]
    underperform_events = [
        _compound_return(equity_window) < _compound_return(cash_window)
        for equity_window, cash_window in zip(equity_windows, cash_windows)
    ]
    expected_returns = [_compound_return(window) for window in windows]
    forecast = TargetForecast(
        next_20d_realized_volatility=_round6(volatility),
        probability_5pct_drawdown=_ratio(sum(drawdown_events), len(drawdown_events)) if drawdown_events else 0.5,
        probability_equities_underperform_cash=_ratio(sum(underperform_events), len(underperform_events)) if underperform_events else 0.5,
        expected_20d_return=_round6(mean(expected_returns) if expected_returns else _compound_return(trailing_normal)),
        trailing_equity_cash_spread_20d=_round6(_compound_return(trailing_equity) - _compound_return(trailing_cash)),
        sample_count=len(windows),
    )
    return forecast, {
        "target_method": "target_only_autoregression_from_prior_20d_windows",
        "history_sessions": len(table),
        "lookback_windows": len(windows),
        "trailing_normal_return_20d": _round6(_compound_return(trailing_normal)),
        "trailing_equity_return_20d": _round6(_compound_return(trailing_equity)),
        "trailing_cash_return_20d": _round6(_compound_return(trailing_cash)),
    }


def _select_action(strategy_id: str, forecast: TargetForecast, config: ETFRiskConfig) -> str:
    if strategy_id == "buy_and_hold":
        return "normal_exposure"
    if strategy_id == "fixed_allocation":
        return "low_exposure"
    if strategy_id == "cash":
        return "cash"
    if strategy_id == "simple_momentum":
        if forecast.trailing_equity_cash_spread_20d <= 0.0:
            return "cash"
        if forecast.next_20d_realized_volatility > config.volatility_low_threshold:
            return "low_exposure"
        return "normal_exposure"
    if strategy_id == "volatility_scaling":
        if forecast.next_20d_realized_volatility >= config.volatility_cash_threshold:
            return "cash"
        if forecast.next_20d_realized_volatility >= config.volatility_low_threshold:
            return "low_exposure"
        return "normal_exposure"
    if strategy_id == "target_only_autoregression":
        if (
            forecast.next_20d_realized_volatility >= config.volatility_cash_threshold
            or forecast.probability_5pct_drawdown >= config.drawdown_cash_threshold
            or forecast.probability_equities_underperform_cash >= config.equity_underperform_cash_threshold
        ):
            return "cash"
        if (
            forecast.next_20d_realized_volatility >= config.volatility_low_threshold
            or forecast.probability_5pct_drawdown >= config.drawdown_low_threshold
        ):
            return "low_exposure"
        return "normal_exposure"
    raise ETFRiskError(f"unknown strategy_id: {strategy_id}")


def _realized_outcome(
    *,
    full_rows: list[dict[str, Any]],
    universe: Universe,
    weights: dict[str, float],
    config: ETFRiskConfig,
    transaction_cost: float,
) -> dict[str, Any]:
    returns_by_action = {}
    for action_id in ACTION_IDS:
        action_returns = _portfolio_daily_returns(full_rows, allocation_weights(universe, action_id))
        returns_by_action[action_id] = _round6(_compound_return(action_returns))
    selected_returns = _portfolio_daily_returns(full_rows, weights)
    equity_returns = _portfolio_daily_returns(full_rows, _equity_weights(universe))
    cash_returns = _portfolio_daily_returns(full_rows, allocation_weights(universe, "cash"))
    gross_return = _compound_return(selected_returns)
    cost_return = transaction_cost / config.paper_notional
    return {
        "gross_return": _round6(gross_return),
        "net_return": _round6(gross_return - cost_return),
        "transaction_cost_return": _round6(cost_return),
        "realized_volatility": _round6(_annualized_volatility(selected_returns)),
        "drawdown_5pct_event": _return_drawdown(selected_returns) <= -0.05,
        "equities_underperform_cash_event": _compound_return(equity_returns) < _compound_return(cash_returns),
        "no_action_return": _round6(returns_by_action["cash"]),
        "returns_by_action": returns_by_action,
        "forward_session_count": max(0, len(full_rows) - 1),
    }


def _price_table(history: list[AdjustedPrice], asset_ids: list[str]) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, float]] = {}
    for price in history:
        by_date.setdefault(price.market_date, {})[price.asset_id] = float(price.adjusted_close)
    rows = []
    for market_date in sorted(by_date):
        if all(asset_id in by_date[market_date] for asset_id in asset_ids):
            rows.append({"market_date": market_date, "prices": dict(by_date[market_date])})
    return rows


def _latest_prices_for_date(history: list[AdjustedPrice], asset_ids: list[str], market_date: str) -> dict[str, AdjustedPrice]:
    prices = {price.asset_id: price for price in history if price.market_date == market_date}
    missing = sorted(set(asset_ids) - set(prices))
    if missing:
        raise ETFRiskError(f"missing complete latest price snapshot: {missing}")
    return {asset_id: prices[asset_id] for asset_id in asset_ids}


def _forward_rows(full_table: list[dict[str, Any]], decision_date: str, horizon: int) -> list[dict[str, Any]]:
    dates = [row["market_date"] for row in full_table]
    if decision_date not in dates:
        raise ETFRiskError(f"decision_date has no full price row: {decision_date}")
    index = dates.index(decision_date)
    rows = full_table[index : index + horizon + 1]
    if len(rows) < horizon + 1:
        raise ETFRiskError("insufficient future rows for historical walk-forward evaluation")
    return rows


def _portfolio_daily_returns(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[float]:
    returns = []
    for previous, current in zip(rows, rows[1:]):
        daily = 0.0
        for asset_id, weight in weights.items():
            daily += weight * (float(current["prices"][asset_id]) / float(previous["prices"][asset_id]) - 1.0)
        returns.append(daily)
    return returns


def _prior_windows(returns: list[float], horizon: int, max_windows: int) -> list[list[float]]:
    windows = []
    for end in range(horizon, len(returns) + 1):
        windows.append(returns[end - horizon : end])
    return windows[-max_windows:]


def _compound_return(returns: list[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1.0 + float(item)
    return value - 1.0


def _return_drawdown(returns: list[float]) -> float:
    value = 1.0
    peak = 1.0
    worst = 0.0
    for item in returns:
        value *= 1.0 + float(item)
        if value > peak:
            peak = value
        drawdown = value / peak - 1.0
        if drawdown < worst:
            worst = drawdown
    return worst


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    keys = set(previous) | set(current)
    return _round6(sum(abs(float(current.get(key, 0.0)) - float(previous.get(key, 0.0))) for key in keys) / 2.0)


def _equity_weights(universe: Universe) -> dict[str, float]:
    weights = {asset.asset_id: 0.0 for asset in universe.assets}
    weights[universe.asset_for_role("us_equity").asset_id] = 0.70
    weights[universe.asset_for_role("international_equity").asset_id] = 0.30
    return weights


def _assert_long_only_fully_invested(weights: dict[str, float]) -> None:
    if any(weight < 0.0 or weight > 1.0 for weight in weights.values()):
        raise ETFRiskError("allocations must be long-only and unlevered")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ETFRiskError("allocations must sum to one")


def _cumulative_return_curve(returns: list[float]) -> list[float]:
    value = 1.0
    curve = []
    for item in returns:
        value *= 1.0 + item
        curve.append(value)
    return curve


def _maximum_drawdown(curve: list[float]) -> dict[str, Any]:
    if not curve:
        return {"maximum_drawdown": 0.0, "peak_index": None, "trough_index": None}
    peak = curve[0]
    peak_index = 0
    worst = 0.0
    worst_peak = 0
    worst_trough = 0
    for index, value in enumerate(curve):
        if value > peak:
            peak = value
            peak_index = index
        drawdown = value / peak - 1.0
        if drawdown < worst:
            worst = drawdown
            worst_peak = peak_index
            worst_trough = index
    return {"maximum_drawdown": _round6(worst), "peak_index": worst_peak, "trough_index": worst_trough}


def _annualized_return(returns: list[float]) -> float:
    if not returns:
        return 0.0
    return _round6(mean(returns) * 52.0)


def _annualized_volatility(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return _round6(stdev(returns) * sqrt(252.0))


def _binary_calibration(rows: list[dict[str, Any]], *, forecast_key: str, outcome_key: str) -> dict[str, Any]:
    bins: dict[int, list[tuple[float, bool]]] = {}
    for row in rows:
        probability = float(row["context"]["forecast"][forecast_key])
        outcome = bool(row["realized"][outcome_key])
        bin_id = min(4, int(probability * 5))
        bins.setdefault(bin_id, []).append((probability, outcome))
    details = []
    weighted_error = 0.0
    total = 0
    for bin_id in range(5):
        values = bins.get(bin_id, [])
        if not values:
            continue
        avg_pred = mean(item[0] for item in values)
        avg_actual = mean(1.0 if item[1] else 0.0 for item in values)
        weighted_error += abs(avg_pred - avg_actual) * len(values)
        total += len(values)
        details.append({"bin": bin_id, "count": len(values), "avg_predicted": _round6(avg_pred), "actual_rate": _round6(avg_actual)})
    return {"expected_calibration_error": _round6(weighted_error / total) if total else 0.0, "bins": details}


def _volatility_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        abs(float(row["context"]["forecast"]["next_20d_realized_volatility"]) - float(row["realized"]["realized_volatility"]))
        for row in rows
    ]
    return {"mean_absolute_error": _round6(mean(errors) if errors else 0.0), "count": len(errors)}


def _regime_period_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[str, int] = {}
    by_regime: dict[str, int] = {}
    for row in rows:
        year = row["decision_date"][:4]
        by_year[year] = by_year.get(year, 0) + 1
        realized_vol = float(row["realized"]["realized_volatility"])
        if realized_vol < 0.12:
            regime = "low_volatility"
        elif realized_vol < 0.24:
            regime = "normal_volatility"
        else:
            regime = "high_volatility"
        by_regime[regime] = by_regime.get(regime, 0) + 1
    total = len(rows)
    return {
        "period_counts": by_year,
        "regime_counts": by_regime,
        "largest_period_share": _round6(max(by_year.values()) / total) if total and by_year else 0.0,
        "largest_regime_share": _round6(max(by_regime.values()) / total) if total and by_regime else 0.0,
    }


def _latest_strategy_weights(ledger_path: str, strategy_id: str, universe: Universe) -> dict[str, float]:
    entries = MoneyLedger(ledger_path).latest_entries()
    for entry in reversed(entries):
        provenance = entry.provenance or {}
        if provenance.get("strategy_id") == strategy_id and isinstance(provenance.get("weights"), dict):
            return {asset.asset_id: float(provenance["weights"].get(asset.asset_id, 0.0)) for asset in universe.assets}
    return allocation_weights(universe, "cash")


def _validate_strategy_ids(strategy_ids: tuple[str, ...]) -> None:
    unknown = sorted(set(strategy_ids) - set(BASELINE_STRATEGY_IDS))
    if unknown:
        raise ETFRiskError(f"unknown strategy IDs: {unknown}")


def _session_cutoff(session: str) -> str:
    return f"{session}T21:30:00+00:00"


def _deadline_after(timestamp: str) -> str:
    return (parse_time(timestamp) + timedelta(days=1)).isoformat()


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return _round6(float(numerator) / float(denominator))


def _round6(value: float) -> float:
    return round(float(value), 6)


def _money(value: float) -> float:
    return round(float(value) + 0.0, 2)
