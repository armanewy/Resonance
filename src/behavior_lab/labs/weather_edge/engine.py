from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.labs.weather_edge.baselines import BaselineProbabilities, compute_baselines
from behavior_lab.labs.weather_edge.fixtures import WeatherEdgeProvider
from behavior_lab.labs.weather_edge.models import (
    CostPolicy,
    DailyHighTemperatureEvent,
    Settlement,
    StrategyConfig,
    WeatherSnapshot,
)
from behavior_lab.money.accounting import summarize_money_entries
from behavior_lab.money.contracts import Action, FinancialDecisionContract
from behavior_lab.money.ledger import MoneyLedgerEntry
from behavior_lab.money.storage import MoneyStorage


WEATHER_EDGE_SOURCE_ID = "weather_edge"
FEATURE_PROGRAM = "weather_edge.paper_lab.v1"
NO_TRADE_ACTION = "no_trade"
BUY_YES_ACTION = "buy_yes"
REALIZED_ZERO_COSTS = {
    "fees": 0.0,
    "slippage": 0.0,
    "shipping": 0.0,
    "taxes_or_tax_assumption_reference": 0.0,
    "holding_costs": 0.0,
    "return_refund_allowance": 0.0,
    "research_api_cost": 0.0,
}


@dataclass(frozen=True)
class DecisionCandidate:
    event: DailyHighTemperatureEvent
    contract: FinancialDecisionContract
    entry: MoneyLedgerEntry
    projected_conservative_net_value: float
    selected_trade: bool


def backfill(
    provider: WeatherEdgeProvider,
    storage_root: str | Path,
    *,
    as_of: str | None = None,
    cost_policy: CostPolicy | None = None,
    strategy: StrategyConfig | None = None,
) -> dict[str, Any]:
    """Run walk-forward historical paper decisions and resolve known outcomes."""

    timestamp = as_of or utc_now()
    policy = cost_policy or CostPolicy()
    config = strategy or StrategyConfig()
    storage = MoneyStorage(storage_root)
    events = [
        event
        for event in provider.discover_events(timestamp, include_resolved=True)
        if provider.settlement(event.event_id) is not None and parse_time(event.close_time) <= parse_time(timestamp)
    ]
    groups = _city_event_groups(events)
    appended = 0
    resolved = 0
    duplicates = 0
    decisions: list[dict[str, Any]] = []
    for city_event_key in sorted(groups):
        group = groups[city_event_key]
        decision_time = fixed_decision_timestamp(group[0], config)
        candidate = _decide_city_event(
            provider,
            group,
            decision_timestamp=decision_time,
            cost_policy=policy,
            strategy=config,
            mode="backfill",
        )
        result = _record_candidate(storage, candidate)
        appended += int(result["appended"])
        duplicates += int(result["duplicate"])
        settlement = provider.settlement(candidate.event.event_id)
        if settlement is not None:
            resolution = _resolve_candidate(storage, candidate, settlement)
            resolved += int(resolution["resolved"])
            result["resolution"] = resolution
        decisions.append({**result, "decision_id": candidate.entry.decision_id})
    return {
        "mode": "backfill",
        "as_of": timestamp,
        "fixed_decision_horizon": config.decision_horizon,
        "discovered_event_count": len(events),
        "city_event_count": len(groups),
        "decisions_appended": appended,
        "decisions_resolved": resolved,
        "duplicate_decisions": duplicates,
        "paper_only": True,
        "authenticates_for_trading": False,
        "submits_orders": False,
        "notifications_allowed": False,
        "decisions": decisions,
    }


def paper_cycle(
    provider: WeatherEdgeProvider,
    storage_root: str | Path,
    *,
    as_of: str,
    cost_policy: CostPolicy | None = None,
    strategy: StrategyConfig | None = None,
) -> dict[str, Any]:
    """Run the current paper-only cycle for open events."""

    parse_time(as_of)
    policy = cost_policy or CostPolicy()
    config = strategy or StrategyConfig()
    storage = MoneyStorage(storage_root)
    events = provider.discover_events(as_of, include_resolved=False)
    groups = _city_event_groups(events)
    appended = 0
    duplicates = 0
    decisions: list[dict[str, Any]] = []
    for city_event_key in sorted(groups):
        candidate = _decide_city_event(
            provider,
            groups[city_event_key],
            decision_timestamp=as_of,
            cost_policy=policy,
            strategy=config,
            mode="paper_cycle",
        )
        result = _record_candidate(storage, candidate)
        appended += int(result["appended"])
        duplicates += int(result["duplicate"])
        decisions.append({**result, "decision_id": candidate.entry.decision_id})
    return {
        "mode": "paper_cycle",
        "as_of": as_of,
        "fixed_decision_horizon": config.decision_horizon,
        "discovered_event_count": len(events),
        "city_event_count": len(groups),
        "decisions_appended": appended,
        "duplicate_decisions": duplicates,
        "paper_only": True,
        "authenticates_for_trading": False,
        "submits_orders": False,
        "notifications_allowed": False,
        "decisions": decisions,
    }


def report(
    storage_root: str | Path,
    *,
    provider: Any | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Summarize Weather Edge paper decisions and evidence gates."""

    generated_at = as_of or utc_now()
    storage = MoneyStorage(storage_root)
    entries = [
        entry.to_dict()
        for entry in storage.ledger.latest_entries()
        if (entry.provenance or {}).get("source_id") == WEATHER_EDGE_SOURCE_ID
    ]
    summary = summarize_money_entries(entries)
    resolved = [entry for entry in entries if entry.get("evidence_state") == "resolved_paper"]
    prospective = [entry for entry in entries if entry.get("evidence_state") == "paper_decision"]
    scorecard = _score_resolved_entries(resolved)
    cost_sensitivity = _pessimistic_cost_sensitivity(resolved)
    concentration = _concentration(resolved)
    historical_available = _historical_available(provider, resolved)
    gate = _evidence_gate(
        resolved_count=len({entry.get("economic_event_key") for entry in resolved}),
        historical_available=historical_available,
        scorecard=scorecard,
        cost_sensitivity=cost_sensitivity,
        concentration=concentration,
        prospective=prospective,
    )
    return {
        "generated_at": generated_at,
        "source_id": WEATHER_EDGE_SOURCE_ID,
        "summary": summary,
        "scorecard": scorecard,
        "pessimistic_cost_sensitivity": cost_sensitivity,
        "concentration": concentration,
        "evidence_gate": gate,
        "paper_only": True,
        "authenticates_for_trading": False,
        "submits_orders": False,
        "notifications_allowed": False,
    }


def fixed_decision_timestamp(event: DailyHighTemperatureEvent, strategy: StrategyConfig | None = None) -> str:
    config = strategy or StrategyConfig()
    candidate = parse_time(event.close_time) - timedelta(hours=int(config.horizon_hours_before_close))
    opened = parse_time(event.open_time)
    if candidate < opened:
        return event.open_time
    return candidate.isoformat()


def build_as_of_weather_snapshot(
    provider: WeatherEdgeProvider,
    event: DailyHighTemperatureEvent,
    as_of: str,
) -> WeatherSnapshot:
    decision_time = parse_time(as_of)
    snapshot = provider.weather_snapshot(event.event_id, as_of)
    if snapshot.event_id != event.event_id:
        raise ValueError("weather snapshot event_id does not match event")
    if snapshot.station_id != event.station_id:
        raise ValueError("weather snapshot station_id does not match event")
    if snapshot.timezone != event.timezone:
        raise ValueError("weather snapshot timezone does not match event")
    if parse_time(snapshot.as_of) > decision_time:
        raise ValueError("weather snapshot is after the decision time")
    if parse_time(snapshot.forecast_issued_at) > decision_time:
        raise ValueError("forecast issue time is after the decision time")
    return snapshot


def weather_event_contract(event: DailyHighTemperatureEvent) -> FinancialDecisionContract:
    actions = [
        Action(
            action_id=NO_TRADE_ACTION,
            action_type="no_action",
            parameters={"reason": "paper_lab_abstention_or_no_positive_edge"},
        ),
        Action(
            action_id=BUY_YES_ACTION,
            action_type="event_market_paper_fill",
            parameters={
                "side": "yes",
                "paper_only": True,
                "requires_executable_order_book_ask": True,
            },
        ),
    ]
    return FinancialDecisionContract(
        contract_id=f"weather_edge_{stable_hash(event.to_dict())[:16]}",
        domain="event_market",
        target={
            "type": "daily_high_temperature_event_market",
            "city_event_key": event.city_event_key,
            "event_id": event.event_id,
            "city": event.city,
            "station_id": event.station_id,
            "local_date": event.local_date,
            "settlement_semantics": event.settlement_semantics(),
        },
        decision_horizon=StrategyConfig().decision_horizon,
        decision_deadline=event.close_time,
        available_actions=actions,
        no_action_id=NO_TRADE_ACTION,
        payoff_specification={
            "instrument": "binary_event_contract",
            "side": "yes",
            "payout_if_bracket_settles_yes": 1.0,
            "gross_value": "paper payout minus executable entry price before fees and slippage",
            "no_midpoint_or_candle_fill": True,
        },
        cost_policy={
            "fees": "explicit per-contract paper fee",
            "slippage": "explicit per-contract paper slippage",
            "liquidity": "quantity is bounded by current order-book quantity",
            "uncertainty_buffer": "probability-point buffer converted to dollars at quantity",
            "unknown_material_cost": "ineligible",
        },
        risk_policy={
            "paper_only": True,
            "leverage_allowed": False,
            "real_trading_authentication_allowed": False,
            "order_submission_allowed": False,
        },
        liquidity_policy={
            "source": "current_order_book_depth",
            "midpoint_executable": False,
            "candle_extremes_executable": False,
            "preserve_order_book_quantity": True,
        },
        resolution_source={
            "settlement_series": event.settlement_series,
            "station_id": event.station_id,
            "report_source": event.report_source,
            "report_name": event.report_name,
            "timezone": event.timezone,
            "dst_status": event.dst_status,
        },
        data_cutoff_policy={"as_of_required": True, "post_decision_weather_forbidden": True},
        prospective_requirement={
            "minimum_resolved_city_days_when_available": 150,
            "minimum_prospective_days_before_later_real_money_review": 30,
            "strategy_changes_during_incubation_allowed": False,
        },
        notification_threshold={"notifications_allowed": False},
        paper_only=True,
        contract_version="weather_edge_event_market.v1",
    )


def _decide_city_event(
    provider: WeatherEdgeProvider,
    events: list[DailyHighTemperatureEvent],
    *,
    decision_timestamp: str,
    cost_policy: CostPolicy,
    strategy: StrategyConfig,
    mode: str,
) -> DecisionCandidate:
    if not events:
        raise ValueError("cannot decide an empty city-event group")
    candidates = [
        _candidate_for_event(
            provider,
            event,
            decision_timestamp=decision_timestamp,
            cost_policy=cost_policy,
            strategy=strategy,
            mode=mode,
        )
        for event in sorted(events, key=lambda item: item.event_id)
    ]
    trades = [candidate for candidate in candidates if candidate.selected_trade]
    if trades:
        return max(trades, key=lambda candidate: candidate.projected_conservative_net_value)
    return max(candidates, key=lambda candidate: candidate.projected_conservative_net_value)


def _candidate_for_event(
    provider: WeatherEdgeProvider,
    event: DailyHighTemperatureEvent,
    *,
    decision_timestamp: str,
    cost_policy: CostPolicy,
    strategy: StrategyConfig,
    mode: str,
) -> DecisionCandidate:
    decision_time = parse_time(decision_timestamp)
    if decision_time > parse_time(event.close_time):
        raise ValueError("decision_timestamp may not be after event close_time")
    contract = weather_event_contract(event)
    market_depth = provider.market_depth(event.event_id, decision_timestamp)
    if market_depth.event_id != event.event_id:
        raise ValueError("market depth event_id does not match event")
    if parse_time(market_depth.as_of) > decision_time:
        raise ValueError("market depth is after the decision time")
    snapshot = build_as_of_weather_snapshot(provider, event, decision_timestamp)
    history = provider.station_history(event.station_id, before_local_date=event.local_date)
    baselines = compute_baselines(
        bracket=event.bracket,
        market_depth=market_depth,
        weather_snapshot=snapshot,
        station_history=history,
    )
    selected_action, execution, projected = _select_action(baselines, market_depth, cost_policy)
    data_cutoff = _latest_timestamp([market_depth.as_of, snapshot.as_of, snapshot.forecast_issued_at])
    costs = _entry_costs(selected_action, execution)
    uncertainty = execution["uncertainty_adjustment"] if selected_action == BUY_YES_ACTION else 0.0
    conservative = projected["conservative_expected_net_value"] if selected_action == BUY_YES_ACTION else 0.0
    entry = MoneyLedgerEntry(
        decision_id=_decision_id(event, decision_timestamp, strategy),
        contract_hash=contract.contract_hash(),
        decision_timestamp=decision_timestamp,
        data_cutoff=data_cutoff,
        target=contract.target,
        action_alternatives=[action.action_id for action in contract.available_actions],
        selected_action=selected_action,
        no_action_alternative=contract.no_action_id,
        capital_required=execution["capital_required"] if selected_action == BUY_YES_ACTION else 0.0,
        maximum_possible_loss=execution["maximum_possible_loss"] if selected_action == BUY_YES_ACTION else 0.0,
        expected_gross_value=projected["expected_gross_value"] if selected_action == BUY_YES_ACTION else 0.0,
        uncertainty_adjustment=uncertainty,
        fees=costs["fees"],
        slippage=costs["slippage"],
        shipping=0.0,
        taxes_or_tax_assumption_reference="not_applicable_event_market_paper",
        holding_costs=0.0,
        return_refund_allowance=0.0,
        research_api_cost=0.0,
        conservative_expected_net_value=conservative,
        decision_deadline=event.close_time,
        feature_program_hash=stable_hash({"feature_program": FEATURE_PROGRAM, "strategy": strategy.to_dict()}),
        evidence_state="paper_decision",
        designation="paper",
        mechanically_defined_no_action_outcome={
            "selected_action": NO_TRADE_ACTION,
            "realized_gross_value": 0.0,
            "realized_net_value": 0.0,
            "city_event_key": event.city_event_key,
        },
        economic_event_key=event.city_event_key,
        provenance={
            "source_id": WEATHER_EDGE_SOURCE_ID,
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.strategy_version,
            "mode": mode,
            "decision_horizon": strategy.decision_horizon,
            "fixed_horizon_hours_before_close": strategy.horizon_hours_before_close,
            "city_event_key": event.city_event_key,
            "event_id": event.event_id,
            "city": event.city,
            "local_date": event.local_date,
            "regime": snapshot.regime,
            "paper_only": True,
            "read_only": True,
            "authenticates_for_trading": False,
            "submits_orders": False,
            "notifications_allowed": False,
            "leverage_allowed": False,
            "event_semantics": event.settlement_semantics(),
            "market_depth": market_depth.to_dict(),
            "weather_snapshot": snapshot.to_dict(),
            "baselines": baselines.to_dict(),
            "model_probability_name": strategy.model_probability_name,
            "model_probability": getattr(baselines, strategy.model_probability_name),
            "cost_policy": cost_policy.to_dict(),
            "execution": execution,
            "projected": projected,
            "no_trade_reasons": execution["no_trade_reasons"],
            "prohibited_fill_sources": ["midpoint", "candle_high", "candle_low"],
        },
        artifact_hashes={
            "event_hash": stable_hash(event.to_dict()),
            "market_depth_hash": stable_hash(market_depth.to_dict()),
            "weather_snapshot_hash": stable_hash(snapshot.to_dict()),
            "baselines_hash": stable_hash(baselines.to_dict()),
        },
        assumption_versions={
            "weather_edge_engine": FEATURE_PROGRAM,
            "strategy": strategy.strategy_version,
            "cost_policy": cost_policy.version,
            "decision_horizon": strategy.decision_horizon,
        },
        material_costs_known=True,
        ineligibility_reasons=[],
    )
    return DecisionCandidate(
        event=event,
        contract=contract,
        entry=entry,
        projected_conservative_net_value=projected["conservative_expected_net_value"],
        selected_trade=selected_action == BUY_YES_ACTION,
    )


def _select_action(
    baselines: BaselineProbabilities,
    market_depth: Any,
    cost_policy: CostPolicy,
) -> tuple[str, dict[str, Any], dict[str, float]]:
    best_ask = market_depth.best_yes_ask
    no_trade_reasons: list[str] = []
    model_probability = baselines.station_bias_corrected
    executable_price = None if best_ask is None else float(best_ask.price)
    raw_book_quantity = 0 if best_ask is None else int(best_ask.quantity)
    liquidity_quantity = min(
        int(cost_policy.max_contracts),
        int(raw_book_quantity * float(cost_policy.liquidity_fraction)),
    )
    if best_ask is None:
        no_trade_reasons.append("no_executable_yes_ask")
    if liquidity_quantity <= 0:
        no_trade_reasons.append("insufficient_order_book_quantity_after_liquidity_limit")
    if executable_price is not None:
        edge_probability = model_probability - executable_price
        buffered_edge = edge_probability - float(cost_policy.uncertainty_buffer_probability)
        if edge_probability < float(cost_policy.min_edge_probability):
            no_trade_reasons.append("edge_below_minimum_probability")
        if buffered_edge <= 0.0:
            no_trade_reasons.append("edge_not_positive_after_uncertainty_buffer")
    else:
        edge_probability = 0.0
        buffered_edge = 0.0
    quantity = max(0, liquidity_quantity)
    expected_gross = _money(quantity * edge_probability)
    fees = _money(quantity * float(cost_policy.per_contract_fee))
    slippage = _money(quantity * cost_policy.slippage_per_contract)
    uncertainty = _money(quantity * float(cost_policy.uncertainty_buffer_probability))
    conservative = _money(expected_gross - fees - slippage - uncertainty)
    if conservative <= 0.0:
        no_trade_reasons.append("conservative_net_value_not_positive")
    selected = BUY_YES_ACTION if not no_trade_reasons else NO_TRADE_ACTION
    execution = {
        "side": "yes",
        "executable_price": executable_price,
        "executable_price_source": "best_yes_ask_order_book_level" if executable_price is not None else None,
        "midpoint_used": False,
        "candle_extreme_used": False,
        "raw_order_book_quantity": raw_book_quantity,
        "liquidity_fraction": cost_policy.liquidity_fraction,
        "quantity": quantity if selected == BUY_YES_ACTION else 0,
        "candidate_quantity": quantity,
        "fees": fees if selected == BUY_YES_ACTION else 0.0,
        "slippage": slippage if selected == BUY_YES_ACTION else 0.0,
        "uncertainty_adjustment": uncertainty if selected == BUY_YES_ACTION else 0.0,
        "capital_required": _money((executable_price or 0.0) * quantity + fees + slippage)
        if selected == BUY_YES_ACTION
        else 0.0,
        "maximum_possible_loss": _money((executable_price or 0.0) * quantity + fees + slippage)
        if selected == BUY_YES_ACTION
        else 0.0,
        "no_trade_reasons": sorted(set(no_trade_reasons)),
    }
    projected = {
        "model_probability": round(model_probability, 6),
        "market_implied_probability": baselines.market_implied if baselines.market_implied is not None else -1.0,
        "edge_probability": round(edge_probability, 6),
        "buffered_edge_probability": round(buffered_edge, 6),
        "expected_gross_value": expected_gross,
        "fees": fees,
        "slippage": slippage,
        "uncertainty_adjustment": uncertainty,
        "conservative_expected_net_value": conservative,
    }
    return selected, execution, projected


def _entry_costs(selected_action: str, execution: dict[str, Any]) -> dict[str, float]:
    if selected_action != BUY_YES_ACTION:
        return {"fees": 0.0, "slippage": 0.0}
    return {
        "fees": float(execution["fees"]),
        "slippage": float(execution["slippage"]),
    }


def _record_candidate(storage: MoneyStorage, candidate: DecisionCandidate) -> dict[str, Any]:
    storage.write_contract(candidate.contract)
    ledger = storage.ledger
    latest = ledger.latest_record(candidate.entry.decision_id)
    if latest is not None:
        return {
            "appended": False,
            "duplicate": True,
            "selected_action": latest["payload"]["selected_action"],
            "evidence_state": latest["payload"]["evidence_state"],
        }
    record = ledger.append_entry(candidate.entry)
    return {
        "appended": True,
        "duplicate": False,
        "selected_action": record["payload"]["selected_action"],
        "evidence_state": record["payload"]["evidence_state"],
        "record_hash": record["record_hash"],
    }


def _resolve_candidate(
    storage: MoneyStorage,
    candidate: DecisionCandidate,
    settlement: Settlement,
) -> dict[str, Any]:
    ledger = storage.ledger
    latest = ledger.latest_record(candidate.entry.decision_id)
    if latest is None:
        raise ValueError("cannot resolve a candidate before recording it")
    if latest["payload"]["evidence_state"] == "resolved_paper":
        return {"resolved": False, "duplicate_resolution": True}
    if latest["payload"]["evidence_state"] != "paper_decision":
        return {"resolved": False, "duplicate_resolution": False}
    payload = latest["payload"]
    realized = _realized_values(payload, candidate.event, settlement)
    record = ledger.append_resolution(
        candidate.entry.decision_id,
        resolution=realized["resolution"],
        realized_gross_value=realized["realized_gross_value"],
        realized_net_value=realized["realized_net_value"],
        mechanically_defined_no_action_outcome=realized["mechanically_defined_no_action_outcome"],
    )
    return {
        "resolved": True,
        "duplicate_resolution": False,
        "record_hash": record["record_hash"],
        "realized_net_value": record["payload"]["realized_net_value"],
    }


def _realized_values(
    payload: dict[str, Any],
    event: DailyHighTemperatureEvent,
    settlement: Settlement,
) -> dict[str, Any]:
    if settlement.station_id != event.station_id:
        raise ValueError("settlement station_id does not match event")
    if settlement.settlement_series != event.settlement_series:
        raise ValueError("settlement series does not match event")
    if settlement.report_source != event.report_source:
        raise ValueError("settlement report source does not match event")
    outcome_yes = event.bracket.contains(settlement.observed_high_f)
    execution = payload.get("provenance", {}).get("execution", {})
    selected_action = payload["selected_action"]
    quantity = int(execution.get("quantity") or 0)
    price = float(execution.get("executable_price") or 0.0)
    if selected_action == BUY_YES_ACTION:
        realized_gross = _money((quantity if outcome_yes else 0.0) - (quantity * price))
        realized_costs = {
            **REALIZED_ZERO_COSTS,
            "fees": float(payload["fees"] or 0.0),
            "slippage": float(payload["slippage"] or 0.0),
        }
    else:
        realized_gross = 0.0
        realized_costs = dict(REALIZED_ZERO_COSTS)
    realized_net = _money(realized_gross - sum(float(value) for value in realized_costs.values()))
    no_action = {
        "selected_action": NO_TRADE_ACTION,
        "realized_gross_value": 0.0,
        "realized_net_value": 0.0,
        "city_event_key": event.city_event_key,
    }
    return {
        "realized_gross_value": realized_gross,
        "realized_net_value": realized_net,
        "mechanically_defined_no_action_outcome": no_action,
        "resolution": {
            "outcome_yes": outcome_yes,
            "observed_high_f": settlement.observed_high_f,
            "settlement": settlement.to_dict(),
            "event_semantics": event.settlement_semantics(),
            "realized_costs": realized_costs,
            "paper_only": True,
            "authenticates_for_trading": False,
            "submits_orders": False,
            "notifications_allowed": False,
        },
    }


def _score_resolved_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for entry in entries:
        resolution = entry.get("resolution") or {}
        if "outcome_yes" not in resolution:
            continue
        outcome = 1.0 if resolution["outcome_yes"] else 0.0
        baselines = (entry.get("provenance") or {}).get("baselines") or {}
        row = {
            "outcome": outcome,
            "strategy": (entry.get("provenance") or {}).get("model_probability"),
            "market_implied": baselines.get("market_implied"),
            "station_climatology": baselines.get("station_climatology"),
            "official_forecast": baselines.get("official_forecast"),
            "station_bias_corrected": baselines.get("station_bias_corrected"),
        }
        rows.append(row)
    return {
        "resolved_count": len(rows),
        "brier": {
            name: _brier(rows, name)
            for name in (
                "strategy",
                "market_implied",
                "station_climatology",
                "official_forecast",
                "station_bias_corrected",
            )
        },
        "market_baseline_comparison": _market_baseline_comparison(rows),
    }


def _brier(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    scored = [row for row in rows if row.get(key) is not None]
    if not scored:
        return {"n": 0, "score": None}
    value = sum((float(row[key]) - float(row["outcome"])) ** 2 for row in scored) / len(scored)
    return {"n": len(scored), "score": round(value, 6)}


def _market_baseline_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategy = _brier(rows, "strategy")
    market = _brier(rows, "market_implied")
    if strategy["score"] is None or market["score"] is None:
        return {"available": False, "strategy_brier_lte_market": False}
    return {
        "available": True,
        "strategy_brier_lte_market": float(strategy["score"]) <= float(market["score"]),
        "strategy_brier": strategy["score"],
        "market_brier": market["score"],
    }


def _pessimistic_cost_sensitivity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0.0
    trade_count = 0
    for entry in entries:
        resolution = entry.get("resolution") or {}
        costs = resolution.get("realized_costs") or {}
        multiplier = float(
            (entry.get("provenance") or {}).get("cost_policy", {}).get("pessimistic_cost_multiplier", 2.0)
        )
        gross = float(entry.get("realized_gross_value") or 0.0)
        selected = entry.get("selected_action")
        if selected == BUY_YES_ACTION:
            trade_count += 1
        pessimistic_costs = 0.0
        for field_name, value in costs.items():
            component = float(value or 0.0)
            if field_name in {"fees", "slippage"}:
                component *= multiplier
            pessimistic_costs += component
        total = _money(total + gross - pessimistic_costs)
    return {
        "trade_count": trade_count,
        "pessimistic_total_net_value": total,
        "passes_positive_value": total > 0.0 if trade_count else False,
    }


def _concentration(entries: list[dict[str, Any]]) -> dict[str, Any]:
    city = _share(entries, lambda entry: str((entry.get("provenance") or {}).get("city", "unknown")))
    month = _share(entries, lambda entry: str((entry.get("provenance") or {}).get("local_date", "unknown"))[:7])
    regime = _share(entries, lambda entry: str((entry.get("provenance") or {}).get("regime", "unknown")))
    enough = len(entries) >= 30
    passes = True
    if enough:
        passes = (
            city["max_share"] <= 0.5
            and month["max_share"] <= 0.4
            and regime["max_share"] <= 0.6
        )
    return {
        "resolved_count": len(entries),
        "city": city,
        "month": month,
        "regime": regime,
        "passes_concentration_gate": passes,
        "minimum_samples_for_strict_gate": 30,
    }


def _share(entries: list[dict[str, Any]], key_fn: Any) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = key_fn(entry)
        counts[key] = counts.get(key, 0) + 1
    if not entries:
        return {"counts": {}, "max_share": 0.0, "max_key": None}
    max_key = max(counts, key=lambda key: counts[key])
    return {
        "counts": counts,
        "max_share": round(counts[max_key] / len(entries), 6),
        "max_key": max_key,
    }


def _evidence_gate(
    *,
    resolved_count: int,
    historical_available: int,
    scorecard: dict[str, Any],
    cost_sensitivity: dict[str, Any],
    concentration: dict[str, Any],
    prospective: list[dict[str, Any]],
) -> dict[str, Any]:
    historical_data_permits_150 = historical_available >= 150
    resolved_gate = (not historical_data_permits_150) or resolved_count >= 150
    market_comparison = scorecard["market_baseline_comparison"]
    market_gate = bool(market_comparison.get("available")) and bool(
        market_comparison.get("strategy_brier_lte_market")
    )
    prospective_days = len(
        {
            (entry.get("provenance") or {}).get("local_date")
            for entry in prospective
            if (entry.get("provenance") or {}).get("mode") == "paper_cycle"
        }
    )
    strategy_versions = {
        (entry.get("provenance") or {}).get("strategy_version")
        for entry in prospective
        if (entry.get("provenance") or {}).get("strategy_version")
    }
    strategy_lock_gate = len(strategy_versions) <= 1
    future_review_gate = (
        resolved_gate
        and market_gate
        and cost_sensitivity["passes_positive_value"]
        and concentration["passes_concentration_gate"]
        and prospective_days >= 30
        and strategy_lock_gate
    )
    return {
        "minimum_resolved_city_days": {
            "required_when_historical_data_permits": 150,
            "historical_available": historical_available,
            "historical_data_permits_150": historical_data_permits_150,
            "resolved_city_days": resolved_count,
            "passes": resolved_gate,
        },
        "market_baseline_comparison": {**market_comparison, "passes": market_gate},
        "pessimistic_cost_sensitivity": {
            **cost_sensitivity,
            "passes": bool(cost_sensitivity["passes_positive_value"]),
        },
        "city_month_regime_concentration": {
            "passes": bool(concentration["passes_concentration_gate"]),
            "details": concentration,
        },
        "prospective_incubation": {
            "minimum_days_before_later_real_money_review": 30,
            "observed_prospective_days": prospective_days,
            "passes": prospective_days >= 30,
        },
        "strategy_lock": {
            "strategy_versions_during_prospective_incubation": sorted(strategy_versions),
            "passes": strategy_lock_gate,
        },
        "future_real_money_review_allowed": future_review_gate,
        "real_money_enabled_in_this_wave": False,
    }


def _historical_available(provider: Any | None, resolved: list[dict[str, Any]]) -> int:
    if provider is not None and hasattr(provider, "historical_resolved_count"):
        return int(provider.historical_resolved_count())
    return len({entry.get("economic_event_key") for entry in resolved})


def _city_event_groups(events: Iterable[DailyHighTemperatureEvent]) -> dict[str, list[DailyHighTemperatureEvent]]:
    groups: dict[str, list[DailyHighTemperatureEvent]] = {}
    for event in events:
        groups.setdefault(event.city_event_key, []).append(event)
    return groups


def _decision_id(event: DailyHighTemperatureEvent, decision_timestamp: str, strategy: StrategyConfig) -> str:
    return "weather_edge_" + stable_hash(
        {
            "city_event_key": event.city_event_key,
            "decision_timestamp": decision_timestamp,
            "strategy_id": strategy.strategy_id,
            "strategy_version": strategy.strategy_version,
        }
    )[:20]


def _latest_timestamp(values: list[str]) -> str:
    return max(values, key=parse_time)


def _money(value: float | int | Decimal) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
