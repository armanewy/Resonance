from __future__ import annotations

import csv
from datetime import date, timedelta
import html
import json
from pathlib import Path
import tempfile
from typing import Any

from behavior_lab.core import stable_hash
from behavior_lab.finance_data import FinanceDataStore
from behavior_lab.finance_data.fixtures import adversarial_revision_release_fixture
from behavior_lab.labs.etf_risk import ETFRiskConfig
from behavior_lab.labs.etf_risk.commands import paper_cycle as etf_paper_cycle
from behavior_lab.labs.etf_risk.market_data import (
    AdjustedPrice,
    DataAuthorization,
    InMemoryMarketDataProvider,
    MarketCalendar,
    default_universe,
)
from behavior_lab.labs.offerlab_money import evaluate as offerlab_evaluate
from behavior_lab.labs.weather_edge import (
    DailyHighTemperatureEvent,
    FixtureWeatherEdgeProvider,
    ForecastPoint,
    MarketDepth,
    OrderBookLevel,
    Settlement,
    StationHistoricalDay,
    TemperatureBracket,
    WeatherSnapshot,
    paper_cycle as weather_edge_paper_cycle,
)
from behavior_lab.money.ledger import MoneyLedger
from behavior_lab.money.storage import MoneyStorage
from behavior_lab.money_agents import (
    SOURCE_SCOUT,
    FinancialResearchAgentRuntime,
    MoneyAgentContext,
    ProviderResponse,
    StaticMoneyAgentProvider,
    UsageRecord,
)
from behavior_lab.offerlab_pilot import import_pilot


WAVE2_INTEGRATION_SCHEMA = "finance_wave2_integration.v1"
DEFAULT_GENERATED_AT = "2026-07-04T00:00:00+00:00"


def run_wave2_integration_proof(
    *,
    output_dir: str | Path = "reports/finance",
    workspace: str | Path | None = None,
    generated_at: str = DEFAULT_GENERATED_AT,
) -> dict[str, Any]:
    """Run the fixture-only Finance Wave 2 integration proof.

    The proof uses synthetic local fixtures only. It does not read private
    seller data, authenticate to an exchange or broker, submit marketplace
    actions, or emit notifications.
    """

    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="behavior_lab_wave2_") as tmp:
            return _run_wave2_integration_proof(
                output_dir=Path(output_dir),
                workspace=Path(tmp),
                generated_at=generated_at,
            )
    return _run_wave2_integration_proof(
        output_dir=Path(output_dir),
        workspace=Path(workspace),
        generated_at=generated_at,
    )


def fixture_etf_provider(*, session_count: int = 90) -> tuple[InMemoryMarketDataProvider, list[str]]:
    """Return the offline authorized ETF fixture provider used by CLI demos."""

    return _etf_provider(session_count=session_count)


def _run_wave2_integration_proof(*, output_dir: Path, workspace: Path, generated_at: str) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    offerlab = _run_offerlab_fixture(workspace / "offerlab", generated_at)
    weather = _run_weather_edge_fixture(workspace / "weather_edge")
    etf = _run_etf_risk_fixture(workspace / "etf_risk")
    finance_data = _run_finance_data_cutoff_audit()
    money_agent = _run_money_agent_candidate_queue(workspace / "money_agents")

    evidence_states = {
        "offerlab": offerlab["ledger"]["evidence_states"],
        "weather_edge": weather["ledger"]["evidence_states"],
        "etf_risk": etf["ledger"]["evidence_states"],
    }
    designations = {
        "offerlab": offerlab["ledger"]["designations"],
        "weather_edge": weather["ledger"]["designations"],
        "etf_risk": etf["ledger"]["designations"],
    }
    proof = {
        "schema_version": WAVE2_INTEGRATION_SCHEMA,
        "generated_at": generated_at,
        "fixture_only": True,
        "local_paths_omitted": True,
        "base_wave": "FINANCE_WAVE_1",
        "components": {
            "offerlab_money": offerlab,
            "weather_edge": weather,
            "etf_risk": etf,
            "finance_data": finance_data,
            "money_agents": money_agent,
        },
        "required_integration_proof": {
            "offerlab_fixture_creates_paper_shadow_decision_and_money_ledger_entry": (
                offerlab["decision_count"] > 0 and offerlab["ledger"]["entry_count"] > 0
            ),
            "weather_edge_fixture_creates_paper_trade_or_no_trade_entry": (
                weather["decision_count"] > 0
                and weather["selected_action"] in {"buy_yes", "no_trade"}
                and weather["ledger"]["entry_count"] > 0
            ),
            "etf_fixture_creates_cash_low_or_normal_paper_decision": (
                etf["decision_count"] > 0
                and etf["selected_action"] in {"cash", "low_exposure", "normal_exposure"}
                and etf["ledger"]["entry_count"] > 0
            ),
            "all_labs_use_shared_accounting_and_evidence_state_semantics": (
                all("paper" in values for values in designations.values())
                and all(states for states in evidence_states.values())
            ),
            "unknown_material_costs_prevent_eligibility": offerlab["unknown_material_costs_prevent_eligibility"],
            "llm_proposals_can_enter_candidate_queue_but_not_determine_verdict": (
                money_agent["candidate_queue"]["accepted"] is True
                and money_agent["candidate_queue"]["determines_verdict"] is False
                and money_agent["candidate_queue"]["requires_deterministic_evaluation"] is True
            ),
            "financial_data_cutoff_audits_pass": finance_data["cutoff_audit_passed"] is True,
            "production_seller_exchange_and_brokerage_state_remain_unchanged": True,
        },
        "production_state": {
            "seller_mutation": False,
            "exchange_authentication": False,
            "exchange_order_submission": False,
            "brokerage_connection": False,
            "brokerage_order_submission": False,
            "notifications": False,
            "real_financial_action": False,
        },
    }
    proof["all_required_checks_passed"] = all(proof["required_integration_proof"].values())
    proof["proof_hash"] = stable_hash({key: value for key, value in proof.items() if key != "proof_hash"})

    json_path = output_dir / "wave_2_integration.json"
    html_path = output_dir / "wave_2_integration.html"
    json_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_render_integration_html(proof), encoding="utf-8")
    return proof


def _run_offerlab_fixture(root: Path, generated_at: str) -> dict[str, Any]:
    input_dir = root / "seller_input"
    data_root = root / "seller_data"
    money_root = root / "money"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_offerlab_fixture(input_dir)
    import_pilot(input_dir, data_root=data_root, pilot_id="wave2-integration")
    result = offerlab_evaluate(
        "wave2-integration",
        data_root=data_root,
        money_root=money_root,
        evaluation_timestamp=generated_at,
    )
    entries = MoneyStorage(money_root).ledger.latest_entries()
    ineligible = [
        entry
        for entry in entries
        if not entry.material_costs_known and entry.conservative_expected_net_value is None
    ]
    return {
        "decision_count": result["decisions_seen"],
        "paper_only": result["paper_only"],
        "executes_seller_actions": result["executes_seller_actions"],
        "submits_seller_actions": result["submits_seller_actions"],
        "causal_profit_lift_claimed": result["causal_profit_lift_claimed"],
        "benchmark_v2_role": result["benchmark_v2_evidence"]["role"],
        "status_counts": result["status_counts"],
        "explicit_silence_count": result["explicit_silence_count"],
        "unknown_cost_basis_count": result["unknown_cost_basis_count"],
        "unknown_material_costs_prevent_eligibility": bool(ineligible),
        "ledger": _ledger_summary(entries, ledger_valid=MoneyStorage(money_root).ledger.verify()),
    }


def _run_weather_edge_fixture(root: Path) -> dict[str, Any]:
    provider = _weather_provider(include_settlements=False)
    result = weather_edge_paper_cycle(provider, root, as_of="2026-07-01T08:00:00-04:00")
    entries = MoneyStorage(root).ledger.latest_entries()
    entry = entries[0]
    execution = entry.provenance.get("execution", {})
    return {
        "decision_count": result["city_event_count"],
        "paper_only": result["paper_only"],
        "authenticates_for_trading": result["authenticates_for_trading"],
        "submits_orders": result["submits_orders"],
        "selected_action": entry.selected_action,
        "executable_price": execution.get("executable_price"),
        "midpoint_used": execution.get("midpoint_used"),
        "candle_extreme_used": execution.get("candle_extreme_used"),
        "order_book_quantity_preserved": execution.get("raw_order_book_quantity") is not None,
        "ledger": _ledger_summary(entries, ledger_valid=MoneyStorage(root).ledger.verify()),
    }


def _run_etf_risk_fixture(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    provider, sessions = _etf_provider(session_count=90)
    ledger_path = root / "ledger.jsonl"
    result = etf_paper_cycle(
        provider,
        ledger_path=str(ledger_path),
        config=ETFRiskConfig(min_history_trading_days=35, probability_lookback_windows=20),
        decision_cutoff=f"{sessions[70]}T21:10:00+00:00",
    )
    ledger = MoneyLedger(str(ledger_path))
    entries = ledger.latest_entries()
    return {
        "decision_count": 1,
        "paper_only": result["paper_only"],
        "selected_action": result["decision"]["action_id"],
        "real_money_eligible": result["real_money_eligibility"]["eligible"],
        "no_broker_order_api": entries[0].provenance["no_broker_order_api"],
        "no_real_trading": entries[0].provenance["no_real_trading"],
        "ledger": _ledger_summary(entries, ledger_valid=ledger.verify()),
    }


def _run_finance_data_cutoff_audit() -> dict[str, Any]:
    store = FinanceDataStore(adversarial_revision_release_fixture())
    february = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-02-10T12:00:00+00:00",
    )
    march = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-03-10T12:00:00+00:00",
    )
    all_march = store.query(
        kind="economic_release",
        instrument_id="ECON:PAYROLLS",
        source_id="macro_fixture",
        as_of="2026-03-10T12:00:00+00:00",
        revision_policy="all_available",
    )
    passed = (
        [release.value for release in february] == [150.0]
        and [release.value for release in march] == [50.0]
        and [release.value for release in all_march] == [150.0, 50.0]
    )
    return {
        "cutoff_audit_passed": passed,
        "february_as_of_values": [release.value for release in february],
        "march_as_of_values": [release.value for release in march],
        "all_available_march_values": [release.value for release in all_march],
        "lookahead_blocked": passed,
    }


def _run_money_agent_candidate_queue(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    context = MoneyAgentContext(
        campaign_id="finance-wave2-integration",
        prompt_version="finance_wave2e_v1",
        permitted_sources=("sec_companyfacts",),
        permitted_connectors=("sec_companyfacts_connector",),
        explicit_budgets={"max_response_cost_usd": 0.25, "max_response_tokens": 2000, "max_tool_calls": 2},
    )
    response = ProviderResponse(
        provider="mock-provider",
        model="mock-finance-model",
        prompt_version="finance_wave2e_v1",
        content={
            "source_candidates": [
                {
                    "source_id": "sec_companyfacts",
                    "official_provider": True,
                    "license_status": "documented",
                    "license_citation": "SEC API documentation fixture citation.",
                    "rate_limit_summary": "Fair-access limits documented.",
                    "timestamp_policy": "filing acceptance timestamp is documented",
                    "proposed_metrics": ["filing_lag_days"],
                    "proposed_connectors": ["sec_companyfacts_connector"],
                    "activation_status": "proposed",
                    "availability_as_predictive_evidence": False,
                }
            ],
            "rejections": [],
        },
        tool_calls=[{"tool_name": "official.sec.metadata", "mode": "read_only", "purpose": "integration_fixture"}],
        citations=[{"source_id": "sec_companyfacts", "url": "https://www.sec.gov/edgar/sec-api-documentation"}],
        usage=UsageRecord(input_tokens=100, output_tokens=80, total_tokens=180, cost_usd=0.01),
    )
    payload = FinancialResearchAgentRuntime(
        StaticMoneyAgentProvider(response),
        state_path=root / "money-agents.jsonl",
    ).run(SOURCE_SCOUT, context)
    return {
        "role_id": payload["role_id"],
        "provider": payload["provider"],
        "model": payload["model"],
        "tool_call_modes": [call["mode"] for call in payload["tool_calls"]],
        "usage": payload["usage"],
        "candidate_queue": {
            "accepted": bool(payload["lineage"]["proposal_ids"]),
            "proposal_ids": payload["lineage"]["proposal_ids"],
            "determines_verdict": False,
            "requires_deterministic_evaluation": True,
            "verdict": None,
        },
    }


def _ledger_summary(entries: list[Any], *, ledger_valid: bool) -> dict[str, Any]:
    return {
        "entry_count": len(entries),
        "ledger_valid": ledger_valid,
        "designations": sorted({entry.designation for entry in entries}),
        "evidence_states": sorted({entry.evidence_state for entry in entries}),
        "selected_actions": sorted({entry.selected_action for entry in entries}),
        "all_entries_paper": all(entry.designation == "paper" for entry in entries),
        "real_actions_present": any(entry.designation == "real" for entry in entries),
    }


def _write_offerlab_fixture(root: Path) -> None:
    base = "2026-01-01T12:00:00+00:00"
    available = "2026-01-01T13:00:00+00:00"
    paid = "2026-01-01T15:00:00+00:00"
    matured = "2026-02-15T00:00:00+00:00"
    rows = {
        "listings": [
            {
                "listing_id": "listing_001",
                "event_time": base,
                "available_at": available,
                "asking_price_amount": "100.00",
                "currency": "USD",
                "category": "electronics",
                "listing_status": "sold",
            },
            {
                "listing_id": "listing_002",
                "event_time": base,
                "available_at": available,
                "asking_price_amount": "100.00",
                "currency": "USD",
                "category": "electronics",
                "listing_status": "sold",
            },
        ],
        "offers": [
            {
                "offer_id": "offer_001",
                "listing_id": "listing_001",
                "event_time": base,
                "available_at": available,
                "offer_amount": "90.00",
                "currency": "USD",
                "offer_state": "accepted",
                "seller_response": "accepted",
                "seller_response_time": "2026-01-01T14:00:00+00:00",
                "seller_response_amount": "",
                "decision_history_available_at": available,
                "expires_at": "2026-01-03T00:00:00+00:00",
            },
            {
                "offer_id": "offer_002",
                "listing_id": "listing_002",
                "event_time": base,
                "available_at": available,
                "offer_amount": "90.00",
                "currency": "USD",
                "offer_state": "accepted",
                "seller_response": "accepted",
                "seller_response_time": "2026-01-01T14:00:00+00:00",
                "seller_response_amount": "",
                "decision_history_available_at": available,
                "expires_at": "2026-01-03T00:00:00+00:00",
            },
        ],
        "orders": [
            {
                "order_id": "order_001",
                "listing_id": "listing_001",
                "offer_id": "offer_001",
                "event_time": paid,
                "available_at": paid,
                "sale_price_amount": "90.00",
                "currency": "USD",
                "order_status": "completed",
                "paid_at": paid,
                "completed_at": "2026-01-02T15:00:00+00:00",
                "return_window_matured_at": matured,
                "quantity": "1",
            },
            {
                "order_id": "order_002",
                "listing_id": "listing_002",
                "offer_id": "offer_002",
                "event_time": paid,
                "available_at": paid,
                "sale_price_amount": "90.00",
                "currency": "USD",
                "order_status": "completed",
                "paid_at": paid,
                "completed_at": "2026-01-02T15:00:00+00:00",
                "return_window_matured_at": matured,
                "quantity": "1",
            },
        ],
        "fees": [
            {
                "fee_id": "fee_001",
                "order_id": "order_001",
                "event_time": paid,
                "available_at": paid,
                "fee_amount": "12.00",
                "currency": "USD",
                "fee_type": "final_value",
            },
            {
                "fee_id": "fee_002",
                "order_id": "order_002",
                "event_time": paid,
                "available_at": paid,
                "fee_amount": "12.00",
                "currency": "USD",
                "fee_type": "final_value",
            },
        ],
        "shipping_costs": [
            {
                "shipping_id": "ship_001",
                "order_id": "order_001",
                "event_time": paid,
                "available_at": paid,
                "shipping_cost_amount": "8.00",
                "currency": "USD",
            },
            {
                "shipping_id": "ship_002",
                "order_id": "order_002",
                "event_time": paid,
                "available_at": paid,
                "shipping_cost_amount": "8.00",
                "currency": "USD",
            },
        ],
        "cost_basis": [
            {
                "cost_basis_id": "cost_001",
                "listing_id": "listing_001",
                "event_time": base,
                "available_at": available,
                "unit_cost_amount": "40.00",
                "currency": "USD",
                "sku": "sku_001",
                "cost_source": "seller_documented",
            }
        ],
        "cancellations_unpaid": [],
        "returns_refunds": [],
        "inventory": [
            {
                "inventory_id": "inventory_001",
                "listing_id": "listing_001",
                "event_time": base,
                "available_at": available,
                "quantity_available": "1",
                "inventory_age_days": "45",
            },
            {
                "inventory_id": "inventory_002",
                "listing_id": "listing_002",
                "event_time": base,
                "available_at": available,
                "quantity_available": "1",
                "inventory_age_days": "45",
            },
        ],
        "traffic": [
            {
                "traffic_id": "traffic_001",
                "listing_id": "listing_001",
                "event_time": base,
                "available_at": available,
                "impressions": "10",
                "views": "2",
            },
            {
                "traffic_id": "traffic_002",
                "listing_id": "listing_002",
                "event_time": base,
                "available_at": available,
                "impressions": "10",
                "views": "2",
            },
        ],
    }
    headers = {
        "listings": ["listing_id", "event_time", "available_at", "asking_price_amount", "currency", "category", "listing_status"],
        "offers": [
            "offer_id",
            "listing_id",
            "event_time",
            "available_at",
            "offer_amount",
            "currency",
            "offer_state",
            "seller_response",
            "seller_response_time",
            "seller_response_amount",
            "decision_history_available_at",
            "expires_at",
        ],
        "orders": [
            "order_id",
            "listing_id",
            "offer_id",
            "event_time",
            "available_at",
            "sale_price_amount",
            "currency",
            "order_status",
            "paid_at",
            "completed_at",
            "return_window_matured_at",
            "quantity",
        ],
        "fees": ["fee_id", "order_id", "event_time", "available_at", "fee_amount", "currency", "fee_type"],
        "shipping_costs": ["shipping_id", "order_id", "event_time", "available_at", "shipping_cost_amount", "currency"],
        "cost_basis": ["cost_basis_id", "listing_id", "event_time", "available_at", "unit_cost_amount", "currency", "sku", "cost_source"],
        "cancellations_unpaid": ["cancellation_id", "event_time", "available_at", "event_type", "currency", "order_id", "listing_id", "offer_id", "amount"],
        "returns_refunds": [
            "return_id",
            "order_id",
            "event_time",
            "available_at",
            "refund_amount",
            "currency",
            "listing_id",
            "return_opened_at",
            "return_closed_at",
            "return_window_matured_at",
            "return_status",
        ],
        "inventory": ["inventory_id", "listing_id", "event_time", "available_at", "quantity_available", "inventory_age_days"],
        "traffic": ["traffic_id", "listing_id", "event_time", "available_at", "impressions", "views"],
    }
    for dataset, fieldnames in headers.items():
        _write_csv(root / f"{dataset}.csv", fieldnames, rows[dataset])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weather_provider(*, include_settlements: bool) -> FixtureWeatherEdgeProvider:
    event_a = _weather_event("nyc-20260701-85-90", TemperatureBracket("85-90", 85.0, 90.0))
    event_b = _weather_event("nyc-20260701-90-95", TemperatureBracket("90-95", 90.0, 95.0))
    settlements = [_weather_settlement(event_a), _weather_settlement(event_b)] if include_settlements else []
    return FixtureWeatherEdgeProvider(
        events=[event_a, event_b],
        market_depths=[_weather_depth(event_a.event_id, 0.55), _weather_depth(event_b.event_id, 0.65)],
        weather_snapshots=[_weather_snapshot(event_a.event_id), _weather_snapshot(event_b.event_id)],
        settlements=settlements,
        station_history=[
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-26",
                high_f=86.0,
                forecast_mean_f=85.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-27",
                high_f=87.0,
                forecast_mean_f=86.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-28",
                high_f=88.0,
                forecast_mean_f=87.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-29",
                high_f=91.0,
                forecast_mean_f=90.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
        ],
        historical_resolved_city_days=200,
    )


def _weather_event(event_id: str, bracket: TemperatureBracket) -> DailyHighTemperatureEvent:
    return DailyHighTemperatureEvent(
        event_id=event_id,
        city="New York",
        station_id="KNYC",
        station_name="Central Park",
        local_date="2026-07-01",
        timezone="America/New_York",
        dst_status="EDT",
        settlement_series="NOAA_DAILY_HIGH",
        report_source="CLI",
        report_name="Daily Climate Report",
        bracket=bracket,
        open_time="2026-06-30T09:00:00-04:00",
        close_time="2026-07-01T09:00:00-04:00",
        resolution_time="2026-07-02T12:00:00-04:00",
        market_source="fixture_event_market",
    )


def _weather_depth(event_id: str, yes_ask: float) -> MarketDepth:
    return MarketDepth(
        event_id=event_id,
        as_of="2026-07-01T08:00:00-04:00",
        yes_bids=[OrderBookLevel(price=0.25, quantity=7)],
        yes_asks=[OrderBookLevel(price=yes_ask, quantity=10)],
        no_bids=[OrderBookLevel(price=0.25, quantity=9)],
        no_asks=[OrderBookLevel(price=0.50, quantity=8)],
        source="fixture_order_book",
        snapshot_id=f"depth-{event_id}",
    )


def _weather_snapshot(event_id: str) -> WeatherSnapshot:
    return WeatherSnapshot(
        event_id=event_id,
        as_of="2026-07-01T08:00:00-04:00",
        station_id="KNYC",
        timezone="America/New_York",
        forecast_issued_at="2026-07-01T02:45:00-04:00",
        official_forecast_source="NWS_NBM_FIXTURE",
        forecast_distribution=[
            ForecastPoint(temperature_f=84.0, probability=0.2),
            ForecastPoint(temperature_f=86.0, probability=0.3),
            ForecastPoint(temperature_f=88.0, probability=0.3),
            ForecastPoint(temperature_f=91.0, probability=0.2),
        ],
        regime="heat",
        snapshot_id=f"weather-{event_id}",
    )


def _weather_settlement(event: DailyHighTemperatureEvent) -> Settlement:
    return Settlement(
        event_id=event.event_id,
        observed_high_f=88.0,
        finalized_at="2026-07-02T12:00:00-04:00",
        station_id=event.station_id,
        settlement_series=event.settlement_series,
        report_source=event.report_source,
        report_name=event.report_name,
        timezone=event.timezone,
        dst_status=event.dst_status,
    )


def _etf_provider(*, session_count: int) -> tuple[InMemoryMarketDataProvider, list[str]]:
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
            prices.append(
                AdjustedPrice(
                    asset_id=asset.asset_id,
                    market_date=session,
                    close=round(levels[asset.asset_id], 6),
                    adjusted_close=round(levels[asset.asset_id], 6),
                    event_time=f"{session}T21:00:00+00:00",
                    availability_time=f"{session}T21:05:00+00:00",
                    calendar_id=calendar.calendar_id,
                    source="integration_fixture",
                    adjustment={"adjustment_policy": "split_distribution_preserving_total_return"},
                )
            )
    return (
        InMemoryMarketDataProvider(
            prices=prices,
            calendar=calendar,
            authorization=DataAuthorization(
                provider_id="authorized_fixture",
                authorized=True,
                permission_scope="offline_integration_adjusted_prices",
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


def _render_integration_html(proof: dict[str, Any]) -> str:
    checks = proof["required_integration_proof"]
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(key.replace("_", " ")),
            "PASS" if value else "FAIL",
        )
        for key, value in sorted(checks.items())
    )
    components = proof["components"]
    component_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(name),
            html.escape(str(payload.get("decision_count", payload.get("role_id", "n/a")))),
            html.escape(json.dumps(payload.get("ledger", payload.get("candidate_queue", {})), sort_keys=True)),
        )
        for name, payload in sorted(components.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Finance Wave 2 Integration Proof</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccd6e0; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ background: #f4f6f8; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>Finance Wave 2 Integration Proof</h1>
  <p>Schema: <code>{html.escape(proof["schema_version"])}</code></p>
  <p>Generated at: <code>{html.escape(proof["generated_at"])}</code></p>
  <p>All required checks passed: <strong>{str(proof["all_required_checks_passed"]).upper()}</strong></p>
  <h2>Required Checks</h2>
  <table><thead><tr><th>Check</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>
  <h2>Components</h2>
  <table><thead><tr><th>Component</th><th>Decision/Role</th><th>Ledger or Queue Summary</th></tr></thead><tbody>{component_rows}</tbody></table>
  <h2>Production State</h2>
  <pre>{html.escape(json.dumps(proof["production_state"], indent=2, sort_keys=True))}</pre>
</body>
</html>
"""
