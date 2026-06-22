from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import tempfile
from typing import Any

from behavior_lab.core import stable_hash
from behavior_lab.labs.etf_risk import ETFRiskConfig
from behavior_lab.labs.etf_risk.commands import backfill as etf_backfill
from behavior_lab.labs.etf_risk.commands import report as etf_report
from behavior_lab.labs.offerlab_money import evaluate as offerlab_evaluate
from behavior_lab.labs.offerlab_money import report as offerlab_report
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
)
from behavior_lab.labs.weather_edge import backfill as weather_backfill
from behavior_lab.labs.weather_edge import report as weather_report
from behavior_lab.money.accounting import summarize_money_entries
from behavior_lab.money.canary import MoneyCanaryManager, start_fixture_canaries
from behavior_lab.money.integration import _write_offerlab_fixture, fixture_etf_provider
from behavior_lab.money.ledger import MoneyLedger
from behavior_lab.money.storage import MoneyStorage
from behavior_lab.offerlab_pilot import import_pilot


TOURNAMENT_SCHEMA_VERSION = "finance_tournament.v1"
DEFAULT_GENERATED_AT = "2026-07-05T12:00:00+00:00"
ALLOWED_TOP_LEVEL_RESULTS = {
    "SELLER_DECISION_WEDGE",
    "WEATHER_RESEARCH_WEDGE",
    "ETF_RISK_WEDGE",
    "CONTINUE_MULTIPLE_CANARIES",
    "NO_FINANCIAL_WEDGE",
}
ALLOWED_CLASSIFICATIONS = {
    "COMMERCIAL_WEDGE_CANDIDATE",
    "CONTINUE_PAPER_RESEARCH",
    "DATA_STARVED",
    "ECONOMICALLY_WEAK",
    "TECHNICALLY_INVALID",
    "STOP",
}
PRODUCTION_STATE = {
    "seller_mutation": False,
    "exchange_authentication": False,
    "exchange_order_submission": False,
    "brokerage_connection": False,
    "brokerage_order_submission": False,
    "notifications": False,
    "real_financial_action": False,
}
DIMENSION_KEYS = (
    "target_data_quality",
    "source_reliability",
    "baseline_strength",
    "candidate_count",
    "blind_survival",
    "prospective_status",
    "paper_or_shadow_value",
    "calibration",
    "capital_required",
    "maximum_possible_loss",
    "maximum_drawdown",
    "turnover_action_frequency",
    "no_action_frequency",
    "economic_concentration",
    "sensitivity_to_costs",
    "source_api_cost",
    "llm_research_cost",
    "connector_maintenance_cost",
    "human_attention_required",
    "expected_path_to_revenue",
    "regulatory_platform_dependency",
)


def run_financial_tournament(
    *,
    output_dir: str | Path = "reports/finance",
    docs_dir: str | Path = "docs/finance",
    workspace: str | Path | None = None,
    generated_at: str = DEFAULT_GENERATED_AT,
) -> dict[str, Any]:
    """Run the paper-only financial evidence tournament.

    The default tournament uses local deterministic fixtures and immutable
    paper canaries. It deliberately refuses to select a commercial wedge from
    fixture-only evidence.
    """

    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="behavior_lab_tournament_") as tmp:
            return _run_financial_tournament(
                output_dir=Path(output_dir),
                docs_dir=Path(docs_dir),
                workspace=Path(tmp),
                generated_at=generated_at,
            )
    return _run_financial_tournament(
        output_dir=Path(output_dir),
        docs_dir=Path(docs_dir),
        workspace=Path(workspace),
        generated_at=generated_at,
    )


def _run_financial_tournament(
    *,
    output_dir: Path,
    docs_dir: Path,
    workspace: Path,
    generated_at: str,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "offerlab_seller_pilot": _offerlab_evidence(workspace / "offerlab", generated_at),
        "weather_edge": _weather_evidence(workspace / "weather_edge"),
        "etf_risk": _etf_evidence(workspace / "etf_risk"),
    }
    canaries = _canary_reports(workspace / "canaries", generated_at)
    assessments = {
        contract_id: _assess_contract(contract_id, evidence[contract_id], canaries.get(contract_id))
        for contract_id in sorted(evidence)
    }
    decision = _tournament_decision(assessments)
    payload = {
        "schema_version": TOURNAMENT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "fixture_only": True,
        "paper_only": True,
        "real_actions_executed": False,
        "notifications_sent": False,
        "top_level_result": decision["top_level_result"],
        "selected_wedge": decision["selected_wedge"],
        "decision_rationale": decision["decision_rationale"],
        "contracts": assessments,
        "guardrails": {
            "does_not_reward_gross_edge_before_costs": True,
            "paper_pnl_not_realized_pnl": True,
            "one_off_gains_not_sufficient": True,
            "concentration_penalized": True,
            "strategy_churn_penalized": True,
            "unavailable_executable_prices_rejected": True,
            "incomplete_seller_cost_accounting_blocks_eligibility": True,
            "winner_not_selected_from_synthetic_tests": True,
        },
        "production_state": dict(PRODUCTION_STATE),
        "no_real_action_policy": "The tournament produces reports only. It cannot trade, submit seller actions, allocate capital, or notify.",
    }
    payload["tournament_hash"] = stable_hash({key: value for key, value in payload.items() if key != "tournament_hash"})

    json_path = output_dir / "FINANCIAL_TOURNAMENT.json"
    html_path = output_dir / "FINANCIAL_TOURNAMENT.html"
    decision_path = docs_dir / "FINANCIAL_WEDGE_DECISION.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_render_tournament_html(payload), encoding="utf-8")
    decision_path.write_text(_render_decision_markdown(payload), encoding="utf-8")
    return payload


def _offerlab_evidence(root: Path, generated_at: str) -> dict[str, Any]:
    input_dir = root / "seller_input"
    data_root = root / "seller_data"
    money_root = root / "money"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_offerlab_fixture(input_dir)
    import_pilot(input_dir, data_root=data_root, pilot_id="tournament-offerlab")
    evaluation = offerlab_evaluate(
        "tournament-offerlab",
        data_root=data_root,
        money_root=money_root,
        evaluation_timestamp=generated_at,
    )
    report = offerlab_report("tournament-offerlab", data_root=data_root, money_root=money_root)
    entries = MoneyStorage(money_root).ledger.latest_entries()
    economic_records = _economic_records(entries)
    return {
        "lab": "offerlab_seller_pilot",
        "raw_economic_records": economic_records,
        "summary_from_raw_ledger": summarize_money_entries(economic_records),
        "fixture_summary": {
            "decision_count": evaluation["decisions_seen"],
            "eligible_count": report["net_profit_claim_eligible_count"],
            "ineligible_count": report["net_profit_claim_ineligible_count"],
            "explicit_silence_count": report["explicit_silence_count"],
            "unknown_cost_basis_count": evaluation["unknown_cost_basis_count"],
            "historical_policy_claim": report["historical_policy_claim"],
            "causal_lift_claimed": report["causal_profit_lift_claimed"],
            "conservative_shadow_value": report["conservative_shadow_value"],
            "status_counts": report["financial_status_counts"],
            "ineligibility_reasons": report["ineligibility_reasons"],
        },
    }


def _weather_evidence(root: Path) -> dict[str, Any]:
    provider = _tournament_weather_provider()
    backfill = weather_backfill(provider, root, as_of="2026-07-03T00:00:00-04:00")
    report = weather_report(root, provider=provider, as_of="2026-07-04T00:00:00-04:00")
    entries = MoneyStorage(root).ledger.latest_entries()
    economic_records = _economic_records(entries)
    return {
        "lab": "weather_edge",
        "raw_economic_records": economic_records,
        "summary_from_raw_ledger": summarize_money_entries(economic_records),
        "fixture_summary": {
            "decision_count": backfill["city_event_count"],
            "resolved_count": backfill["decisions_resolved"],
            "selected_actions": sorted({entry.selected_action for entry in entries}),
            "scorecard": report["scorecard"],
            "evidence_gate": report["evidence_gate"],
            "pessimistic_cost_sensitivity": report["pessimistic_cost_sensitivity"],
            "concentration": report["concentration"],
        },
    }


def _tournament_weather_provider() -> FixtureWeatherEdgeProvider:
    event_a = _weather_event("nyc-20260701-85-90", TemperatureBracket("85-90", 85.0, 90.0))
    event_b = _weather_event("nyc-20260701-90-95", TemperatureBracket("90-95", 90.0, 95.0))
    return FixtureWeatherEdgeProvider(
        events=[event_a, event_b],
        market_depths=[_weather_depth(event_a.event_id, 0.55), _weather_depth(event_b.event_id, 0.65)],
        weather_snapshots=[_weather_snapshot(event_a.event_id), _weather_snapshot(event_b.event_id)],
        settlements=[_weather_settlement(event_a), _weather_settlement(event_b)],
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
        as_of="2026-07-01T03:00:00-04:00",
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
        as_of="2026-07-01T03:00:00-04:00",
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


def _etf_evidence(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    provider, sessions = fixture_etf_provider(session_count=110)
    ledger_path = root / "ledger.jsonl"
    config = ETFRiskConfig(min_history_trading_days=35, probability_lookback_windows=20)
    backfill = etf_backfill(provider, ledger_path=str(ledger_path), config=config)
    report = etf_report(provider, ledger_path=str(ledger_path), config=config)
    entries = MoneyLedger(str(ledger_path)).latest_entries()
    economic_records = _economic_records(entries)
    primary = report["walk_forward_metrics"]["strategies"].get(config.primary_strategy_id, {})
    return {
        "lab": "etf_risk",
        "raw_economic_records": economic_records,
        "summary_from_raw_ledger": summarize_money_entries(economic_records),
        "fixture_summary": {
            "decision_count": backfill["decision_count"],
            "session_count": len(sessions),
            "strategy_count": len(backfill["strategy_ids"]),
            "primary_strategy": config.primary_strategy_id,
            "primary_metrics": primary,
            "real_money_eligibility": report["real_money_eligibility"],
            "walk_forward_metrics": report["walk_forward_metrics"],
        },
    }


def _canary_reports(root: Path, generated_at: str) -> dict[str, dict[str, Any]]:
    fixture = start_fixture_canaries(root, as_of=generated_at)
    manager = MoneyCanaryManager(root)
    reports = {}
    for item in fixture["canaries"]:
        if item.get("status") != "started":
            continue
        reports[str(item["lab"])] = manager.report(str(item["canary_id"]))
    return reports


def _assess_contract(contract_id: str, evidence: dict[str, Any], canary_report: dict[str, Any] | None) -> dict[str, Any]:
    raw_records = evidence["raw_economic_records"]
    ledger_summary = summarize_money_entries(raw_records)
    fixture = evidence["fixture_summary"]
    dimensions = _dimensions(contract_id, fixture, ledger_summary, raw_records, canary_report)
    classification = _classification(contract_id, dimensions, fixture)
    return {
        "classification": classification,
        "dimensions": dimensions,
        "raw_economic_records": raw_records,
        "economic_reproduction": {
            "summary_from_raw_ledger": ledger_summary,
            "formula": "behavior_lab.money.accounting.summarize_money_entries(raw_economic_records)",
            "cost_basis": "net values are after fees, slippage, shipping, holding costs, return/refund allowance, and research/API cost when known",
            "summary_hash": stable_hash(ledger_summary),
        },
        "fixture_evidence": fixture,
        "canary": _canary_summary(canary_report),
    }


def _dimensions(
    contract_id: str,
    fixture: dict[str, Any],
    ledger_summary: dict[str, Any],
    raw_records: list[dict[str, Any]],
    canary_report: dict[str, Any] | None,
) -> dict[str, Any]:
    common = {
        "candidate_count": len(raw_records),
        "paper_or_shadow_value": _ledger_value(raw_records),
        "capital_required": ledger_summary["capital_at_risk"],
        "maximum_possible_loss": ledger_summary["maximum_possible_loss"],
        "maximum_drawdown": ledger_summary["maximum_drawdown"]["maximum_drawdown"],
        "turnover_action_frequency": ledger_summary["action_frequency"],
        "no_action_frequency": ledger_summary["no_action_frequency"],
        "source_api_cost": _sum_field(raw_records, "research_api_cost"),
        "llm_research_cost": 0.0,
        "connector_maintenance_cost": 0.0,
        "human_attention_required": "manual_review_required_before_any_real_action",
        "prospective_status": _prospective_status(canary_report),
        "blind_survival": "not_promoted_to_blind_validated_winner",
    }
    if contract_id == "offerlab_seller_pilot":
        concentration = _status_concentration(fixture.get("status_counts", {}))
        return {
            **common,
            "target_data_quality": {
                "status": "partial_fixture",
                "eligible_decisions": fixture["eligible_count"],
                "ineligible_decisions": fixture["ineligible_count"],
                "unknown_cost_basis_count": fixture["unknown_cost_basis_count"],
            },
            "source_reliability": {"status": "fixture_only", "ledger_append_only": True},
            "baseline_strength": "seller_documented_historical_action_only",
            "calibration": "not_applicable_seller_shadow_without_randomized_outcomes",
            "economic_concentration": concentration,
            "sensitivity_to_costs": {
                "unknown_material_costs_block_eligibility": fixture["unknown_cost_basis_count"] > 0,
                "gross_edge_before_costs_rewarded": False,
            },
            "expected_path_to_revenue": "merchant_shadow_decision_support_after_private_seller_data_and_prospective_shadow_validation",
            "regulatory_platform_dependency": "eBay/platform policy and seller authorization dependency",
        }
    if contract_id == "weather_edge":
        gate = fixture["evidence_gate"]
        return {
            **common,
            "target_data_quality": {
                "status": "fixture_only",
                "resolved_city_days": gate["minimum_resolved_city_days"]["resolved_city_days"],
                "required_resolved_city_days": gate["minimum_resolved_city_days"]["required_when_historical_data_permits"],
            },
            "source_reliability": {"status": "fixture_provider", "executable_price_required": True},
            "baseline_strength": gate["market_baseline_comparison"],
            "calibration": fixture["scorecard"]["brier"],
            "economic_concentration": fixture["concentration"],
            "sensitivity_to_costs": fixture["pessimistic_cost_sensitivity"],
            "expected_path_to_revenue": "weather_research_or_event_market_paper_edge_after_150_resolved_city_days_and_30_prospective_days",
            "regulatory_platform_dependency": "event-market rules, settlement station semantics, and executable order-book availability",
        }
    primary = fixture["primary_metrics"]
    return {
        **common,
        "target_data_quality": {
            "status": "authorized_fixture",
            "decision_count": fixture["decision_count"],
            "session_count": fixture["session_count"],
        },
        "source_reliability": {"status": "authorized_fixture", "no_broker_order_api": True},
        "baseline_strength": {
            "baseline_strategy_ids": fixture["walk_forward_metrics"]["baseline_strategy_ids"],
            "primary_strategy": fixture["primary_strategy"],
        },
        "calibration": primary.get("calibration", {}),
        "economic_concentration": primary.get("regime_period_concentration", {}),
        "sensitivity_to_costs": {
            "transaction_cost_assumption_bps": fixture["walk_forward_metrics"]["transaction_cost_assumption_bps"],
            "parameter_neighborhood_sensitivity": fixture["walk_forward_metrics"]["parameter_neighborhood_sensitivity"],
        },
        "expected_path_to_revenue": "personal_risk_decision_support_only_after_six_months_prospective_paper_evidence",
        "regulatory_platform_dependency": "authorized market data, broker separation, and personal financial review dependency",
    }


def _classification(contract_id: str, dimensions: dict[str, Any], fixture: dict[str, Any]) -> str:
    if contract_id == "weather_edge":
        if not fixture["evidence_gate"]["future_real_money_review_allowed"]:
            return "DATA_STARVED"
    if contract_id == "offerlab_seller_pilot":
        if dimensions["target_data_quality"]["eligible_decisions"] == 0:
            return "DATA_STARVED"
        return "CONTINUE_PAPER_RESEARCH"
    if contract_id == "etf_risk":
        if dimensions["prospective_status"]["final_evidence_available"] is False:
            return "CONTINUE_PAPER_RESEARCH"
    return "CONTINUE_PAPER_RESEARCH"


def _tournament_decision(assessments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    classifications = {key: value["classification"] for key, value in assessments.items()}
    if all(value in {"DATA_STARVED", "CONTINUE_PAPER_RESEARCH"} for value in classifications.values()):
        return {
            "top_level_result": "CONTINUE_MULTIPLE_CANARIES",
            "selected_wedge": None,
            "decision_rationale": [
                "All current evidence is fixture or immature paper evidence.",
                "No contract has enough prospective canary evidence for commercial selection.",
                "No winner is selected from synthetic tests.",
            ],
        }
    return {
        "top_level_result": "NO_FINANCIAL_WEDGE",
        "selected_wedge": None,
        "decision_rationale": ["No contract satisfies the commercial wedge gate."],
    }


def _canary_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {"available": False, "final_evidence_available": False, "reason": "no_canary_report"}
    metrics = report["metrics"]
    return {
        "available": True,
        "canary_id": report["canary_id"],
        "minimum_duration_days": report["protocol"]["minimum_duration_days"],
        "snapshot_count": metrics["snapshot_count"],
        "distinct_observation_periods": metrics["distinct_observation_periods"],
        "consecutive_observation_periods": metrics["consecutive_observation_periods"],
        "minimum_duration_elapsed": metrics["minimum_duration_elapsed"],
        "final_evidence_available": report["final_evidence_report"]["available"],
        "real_money_allowed": report["final_evidence_report"]["real_money_allowed"],
    }


def _prospective_status(report: dict[str, Any] | None) -> dict[str, Any]:
    summary = _canary_summary(report)
    if not summary["available"]:
        return summary
    if summary["final_evidence_available"]:
        status = "canary_duration_gate_passed_paper_only"
    else:
        status = "prospective_canary_active_or_immature"
    return {**summary, "status": status}


def _economic_records(entries: list[Any]) -> list[dict[str, Any]]:
    records = []
    for entry in entries:
        payload = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        records.append(
            {
                key: _sanitize_for_report(value)
                for key, value in payload.items()
                if key not in {"created_at", "supersedes_entry_hash", "correction_reason"}
            }
        )
    return sorted(records, key=lambda item: (str(item.get("decision_timestamp")), str(item.get("decision_id"))))


def _ledger_value(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0.0
    realized = 0.0
    paper_expected = 0.0
    for record in records:
        if record.get("realized_net_value") is not None:
            realized += float(record["realized_net_value"])
            total += float(record["realized_net_value"])
        else:
            value = float(record.get("conservative_expected_net_value") or 0.0)
            paper_expected += value
            total += value
    return {
        "total_value_used_for_ranking": round(total, 2),
        "realized_paper_value": round(realized, 2),
        "unresolved_conservative_expected_value": round(paper_expected, 2),
        "presented_as_realized_pnl": False,
    }


def _sum_field(records: list[dict[str, Any]], field_name: str) -> float:
    return round(sum(float(record.get(field_name) or 0.0) for record in records), 2)


def _status_concentration(counts: dict[str, Any]) -> dict[str, Any]:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {"counts": counts, "max_share": 0.0, "max_key": None}
    max_key = max(counts, key=lambda key: int(counts[key]))
    return {"counts": counts, "max_share": round(int(counts[max_key]) / total, 6), "max_key": max_key}


def _sanitize_for_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_report(item) for key, item in value.items() if not _looks_like_path_key(str(key))}
    if isinstance(value, list):
        return [_sanitize_for_report(item) for item in value]
    if isinstance(value, str):
        if "\\" in value or ":/" in value or value.startswith("/"):
            return "[local_path_omitted]"
    return value


def _looks_like_path_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_path") or lowered in {"path", "money_root", "money_ledger", "contracts_dir", "seller_pilot_ledger"}


def _render_tournament_html(payload: dict[str, Any]) -> str:
    rows = []
    for contract_id, assessment in payload["contracts"].items():
        dimensions = assessment["dimensions"]
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(contract_id),
                html.escape(assessment["classification"]),
                html.escape(str(dimensions["candidate_count"])),
                html.escape(json.dumps(dimensions["paper_or_shadow_value"], sort_keys=True)),
                html.escape(str(dimensions["prospective_status"]["final_evidence_available"])),
            )
        )
    guardrails = "\n".join(
        "<li>{}: {}</li>".format(html.escape(key.replace("_", " ")), "PASS" if value else "FAIL")
        for key, value in sorted(payload["guardrails"].items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Financial Evidence Tournament</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #cbd5df; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ background: #f4f6f8; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>Financial Evidence Tournament</h1>
  <p>Schema: <code>{html.escape(payload["schema_version"])}</code></p>
  <p>Generated at: <code>{html.escape(payload["generated_at"])}</code></p>
  <p>Top-level result: <strong>{html.escape(payload["top_level_result"])}</strong></p>
  <p>Selected wedge: <strong>{html.escape(str(payload["selected_wedge"]))}</strong></p>
  <h2>Contracts</h2>
  <table>
    <thead><tr><th>Contract</th><th>Classification</th><th>Candidates</th><th>Value</th><th>Final Evidence</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Guardrails</h2>
  <ul>{guardrails}</ul>
  <h2>Rationale</h2>
  <pre>{html.escape(json.dumps(payload["decision_rationale"], indent=2))}</pre>
</body>
</html>
"""


def _render_decision_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Financial Wedge Decision",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Top-level result: `{payload['top_level_result']}`",
        f"Selected wedge: `{payload['selected_wedge']}`",
        "",
        "## Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["decision_rationale"])
    lines.extend(["", "## Contract Classifications", ""])
    for contract_id, assessment in payload["contracts"].items():
        lines.append(f"- `{contract_id}`: `{assessment['classification']}`")
    lines.extend(
        [
            "",
            "## Capital Policy",
            "",
            "No real capital is authorized. All entries are paper or shadow decisions, and the production-state flags are false.",
            "",
            "## 90-Day Plan",
            "",
            "- Continue the three prospective canaries without strategy mutation.",
            "- Prioritize private seller-pilot data only if the seller readiness gate passes.",
            "- Keep Weather Edge and ETF Risk as paper research until their duration and evidence gates mature.",
            "- Re-run this tournament only from append-only ledger and canary records.",
            "",
            "## Kill Criteria",
            "",
            "- Stop a lab if source health fails, material costs are unknown, or paper value remains economically weak after prospective validation.",
            "- Do not promote any fixture-only or synthetic-only result into a commercial wedge.",
            "",
            "## Output Artifacts",
            "",
            "- `reports/finance/FINANCIAL_TOURNAMENT.json`",
            "- `reports/finance/FINANCIAL_TOURNAMENT.html`",
        ]
    )
    return "\n".join(lines) + "\n"
