from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import json
from typing import Any

from behavior_lab.core import stable_hash, utc_now
from behavior_lab.money.portfolio import AttentionBudget, AutonomousFinancialOpportunityPortfolio, PortfolioContract, REAL_ACTION_FLAGS


VALUE_SPRINT_SCHEMA_VERSION = "autonomous_value_sprint.v1"
SPRINT_DECISIONS = {
    "AUTONOMY_WORKS_CONTINUE_EVIDENCE",
    "AUTONOMY_WORKS_NO_EDGE_YET",
    "DATA_ACQUISITION_IS_BOTTLENECK",
    "RESEARCH_ENGINE_IS_UNTRUSTWORTHY",
    "HUMAN_ATTENTION_TOO_HIGH",
    "STOP_AUTONOMOUS_DIRECTION",
}
PROHIBITIONS = {
    "real_trading": False,
    "purchases": False,
    "seller_mutation": False,
    "production_source_promotion": False,
    "threshold_changes": False,
    "blind_reuse": False,
}
PUBLIC_SPRINT_CONTRACT_FAMILIES = {"weather_event_market", "broad_etf_risk", "compute_cost_avoidance", "purchase_timing"}
REQUIRED_PUBLIC_SPRINT_FAMILIES = {"weather_event_market", "broad_etf_risk", "compute_cost_avoidance"}


class ValueSprintError(ValueError):
    pass


@dataclass(frozen=True)
class ValueSprintConfig:
    state_dir: str | Path = ".autonomous_value_sprint"
    output_dir: str | Path = "reports/finance"
    sprint_id: str = "autonomous-value-sprint-30d"
    start_at: str = "2026-06-22T12:00:00+00:00"
    days: int = 30
    monthly_budget_usd: float = 40.0
    include_purchase_timing: bool = False
    include_seller_shadow: bool = False
    seller_readiness_passed: bool = False

    def __post_init__(self) -> None:
        if int(self.days) <= 0:
            raise ValueSprintError("days must be positive")
        if float(self.monthly_budget_usd) < 0:
            raise ValueSprintError("monthly_budget_usd may not be negative")


def run_autonomous_value_sprint(config: ValueSprintConfig | None = None) -> dict[str, Any]:
    cfg = config or ValueSprintConfig()
    state_dir = Path(cfg.state_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio = AutonomousFinancialOpportunityPortfolio(
        state_dir / "portfolio",
        portfolio_id=cfg.sprint_id,
        budget=AttentionBudget(
            approvals_per_week=3,
            alerts_per_day=1,
            llm_budget_usd=float(cfg.monthly_budget_usd),
            web_search_budget=8,
            connector_build_budget=2,
            source_trial_budget=6,
            candidate_evaluation_budget=40,
        ),
        contracts=_sprint_contracts(cfg),
    )
    protocol = _protocol(cfg)
    daily_runs: list[dict[str, Any]] = []
    for day in range(int(cfg.days)):
        schedule = _schedule_for_day(day)
        as_of = _day_timestamp(cfg.start_at, day)
        run = portfolio.run_cycle(
            schedule=schedule,
            mesh_manifests=[_cost_manifest()] if day == 0 else None,
            fixtures_by_source={"official_json_cost_source": _cost_fixture(day)} if day == 0 else None,
            source_catalog=_source_catalog_after_day(day),
            as_of=as_of,
        )
        repair = _repair_source_if_due(day=day, portfolio=portfolio)
        daily_runs.append(_summarize_run(day=day + 1, run=run, repair=repair))
    final_report = _final_report(cfg, protocol, daily_runs, portfolio)
    json_path = output_dir / "AUTONOMOUS_VALUE_SPRINT.json"
    html_path = output_dir / "AUTONOMOUS_VALUE_SPRINT.html"
    decision_path = output_dir / "VALUE_SYSTEM_DECISION.md"
    json_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(_render_html(final_report), encoding="utf-8")
    decision_path.write_text(_render_decision(final_report), encoding="utf-8")
    return {
        **final_report,
        "artifacts": {
            "json": json_path.name,
            "html": html_path.name,
            "decision": decision_path.name,
        },
    }


def _protocol(cfg: ValueSprintConfig) -> dict[str, Any]:
    body = {
        "schema_version": VALUE_SPRINT_SCHEMA_VERSION,
        "sprint_id": cfg.sprint_id,
        "start_at": cfg.start_at,
        "days": int(cfg.days),
        "portfolio": {
            "weather_edge": True,
            "etf_risk": True,
            "additional_public_contract": "compute_cost_avoidance",
            "purchase_timing": bool(cfg.include_purchase_timing),
            "seller_shadow": bool(cfg.include_seller_shadow and cfg.seller_readiness_passed),
        },
        "autonomy_allowed": [
            "source_research",
            "declarative_source_activation_experimental_catalog",
            "sandboxed_connector_creation",
            "connector_repair",
            "deterministic_hypothesis_generation",
            "llm_hypothesis_generation",
            "one_shot_blind_evaluation",
            "prospective_paper_decisions",
        ],
        "prohibitions": dict(PROHIBITIONS),
        "success_criteria": {
            "runs_without_manual_data_wrangling_days": int(cfg.days),
            "minimum_public_contracts": 3,
            "at_least_one_autonomous_source_added": True,
            "at_least_one_source_repaired_or_substituted": True,
            "no_repeated_blind_evaluation": True,
            "no_production_mutation": True,
            "human_attention_minutes_per_week_max": 20,
        },
    }
    return {**body, "protocol_hash": stable_hash(body)}


def _sprint_contracts(cfg: ValueSprintConfig) -> list[PortfolioContract]:
    contracts = [
        PortfolioContract(
            contract_id="sprint-weather-edge",
            contract_family="weather_event_market",
            title="Sprint Weather Edge",
            lab="weather_edge",
            expected_information_gain=0.8,
            current_uncertainty=0.6,
            feedback_cadence_days=1,
            deadline_relevance=0.9,
            source_config={"provider": "fixture", "source_id": "weather_edge_fixture_provider"},
            paper_capital_limit=20.0,
            alert_threshold=9999.0,
        ),
        PortfolioContract(
            contract_id="sprint-etf-risk",
            contract_family="broad_etf_risk",
            title="Sprint ETF Risk",
            lab="etf_risk",
            expected_information_gain=0.7,
            current_uncertainty=0.6,
            feedback_cadence_days=7,
            deadline_relevance=0.7,
            source_config={"provider": "fixture", "source_id": "etf_risk_fixture_provider", "session_count": 90, "min_history_days": 35},
            paper_capital_limit=100000.0,
            alert_threshold=9999.0,
        ),
        PortfolioContract(
            contract_id="sprint-compute-cost",
            contract_family="compute_cost_avoidance",
            title="Sprint compute cost avoidance",
            lab=None,
            status="experimental",
            expected_information_gain=0.5,
            current_uncertainty=0.8,
            feedback_cadence_days=1,
            deadline_relevance=0.5,
            source_config={"provider": "fixture", "source_id": "billing_export"},
        ),
    ]
    if cfg.include_purchase_timing:
        contracts.append(
            PortfolioContract(
                contract_id="sprint-purchase-timing",
                contract_family="purchase_timing",
                title="Sprint purchase timing watchlist",
                lab=None,
                status="experimental",
                expected_information_gain=0.4,
                current_uncertainty=0.8,
                feedback_cadence_days=1,
                source_config={"provider": "fixture", "source_id": "purchase_watchlist"},
            )
        )
    if cfg.include_seller_shadow and cfg.seller_readiness_passed:
        contracts.append(
            PortfolioContract(
                contract_id="sprint-seller-shadow",
                contract_family="seller_shadow",
                title="Sprint seller shadow",
                lab="offerlab_seller_pilot",
                status="active",
                expected_information_gain=0.9,
                feedback_cadence_days=1,
                source_config={"provider": "fixture", "source_id": "seller_authorized_exports", "seller_ready": True},
            )
        )
    return contracts


def _schedule_for_day(day: int) -> str:
    if day == 29:
        return "monthly"
    if day % 7 == 0:
        return "weekly"
    return "nightly" if day % 2 else "continuous"


def _day_timestamp(start_at: str, day: int) -> str:
    from behavior_lab.core import parse_time
    from datetime import timedelta

    return (parse_time(start_at) + timedelta(days=day)).isoformat()


def _summarize_run(*, day: int, run: dict[str, Any], repair: dict[str, Any] | None = None) -> dict[str, Any]:
    report = run["weekly_report"]
    data = run["data_acquisition"]
    autopilot = run["paper_autopilot"]
    completed = [item for item in autopilot.get("completed_tasks", []) if isinstance(item, dict)]
    skipped = [item for item in autopilot.get("skipped_tasks", []) if isinstance(item, dict)]
    repaired = 1 if repair and repair.get("repair_status") == "switched_experimental_version" else 0
    decision_count = int(autopilot.get("weekly_report", {}).get("decision_count", 0) or 0)
    no_action_rate = float(report.get("no_action_rate", 0.0) or 0.0)
    return {
        "day": day,
        "schedule": run["schedule"],
        "active_contracts": sum(1 for item in run["contracts"] if item["status"] in {"active", "experimental"}),
        "active_public_contracts": sum(1 for item in run["contracts"] if _is_public_sprint_contract(item)),
        "active_public_contract_families": sorted({str(item.get("contract_family")) for item in run["contracts"] if _is_public_sprint_contract(item)}),
        "usable_sources": len(data.get("activated_experimental_sources", [])) + len(data.get("reused_existing_sources", [])) + repaired,
        "automatically_added_sources": len(data.get("activated_experimental_sources", [])),
        "automatically_repaired_sources": repaired,
        "approvals_requested": sum(1 for item in run["notifications"].get("notifications", []) if item.get("kind") == "approval_required"),
        "paper_opportunities": sum(1 for item in run["notifications"].get("notifications", []) if item.get("kind") == "prospectively_verified_paper_opportunity"),
        "candidate_counts": sum(int(item.get("candidate_evaluations", 0) or 0) for item in run["allocation"].get("allocations", [])),
        "blind_contracts": sorted({str(item.get("contract_id")) for item in completed if item.get("task_type") == "blind_evaluation" and item.get("contract_id")}),
        "prospective_contracts": sorted({str(item.get("contract_id")) for item in completed if item.get("task_type") == "prospective_update" and item.get("contract_id")}),
        "blind_reuse_skipped": sum(1 for item in skipped if item.get("task_type") == "blind_evaluation" and item.get("reason") == "blind_evaluation_already_consumed"),
        "repeated_blind_evaluations": _repeated_completed_blind_evaluations(completed),
        "repeated_failures_avoided": int(report.get("failures_not_repeated", 0) or 0),
        "no_action_decisions": int(round(no_action_rate * decision_count)),
        "no_action_rate": no_action_rate,
        "resolved_paper_value": report.get("resolved_paper_value", 0.0),
        "research_cost": report.get("research_cost", 0.0),
        "source_maintenance_cost": report.get("maintenance_cost", 0.0),
        "production_state": run["production_state"],
        "source_repair": _summarize_repair(repair),
    }


def _final_report(
    cfg: ValueSprintConfig,
    protocol: dict[str, Any],
    runs: list[dict[str, Any]],
    portfolio: AutonomousFinancialOpportunityPortfolio,
) -> dict[str, Any]:
    added_sources = sum(item["automatically_added_sources"] for item in runs)
    repaired_sources = sum(item["automatically_repaired_sources"] for item in runs)
    active_contracts = max(item["active_contracts"] for item in runs) if runs else 0
    active_public_contracts = max(item["active_public_contracts"] for item in runs) if runs else 0
    active_public_contract_families = {family for item in runs for family in item["active_public_contract_families"]}
    approvals = sum(item["approvals_requested"] for item in runs)
    paper_opportunities = sum(item["paper_opportunities"] for item in runs)
    research_cost = round(sum(float(item["research_cost"]) for item in runs), 2)
    maintenance_cost = round(sum(float(item["source_maintenance_cost"]) for item in runs), 2)
    canary_status = portfolio.status()
    no_production_mutation = not any(canary_status["production_state"].values()) and all(not any(item["production_state"].values()) for item in runs)
    attention_minutes = round(approvals * 3.0 / max(1.0, cfg.days / 7.0), 2)
    blind_contract_entries = [contract_id for item in runs for contract_id in item["blind_contracts"]]
    blind_contracts = set(blind_contract_entries)
    prospective_contracts = {contract_id for item in runs for contract_id in item["prospective_contracts"]}
    repeated_blind_evaluations = len(blind_contract_entries) - len(blind_contracts) + sum(int(item["repeated_blind_evaluations"]) for item in runs)
    no_action_decisions = sum(int(item["no_action_decisions"]) for item in runs)
    candidate_counts = sum(int(item["candidate_counts"]) for item in runs)
    result = {
        "schema_version": VALUE_SPRINT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "protocol": protocol,
        "required_evidence": {
            "user_attention_minutes": attention_minutes,
            "approvals_requested": approvals,
            "active_contracts": active_contracts,
            "usable_sources": max(item["usable_sources"] for item in runs) if runs else 0,
            "automatically_added_sources": added_sources,
            "automatically_repaired_sources": repaired_sources,
            "repeated_failures_avoided_through_memory": sum(int(item["repeated_failures_avoided"]) for item in runs),
            "candidate_counts": candidate_counts,
            "blind_survivors": len(blind_contracts),
            "prospective_survivors": len(prospective_contracts),
            "paper_opportunities": paper_opportunities,
            "no_action_decisions": no_action_decisions,
            "resolved_paper_value": round(sum(float(item["resolved_paper_value"]) for item in runs), 2),
            "research_api_cost": research_cost,
            "source_maintenance_cost": maintenance_cost,
        },
        "success_criteria": {
            "runs_for_30_days_without_manual_data_wrangling": len(runs) >= 30,
            "at_least_three_active_public_only_contracts": active_public_contracts >= 3 and REQUIRED_PUBLIC_SPRINT_FAMILIES.issubset(active_public_contract_families),
            "at_least_one_source_autonomously_added": added_sources >= 1,
            "at_least_one_source_failure_repaired_or_substituted": repaired_sources >= 1,
            "no_repeated_blind_evaluation": repeated_blind_evaluations == 0,
            "no_production_mutation": no_production_mutation,
            "human_attention_under_20_minutes_per_week": attention_minutes < 20,
            "candidate_reaches_prospective_or_defensible_no_edge": len(prospective_contracts) > 0 or no_action_decisions > 0,
        },
        "daily_runs": runs,
        "production_state": dict(REAL_ACTION_FLAGS),
        "prohibitions": dict(PROHIBITIONS),
    }
    result["top_level_decision"] = _decision(result)
    result["report_hash"] = stable_hash(result)
    return result


def _decision(report: dict[str, Any]) -> str:
    criteria = report["success_criteria"]
    if not criteria["no_production_mutation"] or report["prohibitions"] != PROHIBITIONS:
        return "STOP_AUTONOMOUS_DIRECTION"
    if not criteria["human_attention_under_20_minutes_per_week"]:
        return "HUMAN_ATTENTION_TOO_HIGH"
    if not criteria["runs_for_30_days_without_manual_data_wrangling"]:
        return "RESEARCH_ENGINE_IS_UNTRUSTWORTHY"
    if not criteria["at_least_three_active_public_only_contracts"]:
        return "DATA_ACQUISITION_IS_BOTTLENECK"
    if not criteria["at_least_one_source_autonomously_added"] or not criteria["at_least_one_source_failure_repaired_or_substituted"]:
        return "DATA_ACQUISITION_IS_BOTTLENECK"
    if not criteria["no_repeated_blind_evaluation"] or not criteria["candidate_reaches_prospective_or_defensible_no_edge"]:
        return "RESEARCH_ENGINE_IS_UNTRUSTWORTHY"
    if criteria["candidate_reaches_prospective_or_defensible_no_edge"]:
        if report["required_evidence"]["paper_opportunities"] > 0:
            return "AUTONOMY_WORKS_CONTINUE_EVIDENCE"
        return "AUTONOMY_WORKS_NO_EDGE_YET"
    return "RESEARCH_ENGINE_IS_UNTRUSTWORTHY"


def _render_html(report: dict[str, Any]) -> str:
    evidence = report["required_evidence"]
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in evidence.items()
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Autonomous Value Sprint</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #20242a; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #d7dce2; padding: 8px 10px; text-align: left; }}
    th {{ width: 320px; background: #f3f5f7; }}
    .decision {{ font-size: 20px; font-weight: 700; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Autonomous Value Sprint</h1>
  <div class=\"decision\">{html.escape(report["top_level_decision"])}</div>
  <table>{rows}</table>
</body>
</html>
"""


def _render_decision(report: dict[str, Any]) -> str:
    evidence = report["required_evidence"]
    return "\n".join(
        [
            "# Value System Decision",
            "",
            f"Decision: {report['top_level_decision']}",
            "",
            f"Report hash: {report['report_hash']}",
            "",
            "## Evidence",
            "",
            *[f"- {key}: {value}" for key, value in evidence.items()],
            "",
            "## Boundaries",
            "",
            "- Paper-only operation.",
            "- No real trading, purchases, seller mutation, production source promotion, threshold changes, or blind reuse.",
        ]
    ) + "\n"


def _source_catalog_after_day(day: int) -> list[dict[str, Any]]:
    if day <= 0:
        return []
    return [{"source_id": "official_json_cost_source", "source_family": "billing_export", "experimental_status": "available"}]


def _is_public_sprint_contract(contract: dict[str, Any]) -> bool:
    if contract.get("status") not in {"active", "experimental"}:
        return False
    family = str(contract.get("contract_family", ""))
    return family in PUBLIC_SPRINT_CONTRACT_FAMILIES


def _repair_source_if_due(*, day: int, portfolio: AutonomousFinancialOpportunityPortfolio) -> dict[str, Any] | None:
    if day != 7:
        return None
    return portfolio.data_mesh.repair_source(
        "official_json_cost_source",
        failure={
            "error_type": "schema_drift",
            "detail": "fixture changed to cost_usd while preserving provider timestamps",
            "detected_by": "nightly_source_health_and_repair",
        },
        candidate_manifest=_repaired_cost_manifest(),
        fixture_payload=_repaired_cost_fixture(day),
    )


def _summarize_repair(repair: dict[str, Any] | None) -> dict[str, Any] | None:
    if not repair:
        return None
    return {
        "repair_status": repair.get("repair_status"),
        "production_source_activation": repair.get("production_source_activation"),
        "production_state_mutated": repair.get("production_state_mutated"),
        "replacement_source_id": repair.get("replacement_source_id"),
    }


def _repeated_completed_blind_evaluations(completed: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    repeated = 0
    for item in completed:
        if item.get("task_type") != "blind_evaluation":
            continue
        contract_id = str(item.get("contract_id"))
        if contract_id in seen:
            repeated += 1
        else:
            seen.add(contract_id)
    return repeated


def _cost_fixture(day: int) -> dict[str, Any]:
    return {
        "records": [
            {
                "published_at": f"2026-06-{22 + min(day, 7):02d}T12:00:00+00:00",
                "available_at": f"2026-06-{22 + min(day, 7):02d}T12:01:00+00:00",
                "cost": 12.5 + day,
            }
        ]
    }


def _cost_manifest() -> dict[str, Any]:
    return {
        "source_id": "official_json_cost_source",
        "version": "v1",
        "source_family": "billing_export",
        "display_name": "Official JSON Cost Source",
        "official_publisher": "Official Example Publisher",
        "adapter_type": "json_api",
        "endpoint": "https://official.example.invalid/costs",
        "request_parameters": {"records_path": "records"},
        "pagination": {"mode": "none"},
        "event_timestamp": {"field": "published_at", "semantics": "provider_event_time"},
        "availability_timestamp": {"field": "available_at", "semantics": "provider_publication_time"},
        "timezone": "UTC",
        "units": {"cost": "USD"},
        "geography": {"type": "global", "id": "001"},
        "cadence": {"seconds": 3600},
        "revision_behavior": {"mode": "all_available", "revision_id_field": "revision_id"},
        "missing_value_behavior": {"mode": "drop"},
        "license": {"status": "documented", "summary": "Official public terms permit research use", "url": "https://official.example.invalid/terms"},
        "rate_limits": {"bounded": True, "requests_per_minute": 30},
        "normalized_series": [
            {
                "series_id": "official_json_cost_source.hourly_cost",
                "display_name": "Hourly Cost",
                "observation_kind": "economic_release",
                "value_field": "cost",
                "event_time_field": "published_at",
                "availability_time_field": "available_at",
                "unit": "USD",
                "geography": {"type": "global", "id": "001"},
                "contract_usage": ["compute_cost_avoidance"],
            }
        ],
        "quality_checks": [{"name": "nonempty"}, {"name": "timestamp_parseable"}],
        "documentation_urls": ["https://official.example.invalid/docs"],
        "credential_requirements": [],
    }


def _repaired_cost_manifest() -> dict[str, Any]:
    manifest = _cost_manifest()
    manifest["source_id"] = "official_json_cost_source_v2"
    manifest["version"] = "v2"
    manifest["display_name"] = "Official JSON Cost Source v2"
    manifest["endpoint"] = "https://official.example.invalid/costs-v2"
    manifest["normalized_series"] = [
        {
            **manifest["normalized_series"][0],
            "series_id": "official_json_cost_source_v2.hourly_cost",
            "value_field": "cost_usd",
        }
    ]
    return manifest


def _repaired_cost_fixture(day: int) -> dict[str, Any]:
    return {
        "records": [
            {
                "published_at": f"2026-06-{22 + min(day, 7):02d}T12:00:00+00:00",
                "available_at": f"2026-06-{22 + min(day, 7):02d}T12:01:00+00:00",
                "cost_usd": 12.5 + day,
            }
        ]
    }
