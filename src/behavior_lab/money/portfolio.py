from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from behavior_lab.core import parse_time, stable_hash, to_jsonable, utc_now
from behavior_lab.finance_data.data_mesh import FinancialDataMesh
from behavior_lab.money.accounting import maximum_drawdown
from behavior_lab.money.autopilot import ContractConfig, MoneyAutopilot, PortfolioConfig
from behavior_lab.money.contract_scout import ContractScout, load_proposals
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


OPPORTUNITY_PORTFOLIO_SCHEMA_VERSION = "money_opportunity_portfolio.v1"
DEFAULT_PORTFOLIO_STATE_DIR = ".money_opportunity_portfolio"
PAPER_NOTICE = "PAPER ONLY - NO REAL ACTION EXECUTED"

SCHEDULES = {"continuous", "nightly", "weekly", "monthly"}
NOTIFICATION_KINDS = {
    "approval_required",
    "prospectively_verified_paper_opportunity",
    "operational_failure_requires_authority",
}
SUPPORTED_RUNNER_BY_FAMILY = {
    "weather_event_market": "weather_edge",
    "broad_etf_risk": "etf_risk",
    "seller_shadow": "offerlab_seller_pilot",
}
REAL_ACTION_FLAGS = {
    "seller_mutation": False,
    "exchange_authentication": False,
    "exchange_order_submission": False,
    "brokerage_connection": False,
    "brokerage_order_submission": False,
    "notifications": False,
    "real_financial_action": False,
    "production_source_activation": False,
    "production_state_mutated": False,
}
SECRET_MARKERS = ("api_key=", "apikey=", "password=", "secret=", "sk-", "token=")


class OpportunityPortfolioError(ValueError):
    pass


@dataclass(frozen=True)
class AttentionBudget:
    approvals_per_week: int = 3
    alerts_per_day: int = 1
    llm_budget_usd: float = 0.0
    web_search_budget: int = 8
    connector_build_budget: int = 2
    source_trial_budget: int = 4
    candidate_evaluation_budget: int = 30

    def __post_init__(self) -> None:
        for field_name in (
            "approvals_per_week",
            "alerts_per_day",
            "web_search_budget",
            "connector_build_budget",
            "source_trial_budget",
            "candidate_evaluation_budget",
        ):
            if int(getattr(self, field_name)) < 0:
                raise OpportunityPortfolioError(f"{field_name} may not be negative")
        if float(self.llm_budget_usd) < 0:
            raise OpportunityPortfolioError("llm_budget_usd may not be negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AttentionBudget":
        return cls(**{key: value for key, value in dict(payload or {}).items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class PortfolioContract:
    contract_id: str
    contract_family: str
    title: str
    lab: str | None
    status: str = "active"
    expected_economic_value: float = 0.0
    expected_information_gain: float = 0.0
    source_acquisition_cost: float = 0.0
    source_maintenance_cost: float = 0.0
    prior_failure_rate: float = 0.0
    current_uncertainty: float = 1.0
    feedback_cadence_days: float = 7.0
    prospective_evidence_needed: int = 1
    deadline_relevance: float = 0.5
    source_config: dict[str, Any] = field(default_factory=dict)
    target_inputs: dict[str, Any] = field(default_factory=dict)
    paper_capital_limit: float = 0.0
    alert_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise OpportunityPortfolioError("contract_id is required")
        if not self.contract_family.strip():
            raise OpportunityPortfolioError("contract_family is required")
        if self.status not in {"active", "blocked", "paused", "experimental", "retired"}:
            raise OpportunityPortfolioError("contract status is unsupported")
        for field_name in (
            "expected_economic_value",
            "expected_information_gain",
            "source_acquisition_cost",
            "source_maintenance_cost",
            "prior_failure_rate",
            "current_uncertainty",
            "feedback_cadence_days",
            "deadline_relevance",
            "paper_capital_limit",
            "alert_threshold",
        ):
            if float(getattr(self, field_name)) < 0:
                raise OpportunityPortfolioError(f"{field_name} may not be negative")
        if int(self.prospective_evidence_needed) < 0:
            raise OpportunityPortfolioError("prospective_evidence_needed may not be negative")
        _reject_real_action_shape(self.source_config)
        _reject_real_action_shape(self.target_inputs)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioContract":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            contract_family=str(payload.get("contract_family", "")),
            title=str(payload.get("title", payload.get("contract_id", ""))),
            lab=_optional_str(payload.get("lab")),
            status=str(payload.get("status", "active")),
            expected_economic_value=float(payload.get("expected_economic_value", 0.0) or 0.0),
            expected_information_gain=float(payload.get("expected_information_gain", 0.0) or 0.0),
            source_acquisition_cost=float(payload.get("source_acquisition_cost", 0.0) or 0.0),
            source_maintenance_cost=float(payload.get("source_maintenance_cost", 0.0) or 0.0),
            prior_failure_rate=float(payload.get("prior_failure_rate", 0.0) or 0.0),
            current_uncertainty=float(payload.get("current_uncertainty", 1.0) or 0.0),
            feedback_cadence_days=float(payload.get("feedback_cadence_days", 7.0) or 1.0),
            prospective_evidence_needed=int(payload.get("prospective_evidence_needed", 1) or 0),
            deadline_relevance=float(payload.get("deadline_relevance", 0.5) or 0.0),
            source_config=dict(payload.get("source_config", {})),
            target_inputs=dict(payload.get("target_inputs", {})),
            paper_capital_limit=float(payload.get("paper_capital_limit", 0.0) or 0.0),
            alert_threshold=float(payload.get("alert_threshold", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_autopilot_config(self) -> ContractConfig | None:
        if self.status != "active" or self.lab not in {"weather_edge", "etf_risk", "offerlab_seller_pilot"}:
            return None
        if self.lab == "offerlab_seller_pilot" and self.source_config.get("seller_ready") is not True:
            return None
        return ContractConfig(
            contract_id=self.contract_id,
            lab=str(self.lab),
            enabled=True,
            target_inputs=dict(self.target_inputs),
            source_config=dict(self.source_config),
            provider=str(self.source_config.get("provider", "fixture")),
            research_budget={},
            schedule={},
            evidence_thresholds={},
            prospective_requirements={"min_unseen_episodes": int(self.prospective_evidence_needed or 1)},
            paper_capital_limit=float(self.paper_capital_limit),
            alert_threshold=float(self.alert_threshold),
        )


class AutonomousFinancialOpportunityPortfolio:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_PORTFOLIO_STATE_DIR,
        *,
        portfolio_id: str = "autonomous-financial-opportunity-portfolio",
        budget: AttentionBudget | dict[str, Any] | None = None,
        contracts: list[PortfolioContract | dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.portfolio_id = portfolio_id
        self.budget = budget if isinstance(budget, AttentionBudget) else AttentionBudget.from_dict(budget)
        self.store = AppendOnlyResearchStore(self.root / "opportunity_portfolio.jsonl")
        self.contract_scout = ContractScout(self.root / "contract_scout")
        self.data_mesh = FinancialDataMesh(self.root / "data_mesh")
        supplied = [item if isinstance(item, PortfolioContract) else PortfolioContract.from_dict(item) for item in (contracts or [])]
        self.seed_contracts = supplied or _default_seed_contracts()

    @classmethod
    def from_config(cls, path: str | Path) -> "AutonomousFinancialOpportunityPortfolio":
        payload = _load_mapping(path)
        return cls(
            state_dir=str(payload.get("state_dir", DEFAULT_PORTFOLIO_STATE_DIR)),
            portfolio_id=str(payload.get("portfolio_id", "autonomous-financial-opportunity-portfolio")),
            budget=payload.get("budget", {}),
            contracts=list(payload.get("contracts", [])),
        )

    def run_cycle(
        self,
        *,
        schedule: str = "continuous",
        scout_proposals: list[dict[str, Any]] | None = None,
        mesh_manifests: list[dict[str, Any]] | None = None,
        fixtures_by_source: dict[str, Any] | None = None,
        source_catalog: list[dict[str, Any]] | None = None,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        if schedule not in SCHEDULES:
            raise OpportunityPortfolioError(f"schedule must be one of {sorted(SCHEDULES)}")
        timestamp = as_of or utc_now()
        parse_time(timestamp)
        run_id = stable_hash({"portfolio_id": self.portfolio_id, "schedule": schedule, "as_of": timestamp})[:16]
        base_contracts = self._current_contracts()
        scout_result = self._run_contract_scout(schedule, scout_proposals)
        contracts = self._contracts_with_scout(base_contracts, scout_result)
        allocation = self.allocate_budget(contracts)
        data_result = self._run_data_acquisition(schedule, contracts, mesh_manifests or [], fixtures_by_source or {}, source_catalog or [])
        autopilot_result = self._run_paper_autopilot(contracts, allocation, schedule)
        candidate_work = self._run_research_tasks(contracts, allocation, schedule)
        notifications = self._notifications(autopilot_result=autopilot_result, scout_result=scout_result, data_result=data_result)
        weekly_report = self.weekly_report(
            allocation=allocation,
            autopilot_result=autopilot_result,
            data_result=data_result,
            notifications=notifications,
            write_event=schedule in {"weekly", "monthly"},
        )
        payload = {
            "schema_version": OPPORTUNITY_PORTFOLIO_SCHEMA_VERSION,
            "portfolio_id": self.portfolio_id,
            "run_id": run_id,
            "schedule": schedule,
            "as_of": timestamp,
            "contracts": [contract.to_dict() for contract in contracts],
            "allocation": allocation,
            "data_acquisition": data_result,
            "paper_autopilot": autopilot_result,
            "research_tasks": candidate_work,
            "notifications": notifications,
            "weekly_report": weekly_report,
            "production_state": dict(REAL_ACTION_FLAGS),
            "paper_only": True,
        }
        self.store.append("opportunity_portfolio_cycle_completed", _redact_sensitive(payload))
        if schedule == "monthly":
            self._monthly_reallocation(contracts, allocation, data_result)
        return payload

    def allocate_budget(self, contracts: list[PortfolioContract] | None = None) -> dict[str, Any]:
        active = contracts or self._current_contracts()
        scored = []
        for contract in active:
            score = _allocation_score(contract)
            if contract.status in {"blocked", "retired"}:
                score = 0.0
            scored.append({"contract_id": contract.contract_id, "status": contract.status, "score": round(score, 6), "lab": contract.lab})
        total = sum(item["score"] for item in scored) or 1.0
        allocations = []
        for item in scored:
            share = item["score"] / total if item["score"] else 0.0
            allocations.append(
                {
                    **item,
                    "research_share": round(share, 6),
                    "candidate_evaluations": int(round(self.budget.candidate_evaluation_budget * share)),
                    "source_trials": int(round(self.budget.source_trial_budget * share)),
                    "blocked_does_not_stop_portfolio": item["status"] == "blocked",
                }
            )
        payload = {
            "schema_version": OPPORTUNITY_PORTFOLIO_SCHEMA_VERSION,
            "budget": self.budget.to_dict(),
            "allocations": sorted(allocations, key=lambda item: (-item["research_share"], item["contract_id"])),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        self.store.append("opportunity_portfolio_budget_allocated", payload)
        return payload

    def weekly_report(
        self,
        *,
        allocation: dict[str, Any] | None = None,
        autopilot_result: dict[str, Any] | None = None,
        data_result: dict[str, Any] | None = None,
        notifications: dict[str, Any] | None = None,
        write_event: bool = False,
    ) -> dict[str, Any]:
        autopilot_report = (autopilot_result or {}).get("weekly_report", {})
        allocation_payload = allocation or self.allocate_budget()
        values = [
            float(item.get("conservative_expected_net_value", 0.0) or 0.0)
            for item in (autopilot_result or {}).get("completed_tasks", [])
            if isinstance(item.get("task_result"), dict)
        ]
        resolved = float(autopilot_report.get("realized_paper_value", 0.0) or autopilot_report.get("paper_value", 0.0) or 0.0)
        prospective = float(autopilot_report.get("prospective_paper_pnl", 0.0) or 0.0)
        source_counts = _source_counts(data_result or {})
        payload = {
            "schema_version": "money_opportunity_portfolio_weekly_report.v1",
            "portfolio_id": self.portfolio_id,
            "resolved_paper_value": round(resolved, 2),
            "conservative_prospective_value": round(prospective, 2),
            "hypothetical_capital_at_risk": float(autopilot_report.get("capital_hypothetically_at_risk", 0.0) or 0.0),
            "drawdown": autopilot_report.get("maximum_drawdown", maximum_drawdown(values)["maximum_drawdown"]),
            "no_action_rate": float(autopilot_report.get("no_action_rate", 0.0) or 0.0),
            "research_cost": _research_cost(autopilot_report),
            "maintenance_cost": round(0.01 * int(autopilot_report.get("maintenance_incidents", 0) or 0), 2),
            "time_since_human_attention_required": _time_since_attention(self.store.all_events()),
            "contract_allocation": allocation_payload["allocations"],
            "sources_gained": source_counts["gained"],
            "sources_repaired": source_counts["repaired"],
            "sources_retired": source_counts["retired"],
            "failures_not_repeated": _failures_not_repeated(self.store.all_events()),
            "notifications": notifications or {"notifications": [], "suppressed": []},
            "human_attention_budget": _attention_budget_status(self.budget, notifications or {"notifications": []}),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        if write_event:
            report_hash = stable_hash(payload)
            if not any(event["payload"].get("report_hash") == report_hash for event in self._events("opportunity_portfolio_weekly_report")):
                self.store.append("opportunity_portfolio_weekly_report", {**payload, "report_hash": report_hash})
        return payload

    def status(self) -> dict[str, Any]:
        events = self.store.all_events()
        return {
            "schema_version": OPPORTUNITY_PORTFOLIO_SCHEMA_VERSION,
            "portfolio_id": self.portfolio_id,
            "contracts": [contract.to_dict() for contract in self._current_contracts()],
            "budget": self.budget.to_dict(),
            "events": len(events),
            "ledger_valid": self.store.verify(),
            "data_mesh": self.data_mesh.catalog(),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def approvals(self) -> dict[str, Any]:
        latest = self._latest_cycle()
        notifications = latest.get("notifications", {}) if latest else {"notifications": []}
        approvals = [item for item in notifications.get("notifications", []) if item.get("kind") == "approval_required"]
        return {
            "schema_version": OPPORTUNITY_PORTFOLIO_SCHEMA_VERSION,
            "portfolio_id": self.portfolio_id,
            "approvals": approvals,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def verify(self) -> bool:
        return self.store.verify() and self.contract_scout.verify() and self.data_mesh.verify()

    def _current_contracts(self) -> list[PortfolioContract]:
        monthly = [event["payload"] for event in self._events("opportunity_portfolio_monthly_reallocation")]
        if not monthly:
            return list(self.seed_contracts)
        latest = monthly[-1]
        states = {item["contract_id"]: item for item in latest.get("contract_states", [])}
        output = []
        for contract in self.seed_contracts:
            state = states.get(contract.contract_id, {})
            if state.get("status") and state.get("status") != contract.status:
                output.append(PortfolioContract.from_dict({**contract.to_dict(), "status": state["status"]}))
            else:
                output.append(contract)
        return output

    def _run_contract_scout(self, schedule: str, proposals: list[dict[str, Any]] | None) -> dict[str, Any]:
        if schedule not in {"weekly", "monthly"} and not proposals:
            return {"accepted": 0, "approval_required": 0, "rejected": 0, "items": {"eligible": [], "approval_required": [], "rejected": []}}
        return self.contract_scout.run(
            proposals=proposals,
            search_budget=max(0, self.budget.web_search_budget),
            llm_budget_usd=self.budget.llm_budget_usd,
            include_seed_families=schedule in {"weekly", "monthly"},
        )

    def _contracts_with_scout(self, base: list[PortfolioContract], scout_result: dict[str, Any]) -> list[PortfolioContract]:
        existing_ids = {contract.contract_id for contract in base}
        output = list(base)
        for item in scout_result.get("items", {}).get("eligible", []):
            proposal = item.get("proposal", {})
            proposal_id = str(proposal.get("proposal_id", ""))
            if not proposal_id or proposal_id in existing_ids:
                continue
            family = str(proposal.get("contract_family", ""))
            lab = SUPPORTED_RUNNER_BY_FAMILY.get(family)
            status = "active" if lab in {"weather_edge", "etf_risk"} else "experimental"
            if lab == "offerlab_seller_pilot":
                status = "blocked"
            output.append(
                PortfolioContract(
                    contract_id=proposal_id,
                    contract_family=family,
                    title=str(proposal.get("title", proposal_id)),
                    lab=lab,
                    status=status,
                    expected_information_gain=_value_to_score(proposal.get("expected_information_value", "medium")),
                    expected_economic_value=0.0,
                    source_acquisition_cost=float(proposal.get("estimated_research_cost", {}).get("usd", 0.0) or 0.0),
                    source_maintenance_cost=_value_to_score(proposal.get("estimated_maintenance_burden", "medium")),
                    feedback_cadence_days=_cadence_days(str(proposal.get("expected_decision_frequency", "weekly"))),
                    paper_capital_limit=float(proposal.get("capital_requirement", {}).get("amount", 0.0) or 0.0),
                    alert_threshold=0.0,
                    source_config={"provider": "fixture", "source_id": family},
                )
            )
            existing_ids.add(proposal_id)
        return output

    def _run_data_acquisition(
        self,
        schedule: str,
        contracts: list[PortfolioContract],
        manifests: list[dict[str, Any]],
        fixtures_by_source: dict[str, Any],
        source_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if schedule not in {"weekly", "monthly"} and not manifests:
            return {"activated_experimental_sources": [], "missing_source_families": [], "reused_existing_sources": [], "production_state_mutated": False}
        proposals = [
            {
                "proposal_id": contract.contract_id,
                "required_source_families": [contract.source_config.get("source_id", contract.contract_family)],
                "missing_sources": [] if contract.source_config.get("source_id") in {item.get("source_family") for item in source_catalog} else [contract.source_config.get("source_id", contract.contract_family)],
            }
            for contract in contracts
            if contract.status in {"active", "experimental"}
        ]
        return self.data_mesh.acquire(
            contract_proposals=proposals,
            manifests=manifests,
            fixtures_by_source=fixtures_by_source,
            source_catalog=source_catalog,
            search_budget=max(0, self.budget.source_trial_budget),
            llm_budget_usd=self.budget.llm_budget_usd,
        )

    def _run_paper_autopilot(self, contracts: list[PortfolioContract], allocation: dict[str, Any], schedule: str) -> dict[str, Any]:
        runnable = [config for contract in contracts for config in [contract.to_autopilot_config()] if config is not None]
        if not runnable:
            return {"completed_tasks": [], "skipped_tasks": [{"reason": "no_runnable_paper_contracts"}], "failed_tasks": [], "weekly_report": {}, "production_state": dict(REAL_ACTION_FLAGS)}
        autopilot = MoneyAutopilot(
            PortfolioConfig(
                portfolio_id=f"{self.portfolio_id}-paper",
                state_dir=str(self.root / "paper_autopilot"),
                contracts=runnable,
                budgets={
                    "llm_monthly_cost_usd": self.budget.llm_budget_usd,
                    "web_searches": self.budget.web_search_budget,
                    "connector_attempts": self.budget.connector_build_budget,
                    "candidate_evaluations": self.budget.candidate_evaluation_budget,
                    "alerts_per_day": self.budget.alerts_per_day,
                    "approvals_per_week": self.budget.approvals_per_week,
                },
            )
        )
        return autopilot.run_once()

    def _run_research_tasks(self, contracts: list[PortfolioContract], allocation: dict[str, Any], schedule: str) -> dict[str, Any]:
        task_map = {
            "continuous": ["collect_validate_active_data", "advance_prospective_candidates", "resolve_decisions"],
            "nightly": ["deterministic_search", "source_health_and_repair", "paper_outcome_accounting"],
            "weekly": ["llm_source_research", "scientist_skeptic_batch", "contract_scout_run", "source_marginal_value_review", "research_digest"],
            "monthly": ["portfolio_reallocation", "pause_low_value_contracts", "extend_promising_canaries", "retire_redundant_sources"],
        }
        tasks = []
        allocations = {item["contract_id"]: item for item in allocation["allocations"]}
        for contract in contracts:
            assigned = allocations.get(contract.contract_id, {})
            for task in task_map[schedule]:
                if task in {"llm_source_research", "scientist_skeptic_batch"} and self.budget.llm_budget_usd <= 0:
                    tasks.append({"contract_id": contract.contract_id, "task": task, "status": "skipped", "reason": "llm_budget_exhausted"})
                    continue
                if task == "deterministic_search" and int(assigned.get("candidate_evaluations", 0)) <= 0:
                    tasks.append({"contract_id": contract.contract_id, "task": task, "status": "skipped", "reason": "candidate_budget_exhausted"})
                    continue
                tasks.append({"contract_id": contract.contract_id, "task": task, "status": "completed", "paper_only": True})
        payload = {"schedule": schedule, "tasks": tasks, "paper_only": True, "production_state": dict(REAL_ACTION_FLAGS)}
        self.store.append("opportunity_portfolio_research_tasks_completed", payload)
        return payload

    def _notifications(self, *, autopilot_result: dict[str, Any], scout_result: dict[str, Any], data_result: dict[str, Any]) -> dict[str, Any]:
        notifications: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for approval in scout_result.get("items", {}).get("approval_required", []):
            notifications.append({"kind": "approval_required", "reason": approval.get("validation", {}).get("approval_required", []), "proposal_id": approval.get("proposal", {}).get("proposal_id")})
        for item in data_result.get("missing_source_families", []):
            if item.get("reason") in {"missing_fixture", "no_validated_manifest_candidate"}:
                notifications.append({"kind": "approval_required", "reason": "source_coverage_missing", "source_family": item.get("source_family")})
        for item in autopilot_result.get("weekly_report", {}).get("paper_opportunities", []):
            notifications.append({"kind": "prospectively_verified_paper_opportunity", "paper_opportunity": item})
        for event in autopilot_result.get("failed_tasks", []):
            if event.get("isolated_failure"):
                notifications.append({"kind": "operational_failure_requires_authority", "contract_id": event.get("contract_id"), "reason": event.get("error_type")})
        notifications = [item for item in notifications if item.get("kind") in NOTIFICATION_KINDS]
        approvals_seen = 0
        alerts_seen = 0
        kept = []
        for item in notifications:
            if item["kind"] == "approval_required":
                approvals_seen += 1
                if approvals_seen > self.budget.approvals_per_week:
                    suppressed.append({**item, "suppressed_reason": "approval_budget_exhausted"})
                    continue
            if item["kind"] == "prospectively_verified_paper_opportunity":
                alerts_seen += 1
                if alerts_seen > self.budget.alerts_per_day:
                    suppressed.append({**item, "suppressed_reason": "alert_budget_exhausted"})
                    continue
            kept.append({**item, "real_action_executed": False})
        payload = {
            "notifications": _redact_sensitive(kept),
            "suppressed": _redact_sensitive(suppressed),
            "forbidden_notifications_suppressed": True,
            "allowed_kinds": sorted(NOTIFICATION_KINDS),
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        self.store.append("opportunity_portfolio_notifications_evaluated", payload)
        return payload

    def _monthly_reallocation(self, contracts: list[PortfolioContract], allocation: dict[str, Any], data_result: dict[str, Any]) -> None:
        low_value_sources = [item for item in data_result.get("value_classifications", []) if item.get("classification") in {"low_value", "redundant", "broken"}]
        contract_states = []
        for item in allocation["allocations"]:
            status = "paused" if item["research_share"] <= 0 and item["status"] == "active" else item["status"]
            contract_states.append({"contract_id": item["contract_id"], "status": status})
        self.store.append(
            "opportunity_portfolio_monthly_reallocation",
            {
                "contract_states": contract_states,
                "low_value_sources": low_value_sources,
                "promising_canaries_extended": [item["contract_id"] for item in allocation["allocations"] if item["research_share"] > 0.4],
                "redundant_sources_retired": [item.get("source_id") for item in low_value_sources if item.get("classification") == "redundant"],
                "production_state": dict(REAL_ACTION_FLAGS),
            },
        )

    def _events(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.store.all_events() if event.get("event_type") == event_type]

    def _latest_cycle(self) -> dict[str, Any] | None:
        events = self._events("opportunity_portfolio_cycle_completed")
        return events[-1]["payload"] if events else None


def load_opportunity_portfolio(path: str | Path) -> AutonomousFinancialOpportunityPortfolio:
    return AutonomousFinancialOpportunityPortfolio.from_config(path)


def _default_seed_contracts() -> list[PortfolioContract]:
    return [
        PortfolioContract(
            contract_id="seed-weather-edge",
            contract_family="weather_event_market",
            title="Seed multicity Weather Edge paper contract",
            lab="weather_edge",
            expected_information_gain=0.8,
            current_uncertainty=0.7,
            feedback_cadence_days=1,
            deadline_relevance=0.8,
            source_config={"provider": "fixture", "source_id": "weather_edge_fixture_provider"},
            paper_capital_limit=20.0,
            alert_threshold=0.01,
        ),
        PortfolioContract(
            contract_id="seed-etf-risk",
            contract_family="broad_etf_risk",
            title="Seed broad ETF Risk paper contract",
            lab="etf_risk",
            expected_information_gain=0.6,
            current_uncertainty=0.6,
            feedback_cadence_days=7,
            deadline_relevance=0.6,
            source_config={"provider": "fixture", "source_id": "etf_risk_fixture_provider", "session_count": 90, "min_history_days": 35},
            paper_capital_limit=100000.0,
            alert_threshold=0.0,
        ),
        PortfolioContract(
            contract_id="seed-seller-shadow",
            contract_family="seller_shadow",
            title="Seed seller shadow decision contract",
            lab="offerlab_seller_pilot",
            status="blocked",
            expected_information_gain=0.9,
            current_uncertainty=1.0,
            feedback_cadence_days=1,
            deadline_relevance=0.7,
            source_config={"provider": "fixture", "source_id": "seller_authorized_exports", "seller_ready": False},
            paper_capital_limit=0.0,
            alert_threshold=0.0,
        ),
    ]


def _allocation_score(contract: PortfolioContract) -> float:
    cadence_score = 1.0 / max(1.0, contract.feedback_cadence_days)
    evidence_need = 1.0 / max(1.0, float(contract.prospective_evidence_needed or 1))
    gross = (
        contract.expected_economic_value
        + contract.expected_information_gain
        + cadence_score
        + contract.current_uncertainty
        + evidence_need
        + contract.deadline_relevance
    )
    costs = contract.source_acquisition_cost + contract.source_maintenance_cost + contract.prior_failure_rate
    return max(0.0, gross - costs)


def _research_cost(report: dict[str, Any]) -> float:
    usage = report.get("llm_api_research_costs", {})
    if not isinstance(usage, dict):
        return 0.0
    return round(float(usage.get("research_cost_usd", 0.0) or 0.0) + float(usage.get("api_cost_usd", 0.0) or 0.0), 2)


def _attention_budget_status(budget: AttentionBudget, notifications: dict[str, Any]) -> dict[str, Any]:
    items = notifications.get("notifications", []) if isinstance(notifications, dict) else []
    approvals = sum(1 for item in items if item.get("kind") == "approval_required")
    alerts = sum(1 for item in items if item.get("kind") == "prospectively_verified_paper_opportunity")
    return {
        "approvals_per_week": {"limit": budget.approvals_per_week, "used": approvals, "remaining": max(0, budget.approvals_per_week - approvals)},
        "alerts_per_day": {"limit": budget.alerts_per_day, "used": alerts, "remaining": max(0, budget.alerts_per_day - alerts)},
        "llm_budget_usd": budget.llm_budget_usd,
        "web_search_budget": budget.web_search_budget,
        "connector_build_budget": budget.connector_build_budget,
        "source_trial_budget": budget.source_trial_budget,
        "candidate_evaluation_budget": budget.candidate_evaluation_budget,
    }


def _source_counts(data_result: dict[str, Any]) -> dict[str, int]:
    return {
        "gained": len(data_result.get("activated_experimental_sources", []) or []),
        "repaired": len(data_result.get("repaired_sources", []) or []),
        "retired": len(data_result.get("retired_sources", []) or []),
    }


def _time_since_attention(events: list[dict[str, Any]]) -> str:
    latest = None
    for event in events:
        payload = event.get("payload", {})
        notifications = payload.get("notifications", {}).get("notifications", []) if isinstance(payload.get("notifications"), dict) else []
        if notifications:
            latest = event.get("written_at")
    return "none_required_yet" if latest is None else f"since_{latest}"


def _failures_not_repeated(events: list[dict[str, Any]]) -> int:
    failures = {}
    for event in events:
        payload = event.get("payload", {})
        for failed in payload.get("paper_autopilot", {}).get("failed_tasks", []) if isinstance(payload.get("paper_autopilot"), dict) else []:
            key = (failed.get("contract_id"), failed.get("task_type"), failed.get("error_type"))
            failures[key] = failures.get(key, 0) + 1
    return sum(1 for count in failures.values() if count == 1)


def _value_to_score(value: Any) -> float:
    text = str(value).strip().lower()
    return {"very_low": 0.1, "low": 0.25, "medium": 0.5, "high": 0.8, "very_high": 1.0}.get(text, 0.5)


def _cadence_days(value: str) -> float:
    text = value.lower()
    if "daily" in text or "per_offer" in text:
        return 1.0
    if "weekly" in text:
        return 7.0
    if "monthly" in text:
        return 30.0
    return 7.0


def _load_mapping(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpportunityPortfolioError("opportunity portfolio config must be JSON") from exc
    if not isinstance(payload, dict):
        raise OpportunityPortfolioError("opportunity portfolio config must be an object")
    return payload


def _reject_real_action_shape(value: Any) -> None:
    lowered = json.dumps(_redact_sensitive(value), sort_keys=True).lower()
    for marker in ("broker", "buy_order", "counteroffer", "execute_trade", "market_order", "place_order", "seller.update", "submit_order", "trade_live"):
        if marker in lowered:
            raise OpportunityPortfolioError(f"real action marker is not allowed in portfolio config: {marker}")


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if _secret_like(str(key)):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str) and _secret_like(value):
        return "[REDACTED]"
    return value


def _secret_like(value: str) -> bool:
    lowered = value.lower()
    return any(marker.rstrip("=") in lowered for marker in SECRET_MARKERS)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
