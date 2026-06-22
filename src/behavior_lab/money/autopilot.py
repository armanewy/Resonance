from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.labs.etf_risk import ETFRiskConfig
from behavior_lab.labs.etf_risk.commands import paper_cycle as etf_paper_cycle
from behavior_lab.labs.offerlab_money import evaluate as offerlab_evaluate
from behavior_lab.labs.weather_edge import (
    DailyHighTemperatureEvent,
    FixtureWeatherEdgeProvider,
    ForecastPoint,
    MarketDepth,
    OrderBookLevel,
    StationHistoricalDay,
    TemperatureBracket,
    WeatherSnapshot,
    paper_cycle as weather_edge_paper_cycle,
)
from behavior_lab.money.accounting import maximum_drawdown
from behavior_lab.money.integration import fixture_etf_provider
from behavior_lab.money.ledger import MoneyLedger
from behavior_lab.money.storage import MoneyStorage
from behavior_lab.offerlab_pilot import import_pilot
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


AUTOPILOT_SCHEMA_VERSION = "money_autopilot.v1"
TASK_TYPES = (
    "source_update",
    "target_update",
    "source_health_audit",
    "deterministic_candidate_search",
    "llm_scientist_skeptic_batch",
    "source_scout_batch",
    "connector_maintenance",
    "tuning_selection",
    "blind_evaluation",
    "prospective_update",
    "paper_decision",
    "outcome_resolution",
)
GLOBAL_TASK_TYPES = ("weekly_report", "monthly_allocation_review")
RESEARCH_TASKS = {
    "deterministic_candidate_search",
    "llm_scientist_skeptic_batch",
    "source_scout_batch",
    "connector_maintenance",
    "tuning_selection",
    "blind_evaluation",
}
APPROVAL_REASONS = {
    "missing_credential",
    "unclear_license",
    "paid_source",
    "private_data_ambiguity",
    "production_source_promotion",
    "proposed_real_action",
}
LABS = {"offerlab_seller_pilot", "weather_edge", "etf_risk"}
REAL_ACTION_MARKERS = ("real_action", "submit_order", "broker", "trade_live", "seller_mutation")
CONNECTOR_DANGER_MARKERS = (
    "broker",
    "buy",
    "exchange",
    "live_order",
    "order",
    "place_live",
    "place_order",
    "purchase",
    "sell",
    "submit",
    "trade",
)
SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "credential",
    "credentials",
    "key",
    "map_key",
    "password",
    "secret",
    "session",
    "sig",
    "signature",
    "token",
)
SECRET_VALUE_MARKERS = ("audit_secret_token",)
PAPER_NOTICE = "PAPER — NO REAL ACTION EXECUTED"


class MoneyAutopilotError(ValueError):
    pass


@dataclass(frozen=True)
class ContractConfig:
    contract_id: str
    lab: str
    enabled: bool = True
    target_inputs: dict[str, Any] = field(default_factory=dict)
    source_config: dict[str, Any] = field(default_factory=dict)
    provider: str = "fixture"
    research_budget: dict[str, float] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    evidence_thresholds: dict[str, Any] = field(default_factory=dict)
    prospective_requirements: dict[str, Any] = field(default_factory=dict)
    paper_capital_limit: float = 0.0
    alert_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise MoneyAutopilotError("contract_id is required")
        if self.lab not in LABS:
            raise MoneyAutopilotError(f"unsupported money lab: {self.lab}")
        if float(self.paper_capital_limit) < 0:
            raise MoneyAutopilotError("paper_capital_limit may not be negative")
        if float(self.alert_threshold) < 0:
            raise MoneyAutopilotError("alert_threshold may not be negative")
        _reject_real_action_config(self.source_config)
        _reject_real_action_config(self.target_inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "lab": self.lab,
            "enabled": self.enabled,
            "target_inputs": dict(self.target_inputs),
            "source_config": dict(self.source_config),
            "provider": self.provider,
            "research_budget": dict(self.research_budget),
            "schedule": dict(self.schedule),
            "evidence_thresholds": dict(self.evidence_thresholds),
            "prospective_requirements": dict(self.prospective_requirements),
            "paper_capital_limit": self.paper_capital_limit,
            "alert_threshold": self.alert_threshold,
        }


@dataclass(frozen=True)
class PortfolioConfig:
    portfolio_id: str
    state_dir: str
    contracts: list[ContractConfig]
    budgets: dict[str, float] = field(default_factory=dict)
    alert_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.portfolio_id.strip():
            raise MoneyAutopilotError("portfolio_id is required")
        if not self.contracts:
            raise MoneyAutopilotError("at least one contract is required")
        for key, value in self.budgets.items():
            if float(value) < 0:
                raise MoneyAutopilotError(f"budget {key!r} may not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "state_dir": self.state_dir,
            "contracts": [contract.to_dict() for contract in self.contracts],
            "budgets": dict(self.budgets),
            "alert_threshold": self.alert_threshold,
        }

    def portfolio_hash(self) -> str:
        return stable_hash(self.to_dict())


class MoneyAutopilot:
    def __init__(self, portfolio: PortfolioConfig) -> None:
        self.portfolio = portfolio
        self.root = Path(portfolio.state_dir) / portfolio.portfolio_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = AppendOnlyResearchStore(self.root / "autopilot.jsonl")

    @classmethod
    def from_path(cls, path: str | Path) -> "MoneyAutopilot":
        return cls(load_portfolio(path))

    def run_once(self) -> dict[str, Any]:
        self.store.verify()
        run_id = stable_hash({"portfolio": self.portfolio.portfolio_hash(), "started_at": utc_now()})[:16]
        self.store.append("autopilot_run_started", self._base_payload({"run_id": run_id}))
        completed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for contract in self.portfolio.contracts:
            if not contract.enabled:
                skipped.append({"contract_id": contract.contract_id, "reason": "disabled"})
                continue
            if self._is_paused(contract.contract_id):
                skipped.append({"contract_id": contract.contract_id, "reason": "paused"})
                continue
            contract_failed = False
            for task_type in TASK_TYPES:
                task_id = self._task_id(contract, task_type)
                if task_type == "connector_maintenance" and self._failed_connector_without_changed_evidence(contract):
                    skipped.append({"contract_id": contract.contract_id, "task_type": task_type, "reason": "failed_connector_without_changed_evidence"})
                    continue
                if self._task_completed(task_id):
                    skipped.append({"contract_id": contract.contract_id, "task_type": task_type, "reason": "already_completed"})
                    continue
                if task_type == "blind_evaluation" and self._completed_task_type(contract.contract_id, "blind_evaluation"):
                    skipped.append({"contract_id": contract.contract_id, "task_type": task_type, "reason": "blind_evaluation_already_consumed"})
                    continue
                budget_reason = self._budget_block_reason(task_type)
                if budget_reason:
                    skipped.append({"contract_id": contract.contract_id, "task_type": task_type, "reason": budget_reason})
                    continue
                try:
                    result = self._run_task(contract, task_type, task_id)
                except Exception as exc:  # isolate one lab failure from the rest
                    event = self.store.append(
                        "autopilot_task_failed",
                        self._base_payload(
                            {
                                "task_id": task_id,
                                "contract_id": contract.contract_id,
                                "task_type": task_type,
                                "error_type": type(exc).__name__,
                                "error": _redact_secrets(str(exc)),
                                "isolated_failure": True,
                            }
                        ),
                    )
                    failed.append(event["payload"])
                    contract_failed = True
                    break
                completed.append(result)
            if contract_failed:
                continue
        weekly = self.weekly_report(write_event=True)
        for task_type in GLOBAL_TASK_TYPES:
            task_id = self._global_task_id(task_type)
            if not self._task_completed(task_id):
                self.store.append(
                    "autopilot_task_completed",
                    self._base_payload(
                        {
                            "task_id": task_id,
                            "contract_id": None,
                            "task_type": task_type,
                            "task_result": {"status": "completed", "paper_only": True},
                            "paper_only": True,
                            "production_state_mutated": False,
                        }
                    ),
                )
        self.store.append(
            "autopilot_run_completed",
            self._base_payload(
                {
                    "run_id": run_id,
                    "completed_tasks": len(completed),
                    "skipped_tasks": len(skipped),
                    "failed_tasks": len(failed),
                    "production_state_mutated": False,
                    "real_actions_executed": False,
                }
            ),
        )
        return {
            "schema_version": AUTOPILOT_SCHEMA_VERSION,
            "portfolio_id": self.portfolio.portfolio_id,
            "portfolio_hash": self.portfolio.portfolio_hash(),
            "ledger_valid": self.store.verify(),
            "completed_tasks": completed,
            "skipped_tasks": skipped,
            "failed_tasks": failed,
            "weekly_report": weekly,
            "production_state": _production_state_flags(),
        }

    def status(self) -> dict[str, Any]:
        events = self.store.all_events()
        return {
            "schema_version": AUTOPILOT_SCHEMA_VERSION,
            "portfolio_id": self.portfolio.portfolio_id,
            "portfolio_hash": self.portfolio.portfolio_hash(),
            "ledger_valid": self.store.verify(),
            "contracts": [
                {
                    "contract_id": contract.contract_id,
                    "lab": contract.lab,
                    "enabled": contract.enabled,
                    "paused": self._is_paused(contract.contract_id),
                    "completed_tasks": self._completed_task_count(contract.contract_id),
                }
                for contract in self.portfolio.contracts
            ],
            "budgets": self._budget_status(),
            "approvals_waiting": len(self.approvals()["approvals"]),
            "paper_opportunities": len(self._events("autopilot_paper_opportunity")),
            "events": len(events),
            "production_state": _production_state_flags(),
        }

    def approvals(self) -> dict[str, Any]:
        approvals = [
            event["payload"]
            for event in self._events("autopilot_approval_requested")
            if not self._approval_resolved(event["payload"]["approval_id"])
        ]
        return {"portfolio_id": self.portfolio.portfolio_id, "approvals": approvals}

    def pause(self, contract_id: str) -> dict[str, Any]:
        self._require_contract(contract_id)
        event = self.store.append(
            "autopilot_contract_paused",
            self._base_payload({"contract_id": contract_id, "paused": True}),
        )
        return event["payload"]

    def resume(self, contract_id: str) -> dict[str, Any]:
        self._require_contract(contract_id)
        event = self.store.append(
            "autopilot_contract_resumed",
            self._base_payload({"contract_id": contract_id, "paused": False}),
        )
        return event["payload"]

    def weekly_report(self, *, write_event: bool = False) -> dict[str, Any]:
        decisions = [event["payload"] for event in self._events("autopilot_paper_decision")]
        opportunities = [event["payload"] for event in self._events("autopilot_paper_opportunity")]
        resolved = [event["payload"] for event in self._events("autopilot_resolved_paper_outcome")]
        approvals = self.approvals()["approvals"]
        prospective_values = [float(item.get("conservative_expected_net_value") or 0.0) for item in decisions]
        realized_values = [float(item.get("realized_net_value") or item.get("paper_realized_net_value") or 0.0) for item in resolved]
        paper_value = round(sum(realized_values), 2)
        prospective_value = round(sum(prospective_values), 2)
        decision_count = len(decisions)
        no_action_count = sum(1 for item in decisions if item.get("selected_action") in {"abstain", "no_trade", "cash"})
        by_contract: dict[str, float] = {}
        by_strategy: dict[str, float] = {}
        by_source: dict[str, float] = {}
        prospective_by_contract: dict[str, float] = {}
        prospective_by_strategy: dict[str, float] = {}
        prospective_by_source: dict[str, float] = {}
        capital = 0.0
        for item in resolved:
            value = float(item.get("realized_net_value") or item.get("paper_realized_net_value") or 0.0)
            by_contract[item["contract_id"]] = round(by_contract.get(item["contract_id"], 0.0) + value, 2)
            by_strategy[str(item.get("strategy_id", "fixture"))] = round(by_strategy.get(str(item.get("strategy_id", "fixture")), 0.0) + value, 2)
            source_key = str(_redact_secrets(str(item.get("source_id", item.get("lab", "unknown")))))
            by_source[source_key] = round(
                by_source.get(source_key, 0.0) + value,
                2,
            )
        for item in decisions:
            value = float(item.get("conservative_expected_net_value") or 0.0)
            prospective_by_contract[item["contract_id"]] = round(prospective_by_contract.get(item["contract_id"], 0.0) + value, 2)
            prospective_by_strategy[str(item.get("strategy_id", "fixture"))] = round(
                prospective_by_strategy.get(str(item.get("strategy_id", "fixture")), 0.0) + value,
                2,
            )
            source_key = str(_redact_secrets(str(item.get("source_id", item.get("lab", "unknown")))))
            prospective_by_source[source_key] = round(
                prospective_by_source.get(source_key, 0.0) + value,
                2,
            )
            capital += float(item.get("capital_required") or 0.0)
        usage = self._usage()
        report = {
            "schema_version": "money_autopilot_weekly_report.v1",
            "portfolio_id": self.portfolio.portfolio_id,
            "realized_paper_value": paper_value,
            "seller_shadow_savings": round(sum(float(item.get("seller_shadow_value") or 0.0) for item in resolved), 2),
            "prospective_paper_pnl": prospective_value,
            "prospective_seller_shadow_value": round(sum(float(item.get("seller_shadow_value") or 0.0) for item in decisions), 2),
            "capital_hypothetically_at_risk": round(capital, 2),
            "maximum_drawdown": maximum_drawdown(_cumulative(realized_values))["maximum_drawdown"],
            "no_action_rate": round(no_action_count / decision_count, 6) if decision_count else 0.0,
            "calibration": {"available": False, "reason": "fixture_paper_runner_no_resolved_probability_sample"},
            "decision_count": decision_count,
            "value_by_contract": by_contract,
            "value_by_strategy": by_strategy,
            "value_by_source": by_source,
            "prospective_value_by_contract": prospective_by_contract,
            "prospective_value_by_strategy": prospective_by_strategy,
            "prospective_value_by_source": prospective_by_source,
            "llm_api_research_costs": usage,
            "maintenance_incidents": len(self._events("autopilot_task_failed")),
            "approvals_waiting": len(approvals),
            "paper_opportunity_count": len(opportunities),
            "notice": PAPER_NOTICE,
            "production_state": _production_state_flags(),
        }
        if write_event:
            event_hash = stable_hash(report)
            if not any(event["payload"].get("report_hash") == event_hash for event in self._events("autopilot_weekly_report")):
                self.store.append("autopilot_weekly_report", self._base_payload({**report, "report_hash": event_hash}))
        return report

    def _run_task(self, contract: ContractConfig, task_type: str, task_id: str) -> dict[str, Any]:
        if contract.source_config.get("simulate_failure") and task_type == "source_update":
            raise MoneyAutopilotError("simulated lab failure")
        if task_type == "source_health_audit":
            self._source_health(contract)
        if task_type == "source_scout_batch":
            self._source_scout(contract)
        if task_type == "connector_maintenance":
            self._connector_maintenance(contract)
        if task_type == "blind_evaluation":
            self._blind_evaluation(contract)
        if task_type == "prospective_update":
            self._prospective_update(contract)
        task_payload: dict[str, Any] = {"status": "completed"}
        if task_type == "paper_decision":
            task_payload = self._paper_decision(contract)
        if task_type == "outcome_resolution":
            task_payload = {"status": "completed", "resolved_outcomes": 0, "paper_only": True}
        if task_type in {"deterministic_candidate_search", "llm_scientist_skeptic_batch", "tuning_selection"}:
            task_payload = {"status": "completed", "research_only": True, "selected_before_blind": task_type == "tuning_selection"}
        event = self.store.append(
            "autopilot_task_completed",
            self._base_payload(
                {
                    "task_id": task_id,
                    "contract_id": contract.contract_id,
                    "task_type": task_type,
                    "task_result": task_payload,
                    "paper_only": True,
                    "production_state_mutated": False,
                }
            ),
        )
        return event["payload"]

    def _source_health(self, contract: ContractConfig) -> None:
        credential_blocked = bool(contract.source_config.get("requires_credential") and not contract.source_config.get("credential_available"))
        if credential_blocked:
            self._request_approval(contract, "missing_credential", "Configured source needs a credential before source update.")
        self.store.append(
            "autopilot_source_health",
            self._base_payload(
                {
                    "contract_id": contract.contract_id,
                    "source_id": _redact_secrets(contract.source_config.get("source_id", contract.lab)),
                    "healthy": not contract.source_config.get("simulate_failure") and not credential_blocked,
                    "paper_only": True,
                }
            ),
        )

    def _source_scout(self, contract: ContractConfig) -> None:
        if contract.source_config.get("license_status") in {"unclear", "unknown"}:
            self._request_approval(contract, "unclear_license", "Source licensing is unclear.")
        if contract.source_config.get("paid_source"):
            self._request_approval(contract, "paid_source", "Paid source requires approval.")
        if contract.source_config.get("private_data_ambiguity"):
            self._request_approval(contract, "private_data_ambiguity", "Private-data use is ambiguous.")
        if contract.source_config.get("production_source_promotion"):
            self._request_approval(contract, "production_source_promotion", "Production source promotion requires explicit approval.")
        equivalent_hash = stable_hash(
            {
                "contract_id": contract.contract_id,
                "source_config": contract.source_config,
                "task_type": "source_scout_batch",
            }
        )
        if any(event["payload"].get("equivalent_hash") == equivalent_hash for event in self._events("autopilot_source_research_completed")):
            return
        self.store.append(
            "autopilot_source_research_completed",
            self._base_payload(
                {
                    "contract_id": contract.contract_id,
                    "equivalent_hash": equivalent_hash,
                    "candidate_queue": [{"candidate_id": stable_hash(contract.to_dict())[:16], "determines_verdict": False}],
                }
            ),
        )

    def _connector_maintenance(self, contract: ContractConfig) -> None:
        evidence_hash = stable_hash(contract.source_config)
        connector_name = str(contract.source_config.get("connector", "fixture"))
        redacted_connector = _redact_secrets(connector_name)
        blocked = any(marker in connector_name.lower() for marker in CONNECTOR_DANGER_MARKERS)
        if blocked:
            self.store.append(
                "autopilot_connector_attempt_failed",
                self._base_payload(
                    {
                        "contract_id": contract.contract_id,
                        "evidence_hash": evidence_hash,
                        "connector": redacted_connector,
                        "reason": "malicious_connector_blocked",
                        "changed_evidence_required_before_retry": True,
                    }
                ),
            )
            return
        self.store.append(
            "autopilot_connector_attempt_completed",
            self._base_payload({"contract_id": contract.contract_id, "evidence_hash": evidence_hash, "connector": redacted_connector}),
        )

    def _blind_evaluation(self, contract: ContractConfig) -> None:
        self.store.append(
            "autopilot_blind_evaluation_consumed",
            self._base_payload(
                {
                    "contract_id": contract.contract_id,
                    "consumed_once": True,
                    "repeat_allowed": False,
                    "program_hash": stable_hash({"contract": contract.to_dict(), "program": "fixture_frozen_v1"}),
                }
            ),
        )

    def _prospective_update(self, contract: ContractConfig) -> None:
        previous = self._latest_event_payload("autopilot_prospective_incubation", contract.contract_id)
        event_count = int((previous or {}).get("unseen_episode_count", 0)) + 1
        self.store.append(
            "autopilot_prospective_incubation",
            self._base_payload(
                {
                    "contract_id": contract.contract_id,
                    "unseen_episode_count": event_count,
                    "refit_performed": False,
                    "frozen_program_hash": stable_hash({"contract": contract.to_dict(), "program": "fixture_frozen_v1"}),
                }
            ),
        )

    def _paper_decision(self, contract: ContractConfig) -> dict[str, Any]:
        if contract.source_config.get("force_no_signal"):
            decision = {
                "contract_id": contract.contract_id,
                "lab": contract.lab,
                "selected_action": "no_trade",
                "capital_required": 0.0,
                "maximum_possible_loss": 0.0,
                "conservative_expected_net_value": 0.0,
                "decision_id": stable_hash({"contract": contract.contract_id, "no_signal": True})[:16],
                "strategy_id": "no_signal",
                "source_id": contract.lab,
                "seller_shadow_value": 0.0,
                "paper_only": True,
                "unknown_cost_basis_count": 0,
                "material_costs_known": True,
                "forecast_current": True,
                "liquidity_capacity_ok": True,
                "deadline_open": True,
                "action_mode": "reactive",
            }
        elif contract.lab == "offerlab_seller_pilot":
            decision = self._run_offerlab_decision(contract)
        elif contract.lab == "weather_edge":
            decision = self._run_weather_decision(contract)
        elif contract.lab == "etf_risk":
            decision = self._run_etf_decision(contract)
        else:
            raise MoneyAutopilotError(f"unsupported lab: {contract.lab}")
        self.store.append("autopilot_paper_decision", self._base_payload(decision))
        if self._opportunity_allowed(decision, contract):
            self.store.append(
                "autopilot_paper_opportunity",
                self._base_payload(
                    {
                        **decision,
                        "notice": PAPER_NOTICE,
                        "real_action_executed": False,
                        "notification_kind": "paper_opportunity",
                    }
                ),
            )
        return decision

    def _run_offerlab_decision(self, contract: ContractConfig) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="offerlab_autopilot_") as tmp:
            root = Path(tmp)
            source = root / "seller_input"
            source.mkdir(parents=True, exist_ok=True)
            _write_offerlab_fixture(source, missing_cost=bool(contract.source_config.get("unknown_cost_basis")))
            data_root = root / "seller_data"
            import_pilot(source, data_root=data_root, pilot_id=contract.contract_id)
            money_root = self.root / "money" / contract.contract_id
            result = offerlab_evaluate(
                contract.contract_id,
                data_root=data_root,
                money_root=money_root,
                evaluation_timestamp=utc_now(),
            )
        storage = MoneyStorage(self.root / "money" / contract.contract_id)
        entries = storage.ledger.latest_entries()
        eligible = [entry for entry in entries if entry.conservative_expected_net_value is not None]
        entry = eligible[0] if eligible else entries[0]
        return {
            "contract_id": contract.contract_id,
            "lab": contract.lab,
            "selected_action": entry.selected_action,
            "capital_required": entry.capital_required,
            "maximum_possible_loss": entry.maximum_possible_loss,
            "conservative_expected_net_value": entry.conservative_expected_net_value or 0.0,
            "decision_id": entry.decision_id,
            "strategy_id": entry.provenance.get("strategy_id", "offerlab_shadow"),
            "source_id": entry.provenance.get("source_id", "offerlab_seller_pilot"),
            "seller_shadow_value": entry.conservative_expected_net_value or 0.0,
            "paper_only": result["paper_only"],
            "unknown_cost_basis_count": result["unknown_cost_basis_count"],
            "material_costs_known": int(result["unknown_cost_basis_count"]) == 0,
            "forecast_current": True,
            "liquidity_capacity_ok": True,
            "deadline_open": True,
            "action_mode": "reactive",
        }

    def _run_weather_decision(self, contract: ContractConfig) -> dict[str, Any]:
        storage_root = self.root / "money" / contract.contract_id
        result = weather_edge_paper_cycle(_weather_provider(no_trade=bool(contract.source_config.get("force_no_action"))), storage_root, as_of="2026-07-01T08:00:00-04:00")
        entry = MoneyStorage(storage_root).ledger.latest_entries()[0]
        return {
            "contract_id": contract.contract_id,
            "lab": contract.lab,
            "selected_action": entry.selected_action,
            "capital_required": entry.capital_required,
            "maximum_possible_loss": entry.maximum_possible_loss,
            "conservative_expected_net_value": entry.conservative_expected_net_value or 0.0,
            "decision_id": entry.decision_id,
            "strategy_id": entry.provenance.get("strategy_id", "weather_edge"),
            "source_id": entry.provenance.get("source_id", "weather_edge"),
            "seller_shadow_value": 0.0,
            "paper_only": result["paper_only"],
            "unknown_cost_basis_count": 0,
            "material_costs_known": True,
            "forecast_current": True,
            "liquidity_capacity_ok": True,
            "deadline_open": True,
            "action_mode": "reactive",
        }

    def _run_etf_decision(self, contract: ContractConfig) -> dict[str, Any]:
        provider, sessions = fixture_etf_provider(session_count=int(contract.source_config.get("session_count", 90)))
        ledger_path = self.root / "money" / contract.contract_id / "money.jsonl"
        result = etf_paper_cycle(
            provider,
            ledger_path=str(ledger_path),
            config=ETFRiskConfig(min_history_trading_days=int(contract.source_config.get("min_history_days", 35))),
            decision_cutoff=f"{sessions[-2]}T21:10:00+00:00",
        )
        entry = MoneyLedger(str(ledger_path)).latest_entries()[0]
        return {
            "contract_id": contract.contract_id,
            "lab": contract.lab,
            "selected_action": entry.selected_action,
            "capital_required": entry.capital_required,
            "maximum_possible_loss": entry.maximum_possible_loss,
            "conservative_expected_net_value": entry.conservative_expected_net_value or 0.0,
            "decision_id": entry.decision_id,
            "strategy_id": entry.provenance.get("strategy_id", "etf_risk"),
            "source_id": entry.provenance.get("source_id", "etf_risk"),
            "seller_shadow_value": 0.0,
            "paper_only": result["paper_only"],
            "unknown_cost_basis_count": 0,
            "material_costs_known": True,
            "forecast_current": True,
            "liquidity_capacity_ok": True,
            "deadline_open": True,
            "action_mode": "reactive",
        }

    def _request_approval(self, contract: ContractConfig, reason: str, message: str) -> None:
        if reason not in APPROVAL_REASONS:
            raise MoneyAutopilotError(f"approval reason is not allowed: {reason}")
        approval_id = self._approval_id(contract, reason)
        if any(event["payload"].get("approval_id") == approval_id for event in self._events("autopilot_approval_requested")):
            return
        if self._budget_status()["approvals_per_week"]["remaining"] <= 0:
            if not any(event["payload"].get("approval_id") == approval_id for event in self._events("autopilot_approval_suppressed")):
                self.store.append(
                    "autopilot_approval_suppressed",
                    self._base_payload(
                        {
                            "approval_id": approval_id,
                            "contract_id": contract.contract_id,
                            "reason": reason,
                            "message": message,
                            "budget": "approvals_per_week",
                            "real_action_executed": False,
                        }
                    ),
                )
            return
        self.store.append(
            "autopilot_approval_requested",
            self._base_payload(
                {
                    "approval_id": approval_id,
                    "contract_id": contract.contract_id,
                    "reason": reason,
                    "message": message,
                    "real_action_executed": False,
                }
            ),
        )

    def _task_id(self, contract: ContractConfig, task_type: str) -> str:
        evidence_hash = stable_hash(contract.source_config) if task_type in {"source_scout_batch", "connector_maintenance"} else "fixed"
        return stable_hash(
            {
                "schema_version": AUTOPILOT_SCHEMA_VERSION,
                "portfolio_hash": self.portfolio.portfolio_hash(),
                "contract_id": contract.contract_id,
                "task_type": task_type,
                "evidence_hash": evidence_hash,
            }
        )[:24]

    def _global_task_id(self, task_type: str) -> str:
        return stable_hash(
            {
                "schema_version": AUTOPILOT_SCHEMA_VERSION,
                "portfolio_hash": self.portfolio.portfolio_hash(),
                "contract_id": "portfolio",
                "task_type": task_type,
            }
        )[:24]

    def _base_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _redact_secrets({
            "schema_version": AUTOPILOT_SCHEMA_VERSION,
            "portfolio_id": self.portfolio.portfolio_id,
            "portfolio_hash": self.portfolio.portfolio_hash(),
            **payload,
        })

    def _events(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.store.all_events() if event.get("event_type") == event_type]

    def _task_completed(self, task_id: str) -> bool:
        return any(event["payload"].get("task_id") == task_id for event in self._events("autopilot_task_completed"))

    def _completed_task_type(self, contract_id: str, task_type: str) -> bool:
        return any(
            event["payload"].get("contract_id") == contract_id and event["payload"].get("task_type") == task_type
            for event in self._events("autopilot_task_completed")
        )

    def _completed_task_count(self, contract_id: str) -> int:
        return sum(1 for event in self._events("autopilot_task_completed") if event["payload"].get("contract_id") == contract_id)

    def _failed_connector_without_changed_evidence(self, contract: ContractConfig) -> bool:
        evidence_hash = stable_hash(contract.source_config)
        return any(
            event["payload"].get("contract_id") == contract.contract_id
            and event["payload"].get("evidence_hash") == evidence_hash
            and event["payload"].get("changed_evidence_required_before_retry")
            for event in self._events("autopilot_connector_attempt_failed")
        )

    def _budget_block_reason(self, task_type: str) -> str | None:
        if task_type not in RESEARCH_TASKS:
            return None
        status = self._budget_status()
        if task_type == "llm_scientist_skeptic_batch" and status["llm_monthly_cost_usd"]["remaining"] <= 0:
            return "research_budget_exhausted"
        if task_type == "source_scout_batch" and status["web_searches"]["remaining"] <= 0:
            return "research_budget_exhausted"
        if task_type == "connector_maintenance" and status["connector_attempts"]["remaining"] <= 0:
            return "research_budget_exhausted"
        if task_type in {"deterministic_candidate_search", "tuning_selection", "blind_evaluation"} and status["candidate_evaluations"]["remaining"] <= 0:
            return "research_budget_exhausted"
        return None

    def _budget_status(self) -> dict[str, dict[str, float]]:
        usage = self._usage()
        limits = {
            "llm_monthly_cost_usd": float(self.portfolio.budgets.get("llm_monthly_cost_usd", 10.0)),
            "web_searches": float(self.portfolio.budgets.get("web_searches", 25.0)),
            "connector_attempts": float(self.portfolio.budgets.get("connector_attempts", 10.0)),
            "candidate_evaluations": float(self.portfolio.budgets.get("candidate_evaluations", 100.0)),
            "max_concurrency": float(self.portfolio.budgets.get("max_concurrency", 1.0)),
            "alerts_per_day": float(self.portfolio.budgets.get("alerts_per_day", 3.0)),
            "approvals_per_week": float(self.portfolio.budgets.get("approvals_per_week", 10.0)),
        }
        used = {
            "llm_monthly_cost_usd": usage["llm_cost_usd"],
            "web_searches": usage["web_searches"],
            "connector_attempts": usage["connector_attempts"],
            "candidate_evaluations": usage["candidate_evaluations"],
            "max_concurrency": 1.0,
            "alerts_per_day": len(self._events("autopilot_paper_opportunity")),
            "approvals_per_week": len(self._events("autopilot_approval_requested")),
        }
        return {key: {"limit": value, "used": used[key], "remaining": max(0.0, value - used[key])} for key, value in limits.items()}

    def _usage(self) -> dict[str, float]:
        completed = [event["payload"] for event in self._events("autopilot_task_completed")]
        return {
            "llm_cost_usd": round(0.01 * sum(1 for item in completed if item.get("task_type") == "llm_scientist_skeptic_batch"), 2),
            "api_cost_usd": 0.0,
            "research_cost_usd": round(0.01 * sum(1 for item in completed if item.get("task_type") in RESEARCH_TASKS), 2),
            "web_searches": float(sum(1 for item in completed if item.get("task_type") == "source_scout_batch")),
            "connector_attempts": float(len(self._events("autopilot_connector_attempt_completed")) + len(self._events("autopilot_connector_attempt_failed"))),
            "candidate_evaluations": float(sum(1 for item in completed if item.get("task_type") in {"deterministic_candidate_search", "tuning_selection", "blind_evaluation"})),
        }

    def _is_paused(self, contract_id: str) -> bool:
        state = False
        for event in self.store.all_events():
            payload = event["payload"]
            if payload.get("contract_id") != contract_id:
                continue
            if event["event_type"] == "autopilot_contract_paused":
                state = True
            if event["event_type"] == "autopilot_contract_resumed":
                state = False
        return state

    def _approval_resolved(self, approval_id: str) -> bool:
        return any(event["payload"].get("approval_id") == approval_id for event in self._events("autopilot_approval_resolved"))

    def _approval_id(self, contract: ContractConfig, reason: str) -> str:
        return stable_hash({"portfolio": self.portfolio.portfolio_hash(), "contract": contract.contract_id, "reason": reason})[:16]

    def _approval_reason_resolved(self, contract: ContractConfig, reason: str) -> bool:
        return self._approval_resolved(self._approval_id(contract, reason))

    def _waiting_approvals(self, contract_id: str) -> list[dict[str, Any]]:
        return [
            event["payload"]
            for event in self._events("autopilot_approval_requested")
            if event["payload"].get("contract_id") == contract_id and not self._approval_resolved(event["payload"]["approval_id"])
        ]

    def _source_approval_blockers(self, contract: ContractConfig) -> list[str]:
        checks = {
            "missing_credential": bool(contract.source_config.get("requires_credential") and not contract.source_config.get("credential_available")),
            "unclear_license": contract.source_config.get("license_status") in {"unclear", "unknown"},
            "paid_source": bool(contract.source_config.get("paid_source")),
            "private_data_ambiguity": bool(contract.source_config.get("private_data_ambiguity")),
            "production_source_promotion": bool(contract.source_config.get("production_source_promotion")),
        }
        return [reason for reason, blocked in checks.items() if blocked and not self._approval_reason_resolved(contract, reason)]

    def _latest_event_payload(self, event_type: str, contract_id: str) -> dict[str, Any] | None:
        match = None
        for event in self._events(event_type):
            if event["payload"].get("contract_id") == contract_id:
                match = event["payload"]
        return match

    def _opportunity_allowed(self, decision: dict[str, Any], contract: ContractConfig) -> bool:
        if not _is_notifiable(decision, contract):
            return False
        if self._duplicate_alert(decision) or self._alert_budget_exhausted():
            return False
        if not decision.get("paper_only"):
            return False
        if self._waiting_approvals(contract.contract_id) or self._source_approval_blockers(contract):
            return False
        source_health = self._latest_event_payload("autopilot_source_health", contract.contract_id)
        if not source_health or source_health.get("healthy") is not True:
            return False
        if not self._latest_event_payload("autopilot_blind_evaluation_consumed", contract.contract_id):
            return False
        prospective = self._latest_event_payload("autopilot_prospective_incubation", contract.contract_id)
        min_episodes = int(contract.prospective_requirements.get("min_unseen_episodes", 1) or 1)
        if not prospective or int(prospective.get("unseen_episode_count", 0)) < min_episodes:
            return False
        if prospective.get("refit_performed") is not False:
            return False
        if int(decision.get("unknown_cost_basis_count", 0) or 0) > 0:
            return False
        if decision.get("material_costs_known") is not True:
            return False
        if decision.get("forecast_current") is not True:
            return False
        if decision.get("liquidity_capacity_ok") is not True:
            return False
        if decision.get("deadline_open") is not True:
            return False
        if decision.get("action_mode") != "reactive":
            return False
        return True

    def _duplicate_alert(self, decision: dict[str, Any]) -> bool:
        key = (decision.get("contract_id"), decision.get("decision_id"), decision.get("selected_action"))
        return any(
            (event["payload"].get("contract_id"), event["payload"].get("decision_id"), event["payload"].get("selected_action")) == key
            for event in self._events("autopilot_paper_opportunity")
        )

    def _alert_budget_exhausted(self) -> bool:
        return self._budget_status()["alerts_per_day"]["remaining"] <= 0

    def _require_contract(self, contract_id: str) -> ContractConfig:
        for contract in self.portfolio.contracts:
            if contract.contract_id == contract_id:
                return contract
        raise MoneyAutopilotError(f"unknown contract_id: {contract_id}")


def load_portfolio(path: str | Path) -> PortfolioConfig:
    payload = _load_mapping(path)
    state_dir = str(payload.get("state_dir") or Path(path).resolve().parent / ".money_autopilot")
    contracts = [ContractConfig(**_normalize_contract(item)) for item in payload.get("contracts", [])]
    return PortfolioConfig(
        portfolio_id=str(payload.get("portfolio_id", "money-lab")),
        state_dir=state_dir,
        contracts=contracts,
        budgets={key: float(value) for key, value in dict(payload.get("budgets", {})).items()},
        alert_threshold=float(payload.get("alert_threshold", 0.0)),
    )


def _normalize_contract(item: dict[str, Any]) -> dict[str, Any]:
    output = dict(item)
    output["paper_capital_limit"] = float(output.get("paper_capital_limit", 0.0))
    output["alert_threshold"] = float(output.get("alert_threshold", 0.0))
    for key in ("target_inputs", "source_config", "research_budget", "schedule", "evidence_thresholds", "prospective_requirements"):
        output[key] = dict(output.get(key, {}))
    output.setdefault("provider", output.get("source_config", {}).get("provider", "fixture"))
    output.setdefault("enabled", True)
    return output


def _load_mapping(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]

            payload = yaml.safe_load(text)
        except Exception:
            payload = _parse_simple_yaml(text)
    if not isinstance(payload, dict):
        raise MoneyAutopilotError("portfolio file must contain an object")
    return payload


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None
    current_nested: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = _parse_scalar(value)
                current_key = None
            else:
                result[key] = [] if key == "contracts" else {}
                current_key = key
            continue
        stripped = line.strip()
        if current_key == "contracts" and indent == 2 and stripped.startswith("- "):
            current_item = {}
            current_nested = None
            result["contracts"].append(current_item)
            body = stripped[2:]
            if body and ":" in body:
                key, value = body.split(":", 1)
                current_item[key.strip()] = _parse_scalar(value.strip())
            continue
        if current_key == "contracts" and current_item is not None and indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                current_item[key] = _parse_scalar(value)
                current_nested = None
            else:
                current_item[key] = {}
                current_nested = key
            continue
        if current_key == "contracts" and current_item is not None and current_nested and indent >= 6 and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[current_nested][key.strip()] = _parse_scalar(value.strip())
            continue
        if current_key and isinstance(result.get(current_key), dict) and ":" in stripped:
            key, value = stripped.split(":", 1)
            result[current_key][key.strip()] = _parse_scalar(value.strip())
            continue
        raise MoneyAutopilotError(f"unsupported portfolio YAML line: {raw_line}")
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _is_notifiable(decision: dict[str, Any], contract: ContractConfig) -> bool:
    if decision.get("selected_action") in {"abstain", "no_trade", "cash"}:
        return False
    if float(decision.get("conservative_expected_net_value") or 0.0) < float(contract.alert_threshold):
        return False
    if float(decision.get("capital_required") or 0.0) > float(contract.paper_capital_limit or 0.0):
        return False
    return True


def _reject_real_action_config(value: Any) -> None:
    if isinstance(value, dict):
        sanitized = {key: child for key, child in value.items() if key != "connector"}
    else:
        sanitized = value
    lowered = json.dumps(sanitized, sort_keys=True).lower() if sanitized else ""
    if any(marker in lowered for marker in REAL_ACTION_MARKERS):
        raise MoneyAutopilotError("portfolio config may not request real actions")


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).lower()
            redacted_key = _redact_secret_text(str(key))
            if any(marker in key_text for marker in SECRET_KEY_MARKERS):
                output[redacted_key] = "[REDACTED]"
            else:
                output[redacted_key] = _redact_secrets(child)
        return output
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if not isinstance(value, str):
        return value
    return _redact_secret_text(value)


def _redact_secret_text(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in SECRET_VALUE_MARKERS):
        return "[REDACTED]"
    if re.search(r"(?i)(auth|authorization|bearer|credential|key|password|secret|session|sig|signature|token)=", value):
        value = re.sub(
            r"(?i)((?:[?&]|^)(?:api[_-]?key|auth|authorization|credential|credentials|key|map[_-]?key|password|secret|session|sig|signature|token|access[_-]?token)=)[^&\s]+",
            r"\1[REDACTED]",
            value,
        )
    value = re.sub(
        r"(?i)((?:authorization|auth|api[_-]?key|x-api-key|token)\s*:\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", value)
    return value


def _production_state_flags() -> dict[str, bool]:
    return {
        "seller_mutation": False,
        "exchange_authentication": False,
        "exchange_order_submission": False,
        "brokerage_connection": False,
        "brokerage_order_submission": False,
        "notifications": False,
        "real_financial_action": False,
    }


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    output = []
    for value in values:
        total += value
        output.append(round(total, 2))
    return output


def _write_offerlab_fixture(root: Path, *, missing_cost: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = "2026-01-01T12:00:00+00:00"
    available = "2026-01-01T13:00:00+00:00"
    paid = "2026-01-01T15:00:00+00:00"
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
            }
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
            }
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
                "return_window_matured_at": "2026-02-15T00:00:00+00:00",
                "quantity": "1",
            }
        ],
        "fees": [
            {"fee_id": "fee_001", "order_id": "order_001", "event_time": paid, "available_at": paid, "fee_amount": "12.00", "currency": "USD", "fee_type": "final_value"}
        ],
        "shipping_costs": [
            {"shipping_id": "ship_001", "order_id": "order_001", "event_time": paid, "available_at": paid, "shipping_cost_amount": "8.00", "currency": "USD"}
        ],
        "cost_basis": []
        if missing_cost
        else [
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
            {"inventory_id": "inventory_001", "listing_id": "listing_001", "event_time": base, "available_at": available, "quantity_available": "1", "inventory_age_days": "45"}
        ],
        "traffic": [
            {"traffic_id": "traffic_001", "listing_id": "listing_001", "event_time": base, "available_at": available, "impressions": "10", "views": "2"}
        ],
    }
    headers = {
        "listings": ["listing_id", "event_time", "available_at", "asking_price_amount", "currency", "category", "listing_status"],
        "offers": ["offer_id", "listing_id", "event_time", "available_at", "offer_amount", "currency", "offer_state", "seller_response", "seller_response_time", "seller_response_amount", "decision_history_available_at", "expires_at"],
        "orders": ["order_id", "listing_id", "offer_id", "event_time", "available_at", "sale_price_amount", "currency", "order_status", "paid_at", "completed_at", "return_window_matured_at", "quantity"],
        "fees": ["fee_id", "order_id", "event_time", "available_at", "fee_amount", "currency", "fee_type"],
        "shipping_costs": ["shipping_id", "order_id", "event_time", "available_at", "shipping_cost_amount", "currency"],
        "cost_basis": ["cost_basis_id", "listing_id", "event_time", "available_at", "unit_cost_amount", "currency", "sku", "cost_source"],
        "cancellations_unpaid": ["cancellation_id", "event_time", "available_at", "event_type", "currency", "order_id", "listing_id", "offer_id", "amount"],
        "returns_refunds": ["return_id", "order_id", "event_time", "available_at", "refund_amount", "currency", "listing_id", "return_opened_at", "return_closed_at", "return_window_matured_at", "return_status"],
        "inventory": ["inventory_id", "listing_id", "event_time", "available_at", "quantity_available", "inventory_age_days"],
        "traffic": ["traffic_id", "listing_id", "event_time", "available_at", "impressions", "views"],
    }
    for dataset, fieldnames in headers.items():
        with (root / f"{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[dataset])


def _weather_provider(*, no_trade: bool = False) -> FixtureWeatherEdgeProvider:
    event_a = _weather_event("nyc-20260701-85-90", TemperatureBracket("85-90", 85.0, 90.0))
    event_b = _weather_event("nyc-20260701-90-95", TemperatureBracket("90-95", 90.0, 95.0))
    first_ask = 0.95 if no_trade else 0.55
    return FixtureWeatherEdgeProvider(
        events=[event_a, event_b],
        market_depths=[_weather_depth(event_a.event_id, first_ask), _weather_depth(event_b.event_id, 0.65)],
        weather_snapshots=[_weather_snapshot(event_a.event_id), _weather_snapshot(event_b.event_id)],
        station_history=[
            StationHistoricalDay(station_id="KNYC", local_date="2026-06-26", high_f=86.0, forecast_mean_f=85.0, settlement_series="NOAA_DAILY_HIGH", report_source="CLI", regime="heat"),
            StationHistoricalDay(station_id="KNYC", local_date="2026-06-27", high_f=87.0, forecast_mean_f=86.0, settlement_series="NOAA_DAILY_HIGH", report_source="CLI", regime="heat"),
            StationHistoricalDay(station_id="KNYC", local_date="2026-06-28", high_f=88.0, forecast_mean_f=87.0, settlement_series="NOAA_DAILY_HIGH", report_source="CLI", regime="heat"),
            StationHistoricalDay(station_id="KNYC", local_date="2026-06-29", high_f=91.0, forecast_mean_f=90.0, settlement_series="NOAA_DAILY_HIGH", report_source="CLI", regime="heat"),
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
