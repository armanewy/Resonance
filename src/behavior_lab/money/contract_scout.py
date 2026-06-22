from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from behavior_lab.core import stable_hash, to_jsonable, utc_now
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


CONTRACT_SCOUT_SCHEMA_VERSION = "money_contract_scout.v1"
DEFAULT_STATE_DIR = ".money_contract_scout"

CONTRACT_FAMILIES = {
    "weather_event_market",
    "broad_etf_risk",
    "seller_shadow",
    "purchase_timing",
    "compute_cost_avoidance",
    "energy_load_shift",
    "internet_failover",
}

APPROVAL_REASONS = {
    "missing_credential",
    "unclear_license",
    "paid_source",
    "private_data_ambiguity",
    "production_source_promotion",
    "proposed_real_action",
}

REJECTION_REASONS = {
    "ambiguous_resolution",
    "duplicate_prior_proposal",
    "high_maintenance_low_value",
    "missing_available_actions",
    "missing_no_action",
    "private_data_dependency_without_acquisition_path",
    "proposed_real_action",
    "real_account_mutation_required",
    "unbounded_capital_requirement",
    "unbounded_loss",
    "unknown_material_costs",
    "unrepresented_material_costs",
    "unusable_payoff",
    "unusable_source_coverage",
}

REAL_ACTION_MARKERS = (
    "accept_offer",
    "broker",
    "buy_order",
    "counteroffer",
    "exchange_order",
    "live_order",
    "make_offer",
    "market_order",
    "order_submission",
    "place_order",
    "place_trade",
    "purchase",
    "sell_order",
    "seller_mutation",
    "submit_market_order",
    "submit_offer",
    "submit_order",
    "trade_live",
    "transfer",
)

SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "credential",
    "key",
    "password",
    "secret",
    "session",
    "sig",
    "signature",
    "token",
)

SECRET_VALUE_MARKERS = ("sk-", "api_key=", "apikey=", "password=", "secret=", "token=")


class ContractScoutError(ValueError):
    pass


@dataclass(frozen=True)
class OpportunityContractProposal:
    proposal_id: str
    title: str
    contract_family: str
    outcome: str
    resolution_source: dict[str, Any]
    resolution_cadence: str
    decision_deadline: str
    available_actions: list[dict[str, Any]]
    no_action_alternative: str
    payoff_formula: dict[str, Any]
    material_costs: list[dict[str, Any]]
    capital_requirement: dict[str, Any]
    maximum_possible_loss: dict[str, Any]
    required_source_families: list[str]
    currently_available_sources: list[str]
    missing_sources: list[str]
    historical_depth: dict[str, Any]
    prospective_duration_required: str
    expected_decision_frequency: str
    paper_mode_feasibility: dict[str, Any]
    platform_regulatory_dependencies: list[str] = field(default_factory=list)
    credential_requirements: list[str] = field(default_factory=list)
    licensing_concerns: list[str] = field(default_factory=list)
    estimated_research_cost: dict[str, Any] = field(default_factory=dict)
    estimated_maintenance_burden: str = "medium"
    expected_information_value: str = "medium"
    reason_it_may_fail: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    status: str = "proposed"

    def __post_init__(self) -> None:
        for field_name in (
            "proposal_id",
            "title",
            "contract_family",
            "outcome",
            "resolution_cadence",
            "decision_deadline",
            "no_action_alternative",
            "prospective_duration_required",
            "expected_decision_frequency",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ContractScoutError(f"{field_name} must be a non-empty string")
        if self.contract_family not in CONTRACT_FAMILIES:
            raise ContractScoutError(f"unsupported contract_family: {self.contract_family}")
        if not isinstance(self.resolution_source, dict) or not self.resolution_source:
            raise ContractScoutError("resolution_source must be a non-empty object")
        if not self.available_actions:
            raise ContractScoutError("available_actions may not be empty")
        if not isinstance(self.payoff_formula, dict) or not self.payoff_formula:
            raise ContractScoutError("payoff_formula must be a non-empty object")
        if not isinstance(self.capital_requirement, dict) or not self.capital_requirement:
            raise ContractScoutError("capital_requirement must be a non-empty object")
        if not isinstance(self.maximum_possible_loss, dict) or not self.maximum_possible_loss:
            raise ContractScoutError("maximum_possible_loss must be a non-empty object")
        if not isinstance(self.paper_mode_feasibility, dict) or not self.paper_mode_feasibility:
            raise ContractScoutError("paper_mode_feasibility must be a non-empty object")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpportunityContractProposal":
        if not isinstance(payload, dict):
            raise ContractScoutError("proposal must be an object")
        return cls(
            proposal_id=str(payload.get("proposal_id", "")),
            title=str(payload.get("title", "")),
            contract_family=str(payload.get("contract_family", "")),
            outcome=str(payload.get("outcome", "")),
            resolution_source=dict(payload.get("resolution_source", {})),
            resolution_cadence=str(payload.get("resolution_cadence", "")),
            decision_deadline=str(payload.get("decision_deadline", "")),
            available_actions=list(payload.get("available_actions", [])),
            no_action_alternative=str(payload.get("no_action_alternative", "")),
            payoff_formula=dict(payload.get("payoff_formula", {})),
            material_costs=list(payload.get("material_costs", [])),
            capital_requirement=dict(payload.get("capital_requirement", {})),
            maximum_possible_loss=dict(payload.get("maximum_possible_loss", {})),
            required_source_families=list(payload.get("required_source_families", [])),
            currently_available_sources=list(payload.get("currently_available_sources", [])),
            missing_sources=list(payload.get("missing_sources", [])),
            historical_depth=dict(payload.get("historical_depth", {})),
            prospective_duration_required=str(payload.get("prospective_duration_required", "")),
            expected_decision_frequency=str(payload.get("expected_decision_frequency", "")),
            paper_mode_feasibility=dict(payload.get("paper_mode_feasibility", {})),
            platform_regulatory_dependencies=list(payload.get("platform_regulatory_dependencies", [])),
            credential_requirements=list(payload.get("credential_requirements", [])),
            licensing_concerns=list(payload.get("licensing_concerns", [])),
            estimated_research_cost=dict(payload.get("estimated_research_cost", {})),
            estimated_maintenance_burden=str(payload.get("estimated_maintenance_burden", "medium")),
            expected_information_value=str(payload.get("expected_information_value", "medium")),
            reason_it_may_fail=list(payload.get("reason_it_may_fail", [])),
            citations=list(payload.get("citations", [])),
            status=str(payload.get("status", "proposed")),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def proposal_hash(self) -> str:
        return stable_hash({k: v for k, v in self.to_dict().items() if k != "status"})

    def equivalence_key(self) -> str:
        action_ids = sorted(str(action.get("action_id", "")) for action in self.available_actions)
        return stable_hash(
            {
                "contract_family": self.contract_family,
                "outcome": self.outcome.strip().lower(),
                "resolution_source": str(self.resolution_source.get("source_id") or self.resolution_source.get("publisher") or "").lower(),
                "resolution_cadence": self.resolution_cadence.strip().lower(),
                "decision_deadline": self.decision_deadline.strip().lower(),
                "available_actions": action_ids,
                "no_action_alternative": self.no_action_alternative.strip().lower(),
            }
        )


@dataclass(frozen=True)
class ValidationResult:
    status: str
    eligible_for_experimental_portfolio: bool
    reasons: list[str]
    approval_required: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class ContractValidator:
    def validate(self, proposal: OpportunityContractProposal) -> ValidationResult:
        reasons: list[str] = []
        approvals: list[str] = []
        capital_amount = _nonnegative_number(proposal.capital_requirement.get("amount"))
        maximum_loss_amount = _nonnegative_number(proposal.maximum_possible_loss.get("amount"))
        checks = {
            "paper_only": proposal.paper_mode_feasibility.get("paper_only") is True,
            "outcome_unambiguous": bool(proposal.outcome.strip()) and proposal.resolution_source.get("ambiguous") is not True,
            "actions_defined": bool(proposal.available_actions),
            "no_action_defined": proposal.no_action_alternative in _action_ids(proposal.available_actions),
            "payoff_executable": proposal.payoff_formula.get("executable") is True and bool(proposal.payoff_formula.get("formula")),
            "material_costs_represented": bool(proposal.material_costs) and all(cost.get("represented") is True for cost in proposal.material_costs),
            "no_unknown_material_costs": not any(cost.get("unknown") is True for cost in proposal.material_costs),
            "capital_requirement_bounded": proposal.capital_requirement.get("bounded") is True and capital_amount is not None,
            "maximum_loss_bounded": proposal.maximum_possible_loss.get("bounded") is True and maximum_loss_amount is not None,
            "source_coverage_usable": _source_coverage_usable(proposal),
            "no_real_action_shape": not _contains_real_action_shape(proposal.available_actions),
            "no_real_account_mutation": proposal.paper_mode_feasibility.get("real_account_mutation_required") is not True,
            "maintenance_value_reasonable": _maintenance_value_reasonable(proposal),
            "private_data_available_or_acquirable": _private_data_available_or_acquirable(proposal),
        }
        if not checks["paper_only"]:
            reasons.append("proposed_real_action")
        if not checks["outcome_unambiguous"]:
            reasons.append("ambiguous_resolution")
        if not checks["actions_defined"]:
            reasons.append("missing_available_actions")
        if not checks["no_action_defined"]:
            reasons.append("missing_no_action")
        if not checks["payoff_executable"]:
            reasons.append("unusable_payoff")
        if not checks["material_costs_represented"]:
            reasons.append("unrepresented_material_costs")
        if not checks["no_unknown_material_costs"]:
            reasons.append("unknown_material_costs")
        if not checks["capital_requirement_bounded"]:
            reasons.append("unbounded_capital_requirement")
        if not checks["maximum_loss_bounded"]:
            reasons.append("unbounded_loss")
        if not checks["source_coverage_usable"]:
            reasons.append("unusable_source_coverage")
        if not checks["no_real_action_shape"]:
            reasons.append("proposed_real_action")
        if not checks["no_real_account_mutation"]:
            reasons.append("real_account_mutation_required")
        if not checks["maintenance_value_reasonable"]:
            reasons.append("high_maintenance_low_value")
        if not checks["private_data_available_or_acquirable"]:
            reasons.append("private_data_dependency_without_acquisition_path")
        if any(_unclear_license(item) for item in proposal.licensing_concerns):
            approvals.append("unclear_license")
        if proposal.credential_requirements:
            approvals.append("missing_credential")
        if proposal.paper_mode_feasibility.get("requires_private_data") is True and not proposal.currently_available_sources:
            approvals.append("private_data_ambiguity")

        if reasons:
            status = "rejected"
        elif approvals:
            status = "approval_required"
        else:
            status = "eligible_experimental"
        return ValidationResult(
            status=status,
            eligible_for_experimental_portfolio=status == "eligible_experimental",
            reasons=reasons,
            approval_required=sorted(set(approvals)),
            checks=checks,
        )


class ContractScout:
    def __init__(self, state_dir: str | Path = DEFAULT_STATE_DIR, *, operations_state_dir: str | Path | None = None) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = AppendOnlyResearchStore(self.root / "contract_scout.jsonl")
        self.validator = ContractValidator()
        self.operations_state_dir = Path(operations_state_dir).resolve() if operations_state_dir else None

    def run(
        self,
        *,
        proposals: list[dict[str, Any]] | None = None,
        search_budget: int = 8,
        llm_budget_usd: float = 0.0,
        include_seed_families: bool = True,
    ) -> dict[str, Any]:
        if search_budget < 0:
            raise ContractScoutError("search_budget may not be negative")
        if llm_budget_usd < 0:
            raise ContractScoutError("llm_budget_usd may not be negative")
        prior_keys = {item["equivalence_key"] for item in self._proposal_records(include_rejected=True)}
        operations_context = self._operations_context()
        candidates = []
        if include_seed_families:
            candidates.extend(_seed_proposals())
        candidates.extend(proposals or [])
        candidates = candidates[:search_budget]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        approvals: list[dict[str, Any]] = []
        for raw in candidates:
            proposal = OpportunityContractProposal.from_dict(raw)
            key = proposal.equivalence_key()
            validation = self.validator.validate(proposal)
            payload = {
                "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
                "proposal": _redact_sensitive(proposal.to_dict()),
                "proposal_hash": proposal.proposal_hash(),
                "equivalence_key": key,
                "validation": validation.to_dict(),
                "operations_context_hash": stable_hash(operations_context),
                "paper_only": proposal.paper_mode_feasibility.get("paper_only") is True,
                "production_source_activation": False,
                "money_allocation": False,
                "generated_at": utc_now(),
            }
            if key in prior_keys:
                payload["validation"] = {
                    **payload["validation"],
                    "status": "duplicate",
                    "eligible_for_experimental_portfolio": False,
                    "reasons": sorted(set(payload["validation"].get("reasons", []) + ["duplicate_prior_proposal"])),
                }
                self.store.append("contract_scout_duplicate", payload)
                duplicates.append(payload)
                continue
            prior_keys.add(key)
            if validation.status == "eligible_experimental":
                event = self.store.append("contract_scout_proposal", payload)
                accepted.append(event["payload"])
            elif validation.status == "approval_required":
                event = self.store.append("contract_scout_approval_required", payload)
                approvals.append(event["payload"])
            else:
                event = self.store.append("contract_scout_rejected", payload)
                rejected.append(event["payload"])
        report = {
            "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
            "state_dir": str(self.root),
            "run_at": utc_now(),
            "search_budget": search_budget,
            "llm_budget_usd": llm_budget_usd,
            "llm_used": False,
            "operations_context": operations_context,
            "accepted": len(accepted),
            "approval_required": len(approvals),
            "rejected": len(rejected),
            "duplicates": len(duplicates),
            "paper_only": True,
            "production_source_activation": False,
            "money_allocation": False,
            "items": {
                "eligible": accepted,
                "approval_required": approvals,
                "rejected": rejected,
                "duplicates": duplicates,
            },
        }
        self.store.append("contract_scout_run", report)
        return report

    def proposals(self) -> dict[str, Any]:
        records = self._proposal_records(include_rejected=True)
        return {
            "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
            "state_dir": str(self.root),
            "proposals": records,
            "counts": _counts(records),
            "paper_only": True,
        }

    def approve(self, proposal_id: str) -> dict[str, Any]:
        record = self._find_proposal(proposal_id)
        validation = record["validation"]
        if validation.get("status") != "eligible_experimental":
            raise ContractScoutError(f"proposal {proposal_id!r} is not eligible for approval: {validation.get('status')}")
        payload = {
            "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "status": "approved_for_experimental_portfolio",
            "approved_at": utc_now(),
            "proposal_hash": record["proposal_hash"],
            "equivalence_key": record["equivalence_key"],
            "paper_only": True,
            "production_source_activation": False,
            "money_allocation": False,
        }
        event = self.store.append("contract_scout_approved", payload)
        return event["payload"]

    def reject(self, proposal_id: str, *, reason: str = "manual_rejection") -> dict[str, Any]:
        record = self._find_proposal(proposal_id)
        payload = {
            "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "status": "manually_rejected",
            "reason": reason,
            "rejected_at": utc_now(),
            "proposal_hash": record["proposal_hash"],
            "equivalence_key": record["equivalence_key"],
            "paper_only": True,
            "production_source_activation": False,
            "money_allocation": False,
        }
        event = self.store.append("contract_scout_manually_rejected", payload)
        return event["payload"]

    def report(self) -> dict[str, Any]:
        records = self._proposal_records(include_rejected=True)
        approved = [
            event["payload"]
            for event in self.store.all_events()
            if event.get("event_type") == "contract_scout_approved"
        ]
        rejected = [
            event["payload"]
            for event in self.store.all_events()
            if event.get("event_type") == "contract_scout_manually_rejected"
        ]
        return {
            "schema_version": CONTRACT_SCOUT_SCHEMA_VERSION,
            "state_dir": str(self.root),
            "operations_context": self._operations_context(),
            "proposal_counts": _counts(records),
            "approved_count": len(approved),
            "manual_rejection_count": len(rejected),
            "approval_inbox": [
                {
                    "proposal_id": item["proposal"]["proposal_id"],
                    "title": item["proposal"]["title"],
                    "reasons": item["validation"].get("approval_required", []),
                }
                for item in records
                if item["validation"].get("status") == "approval_required"
            ],
            "eligible_experimental_contracts": [
                {
                    "proposal_id": item["proposal"]["proposal_id"],
                    "title": item["proposal"]["title"],
                    "family": item["proposal"]["contract_family"],
                }
                for item in records
                if item["validation"].get("status") == "eligible_experimental"
            ],
            "paper_only": True,
            "production_source_activation": False,
            "money_allocation": False,
        }

    def verify(self) -> bool:
        return self.store.verify()

    def _proposal_records(self, *, include_rejected: bool) -> list[dict[str, Any]]:
        event_types = {"contract_scout_proposal", "contract_scout_approval_required"}
        if include_rejected:
            event_types.add("contract_scout_rejected")
            event_types.add("contract_scout_duplicate")
        records = [event["payload"] for event in self.store.all_events() if event.get("event_type") in event_types]
        return sorted(records, key=lambda item: item["proposal"]["proposal_id"])

    def _find_proposal(self, proposal_id: str) -> dict[str, Any]:
        for record in self._proposal_records(include_rejected=True):
            if record["proposal"]["proposal_id"] == proposal_id:
                return record
        raise ContractScoutError(f"unknown proposal_id: {proposal_id}")

    def _operations_context(self) -> dict[str, Any]:
        if self.operations_state_dir is None:
            return {
                "available": False,
                "reason": "operations_state_dir_not_configured",
                "active_contracts": [],
                "blocked_contracts": [],
                "paused_contracts": [],
                "source_coverage": {},
            }
        manifest_path = self.operations_state_dir / "release_manifest.json"
        if not manifest_path.exists():
            return {
                "available": False,
                "reason": "release_manifest_not_found",
                "state_dir": str(self.operations_state_dir),
                "active_contracts": [],
                "blocked_contracts": [],
                "paused_contracts": [],
                "source_coverage": {},
            }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canaries = manifest.get("canary_hashes", {})
        active = []
        blocked = []
        for lab, payload in sorted(canaries.items()):
            if payload.get("canary_id"):
                active.append({"lab": lab, "canary_id": payload["canary_id"], "material_hash": payload.get("material_hash")})
            else:
                blocked.append({"lab": lab, "reason": payload.get("reason", "not_started")})
        if manifest.get("seller_readiness", {}).get("passed") is not True and not any(item["lab"] == "offerlab_seller_pilot" for item in blocked):
            blocked.append({"lab": "offerlab_seller_pilot", "reason": manifest.get("seller_readiness", {}).get("reason", "seller_readiness_not_passed")})
        lock_path = self.operations_state_dir / "operations.lock.json"
        return {
            "available": True,
            "state_dir": str(self.operations_state_dir),
            "release_commit": manifest.get("release_commit"),
            "release_hash": manifest.get("release_hash"),
            "running": lock_path.exists(),
            "active_contracts": active,
            "blocked_contracts": blocked,
            "paused_contracts": [] if lock_path.exists() else active,
            "source_coverage": manifest.get("source_versions", {}),
            "seller_readiness": manifest.get("seller_readiness", {}),
        }


def load_proposals(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("proposals", [])
    if not isinstance(payload, list):
        raise ContractScoutError("proposal input must be a list or an object with proposals")
    return [dict(item) for item in payload]


def _action_ids(actions: list[dict[str, Any]]) -> set[str]:
    return {str(action.get("action_id", "")).strip() for action in actions}


def _nonnegative_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _contains_real_action_shape(value: Any) -> bool:
    for key, item in _walk(value):
        text = f"{key} {item}".lower()
        if any(marker in text for marker in REAL_ACTION_MARKERS):
            return True
    return False


def _walk(value: Any, key: str = "") -> list[tuple[str, Any]]:
    found = [(key, value)]
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(_walk(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk(child, key))
    return found


def _redact_sensitive(value: Any, key: str = "") -> Any:
    lowered_key = key.lower()
    if isinstance(value, dict):
        return {item_key: _redact_sensitive(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive(item, key) for item in value]
    if _looks_secret(key=lowered_key, value=value):
        return "[REDACTED]"
    return value


def _looks_secret(*, key: str, value: Any) -> bool:
    if any(marker in key for marker in SECRET_KEY_MARKERS):
        text = str(value).strip().lower()
        if any(marker in text for marker in SECRET_VALUE_MARKERS):
            return True
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_VALUE_MARKERS)
    return False


def _source_coverage_usable(proposal: OpportunityContractProposal) -> bool:
    if proposal.currently_available_sources:
        return True
    if proposal.historical_depth.get("days", 0) or proposal.historical_depth.get("years", 0):
        return True
    return proposal.paper_mode_feasibility.get("can_collect_prospectively") is True


def _private_data_available_or_acquirable(proposal: OpportunityContractProposal) -> bool:
    requires_private = proposal.paper_mode_feasibility.get("requires_private_data") is True
    if not requires_private:
        return True
    return bool(proposal.paper_mode_feasibility.get("private_data_acquisition_path"))


def _maintenance_value_reasonable(proposal: OpportunityContractProposal) -> bool:
    maintenance = proposal.estimated_maintenance_burden.strip().lower()
    information_value = proposal.expected_information_value.strip().lower()
    return not (maintenance in {"high", "very_high"} and information_value in {"low", "very_low"})


def _unclear_license(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"unclear", "unknown", "restrictive", "requires_approval"} or "unclear" in text


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "eligible_experimental": 0,
        "approval_required": 0,
        "rejected": 0,
        "duplicate": 0,
    }
    for record in records:
        status = str(record.get("validation", {}).get("status", "rejected"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _seed_proposals() -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": "seed_weather_event_market_daily_high_v1",
            "title": "Daily maximum-temperature event-market paper contract",
            "contract_family": "weather_event_market",
            "outcome": "Official daily maximum temperature resolves an exchange event bracket.",
            "resolution_source": {
                "source_id": "official_station_daily_climate_report",
                "publisher": "NOAA/NWS station observations",
                "ambiguous": False,
            },
            "resolution_cadence": "daily",
            "decision_deadline": "before_event_market_close",
            "available_actions": [
                {"action_id": "no_trade", "action_type": "no_action"},
                {"action_id": "paper_buy_yes_or_no_trade", "action_type": "paper_event_contract"},
            ],
            "no_action_alternative": "no_trade",
            "payoff_formula": {"formula": "settlement_value - executable_price - fees - spread - slippage", "executable": True},
            "material_costs": [
                {"name": "fees", "represented": True, "unknown": False},
                {"name": "spread", "represented": True, "unknown": False},
                {"name": "slippage", "represented": True, "unknown": False},
            ],
            "capital_requirement": {"amount": 1.0, "currency": "USD", "bounded": True},
            "maximum_possible_loss": {"amount": 1.0, "currency": "USD", "bounded": True},
            "required_source_families": ["event_market_order_book", "official_weather_observations", "official_forecasts"],
            "currently_available_sources": ["weather_edge_fixture_provider"],
            "missing_sources": [],
            "historical_depth": {"days": 365},
            "prospective_duration_required": "60 days",
            "expected_decision_frequency": "daily",
            "paper_mode_feasibility": {"paper_only": True, "can_collect_prospectively": True, "real_account_mutation_required": False},
            "platform_regulatory_dependencies": ["event-market terms must be reviewed before any future real-money use"],
            "credential_requirements": [],
            "licensing_concerns": [],
            "estimated_research_cost": {"usd": 5.0},
            "estimated_maintenance_burden": "medium",
            "expected_information_value": "medium",
            "reason_it_may_fail": ["market spread overwhelms forecast edge", "weather source revision semantics are wrong"],
            "citations": [{"title": "Weather Edge existing lab", "url": "docs/finance/WEATHER_EDGE.md"}],
        },
        {
            "proposal_id": "seed_broad_etf_risk_weekly_v1",
            "title": "Broad ETF weekly risk-exposure paper contract",
            "contract_family": "broad_etf_risk",
            "outcome": "Next-20-trading-day broad equity drawdown and realized volatility.",
            "resolution_source": {"source_id": "adjusted_total_return_bars", "publisher": "authorized market-data provider", "ambiguous": False},
            "resolution_cadence": "weekly",
            "decision_deadline": "weekly_rebalance_cutoff",
            "available_actions": [
                {"action_id": "cash", "action_type": "no_action"},
                {"action_id": "low_exposure", "action_type": "paper_allocation"},
                {"action_id": "normal_exposure", "action_type": "paper_allocation"},
            ],
            "no_action_alternative": "cash",
            "payoff_formula": {"formula": "paper_return - cash_return - turnover_costs", "executable": True},
            "material_costs": [{"name": "turnover_costs", "represented": True, "unknown": False}],
            "capital_requirement": {"amount": 10000.0, "currency": "USD", "bounded": True},
            "maximum_possible_loss": {"amount": 10000.0, "currency": "USD", "bounded": True},
            "required_source_families": ["adjusted_total_return_bars", "market_calendar", "risk_free_benchmark"],
            "currently_available_sources": ["etf_risk_fixture_provider"],
            "missing_sources": [],
            "historical_depth": {"years": 5},
            "prospective_duration_required": "six months",
            "expected_decision_frequency": "weekly",
            "paper_mode_feasibility": {"paper_only": True, "can_collect_prospectively": True, "real_account_mutation_required": False},
            "platform_regulatory_dependencies": ["future real-money use requires investment-advice review"],
            "credential_requirements": [],
            "licensing_concerns": [],
            "estimated_research_cost": {"usd": 8.0},
            "estimated_maintenance_burden": "medium",
            "expected_information_value": "medium",
            "reason_it_may_fail": ["simple baselines are hard to beat", "regime concentration"],
            "citations": [{"title": "ETF Risk existing lab", "url": "docs/finance/ETF_RISK.md"}],
        },
        {
            "proposal_id": "seed_seller_shadow_v1",
            "title": "Seller Best Offer shadow-decision contract",
            "contract_family": "seller_shadow",
            "outcome": "Mature contribution margin after accepted/countered/declined buyer offers.",
            "resolution_source": {"source_id": "seller_authorized_exports", "publisher": "seller private records", "ambiguous": False},
            "resolution_cadence": "per_offer",
            "decision_deadline": "before_seller_response",
            "available_actions": [
                {"action_id": "abstain", "action_type": "no_action"},
                {"action_id": "accept", "action_type": "shadow_decision"},
                {"action_id": "decline", "action_type": "shadow_decision"},
                {"action_id": "counter", "action_type": "shadow_decision"},
            ],
            "no_action_alternative": "abstain",
            "payoff_formula": {"formula": "mature_margin_after_costs - historical_no_action_margin", "executable": True},
            "material_costs": [
                {"name": "cost_basis", "represented": True, "unknown": False},
                {"name": "fees", "represented": True, "unknown": False},
                {"name": "shipping", "represented": True, "unknown": False},
                {"name": "refunds_returns_cancellations", "represented": True, "unknown": False},
            ],
            "capital_requirement": {"amount": 0.0, "currency": "USD", "bounded": True},
            "maximum_possible_loss": {"amount": 0.0, "currency": "USD", "bounded": True},
            "required_source_families": ["seller_orders", "seller_offers", "seller_fees", "seller_shipping", "seller_cost_basis"],
            "currently_available_sources": [],
            "missing_sources": ["seller_readiness_report"],
            "historical_depth": {},
            "prospective_duration_required": "30-60 days",
            "expected_decision_frequency": "per_offer",
            "paper_mode_feasibility": {
                "paper_only": True,
                "requires_private_data": True,
                "private_data_acquisition_path": "seller-provided local CSV exports",
                "can_collect_prospectively": True,
                "real_account_mutation_required": False,
            },
            "platform_regulatory_dependencies": ["seller consent required"],
            "credential_requirements": [],
            "licensing_concerns": [],
            "estimated_research_cost": {"usd": 4.0},
            "estimated_maintenance_burden": "medium",
            "expected_information_value": "high",
            "reason_it_may_fail": ["seller cannot provide cost basis", "offer history unavailable"],
            "citations": [{"title": "OfferLab seller acquisition packet", "url": "C:/OfferLabData/seller_acquisition/seller_data_request.md"}],
        },
    ]
