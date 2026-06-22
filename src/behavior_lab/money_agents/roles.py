from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ROLE_SOURCE_SCOUT = "financial_source_scout"
ROLE_HYPOTHESIS_SCIENTIST = "financial_hypothesis_scientist"
ROLE_SKEPTIC = "financial_skeptic"
ROLE_CONNECTOR_DIAGNOSTICIAN = "connector_maintenance_diagnostician"
ROLE_WEEKLY_ALLOCATOR = "weekly_research_allocator"


class MoneyAgentError(ValueError):
    pass


class MoneyAgentPermissionError(PermissionError):
    pass


class MoneyAgentBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class MoneyAgentContext:
    campaign_id: str
    prompt_version: str
    permitted_sources: tuple[str, ...] = ()
    permitted_connectors: tuple[str, ...] = ()
    explicit_budgets: dict[str, float] = field(default_factory=dict)
    prior_proposal_ids: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id.strip():
            raise MoneyAgentError("campaign_id is required")
        if not self.prompt_version.strip():
            raise MoneyAgentError("prompt_version is required")
        for name, value in self.explicit_budgets.items():
            if float(value) < 0:
                raise MoneyAgentBudgetError(f"budget {name!r} may not be negative")

    def to_request_context(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "prompt_version": self.prompt_version,
            "permitted_sources": list(self.permitted_sources),
            "permitted_connectors": list(self.permitted_connectors),
            "explicit_budgets": dict(self.explicit_budgets),
            "prior_proposal_ids": list(self.prior_proposal_ids),
            "notes": dict(self.notes),
            "authority_boundaries": list(AUTHORITY_BOUNDARIES),
        }


AUTHORITY_BOUNDARIES = (
    "advisory research labor only",
    "no trades, offers, purchases, or marketplace mutations",
    "no source activation or production promotion",
    "no statistical-threshold changes",
    "no blind-outcome access, blind winner selection, or consumed blind reruns",
    "no declaration that a strategy is valid",
    "no budget changes or spend beyond explicit budgets",
)


FORBIDDEN_AUTHORITY_KEYS = {
    "activate_source",
    "activation_approved",
    "activation_requested",
    "blind_outcome",
    "blind_outcomes",
    "blind_results",
    "blind_winner",
    "broker_order",
    "buy_order",
    "change_threshold",
    "counteroffer",
    "execute_trade",
    "hidden_results",
    "hidden_rows",
    "lockbox_rows",
    "make_offer",
    "market_order",
    "marketplace_action",
    "offer_submission",
    "order_ticket",
    "place_trade",
    "production_promotion",
    "promote_source",
    "purchase_order",
    "rerun_blind",
    "sell_order",
    "set_budget",
    "source_activation",
    "statistical_threshold",
    "strategy_valid",
    "strategy_validated",
    "threshold_change",
    "trade_order",
    "valid_strategy",
}


FORBIDDEN_AUTHORITY_PHRASES = (
    "activate this source",
    "choose the blind winner",
    "declare strategy valid",
    "execute the trade",
    "make an offer",
    "place the trade",
    "promote to production",
    "purchase the shares",
    "rerun the blind",
    "strategy is valid",
    "submit an offer",
    "submit the order",
)


MUTATING_TOOL_FRAGMENTS = (
    "accept_offer",
    "activate_source",
    "broker",
    "buy",
    "counteroffer",
    "order",
    "place_trade",
    "promote_source",
    "purchase",
    "sell",
    "submit_offer",
    "trade",
)


@dataclass(frozen=True)
class FinancialAgentRole:
    role_id: str
    display_name: str
    instructions: tuple[str, ...]
    output_contract: dict[str, Any]

    def build_request(self, context: MoneyAgentContext, *, parent_ids: list[str] | None = None) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "display_name": self.display_name,
            "prompt_version": context.prompt_version,
            "instructions": list(self.instructions),
            "output_contract": dict(self.output_contract),
            "context": context.to_request_context(),
            "parent_ids": list(parent_ids or []),
        }

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        _validate_common_boundaries(content)


class FinancialSourceScout(FinancialAgentRole):
    def __init__(self) -> None:
        super().__init__(
            ROLE_SOURCE_SCOUT,
            "Financial Source Scout",
            (
                "Search or read official financial data providers only.",
                "Compare documented licenses, rate limits, timestamp semantics, and connector feasibility.",
                "Return source candidates, proposed metrics, proposed connectors, citations, and rejected sources.",
                "Do not activate sources, accept unclear licensing, infer undocumented timestamps, or treat source availability as predictive evidence.",
            ),
            {
                "source_candidates": "list of official provider candidates with documented license and timestamp policy",
                "rejections": "sources rejected because licensing, timestamps, authority, or provider status is unclear",
            },
        )

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        super().validate_content(content, context)
        candidates = _require_list(content, "source_candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise MoneyAgentError("source candidate must be an object")
            source_id = _require_nonempty(candidate, "source_id")
            if context.permitted_sources and source_id not in context.permitted_sources:
                raise MoneyAgentPermissionError(f"source {source_id!r} is not in the permitted source list")
            if candidate.get("official_provider") is not True:
                raise MoneyAgentPermissionError("source scout may only propose official provider candidates")
            if str(candidate.get("license_status", "")).lower() != "documented":
                raise MoneyAgentPermissionError("source scout may not accept unclear or undocumented licensing")
            _require_nonempty(candidate, "license_citation")
            _require_nonempty(candidate, "rate_limit_summary")
            timestamp_policy = str(candidate.get("timestamp_policy", "")).strip().lower()
            if timestamp_policy in {"", "inferred", "undocumented", "unknown"}:
                raise MoneyAgentPermissionError("source scout may not infer undocumented timestamps")
            if candidate.get("activation_status", "proposed") not in {"proposed", "research_only"}:
                raise MoneyAgentPermissionError("source scout may not activate a source")
            if candidate.get("availability_as_predictive_evidence") is not False:
                raise MoneyAgentPermissionError("source availability may not be treated as predictive evidence")
            _require_list(candidate, "proposed_metrics")
            _require_list(candidate, "proposed_connectors")


class FinancialHypothesisScientist(FinancialAgentRole):
    allowed_hypothesis_types = {
        "forecast_revision_effect",
        "forecast_revision_effects",
        "interaction",
        "interactions",
        "lagged_feature",
        "lagged_features",
        "liquidity_effect",
        "liquidity_effects",
        "regime",
        "regimes",
        "risk_state",
        "risk_state_hypothesis",
        "seller_policy",
        "seller_policy_hypothesis",
    }

    def __init__(self) -> None:
        super().__init__(
            ROLE_HYPOTHESIS_SCIENTIST,
            "Financial Hypothesis Scientist",
            (
                "Propose structured, executable financial research hypotheses.",
                "Allowed families include lagged features, interactions, regimes, forecast revisions, liquidity effects, seller-policy hypotheses, and risk-state hypotheses.",
                "Return falsification tests, required data, execution sketch, and lineage.",
                "Do not generate trading strategy source code or action instructions.",
            ),
            {
                "hypotheses": "structured executable hypothesis proposals",
                "rejections": "candidate ideas rejected before proposal",
            },
        )

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        super().validate_content(content, context)
        hypotheses = _require_list(content, "hypotheses")
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                raise MoneyAgentError("hypothesis must be an object")
            _require_nonempty(hypothesis, "hypothesis_id")
            hypothesis_type = str(hypothesis.get("hypothesis_type", "")).strip().lower()
            if hypothesis_type not in self.allowed_hypothesis_types:
                raise MoneyAgentPermissionError(f"unsupported financial hypothesis type: {hypothesis_type!r}")
            executable_spec = hypothesis.get("executable_spec")
            if not isinstance(executable_spec, dict) or not executable_spec:
                raise MoneyAgentError("hypothesis requires a non-empty executable_spec")
            _require_list(hypothesis, "falsification_tests")
            _reject_code_payload(hypothesis)


class FinancialSkeptic(FinancialAgentRole):
    required_checks = {
        "corporate_action_leakage",
        "correlated_outcomes",
        "omitted_costs",
        "non_executable_prices",
        "prior_failed_equivalent_hypotheses",
        "regime_concentration",
        "selection_bias",
        "stale_pricing",
        "survivorship_bias",
        "target_leakage",
        "timing_leakage",
    }

    def __init__(self) -> None:
        super().__init__(
            ROLE_SKEPTIC,
            "Financial Skeptic",
            (
                "Search candidate proposals for timing leakage, survivorship bias, corporate-action leakage, selection bias, target leakage, stale pricing, non-executable prices, omitted costs, correlated outcomes, regime concentration, and prior failed equivalent hypotheses.",
                "Return risks, affected proposal IDs, evidence, and rejection or remediation recommendations.",
                "Do not access blind outcomes, select winners, rerun consumed blind evaluations, or declare a strategy valid.",
            ),
            {
                "checked_risk_types": "complete list of audited risk types",
                "audit_findings": "risk findings or explicit no-finding records",
            },
        )

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        super().validate_content(content, context)
        checked = {str(item).strip().lower() for item in _require_list(content, "checked_risk_types")}
        missing = sorted(self.required_checks - checked)
        if missing:
            raise MoneyAgentError(f"skeptic did not check required risks: {missing}")
        findings = _require_list(content, "audit_findings")
        for finding in findings:
            if not isinstance(finding, dict):
                raise MoneyAgentError("skeptic finding must be an object")
            _require_nonempty(finding, "finding_id")
            if str(finding.get("verdict", "")).strip().lower() in {"valid", "approved", "production_ready"}:
                raise MoneyAgentPermissionError("skeptic may not declare a strategy valid")


class ConnectorMaintenanceDiagnostician(FinancialAgentRole):
    def __init__(self) -> None:
        super().__init__(
            ROLE_CONNECTOR_DIAGNOSTICIAN,
            "Connector Maintenance Diagnostician",
            (
                "Diagnose connector maintenance issues using read-only or offline replay evidence.",
                "Return symptoms, affected connector IDs, read-only checks, proposed repairs, and escalation needs.",
                "Do not mutate external systems, promote a source, activate a connector, or change thresholds.",
            ),
            {
                "diagnostics": "connector diagnostics with read-only check plans",
                "maintenance_tickets": "bounded repair tickets for humans or connector owners",
            },
        )

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        super().validate_content(content, context)
        diagnostics = _require_list(content, "diagnostics")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                raise MoneyAgentError("diagnostic must be an object")
            connector_id = _require_nonempty(diagnostic, "connector_id")
            if context.permitted_connectors and connector_id not in context.permitted_connectors:
                raise MoneyAgentPermissionError(f"connector {connector_id!r} is not permitted")
            mode = str(diagnostic.get("check_mode", "")).strip().lower()
            if mode not in {"read_only", "offline_replay", "mock_replay"}:
                raise MoneyAgentPermissionError("connector diagnostics must use read-only, offline, or mock checks")
            if diagnostic.get("mutation_required") is True:
                raise MoneyAgentPermissionError("connector diagnostician may not require mutations")
            if diagnostic.get("activation_change") not in {None, False, "none", "no_change"}:
                raise MoneyAgentPermissionError("connector diagnostician may not activate connectors")


class WeeklyResearchAllocator(FinancialAgentRole):
    def __init__(self) -> None:
        super().__init__(
            ROLE_WEEKLY_ALLOCATOR,
            "Weekly Research Allocator",
            (
                "Allocate already-approved weekly research labor across the finance research roles.",
                "Stay within explicit hours, cost, and tool-call budgets supplied by the caller.",
                "Return allocations, deferrals, dependencies, and budget leftovers.",
                "Do not create new budgets, change thresholds, or authorize blind access or real actions.",
            ),
            {
                "allocations": "bounded work allocations against explicit caller budgets",
                "deferred": "work deferred because budget or authority is insufficient",
            },
        )

    def validate_content(self, content: dict[str, Any], context: MoneyAgentContext) -> None:
        super().validate_content(content, context)
        allocations = _require_list(content, "allocations")
        known_roles = set(ROLE_REGISTRY)
        hours = 0.0
        cost = 0.0
        tool_calls = 0.0
        for allocation in allocations:
            if not isinstance(allocation, dict):
                raise MoneyAgentError("allocation must be an object")
            _require_nonempty(allocation, "work_item_id")
            role_id = str(allocation.get("role_id", "")).strip()
            if role_id not in known_roles:
                raise MoneyAgentError(f"unknown allocation role_id: {role_id!r}")
            hours += float(allocation.get("hours", 0.0))
            cost += float(allocation.get("cost_usd", 0.0))
            tool_calls += float(allocation.get("tool_calls", 0.0))
            if float(allocation.get("hours", 0.0)) < 0 or float(allocation.get("cost_usd", 0.0)) < 0:
                raise MoneyAgentBudgetError("allocations may not be negative")
        _assert_within_budget(context, "weekly_hours", hours)
        _assert_within_budget(context, "cost_usd", cost)
        _assert_within_budget(context, "tool_calls", tool_calls)
        if any(key in content for key in {"new_budget", "budget_change", "threshold_change"}):
            raise MoneyAgentPermissionError("weekly allocator may not create budgets or thresholds")


SOURCE_SCOUT = FinancialSourceScout()
HYPOTHESIS_SCIENTIST = FinancialHypothesisScientist()
SKEPTIC = FinancialSkeptic()
CONNECTOR_DIAGNOSTICIAN = ConnectorMaintenanceDiagnostician()
WEEKLY_ALLOCATOR = WeeklyResearchAllocator()

ROLE_REGISTRY: dict[str, FinancialAgentRole] = {
    role.role_id: role
    for role in (
        SOURCE_SCOUT,
        HYPOTHESIS_SCIENTIST,
        SKEPTIC,
        CONNECTOR_DIAGNOSTICIAN,
        WEEKLY_ALLOCATOR,
    )
}


def role_by_id(role_id: str) -> FinancialAgentRole:
    try:
        return ROLE_REGISTRY[role_id]
    except KeyError as exc:
        raise MoneyAgentError(f"unknown money-agent role {role_id!r}") from exc


def _require_nonempty(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MoneyAgentError(f"{key} must be a non-empty string")
    return value.strip()


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise MoneyAgentError(f"{key} must be a list")
    return value


def _assert_within_budget(context: MoneyAgentContext, key: str, observed: float) -> None:
    limit = context.explicit_budgets.get(key)
    if limit is not None and observed > float(limit) + 1e-9:
        raise MoneyAgentBudgetError(f"{key} allocation exceeds explicit budget")


def _validate_common_boundaries(content: dict[str, Any]) -> None:
    if not isinstance(content, dict):
        raise MoneyAgentError("agent content must be an object")
    for path, key, value in _walk(content):
        lowered = key.lower()
        if lowered in FORBIDDEN_AUTHORITY_KEYS:
            raise MoneyAgentPermissionError(f"agent output crosses forbidden authority at {'.'.join(path)}")
        if "threshold" in lowered:
            raise MoneyAgentPermissionError("agents may not change statistical thresholds")
        if lowered in {"budget", "budgets"} and isinstance(value, dict):
            raise MoneyAgentPermissionError("agents may not choose or modify budgets")
        if isinstance(value, str):
            text = value.lower()
            for phrase in FORBIDDEN_AUTHORITY_PHRASES:
                if phrase in text:
                    raise MoneyAgentPermissionError(f"agent output contains forbidden instruction: {phrase}")
        if ("blind" in lowered or "hidden" in lowered or "lockbox" in lowered) and not _is_explicitly_no_blind_access(value):
            raise MoneyAgentPermissionError("agents may not access blind or hidden outcomes")


def _walk(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, Any]]:
    found: list[tuple[tuple[str, ...], str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = path + (key_text,)
            found.append((child_path, key_text, child))
            found.extend(_walk(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, path + (str(index),)))
    return found


def _reject_code_payload(payload: dict[str, Any]) -> None:
    for path, key, value in _walk(payload):
        lowered = key.lower()
        if lowered in {"code", "python", "source_code", "strategy_code", "trading_strategy_code", "trade_rules"}:
            if not _is_empty_value(value):
                raise MoneyAgentPermissionError("hypothesis scientist may not generate trading strategy source code")
        if isinstance(value, str) and ("import " in value or "def " in value or "class " in value):
            raise MoneyAgentPermissionError("hypothesis scientist may not generate source code")


def _is_explicitly_no_blind_access(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return not value
    if isinstance(value, dict):
        return not value
    if value in {None, False, "none", "not_requested", "not requested"}:
        return True
    return False


def _is_empty_value(value: Any) -> bool:
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return value in {None, "", False}
