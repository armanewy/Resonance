from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from behavior_lab.core import stable_hash, to_jsonable
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore
from behavior_lab.money_agents.roles import (
    MUTATING_TOOL_FRAGMENTS,
    FinancialAgentRole,
    MoneyAgentBudgetError,
    MoneyAgentContext,
    MoneyAgentError,
    MoneyAgentPermissionError,
)


@dataclass(frozen=True)
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0 or self.total_tokens < 0:
            raise MoneyAgentBudgetError("token usage may not be negative")
        if self.cost_usd < 0:
            raise MoneyAgentBudgetError("cost usage may not be negative")
        if self.total_tokens and self.total_tokens < self.input_tokens + self.output_tokens:
            raise MoneyAgentBudgetError("total_tokens may not be less than input plus output tokens")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "UsageRecord":
        payload = payload or {}
        input_tokens = int(payload.get("input_tokens", 0))
        output_tokens = int(payload.get("output_tokens", 0))
        total_tokens = int(payload.get("total_tokens", input_tokens + output_tokens))
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=float(payload.get("cost_usd", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    prompt_version: str
    content: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise MoneyAgentError("provider is required")
        if not self.model.strip():
            raise MoneyAgentError("model is required")
        if not self.prompt_version.strip():
            raise MoneyAgentError("prompt_version is required")
        if not isinstance(self.content, dict):
            raise MoneyAgentError("provider content must be an object")
        for field_name in ("tool_calls", "citations"):
            value = getattr(self, field_name)
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise MoneyAgentError(f"{field_name} must be a list of objects")

    @classmethod
    def from_payload(cls, payload: "ProviderResponse | dict[str, Any]") -> "ProviderResponse":
        if isinstance(payload, ProviderResponse):
            return payload
        if not isinstance(payload, dict):
            raise MoneyAgentError("provider must return ProviderResponse or dict")
        return cls(
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            prompt_version=str(payload.get("prompt_version", "")),
            content=dict(payload.get("content", {})),
            tool_calls=list(payload.get("tool_calls", [])),
            citations=list(payload.get("citations", [])),
            usage=UsageRecord.from_dict(payload.get("usage")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "content": to_jsonable(self.content),
            "tool_calls": to_jsonable(self.tool_calls),
            "citations": to_jsonable(self.citations),
            "usage": self.usage.to_dict(),
        }


class MoneyAgentProvider(Protocol):
    def complete(self, request: dict[str, Any]) -> ProviderResponse | dict[str, Any]: ...


class StaticMoneyAgentProvider:
    """Test and fixture provider that never performs network or LLM calls."""

    def __init__(self, response: ProviderResponse | dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def complete(self, request: dict[str, Any]) -> ProviderResponse | dict[str, Any]:
        self.requests.append(request)
        return self.response


class FinancialResearchAgentRuntime:
    def __init__(self, provider: MoneyAgentProvider, *, state_path: str | Path) -> None:
        self.provider = provider
        self.store = AppendOnlyResearchStore(state_path)

    def run(
        self,
        role: FinancialAgentRole,
        context: MoneyAgentContext,
        *,
        parent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        request = role.build_request(context, parent_ids=parent_ids)
        response: ProviderResponse | None = None
        try:
            response = ProviderResponse.from_payload(self.provider.complete(request))
            _validate_response_metadata(response, context)
            _validate_tool_calls(response.tool_calls)
            _validate_usage(response, context)
            role.validate_content(response.content, context)
        except Exception as exc:
            rejection = _rejection_payload(role, context, request, response, parent_ids or [], exc)
            self.store.append("money_agent_rejected", rejection)
            raise

        payload = {
            "campaign_id": context.campaign_id,
            "role_id": role.role_id,
            "display_name": role.display_name,
            "prompt_version": context.prompt_version,
            "provider": response.provider,
            "model": response.model,
            "request_hash": stable_hash(request),
            "response_hash": stable_hash(response.to_dict()),
            "tool_calls": to_jsonable(response.tool_calls),
            "citations": to_jsonable(response.citations),
            "usage": response.usage.to_dict(),
            "content": to_jsonable(response.content),
            "lineage": _lineage(response.content, parent_ids or []),
            "authority_boundaries": list(context.to_request_context()["authority_boundaries"]),
        }
        event = self.store.append("money_agent_completed", payload)
        return event["payload"]


def _validate_response_metadata(response: ProviderResponse, context: MoneyAgentContext) -> None:
    if response.prompt_version != context.prompt_version:
        raise MoneyAgentError("provider response prompt_version does not match context")


def _validate_usage(response: ProviderResponse, context: MoneyAgentContext) -> None:
    max_cost = context.explicit_budgets.get("max_response_cost_usd")
    if max_cost is not None and response.usage.cost_usd > float(max_cost) + 1e-9:
        raise MoneyAgentBudgetError("provider response exceeds explicit cost budget")
    max_tokens = context.explicit_budgets.get("max_response_tokens")
    if max_tokens is not None and response.usage.total_tokens > int(max_tokens):
        raise MoneyAgentBudgetError("provider response exceeds explicit token budget")
    max_tool_calls = context.explicit_budgets.get("max_tool_calls")
    if max_tool_calls is not None and len(response.tool_calls) > int(max_tool_calls):
        raise MoneyAgentBudgetError("provider response exceeds explicit tool-call budget")


def _validate_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    for call in tool_calls:
        name = str(call.get("tool_name") or call.get("name") or "").strip().lower()
        if not name:
            raise MoneyAgentError("tool call requires tool_name")
        if any(fragment in name for fragment in MUTATING_TOOL_FRAGMENTS):
            raise MoneyAgentPermissionError(f"mutating or trading tool call is forbidden: {name}")
        mode = str(call.get("mode", "read_only")).strip().lower()
        if mode not in {"read_only", "offline", "mock", "metadata"}:
            raise MoneyAgentPermissionError("tool calls must be read_only, offline, mock, or metadata")


def _rejection_payload(
    role: FinancialAgentRole,
    context: MoneyAgentContext,
    request: dict[str, Any],
    response: ProviderResponse | None,
    parent_ids: list[str],
    exc: Exception,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "campaign_id": context.campaign_id,
        "role_id": role.role_id,
        "prompt_version": context.prompt_version,
        "request_hash": stable_hash(request),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "lineage": {"parent_ids": list(parent_ids), "proposal_ids": [], "rejection_ids": []},
    }
    if response is not None:
        payload.update(
            {
                "provider": response.provider,
                "model": response.model,
                "response_hash": stable_hash(response.to_dict()),
                "tool_calls": to_jsonable(response.tool_calls),
                "citations": to_jsonable(response.citations),
                "usage": response.usage.to_dict(),
                "content_hash": stable_hash(response.content),
                "lineage": _lineage(response.content, parent_ids),
            }
        )
    return payload


def _lineage(content: dict[str, Any], parent_ids: list[str]) -> dict[str, Any]:
    proposal_ids: set[str] = set()
    rejection_ids: set[str] = set()
    _collect_lineage_ids(content, proposal_ids, rejection_ids, in_rejection=False)
    return {
        "parent_ids": list(parent_ids),
        "proposal_ids": sorted(proposal_ids),
        "rejection_ids": sorted(rejection_ids),
    }


def _collect_lineage_ids(
    value: Any,
    proposal_ids: set[str],
    rejection_ids: set[str],
    *,
    in_rejection: bool,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"proposal_id", "hypothesis_id", "source_id", "diagnostic_id", "work_item_id", "finding_id"}:
                text = str(child).strip()
                if text:
                    if in_rejection:
                        rejection_ids.add(text)
                    else:
                        proposal_ids.add(text)
            _collect_lineage_ids(
                child,
                proposal_ids,
                rejection_ids,
                in_rejection=in_rejection or lowered in {"rejections", "rejected", "deferred"},
            )
    elif isinstance(value, list):
        for item in value:
            _collect_lineage_ids(item, proposal_ids, rejection_ids, in_rejection=in_rejection)
