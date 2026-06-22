from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from behavior_lab.core import parse_time, to_jsonable, utc_now
from behavior_lab.finance_data.contracts import FinanceDataError, SUPPORTED_OBSERVATION_KINDS


PROPOSAL_STATUSES = {"proposed", "needs_human_review", "rejected"}


@dataclass(frozen=True)
class SourceProposal:
    proposal_id: str
    title: str
    researcher_prompt: str
    proposed_provider: str
    observation_types: list[str]
    coverage: dict[str, Any]
    access_method: str
    license_summary: str
    permitted_uses: dict[str, bool]
    prohibited_uses: list[str]
    artifact_hash_plan: str
    time_semantics: dict[str, Any]
    risks: list[str]
    open_questions: list[str]
    created_at: str = field(default_factory=utc_now)
    status: str = "proposed"
    scraping_allowed: bool = False
    approved_source_contract_id: str | None = None
    activation_requested: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "proposal_id",
            "title",
            "researcher_prompt",
            "proposed_provider",
            "access_method",
            "license_summary",
            "artifact_hash_plan",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        parse_time(self.created_at)
        if self.status not in PROPOSAL_STATUSES:
            raise FinanceDataError(f"status must be one of {sorted(PROPOSAL_STATUSES)}")
        if not self.observation_types:
            raise FinanceDataError("observation_types may not be empty")
        unknown_types = set(self.observation_types) - SUPPORTED_OBSERVATION_KINDS
        if unknown_types:
            raise FinanceDataError(f"unknown observation_types: {sorted(unknown_types)}")
        for field_name in ("coverage", "permitted_uses", "time_semantics"):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or not value:
                raise FinanceDataError(f"{field_name} must be a non-empty object")
        for field_name in ("prohibited_uses", "risks", "open_questions"):
            value = getattr(self, field_name)
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                raise FinanceDataError(f"{field_name} must contain strings")
        if self.activation_requested:
            raise FinanceDataError("source proposals cannot request activation automatically")
        if self.approved_source_contract_id is not None:
            raise FinanceDataError("approved source contracts must be registered outside SourceProposal")

    @property
    def is_active_source(self) -> bool:
        return False

    @property
    def requires_human_source_contract(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            **to_jsonable(self),
            "is_active_source": self.is_active_source,
            "requires_human_source_contract": self.requires_human_source_contract,
        }

    def activation_contract(self) -> None:
        raise FinanceDataError("SourceProposal is research-only and cannot activate sources")


def _require_nonempty(value: str | None, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FinanceDataError(f"{field_name} must be a non-empty string")
