from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import Field, ValidationError

from resonance.science.contracts import HypothesisSpec, StrictModel, canonical_json, stable_hash
from resonance.science.discovery_brief import DiscoveryBrief


MAX_HYPOTHESES_PER_REQUEST = 8


class ProviderError(ValueError):
    """Raised when a provider request cannot be represented safely."""


class HypothesisProvider(Protocol):
    name: str
    model: str
    prompt_version: str
    request_config: dict[str, Any]

    def propose(
        self,
        brief: DiscoveryBrief,
        max_hypotheses: int,
        seed: int,
    ) -> list[HypothesisSpec]:
        """Return provider-neutral hypothesis proposals."""


class RejectedProposal(StrictModel):
    index: int
    proposal_hash: str
    error: str


class ProviderRunMetadata(StrictModel):
    provider_name: str
    model: str
    request_config: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str
    seed: int
    max_hypotheses: int


class ProviderRun(StrictModel):
    metadata: ProviderRunMetadata
    brief_sha256: str
    raw_proposals_sha256: str
    hypotheses: tuple[HypothesisSpec, ...]
    rejected_proposals: tuple[RejectedProposal, ...] = Field(default_factory=tuple)

    def artifact_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        return canonical_json(self.artifact_payload())

    def artifact_hash(self) -> str:
        return stable_hash(self.artifact_payload())


def run_provider(
    provider: HypothesisProvider,
    brief: DiscoveryBrief,
    *,
    max_hypotheses: int,
    seed: int,
) -> ProviderRun:
    normalized_max = validate_max_hypotheses(max_hypotheses)
    raw_result = provider.propose(brief, normalized_max, seed)
    raw_proposals = tuple(getattr(provider, "last_raw_proposals", raw_result))
    hypotheses, rejected = validate_hypothesis_proposals(raw_proposals)
    hypotheses = hypotheses[:normalized_max]
    metadata = ProviderRunMetadata(
        provider_name=provider.name,
        model=provider.model,
        request_config=dict(provider.request_config),
        prompt_version=provider.prompt_version,
        seed=seed,
        max_hypotheses=normalized_max,
    )
    return ProviderRun(
        metadata=metadata,
        brief_sha256=brief.artifact_hash(),
        raw_proposals_sha256=hash_artifact(raw_proposals),
        hypotheses=tuple(hypotheses),
        rejected_proposals=tuple(rejected),
    )


def validate_max_hypotheses(max_hypotheses: int) -> int:
    if max_hypotheses < 1:
        raise ProviderError("max_hypotheses must be positive")
    if max_hypotheses > MAX_HYPOTHESES_PER_REQUEST:
        raise ProviderError(
            f"max_hypotheses must be at most {MAX_HYPOTHESES_PER_REQUEST}"
        )
    return int(max_hypotheses)


def validate_hypothesis_proposals(
    proposals: Sequence[Any],
) -> tuple[list[HypothesisSpec], list[RejectedProposal]]:
    accepted: list[HypothesisSpec] = []
    rejected: list[RejectedProposal] = []
    for index, proposal in enumerate(proposals):
        try:
            accepted.append(_validate_hypothesis(proposal))
        except ValidationError as exc:
            rejected.append(
                RejectedProposal(
                    index=index,
                    proposal_hash=hash_artifact(proposal),
                    error=str(exc),
                )
            )
    return accepted, rejected


def serialize_artifact(value: Any) -> str:
    return canonical_json(_artifact_value(value))


def hash_artifact(value: Any) -> str:
    return stable_hash(_artifact_value(value))


def store_json_artifact(value: Any, path: str | Path) -> dict[str, str]:
    content = serialize_artifact(value)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(content, encoding="utf-8")
    return {"sha256": stable_hash(_artifact_value(value)), "path": str(artifact_path)}


def _validate_hypothesis(proposal: Any) -> HypothesisSpec:
    if isinstance(proposal, HypothesisSpec):
        return proposal
    return HypothesisSpec.model_validate(proposal)


def _artifact_value(value: Any) -> Any:
    if isinstance(value, HypothesisSpec):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, StrictModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {
            str(key): _artifact_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_value(item) for item in value]
    return value


__all__ = [
    "MAX_HYPOTHESES_PER_REQUEST",
    "HypothesisProvider",
    "ProviderError",
    "ProviderRun",
    "ProviderRunMetadata",
    "RejectedProposal",
    "hash_artifact",
    "run_provider",
    "serialize_artifact",
    "store_json_artifact",
    "validate_hypothesis_proposals",
    "validate_max_hypotheses",
]
