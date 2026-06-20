"""Provider-neutral hypothesis proposal interfaces."""

from resonance.science.providers.base import (
    MAX_HYPOTHESES_PER_REQUEST,
    HypothesisProvider,
    ProviderError,
    ProviderRun,
    ProviderRunMetadata,
    RejectedProposal,
    hash_artifact,
    run_provider,
    serialize_artifact,
    store_json_artifact,
    validate_hypothesis_proposals,
    validate_max_hypotheses,
)
from resonance.science.providers.mock import FileProvider, MockProvider

__all__ = [
    "MAX_HYPOTHESES_PER_REQUEST",
    "FileProvider",
    "HypothesisProvider",
    "MockProvider",
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
