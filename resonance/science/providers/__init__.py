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
from resonance.science.providers.command_provider import CommandProvider
from resonance.science.providers.mock import FileProvider, MockProvider
from resonance.science.providers.openai_provider import OpenAIProvider

__all__ = [
    "MAX_HYPOTHESES_PER_REQUEST",
    "CommandProvider",
    "FileProvider",
    "HypothesisProvider",
    "MockProvider",
    "OpenAIProvider",
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
