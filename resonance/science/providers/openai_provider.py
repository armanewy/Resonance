from __future__ import annotations

import importlib
import json
from typing import Any

from pydantic import ValidationError

from resonance.science.contracts import HypothesisSpec, canonical_json
from resonance.science.discovery_brief import DiscoveryBrief
from resonance.science.providers.base import ProviderError, validate_max_hypotheses


PROMPT_VERSION = "science-openai-provider-v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2

_INSTRUCTIONS = (
    "Propose observational_prediction hypotheses as JSON that matches the supplied schema. "
    "Use only the DiscoveryBrief JSON provided by the user message."
)


class OpenAIProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any | None = None,
        name: str = "openai",
        prompt_version: str = PROMPT_VERSION,
        request_config: dict[str, Any] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ProviderError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ProviderError("max_retries must be non-negative")
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._client = client
        self.request_config = {
            "model": model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            **dict(request_config or {}),
        }
        self.last_raw_proposals: tuple[Any, ...] = ()

    def propose(
        self,
        brief: DiscoveryBrief,
        max_hypotheses: int,
        seed: int,
    ) -> list[HypothesisSpec]:
        normalized_max = validate_max_hypotheses(max_hypotheses)
        request_payload = canonical_json(
            {
                "discovery_brief": brief.model_dump(mode="json", exclude_none=True),
                "max_hypotheses": normalized_max,
                "requested_seed": int(seed),
                "seed_note": "provenance only; the remote model is not assumed deterministic",
            }
        )
        response = self._responses_create(
            model=self.model,
            input=[
                {"role": "system", "content": _INSTRUCTIONS},
                {"role": "user", "content": request_payload},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "hypothesis_proposals",
                    "strict": True,
                    "schema": _proposal_batch_schema(normalized_max),
                }
            },
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderError("OpenAI response did not include output_text")
        proposals = _parse_proposals(output_text)
        self.last_raw_proposals = tuple(
            proposal.model_dump(mode="json", exclude_none=True) for proposal in proposals
        )
        self.request_config = {
            **self.request_config,
            "requested_seed": int(seed),
            "deterministic_seed_applied": False,
            "max_hypotheses": normalized_max,
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None) or self.model,
            "response_metadata": _response_metadata(response),
        }
        return proposals[:normalized_max]

    def _responses_create(self, **kwargs: Any) -> Any:
        client = self._client or self._build_client()
        try:
            return client.responses.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - exact SDK exceptions are optional.
            raise ProviderError(f"OpenAI provider request failed: {exc}") from exc

    def _build_client(self) -> Any:
        try:
            openai = importlib.import_module("openai")
        except ImportError as exc:
            raise ProviderError(
                "OpenAIProvider requires the optional 'openai' package at runtime"
            ) from exc
        try:
            return openai.OpenAI(
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        except Exception as exc:  # pragma: no cover - depends on optional SDK.
            raise ProviderError(f"OpenAI client initialization failed: {exc}") from exc


def _proposal_batch_schema(max_hypotheses: int) -> dict[str, Any]:
    hypothesis_schema = HypothesisSpec.model_json_schema()
    hypothesis_defs = hypothesis_schema.pop("$defs", {})
    return {
        "$defs": hypothesis_defs,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_hypotheses,
                "items": hypothesis_schema,
            }
        },
        "required": ["proposals"],
    }


def _parse_proposals(output_text: str) -> list[HypothesisSpec]:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
    raw_proposals = _extract_raw_proposals(payload)
    proposals: list[HypothesisSpec] = []
    for index, proposal in enumerate(raw_proposals):
        try:
            proposals.append(HypothesisSpec.model_validate(proposal))
        except ValidationError as exc:
            raise ProviderError(f"provider proposal {index} failed validation: {exc}") from exc
    if not proposals:
        raise ProviderError("provider returned no proposals")
    return proposals


def _extract_raw_proposals(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and "proposals" in payload:
        proposals = payload["proposals"]
    elif isinstance(payload, list):
        proposals = payload
    elif isinstance(payload, dict):
        proposals = [payload]
    else:
        raise ProviderError("provider JSON must be an object or list")
    if not isinstance(proposals, list):
        raise ProviderError("provider proposals must be a JSON array")
    return proposals


def _response_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attr in ("created_at", "status", "usage"):
        value = getattr(response, attr, None)
        if value is not None:
            metadata[attr] = _jsonable(value)
    return metadata


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = ["OpenAIProvider"]
