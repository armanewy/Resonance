from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from resonance.science.contracts import HypothesisSpec
from resonance.science.discovery_brief import DiscoveryBrief
from resonance.science.providers.base import (
    validate_hypothesis_proposals,
    validate_max_hypotheses,
)


class MockProvider:
    def __init__(
        self,
        proposals: Sequence[Any],
        *,
        name: str = "mock",
        model: str = "mock-model",
        prompt_version: str = "mock-prompt-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = dict(request_config or {})
        self._proposals = tuple(proposals)
        self.last_raw_proposals: tuple[Any, ...] = ()

    def propose(
        self,
        brief: DiscoveryBrief,
        max_hypotheses: int,
        seed: int,
    ) -> list[HypothesisSpec]:
        validate_max_hypotheses(max_hypotheses)
        self.last_raw_proposals = self._proposals[:max_hypotheses]
        accepted, _rejected = validate_hypothesis_proposals(self.last_raw_proposals)
        return accepted


class FileProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "file",
        model: str = "file",
        prompt_version: str = "file-prompt-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = dict(request_config or {})
        self.last_raw_proposals: tuple[Any, ...] = ()

    def propose(
        self,
        brief: DiscoveryBrief,
        max_hypotheses: int,
        seed: int,
    ) -> list[HypothesisSpec]:
        validate_max_hypotheses(max_hypotheses)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        proposals = payload.get("proposals", payload) if isinstance(payload, dict) else payload
        self.last_raw_proposals = tuple(proposals[:max_hypotheses])
        accepted, _rejected = validate_hypothesis_proposals(self.last_raw_proposals)
        return accepted


__all__ = ["FileProvider", "MockProvider"]
