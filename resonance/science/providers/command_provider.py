from __future__ import annotations

import subprocess
from typing import Any, Sequence

from resonance.science.contracts import HypothesisSpec
from resonance.science.discovery_brief import DiscoveryBrief, serialize_discovery_brief
from resonance.science.providers.base import ProviderError, validate_max_hypotheses
from resonance.science.providers.openai_provider import _parse_proposals


PROMPT_VERSION = "science-command-provider-v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


class CommandProvider:
    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        name: str = "command",
        model: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        request_config: dict[str, Any] | None = None,
    ) -> None:
        if not command:
            raise ProviderError("command must be a non-empty argument vector")
        if any(not isinstance(part, str) or not part for part in command):
            raise ProviderError("command arguments must be non-empty strings")
        if timeout_seconds <= 0:
            raise ProviderError("timeout_seconds must be positive")
        if max_output_bytes <= 0:
            raise ProviderError("max_output_bytes must be positive")
        self.command = tuple(command)
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)
        self.name = name
        self.model = model or self.command[0]
        self.prompt_version = prompt_version
        self.request_config = {
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
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
        try:
            completed = subprocess.run(
                list(self.command),
                input=serialize_discovery_brief(brief),
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                f"command provider timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise ProviderError(f"command provider failed to start: {exc}") from exc

        stdout_bytes = completed.stdout.encode("utf-8")
        if len(stdout_bytes) > self.max_output_bytes:
            raise ProviderError(
                f"command provider stdout exceeded {self.max_output_bytes} bytes"
            )
        stderr = completed.stderr.strip()
        self.request_config = {
            **self.request_config,
            "returncode": completed.returncode,
            "stderr_bytes": len(completed.stderr.encode("utf-8")),
            "stdout_bytes": len(stdout_bytes),
        }
        if completed.returncode != 0:
            detail = f": {stderr[:500]}" if stderr else ""
            raise ProviderError(
                f"command provider exited with status {completed.returncode}{detail}"
            )
        proposals = _parse_proposals(completed.stdout)
        self.last_raw_proposals = tuple(
            proposal.model_dump(mode="json", exclude_none=True) for proposal in proposals
        )
        return proposals[:normalized_max]


__all__ = ["CommandProvider"]
