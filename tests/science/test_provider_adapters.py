from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from resonance.science.discovery_brief import (
    DiscoveryBrief,
    discovery_brief_from_exploration_view,
)
from resonance.science.providers import CommandProvider, OpenAIProvider, ProviderError, run_provider
from resonance.science.providers import openai_provider


def test_openai_provider_uses_responses_structured_outputs_without_tools() -> None:
    valid = _valid_hypothesis()
    fake_client = _FakeOpenAIClient({"proposals": [valid]})
    provider = OpenAIProvider(
        model="gpt-test",
        timeout_seconds=12.5,
        max_retries=1,
        client=fake_client,
    )

    run = run_provider(provider, _brief_with_hidden_split_sentinels(), max_hypotheses=1, seed=99)

    request = fake_client.responses.requests[0]
    assert request["model"] == "gpt-test"
    assert request["store"] is False
    assert "tools" not in request
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["properties"]["proposals"]["items"]["title"] == "HypothesisSpec"

    sent_payload = json.dumps(request["input"], sort_keys=True)
    assert "TUNING_SENTINEL_SHOULD_NOT_LEAK" not in sent_payload
    assert "BLIND_SENTINEL_SHOULD_NOT_LEAK" not in sent_payload
    assert "SECRET_EVALUATOR_SENTINEL" not in sent_payload
    assert "cpu_percent" in sent_payload
    assert run.metadata.request_config["response_id"] == "resp_test"
    assert run.metadata.request_config["response_model"] == "gpt-test"
    assert run.metadata.request_config["response_metadata"]["usage"] == {"input_tokens": 10}


def test_openai_provider_missing_optional_package_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str) -> Any:
        if name == "openai":
            raise ImportError("missing openai")
        raise AssertionError(name)

    monkeypatch.setattr(openai_provider.importlib, "import_module", missing_import)
    provider = OpenAIProvider()

    with pytest.raises(ProviderError, match="optional 'openai' package"):
        provider.propose(_brief(), max_hypotheses=1, seed=1)


def test_command_provider_passes_discovery_brief_on_stdin_and_validates_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _valid_hypothesis()
    calls: list[dict[str, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, "kwargs": kwargs})
        assert args == (["provider-bin", "--json"],)
        assert kwargs["input"] == _brief().canonical_json()
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"proposals": [valid]}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CommandProvider(["provider-bin", "--json"], timeout_seconds=5, max_output_bytes=4096)

    run = run_provider(provider, _brief(), max_hypotheses=1, seed=7)

    assert len(calls) == 1
    assert run.hypotheses[0].title == valid["title"]
    assert run.metadata.request_config["command"] == ["provider-bin", "--json"]
    assert run.metadata.request_config["returncode"] == 0
    assert run.metadata.request_config["stdout_bytes"] > 0


def test_command_provider_timeout_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CommandProvider(["provider-bin"], timeout_seconds=0.25)

    with pytest.raises(ProviderError, match="timed out"):
        provider.propose(_brief(), max_hypotheses=1, seed=1)


def test_command_provider_output_size_limit_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="x" * 20,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CommandProvider(["provider-bin"], max_output_bytes=8)

    with pytest.raises(ProviderError, match="stdout exceeded"):
        provider.propose(_brief(), max_hypotheses=1, seed=1)


def test_command_provider_invalid_json_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="{not json",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CommandProvider(["provider-bin"])

    with pytest.raises(ProviderError, match="invalid JSON"):
        provider.propose(_brief(), max_hypotheses=1, seed=1)


class _FakeResponses:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            model=kwargs["model"],
            output_text=json.dumps(self.payload),
            status="completed",
            usage=SimpleNamespace(model_dump=lambda mode, exclude_none: {"input_tokens": 10}),
        )


class _FakeOpenAIClient:
    def __init__(self, payload: Any) -> None:
        self.responses = _FakeResponses(payload)


def _brief_with_hidden_split_sentinels() -> DiscoveryBrief:
    exploration_view = {
        "snapshot_id": "snapshot-123",
        "partition": "exploration",
        "rows": [
            {
                "timestamp_utc": "2026-06-01T00:00:00Z",
                "metrics": {
                    "cpu_percent": [
                        {"value": 10.0, "unit": "percent", "source": "sensor-a"}
                    ],
                    "memory_percent": [
                        {"value": 20.0, "unit": "percent", "source": "sensor-a"}
                    ],
                },
            },
        ],
        "metadata": {
            "time_range_utc": {
                "tuning": {"sentinel": "TUNING_SENTINEL_SHOULD_NOT_LEAK"},
                "blind": {"sentinel": "BLIND_SENTINEL_SHOULD_NOT_LEAK"},
            },
            "secret_evaluator_details": "SECRET_EVALUATOR_SENTINEL",
        },
    }
    return discovery_brief_from_exploration_view(
        exploration_view,
        metric_catalog={
            "catalog_id": "a" * 64,
            "metrics": [
                {
                    "name": "cpu_percent",
                    "units": ["percent"],
                    "sources": ["sensor-a"],
                    "coverage": {"sample_count": 1},
                    "cadence": {"median_seconds": 60.0},
                },
                {
                    "name": "memory_percent",
                    "units": ["percent"],
                    "sources": ["sensor-a"],
                    "coverage": {"sample_count": 1},
                    "cadence": {"median_seconds": 60.0},
                },
                {
                    "name": "temperature_2m",
                    "units": ["celsius"],
                    "sources": ["sensor-b"],
                    "coverage": {"sample_count": 1},
                    "cadence": {"median_seconds": 60.0},
                },
            ],
        },
    )


def _brief() -> DiscoveryBrief:
    return DiscoveryBrief(
        snapshot_id="snapshot-123",
        metric_catalog_id="a" * 64,
        metrics=(
            {
                "name": "cpu_percent",
                "units": ("percent",),
                "sources": ("sensor-a",),
                "coverage": {"sample_count": 10},
                "cadence": {"median_seconds": 60.0},
            },
            {
                "name": "memory_percent",
                "units": ("percent",),
                "sources": ("sensor-a",),
                "coverage": {"sample_count": 10},
                "cadence": {"median_seconds": 60.0},
            },
            {
                "name": "temperature_2m",
                "units": ("celsius",),
                "sources": ("sensor-b",),
                "coverage": {"sample_count": 10},
                "cadence": {"median_seconds": 60.0},
            },
        ),
        exploration_boundary={
            "start_utc": "2026-06-01T00:00:00Z",
            "end_utc": "2026-06-01T01:00:00Z",
            "row_count": 10,
        },
        descriptive_stats={"cpu_percent": {"count": 10, "mean": 50.0}},
    )


def _valid_hypothesis() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "hypothesis_type": "observational_prediction",
        "title": "CPU pressure predicts warmer local residuals",
        "concise_claim": "CPU utilization is associated with transformed local temperature residuals.",
        "rationale": "A local heat/load proxy may move with weather-station residuals after robust scaling.",
        "target_metric": "temperature_2m",
        "input_metrics": ["cpu_percent", "memory_percent"],
        "target_transform": "robust_zscore",
        "expression": {
            "node": "safe_divide",
            "numerator": {
                "node": "add",
                "left": {
                    "node": "multiply",
                    "left": {"node": "fitted_parameter", "parameter": "scale"},
                    "right": {
                        "node": "lag",
                        "input": {
                            "node": "rolling_mean",
                            "input": {"node": "metric", "metric": "cpu_percent"},
                            "window_seconds": 900,
                            "min_periods": 3,
                        },
                        "lag_seconds": 300,
                    },
                },
                "right": {"node": "fitted_parameter", "parameter": "offset"},
            },
            "denominator": {
                "node": "rolling_std",
                "input": {"node": "metric", "metric": "memory_percent"},
                "window_seconds": 900,
                "min_periods": 3,
            },
            "epsilon": 0.000001,
            "near_zero_behavior": "return_zero",
        },
        "parameter_bounds": {
            "scale": {"lower": 0.0, "upper": 3.0},
            "offset": {"lower": -1.0, "upper": 1.0},
        },
        "expected_direction": "positive",
        "maximum_lag_seconds": 900,
        "fitting_metric": "rmse",
        "tuning_metric": "mae",
        "blind_metrics": ["rmse", "spearman_r"],
        "minimum_blind_effect": 0.1,
        "minimum_baseline_improvement": 0.05,
        "negative_controls": [{"metric": "memory_percent", "rationale": "Check specificity."}],
        "falsification_conditions": [{"description": "Fails if association reverses."}],
        "complexity_budget": {"max_ast_nodes": 15, "max_source_metrics": 3},
        "origin": "llm",
        "parent_hypothesis_ids": [],
        "snapshot_metric_catalog_id": "a" * 64,
        "random_seed": 42,
    }
