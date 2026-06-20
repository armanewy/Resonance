from __future__ import annotations

import json
from pathlib import Path

import pytest

from resonance.science.discovery_brief import (
    DiscoveryBrief,
    discovery_brief_from_exploration_view,
    serialize_discovery_brief,
)
from resonance.science.providers import (
    FileProvider,
    MockProvider,
    ProviderError,
    hash_artifact,
    run_provider,
)


def test_discovery_brief_serialization_excludes_tuning_and_blind_sentinels() -> None:
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
            {
                "timestamp_utc": "2026-06-01T01:00:00Z",
                "metrics": {
                    "cpu_percent": [
                        {"value": 11.0, "unit": "percent", "source": "sensor-a"}
                    ],
                    "memory_percent": [
                        {"value": 21.0, "unit": "percent", "source": "sensor-a"}
                    ],
                },
            },
        ],
        "metadata": {
            "time_range_utc": {
                "exploration": {
                    "start_utc": "2026-06-01T00:00:00Z",
                    "end_utc": "2026-06-01T01:00:00Z",
                },
                "tuning": {"sentinel": "TUNING_SENTINEL_SHOULD_NOT_LEAK"},
                "blind": {"sentinel": "BLIND_SENTINEL_SHOULD_NOT_LEAK"},
            },
            "row_counts": {
                "exploration": 2,
                "tuning": "TUNING_ROW_COUNT_SENTINEL",
                "blind": "BLIND_ROW_COUNT_SENTINEL",
            },
            "secret_evaluator_details": "SECRET_EVALUATOR_SENTINEL",
        },
    }
    metric_catalog = {
        "catalog_id": "a" * 64,
        "metrics": [
            {
                "name": "cpu_percent",
                "units": ["percent"],
                "sources": ["sensor-a"],
                "coverage": {"sample_count": 2},
                "cadence": {"median_seconds": 3600.0},
            },
            {
                "name": "memory_percent",
                "units": ["percent"],
                "sources": ["sensor-a"],
                "coverage": {"sample_count": 2},
                "cadence": {"median_seconds": 3600.0},
            },
        ],
    }

    brief = discovery_brief_from_exploration_view(
        exploration_view,
        metric_catalog=metric_catalog,
        selected_memory_summaries=["prior selected exploration memory"],
    )
    serialized = serialize_discovery_brief(brief)

    assert "TUNING_SENTINEL_SHOULD_NOT_LEAK" not in serialized
    assert "BLIND_SENTINEL_SHOULD_NOT_LEAK" not in serialized
    assert "TUNING_ROW_COUNT_SENTINEL" not in serialized
    assert "BLIND_ROW_COUNT_SENTINEL" not in serialized
    assert "SECRET_EVALUATOR_SENTINEL" not in serialized
    assert "prior selected exploration memory" in serialized
    assert json.loads(serialized)["exploration_boundary"] == {
        "start_utc": "2026-06-01T00:00:00Z",
        "end_utc": "2026-06-01T01:00:00Z",
        "row_count": 2,
    }


def test_provider_request_rejects_more_than_eight_hypotheses() -> None:
    provider = MockProvider([_valid_hypothesis()])

    with pytest.raises(ProviderError, match="at most 8"):
        run_provider(provider, _brief(), max_hypotheses=9, seed=123)


def test_mock_provider_validates_and_rejects_invalid_proposals_without_repair() -> None:
    valid = _valid_hypothesis()
    invalid = _valid_hypothesis()
    invalid["expression"] = {"node": "eval", "source": "secret()"}
    provider = MockProvider(
        [invalid, valid],
        name="mock-provider",
        model="mock-v1",
        prompt_version="prompt-v2",
        request_config={"temperature": 0.0},
    )

    run = run_provider(provider, _brief(), max_hypotheses=2, seed=42)

    assert run.metadata.provider_name == "mock-provider"
    assert run.metadata.model == "mock-v1"
    assert run.metadata.request_config == {"temperature": 0.0}
    assert run.metadata.prompt_version == "prompt-v2"
    assert run.metadata.seed == 42
    assert len(run.hypotheses) == 1
    assert run.hypotheses[0].title == valid["title"]
    assert len(run.rejected_proposals) == 1
    assert run.rejected_proposals[0].index == 0
    assert run.rejected_proposals[0].proposal_hash == hash_artifact(invalid)
    assert run.raw_proposals_sha256 == hash_artifact((invalid, valid))
    assert len(run.artifact_hash()) == 64


def test_file_provider_loads_proposals_for_tests(tmp_path: Path) -> None:
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(
        json.dumps({"proposals": [_valid_hypothesis()]}),
        encoding="utf-8",
    )
    provider = FileProvider(proposal_path)

    run = run_provider(provider, _brief(), max_hypotheses=1, seed=7)

    assert len(run.hypotheses) == 1
    assert run.metadata.provider_name == "file"
    assert run.metadata.seed == 7


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


def _valid_hypothesis() -> dict:
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
