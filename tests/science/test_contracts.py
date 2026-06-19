from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from resonance.science import HypothesisSpec


def test_valid_hypothesis_round_trips_through_json() -> None:
    spec = HypothesisSpec.model_validate(_valid_hypothesis())

    parsed = HypothesisSpec.model_validate_json(spec.canonical_json())

    assert parsed == spec
    assert json.loads(parsed.canonical_json())["hypothesis_type"] == "observational_prediction"


def test_invalid_operator_fails() -> None:
    payload = _valid_hypothesis()
    payload["expression"] = {"node": "eval", "source": "metric('cpu_percent')"}

    with pytest.raises(ValidationError):
        HypothesisSpec.model_validate(payload)


def test_excessive_complexity_fails() -> None:
    payload = _valid_hypothesis()
    expression = {"node": "metric", "metric": "cpu_percent"}
    for _ in range(15):
        expression = {
            "node": "add",
            "left": expression,
            "right": {"node": "numeric_constant", "value": 1.0},
        }
    payload["expression"] = expression

    with pytest.raises(ValidationError, match="complexity"):
        HypothesisSpec.model_validate(payload)


def test_unknown_metrics_can_be_rejected_against_supplied_catalog() -> None:
    payload = _valid_hypothesis()

    with pytest.raises(ValidationError, match="unknown metrics"):
        HypothesisSpec.model_validate(
            payload,
            context={"metric_catalog": {"cpu_percent", "not_the_target"}},
        )


def test_snapshot_metric_catalog_can_validate_hypothesis_references() -> None:
    payload = _valid_hypothesis()
    catalog = {
        "catalog_id": "a" * 64,
        "metric_names": [
            "battery_percent",
            "cpu_percent",
            "memory_percent",
            "temperature_2m",
        ],
        "metrics": [],
    }
    payload["snapshot_metric_catalog_id"] = catalog["catalog_id"]

    spec = HypothesisSpec.model_validate(payload, context={"metric_catalog": catalog})

    assert spec.snapshot_metric_catalog_id == catalog["catalog_id"]


def test_snapshot_metric_catalog_id_must_match_supplied_catalog() -> None:
    payload = _valid_hypothesis()
    payload["snapshot_metric_catalog_id"] = "b" * 64

    with pytest.raises(ValidationError, match="catalog id"):
        HypothesisSpec.model_validate(
            payload,
            context={
                "metric_catalog": {
                    "catalog_id": "a" * 64,
                    "metric_names": [
                        "battery_percent",
                        "cpu_percent",
                        "memory_percent",
                        "temperature_2m",
                    ],
                }
            },
        )


def test_lag_cannot_exceed_declared_maximum() -> None:
    payload = _valid_hypothesis()
    payload["maximum_lag_seconds"] = 60

    with pytest.raises(ValidationError, match="maximum lag"):
        HypothesisSpec.model_validate(payload)


def test_hashes_are_stable_under_dictionary_ordering() -> None:
    first = _valid_hypothesis()
    second = _valid_hypothesis()
    second["parameter_bounds"] = {
        "offset": {"upper": 1.0, "lower": -1.0},
        "scale": {"upper": 3.0, "lower": 0.0},
    }

    assert HypothesisSpec.model_validate(first).hypothesis_hash() == HypothesisSpec.model_validate(second).hypothesis_hash()


def test_changing_executable_claim_changes_hash() -> None:
    original = HypothesisSpec.model_validate(_valid_hypothesis())
    changed = _valid_hypothesis()
    changed["expression"] = {
        "node": "add",
        "left": changed["expression"],
        "right": {"node": "numeric_constant", "value": 0.5},
    }

    assert HypothesisSpec.model_validate(changed).hypothesis_hash() != original.hypothesis_hash()


def test_changing_only_presentation_formatting_does_not_change_hash() -> None:
    original = HypothesisSpec.model_validate(_valid_hypothesis())
    changed = _valid_hypothesis()
    changed["title"] = "  CPU residual claim  "
    changed["concise_claim"] = "CPU predicts temperature residuals in this dataset."
    changed["rationale"] = "Same scientific content with different wrapping.\n\nNo executable field changed."

    assert HypothesisSpec.model_validate(changed).hypothesis_hash() == original.hypothesis_hash()


def test_checked_in_json_schema_matches_model_schema() -> None:
    schema_path = Path(__file__).parents[2] / "resonance" / "science" / "hypothesis_schema.json"
    checked_in = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = HypothesisSpec.model_json_schema()

    assert checked_in == generated


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
        "minimum_baseline_improvement": 0.02,
        "negative_controls": [
            {"metric": "battery_percent", "rationale": "Battery level should not predict the same residual."}
        ],
        "falsification_conditions": [
            {"description": "Blind RMSE fails to improve over the preregistered baseline."},
            {"description": "Negative control effect meets or exceeds the target effect."},
        ],
        "complexity_budget": {"max_ast_nodes": 15, "max_source_metrics": 3},
        "origin": "manual",
        "parent_hypothesis_ids": [],
        "random_seed": 8675309,
    }
