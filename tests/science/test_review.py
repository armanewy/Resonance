from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from resonance.science.contracts import HypothesisSpec
from resonance.science.review import (
    DeterministicReviewError,
    ReviewRecommendation,
    ReviewSpec,
    assert_hypothesis_valid,
    validate_hypotheses,
    validate_hypothesis,
)


def test_review_spec_rejects_unrequested_statistics() -> None:
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(
            {
                "confounders": ["seasonality"],
                "simpler_explanation": "Both metrics share a daily cycle.",
                "leakage_risk": "No direct target metric input is visible.",
                "mechanical_correlation_risk": "Shared denominator should be checked.",
                "suggested_controls_or_falsifications": ["Shuffle timestamps within day."],
                "executable": True,
                "distinct_from_prior": True,
                "recommendation": ReviewRecommendation.REVISE,
                "p_value": 0.01,
            }
        )


def test_valid_hypothesis_passes_deterministic_review() -> None:
    spec = assert_hypothesis_valid(
        _valid_hypothesis(),
        metric_catalog=_catalog(),
        snapshot_max_lag_seconds=900,
    )

    assert isinstance(spec, HypothesisSpec)


def test_review_rejects_unsupported_metrics() -> None:
    payload = _valid_hypothesis()
    payload["input_metrics"] = ["missing_metric"]
    payload["expression"] = {"node": "metric", "metric": "missing_metric"}

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert _codes(review) == {"unsupported_metrics"}


def test_review_rejects_unbounded_lag() -> None:
    payload = _valid_hypothesis()
    payload["expression"] = {
        "node": "lag",
        "input": {"node": "metric", "metric": "cpu_percent"},
    }

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "unbounded_lag" in _codes(review)


def test_review_rejects_lag_exceeding_snapshot_maximum() -> None:
    payload = _valid_hypothesis()

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=60)

    assert not review.accepted
    assert "lag_exceeds_snapshot" in _codes(review)


def test_review_rejects_lag_exceeding_declared_maximum() -> None:
    payload = _valid_hypothesis()
    payload["maximum_lag_seconds"] = 60

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "lag_exceeds_declared_maximum" in _codes(review)


def test_review_rejects_excessive_complexity() -> None:
    payload = _valid_hypothesis()
    expression = {"node": "metric", "metric": "cpu_percent"}
    for _ in range(4):
        expression = {
            "node": "add",
            "left": expression,
            "right": {"node": "numeric_constant", "value": 1.0},
        }
    payload["expression"] = expression
    payload["complexity_budget"] = {"max_ast_nodes": 3, "max_source_metrics": 3}

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "excessive_complexity" in _codes(review)


def test_review_rejects_duplicate_hypotheses() -> None:
    payload = _valid_hypothesis()

    first, second = validate_hypotheses(
        [payload, deepcopy(payload)],
        metric_catalog=_catalog(),
        snapshot_max_lag_seconds=900,
    )

    assert first.accepted
    assert not second.accepted
    assert _codes(second) == {"duplicate_hypothesis"}


def test_review_rejects_direct_target_leakage() -> None:
    payload = _valid_hypothesis()
    payload["input_metrics"] = ["cpu_percent", "temperature_2m"]

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "direct_target_leakage" in _codes(review)


def test_review_rejects_expression_using_future_target_values() -> None:
    payload = _valid_hypothesis()
    payload["expression"] = {"node": "lag", "input": {"node": "metric", "metric": "temperature_2m"}, "lag_seconds": -60}

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "future_target_values" in _codes(review)


def test_review_rejects_missing_negative_controls() -> None:
    payload = _valid_hypothesis()
    payload["negative_controls"] = []

    review = validate_hypothesis(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)

    assert not review.accepted
    assert "missing_negative_controls" in _codes(review)


def test_raise_for_issues_reports_deterministic_rejections() -> None:
    payload = _valid_hypothesis()
    payload["negative_controls"] = []

    with pytest.raises(DeterministicReviewError, match="negative controls"):
        assert_hypothesis_valid(payload, metric_catalog=_catalog(), snapshot_max_lag_seconds=900)


def _codes(review) -> set[str]:
    return {issue.code for issue in review.issues}


def _catalog() -> dict:
    return {
        "metric_names": [
            "battery_percent",
            "cpu_percent",
            "memory_percent",
            "temperature_2m",
        ]
    }


def _valid_hypothesis() -> dict:
    return {
        "schema_version": "1.0",
        "hypothesis_type": "observational_prediction",
        "title": "CPU pressure is associated with warmer local residuals",
        "concise_claim": "CPU utilization is associated with transformed local temperature residuals.",
        "rationale": "A local heat/load proxy may move with weather-station residuals after robust scaling.",
        "target_metric": "temperature_2m",
        "input_metrics": ["cpu_percent", "memory_percent"],
        "target_transform": "robust_zscore",
        "expression": {
            "node": "add",
            "left": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {
                    "node": "lag",
                    "input": {"node": "metric", "metric": "cpu_percent"},
                    "lag_seconds": 300,
                },
            },
            "right": {"node": "metric", "metric": "memory_percent"},
        },
        "parameter_bounds": {"scale": {"lower": 0.0, "upper": 3.0}},
        "expected_direction": "positive",
        "maximum_lag_seconds": 900,
        "fitting_metric": "rmse",
        "tuning_metric": "mae",
        "blind_metrics": ["rmse", "spearman_r"],
        "minimum_blind_effect": 0.1,
        "minimum_baseline_improvement": 0.02,
        "negative_controls": [
            {"metric": "battery_percent", "rationale": "Battery level should not match the target association."}
        ],
        "falsification_conditions": [
            {"description": "Fails if the association vanishes under a seasonal block control."}
        ],
        "complexity_budget": {"max_ast_nodes": 15, "max_source_metrics": 3},
        "origin": "manual",
        "parent_hypothesis_ids": [],
        "random_seed": 8675309,
    }
