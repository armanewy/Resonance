from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resonance.science import HypothesisSpec
from resonance.science.fitting import FittingError, fit_hypothesis


def test_known_linear_parameters_are_recovered_approximately() -> None:
    x = np.linspace(-4.0, 4.0, 40)
    frame = pd.DataFrame({"x": x, "y": 2.5 * x - 1.25})
    hypothesis = _hypothesis(
        {
            "node": "add",
            "left": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {"node": "metric", "metric": "x"},
            },
            "right": {"node": "fitted_parameter", "parameter": "offset"},
        },
        bounds={"scale": {"lower": -10.0, "upper": 10.0}, "offset": {"lower": -10.0, "upper": 10.0}},
    )

    result = fit_hypothesis(hypothesis, frame)

    assert result.fitted_parameters["scale"] == pytest.approx(2.5, abs=1.0e-6)
    assert result.fitted_parameters["offset"] == pytest.approx(-1.25, abs=1.0e-6)
    assert result.exploration_metrics["rmse"] < 1.0e-8
    assert result.deterministic_fit_artifact["optimizer_seed"] == hypothesis.random_seed


def test_lagged_relationship_is_fit_with_documented_lag_semantics() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="s", tz="UTC")
    x = np.arange(30, dtype="float64")
    y = pd.Series(x, index=index).shift(2) * 3.0 + 1.0
    frame = pd.DataFrame({"x": x, "y": y}, index=index)
    hypothesis = _hypothesis(
        {
            "node": "add",
            "left": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {
                    "node": "lag",
                    "input": {"node": "metric", "metric": "x"},
                    "lag_seconds": 2,
                },
            },
            "right": {"node": "fitted_parameter", "parameter": "offset"},
        },
        bounds={"scale": {"lower": 0.0, "upper": 10.0}, "offset": {"lower": -5.0, "upper": 5.0}},
        maximum_lag_seconds=2,
    )

    result = fit_hypothesis(hypothesis, frame)

    assert result.fitted_parameters["scale"] == pytest.approx(3.0, abs=1.0e-6)
    assert result.fitted_parameters["offset"] == pytest.approx(1.0, abs=1.0e-6)


def test_missing_data_is_ignored_without_losing_alignment() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [2.0, 4.0, 6.0, np.nan]})
    hypothesis = _hypothesis(
        {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": {"node": "metric", "metric": "x"},
        },
        bounds={"scale": {"lower": 0.0, "upper": 4.0}},
    )

    result = fit_hypothesis(hypothesis, frame)

    assert result.exploration_metrics["n"] == 2
    assert result.fitted_parameters["scale"] == pytest.approx(2.0)


def test_invalid_nonfinite_bounds_are_rejected() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    hypothesis = _hypothesis(
        {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": {"node": "metric", "metric": "x"},
        },
        bounds={"scale": {"lower": 0.0, "upper": float("inf")}},
    )

    with pytest.raises(FittingError, match="finite"):
        fit_hypothesis(hypothesis, frame)


def test_constant_inputs_fit_and_report_undefined_rank_correlation() -> None:
    frame = pd.DataFrame({"x": [1.0] * 8, "y": [5.0] * 8})
    hypothesis = _hypothesis(
        {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": {"node": "metric", "metric": "x"},
        },
        bounds={"scale": {"lower": 0.0, "upper": 10.0}},
    )

    result = fit_hypothesis(hypothesis, frame)

    assert result.fitted_parameters["scale"] == pytest.approx(5.0)
    assert result.exploration_metrics["spearman_rho"] is None
    assert any("spearman_rho is undefined" in warning for warning in result.warnings)


def test_division_near_zero_can_be_fit_without_infinite_residuals() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "denominator": [1.0, 0.0, 1.0e-12], "y": [2.0, 0.0, 0.0]})
    hypothesis = _hypothesis(
        {
            "node": "safe_divide",
            "numerator": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {"node": "metric", "metric": "x"},
            },
            "denominator": {"node": "metric", "metric": "denominator"},
            "epsilon": 1.0e-6,
            "near_zero_behavior": "return_zero",
        },
        bounds={"scale": {"lower": 0.0, "upper": 4.0}},
        inputs=("x", "denominator"),
    )

    result = fit_hypothesis(hypothesis, frame)

    assert result.fitted_parameters["scale"] == pytest.approx(2.0)
    assert np.isfinite(result.exploration_metrics["rmse"])


def test_fitting_is_deterministic() -> None:
    frame = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0) * 1.5})
    hypothesis = _hypothesis(
        {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": {"node": "metric", "metric": "x"},
        },
        bounds={"scale": {"lower": 0.0, "upper": 5.0}},
    )

    first = fit_hypothesis(hypothesis, frame)
    second = fit_hypothesis(hypothesis, frame)

    assert first.fitted_parameters == second.fitted_parameters
    assert first.deterministic_fit_artifact == second.deterministic_fit_artifact


def test_null_expression_does_not_beat_zero_residual_baseline() -> None:
    frame = pd.DataFrame({"x": np.arange(8.0), "y": np.zeros(8)})
    hypothesis = _hypothesis({"node": "numeric_constant", "value": 0.0}, bounds={})

    result = fit_hypothesis(hypothesis, frame)

    assert result.exploration_metrics["rmse"] == result.baseline_metrics["zero_residual"]["rmse"]
    assert result.exploration_metrics["rmse"] == 0.0
    assert result.complexity["penalty"] > 0.0


def _hypothesis(
    expression: dict,
    *,
    bounds: dict[str, dict[str, float]],
    inputs: tuple[str, ...] = ("x",),
    maximum_lag_seconds: int = 0,
) -> HypothesisSpec:
    return HypothesisSpec.model_validate(
        {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": "Synthetic fit",
            "concise_claim": "Synthetic input is associated with the target.",
            "rationale": "Deterministic unit-test fixture.",
            "target_metric": "y",
            "input_metrics": list(inputs),
            "target_transform": "identity",
            "expression": expression,
            "parameter_bounds": bounds,
            "expected_direction": "positive",
            "maximum_lag_seconds": maximum_lag_seconds,
            "fitting_metric": "rmse",
            "tuning_metric": "rmse",
            "blind_metrics": ["rmse", "spearman_r"],
            "minimum_blind_effect": 0.1,
            "minimum_baseline_improvement": 0.0,
            "negative_controls": [],
            "falsification_conditions": [{"description": "Synthetic falsification condition."}],
            "complexity_budget": {"max_ast_nodes": 20, "max_source_metrics": 3},
            "origin": "manual",
            "parent_hypothesis_ids": [],
            "random_seed": 12345,
        }
    )
