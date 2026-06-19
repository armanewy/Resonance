from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resonance.science.contracts import (
    AddNode,
    LagNode,
    MetricNode,
    NearZeroBehavior,
    NumericConstantNode,
    RollingMeanNode,
    SafeDivideNode,
)
from resonance.science.interpreter import ExecutionLimits, ExpressionExecutionError, evaluate_expression


def test_lag_moves_observations_forward_without_future_leakage() -> None:
    frame = pd.DataFrame(
        {"x": [10.0, 20.0, 30.0, 40.0]},
        index=pd.date_range("2026-01-01", periods=4, freq="s", tz="UTC"),
    )
    expression = LagNode(node="lag", input=MetricNode(node="metric", metric="x"), lag_seconds=2)

    result = evaluate_expression(expression, frame)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2:].tolist() == [10.0, 20.0]


def test_rolling_mean_uses_only_present_and_past_observations() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, 1.0, 100.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC"),
    )
    expression = RollingMeanNode(
        node="rolling_mean",
        input=MetricNode(node="metric", metric="x"),
        window_seconds=2,
        min_periods=1,
    )

    result = evaluate_expression(expression, frame)

    assert result.iloc[1] == 1.0
    assert result.iloc[2] > 1.0


def test_missing_values_are_preserved_through_arithmetic() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    expression = AddNode(
        node="add",
        left=MetricNode(node="metric", metric="x"),
        right=NumericConstantNode(node="numeric_constant", value=1.0),
    )

    result = evaluate_expression(expression, frame)

    assert result.tolist()[0] == 2.0
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 4.0


def test_safe_division_near_zero_does_not_create_infinities() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "denominator": [1.0, 0.0, 1.0e-12]})
    expression = SafeDivideNode(
        node="safe_divide",
        numerator=MetricNode(node="metric", metric="x"),
        denominator=MetricNode(node="metric", metric="denominator"),
        epsilon=1.0e-6,
        near_zero_behavior=NearZeroBehavior.RETURN_ZERO,
    )

    result = evaluate_expression(expression, frame)

    assert np.isfinite(result).all()
    assert result.tolist() == [1.0, 0.0, 0.0]


def test_unknown_metric_is_rejected_before_execution() -> None:
    frame = pd.DataFrame({"x": [1.0]})
    expression = MetricNode(node="metric", metric="missing")

    with pytest.raises(ExpressionExecutionError, match="unknown metrics"):
        evaluate_expression(expression, frame)


def test_excessive_expression_is_rejected_before_execution() -> None:
    frame = pd.DataFrame({"x": [1.0]})
    expression = AddNode(
        node="add",
        left=MetricNode(node="metric", metric="x"),
        right=NumericConstantNode(node="numeric_constant", value=1.0),
    )

    with pytest.raises(ExpressionExecutionError, match="complexity"):
        evaluate_expression(expression, frame, limits=ExecutionLimits(max_ast_nodes=2))
