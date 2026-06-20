from __future__ import annotations

import pytest

from resonance.science.contracts import (
    HypothesisSpec,
    Origin,
    expression_lag_seconds,
    expression_node_count,
    expression_parameters,
)
from resonance.science.mutation import (
    MutationConfig,
    MutationError,
    MutationOperator,
    mutate_hypothesis,
    mutate_one,
)


def test_mutation_is_deterministic_for_seed_and_preserves_parent_lineage() -> None:
    parent = _hypothesis()

    first = mutate_hypothesis(parent, seed=22, config=MutationConfig(max_children=8))
    second = mutate_hypothesis(parent, seed=22, config=MutationConfig(max_children=8))
    third = mutate_hypothesis(parent, seed=23, config=MutationConfig(max_children=8))

    assert [child.hypothesis_hash() for child in first] == [child.hypothesis_hash() for child in second]
    assert [child.hypothesis_hash() for child in first] != [child.hypothesis_hash() for child in third]
    assert first
    for child in first:
        assert child.origin == Origin.MUTATION
        assert child.parent_hypothesis_ids == ("prior-parent", parent.hypothesis_hash())
        assert child.hypothesis_hash() != parent.hypothesis_hash()
        assert expression_node_count(child.expression) <= child.complexity_budget.max_ast_nodes


@pytest.mark.parametrize(
    ("operator", "predicate"),
    [
        (MutationOperator.ADD_LAG, lambda parent, child: len(expression_lag_seconds(child.expression)) > len(expression_lag_seconds(parent.expression))),
        (MutationOperator.REMOVE_LAG, lambda parent, child: len(expression_lag_seconds(child.expression)) < len(expression_lag_seconds(parent.expression))),
        (MutationOperator.CHANGE_LAG, lambda parent, child: expression_lag_seconds(child.expression) != expression_lag_seconds(parent.expression)),
        (MutationOperator.ADD_ROLLING, lambda parent, child: _node_count(child, "rolling_mean") + _node_count(child, "rolling_std") + _node_count(child, "robust_zscore") > _node_count(parent, "rolling_mean") + _node_count(parent, "rolling_std") + _node_count(parent, "robust_zscore")),
        (MutationOperator.REMOVE_ROLLING, lambda parent, child: _node_count(child, "rolling_mean") + _node_count(child, "rolling_std") + _node_count(child, "robust_zscore") < _node_count(parent, "rolling_mean") + _node_count(parent, "rolling_std") + _node_count(parent, "robust_zscore")),
        (MutationOperator.CHANGE_ROLLING_WINDOW, lambda parent, child: _rolling_windows(parent) != _rolling_windows(child)),
        (MutationOperator.ADD_INTERACTION, lambda parent, child: _node_count(child, "multiply") > _node_count(parent, "multiply")),
        (MutationOperator.REMOVE_BRANCH, lambda parent, child: expression_node_count(child.expression) < expression_node_count(parent.expression)),
        (MutationOperator.REPLACE_OPERATOR, lambda parent, child: child.expression.model_dump(mode="json") != parent.expression.model_dump(mode="json")),
        (MutationOperator.ADD_COEFFICIENT, lambda parent, child: len(expression_parameters(child.expression)) > len(expression_parameters(parent.expression))),
        (MutationOperator.REMOVE_COEFFICIENT, lambda parent, child: len(expression_parameters(child.expression)) < len(expression_parameters(parent.expression))),
    ],
)
def test_each_mutation_operator_returns_valid_bounded_child(operator, predicate) -> None:
    parent = _hypothesis(max_ast_nodes=20)

    child = mutate_one(parent, operator, seed=101, config=MutationConfig(lag_seconds=(0, 300, 600, 900)))

    assert isinstance(child, HypothesisSpec)
    assert predicate(parent, child)
    assert expression_node_count(child.expression) <= child.complexity_budget.max_ast_nodes
    assert all(lag <= child.maximum_lag_seconds for lag in expression_lag_seconds(child.expression))
    assert child.hypothesis_hash() != parent.hypothesis_hash()


def test_change_lag_stays_within_configured_bounds() -> None:
    parent = _hypothesis(maximum_lag_seconds=600)

    child = mutate_one(
        parent,
        MutationOperator.CHANGE_LAG,
        seed=7,
        config=MutationConfig(lag_seconds=(0, 300, 600), max_children=1),
    )

    assert set(expression_lag_seconds(child.expression)) <= {0, 300, 600}
    assert child.maximum_lag_seconds == 600


def test_simplify_removes_algebraically_equivalent_structure() -> None:
    parent = _hypothesis(
        expression={
            "node": "add",
            "left": _linear_expression({"node": "metric", "metric": "x"}),
            "right": {"node": "numeric_constant", "value": 0.0},
        },
        max_ast_nodes=20,
    )

    child = mutate_one(parent, MutationOperator.SIMPLIFY, seed=31)

    assert expression_node_count(child.expression) < expression_node_count(parent.expression)
    child_expression = child.expression.model_dump(mode="json")
    assert _node_count_raw(child_expression, "numeric_constant") == 0
    assert expression_parameters(child.expression) == {"scale", "offset"}


def test_mutations_are_deduplicated_by_equivalent_expression() -> None:
    parent = _hypothesis(
        expression={
            "node": "add",
            "left": {"node": "numeric_constant", "value": 0.0},
            "right": _linear_expression({"node": "metric", "metric": "x"}),
        },
        max_ast_nodes=20,
    )

    children = mutate_hypothesis(
        parent,
        seed=9,
        config=MutationConfig(max_children=12, operators=(MutationOperator.SIMPLIFY, MutationOperator.REMOVE_BRANCH)),
    )

    expressions = [child.expression.model_dump(mode="json") for child in children]
    assert len(expressions) == len({str(expression) for expression in expressions})


def test_mutation_filters_children_that_exceed_complexity_budget() -> None:
    parent = _hypothesis(max_ast_nodes=1, expression={"node": "metric", "metric": "x"}, parameter_bounds={})

    children = mutate_hypothesis(
        parent,
        seed=3,
        config=MutationConfig(
            max_children=4,
            operators=(MutationOperator.ADD_LAG, MutationOperator.ADD_ROLLING, MutationOperator.ADD_COEFFICIENT),
        ),
    )

    assert children == ()


def test_mutation_refuses_target_leakage_and_future_target_references() -> None:
    valid = _hypothesis()
    leaking = {
        **valid.model_dump(mode="json"),
        "target_metric": "y",
        "input_metrics": ["y"],
        "expression": {"node": "metric", "metric": "y"},
        "parameter_bounds": {},
    }

    with pytest.raises(MutationError, match="contract-valid"):
        mutate_hypothesis(leaking, seed=1)

    parent = _hypothesis(maximum_lag_seconds=900)
    with pytest.raises(MutationError, match="snapshot"):
        mutate_hypothesis(parent, seed=1, snapshot_max_lag_seconds=300)


def test_child_hashes_are_distinct_for_distinct_child_content() -> None:
    parent = _hypothesis(max_ast_nodes=20)

    children = mutate_hypothesis(parent, seed=5, config=MutationConfig(max_children=12))
    hashes = [child.hypothesis_hash() for child in children]
    expressions = [child.expression.model_dump(mode="json") for child in children]

    assert len(hashes) == len(set(hashes))
    assert len(expressions) == len({str(expression) for expression in expressions})


def test_add_coefficient_uses_configured_bounds() -> None:
    parent = _hypothesis(max_ast_nodes=20)

    child = mutate_one(
        parent,
        MutationOperator.ADD_COEFFICIENT,
        seed=12,
        config=MutationConfig(coefficient_bounds={"lower": -0.25, "upper": 0.25}),
    )

    assert child.parameter_bounds["coef_1"].lower == -0.25
    assert child.parameter_bounds["coef_1"].upper == 0.25


def _hypothesis(
    *,
    target_metric: str = "y",
    input_metrics: list[str] | None = None,
    expression: dict | None = None,
    parameter_bounds: dict | None = None,
    maximum_lag_seconds: int = 900,
    max_ast_nodes: int = 15,
) -> HypothesisSpec:
    expression = expression or _linear_expression(
        {
            "node": "lag",
            "input": {
                "node": "rolling_mean",
                "input": {"node": "metric", "metric": "x"},
                "window_seconds": 900,
                "min_periods": 3,
            },
            "lag_seconds": 300,
        }
    )
    if parameter_bounds is None:
        parameter_bounds = {
            "scale": {"lower": -5.0, "upper": 5.0},
            "offset": {"lower": -20.0, "upper": 20.0},
        }
    return HypothesisSpec.model_validate(
        {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": "x predicts y",
            "concise_claim": "x is associated with y.",
            "rationale": "Test fixture.",
            "target_metric": target_metric,
            "input_metrics": input_metrics or ["x"],
            "target_transform": "identity",
            "expression": expression,
            "parameter_bounds": parameter_bounds,
            "expected_direction": "positive",
            "maximum_lag_seconds": maximum_lag_seconds,
            "fitting_metric": "rmse",
            "tuning_metric": "rmse",
            "blind_metrics": ["rmse", "mae", "spearman_r"],
            "minimum_blind_effect": 0.1,
            "minimum_baseline_improvement": 0.01,
            "negative_controls": [{"metric": "control", "rationale": "Control should not reproduce the effect."}],
            "falsification_conditions": [{"description": "No tuning improvement."}],
            "complexity_budget": {"max_ast_nodes": max_ast_nodes, "max_source_metrics": 3},
            "origin": "manual",
            "parent_hypothesis_ids": ["prior-parent"],
            "random_seed": 44,
        }
    )


def _linear_expression(input_expression: dict) -> dict:
    return {
        "node": "add",
        "left": {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": input_expression,
        },
        "right": {"node": "fitted_parameter", "parameter": "offset"},
    }


def _node_count(hypothesis: HypothesisSpec, node_name: str) -> int:
    return _node_count_raw(hypothesis.expression.model_dump(mode="json"), node_name)


def _node_count_raw(expression: dict, node_name: str) -> int:
    total = 1 if expression.get("node") == node_name else 0
    for child_name in ("left", "right", "numerator", "denominator", "input"):
        child = expression.get(child_name)
        if isinstance(child, dict):
            total += _node_count_raw(child, node_name)
    return total


def _rolling_windows(hypothesis: HypothesisSpec) -> tuple[int, ...]:
    return tuple(_rolling_windows_raw(hypothesis.expression.model_dump(mode="json")))


def _rolling_windows_raw(expression: dict) -> list[int]:
    values = [int(expression["window_seconds"])] if expression.get("node") in {"rolling_mean", "rolling_std", "robust_zscore"} else []
    for child_name in ("left", "right", "numerator", "denominator", "input"):
        child = expression.get(child_name)
        if isinstance(child, dict):
            values.extend(_rolling_windows_raw(child))
    return values
