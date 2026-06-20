from __future__ import annotations

import random
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import Field, ValidationError

from resonance.science.contracts import (
    HypothesisSpec,
    NearZeroBehavior,
    Origin,
    ParameterBounds,
    StrictModel,
    expression_lag_seconds,
    expression_metrics,
    expression_node_count,
    expression_parameters,
    stable_hash,
)
from resonance.science.review import validate_hypothesis


DEFAULT_LAG_SECONDS = (0, 300, 900)
DEFAULT_ROLLING_WINDOWS_SECONDS = (300, 900, 1800)
DEFAULT_PARAMETER_BOUNDS = ParameterBounds(lower=-5.0, upper=5.0)


class MutationError(ValueError):
    """Raised when a parent hypothesis cannot be mutated safely."""


class MutationOperator(str, Enum):
    ADD_LAG = "add_lag"
    REMOVE_LAG = "remove_lag"
    CHANGE_LAG = "change_lag"
    ADD_ROLLING = "add_rolling"
    REMOVE_ROLLING = "remove_rolling"
    CHANGE_ROLLING_WINDOW = "change_rolling_window"
    ADD_INTERACTION = "add_interaction"
    REMOVE_BRANCH = "remove_branch"
    REPLACE_OPERATOR = "replace_operator"
    ADD_COEFFICIENT = "add_coefficient"
    REMOVE_COEFFICIENT = "remove_coefficient"
    SIMPLIFY = "simplify"


class MutationConfig(StrictModel):
    """Bounded deterministic mutation options for Wave 4B callers."""

    max_children: int = Field(default=16, gt=0)
    lag_seconds: tuple[int, ...] = DEFAULT_LAG_SECONDS
    rolling_windows_seconds: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS_SECONDS
    permitted_interactions: tuple[str, ...] = ("multiply",)
    coefficient_bounds: ParameterBounds = DEFAULT_PARAMETER_BOUNDS
    operators: tuple[MutationOperator, ...] = tuple(MutationOperator)


def mutate_hypothesis(
    parent: HypothesisSpec | Mapping[str, Any],
    *,
    seed: int,
    config: MutationConfig | None = None,
    metric_catalog: Any | None = None,
    snapshot_max_lag_seconds: int | None = None,
) -> tuple[HypothesisSpec, ...]:
    """Return deterministic, contract-valid one-step mutations of a hypothesis."""

    config = config or MutationConfig()
    parsed_parent = _parse_parent(parent, metric_catalog)
    _raise_for_leakage(parsed_parent, metric_catalog, snapshot_max_lag_seconds)

    parent_payload = parsed_parent.model_dump(mode="json", exclude_none=True)
    parent_hash = parsed_parent.hypothesis_hash()
    raw_candidates: list[dict[str, Any]] = []
    for operator in config.operators:
        for expression in _operator_expressions(operator, parsed_parent, config):
            raw_candidates.append(
                {
                    "operator": operator.value,
                    "payload": _child_payload(
                        parent_payload,
                        expression=expression,
                        parent_hash=parent_hash,
                        seed=seed,
                    ),
                }
            )

    raw_candidates.sort(key=stable_hash)
    rng = random.Random(seed)
    rng.shuffle(raw_candidates)

    children: list[HypothesisSpec] = []
    seen_hashes = {parent_hash}
    seen_expressions: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        payload = dict(raw_candidate["payload"])
        payload["random_seed"] = _child_seed(seed, raw_candidate["operator"], index, payload["expression"])
        try:
            child = HypothesisSpec.model_validate(
                _with_expression_derived_bounds(payload, config),
                context={"metric_catalog": metric_catalog} if metric_catalog is not None else None,
            )
        except ValidationError:
            continue
        review = validate_hypothesis(
            child,
            metric_catalog=metric_catalog or _catalog_from_hypothesis(child),
            snapshot_max_lag_seconds=snapshot_max_lag_seconds,
        )
        if not review.accepted:
            continue
        if expression_node_count(child.expression) > child.complexity_budget.max_ast_nodes:
            continue
        child_hash = child.hypothesis_hash()
        expression_key = stable_hash(_simplified_expression(child.expression.model_dump(mode="json")))
        if child_hash in seen_hashes or expression_key in seen_expressions:
            continue
        seen_hashes.add(child_hash)
        seen_expressions.add(expression_key)
        children.append(child)
        if len(children) >= config.max_children:
            break
    return tuple(children)


def mutate_one(
    parent: HypothesisSpec | Mapping[str, Any],
    operator: MutationOperator,
    *,
    seed: int,
    config: MutationConfig | None = None,
    metric_catalog: Any | None = None,
    snapshot_max_lag_seconds: int | None = None,
) -> HypothesisSpec:
    """Return the first deterministic valid child for one mutation operator."""

    base = config or MutationConfig()
    scoped = base.model_copy(update={"max_children": 1, "operators": (operator,)})
    children = mutate_hypothesis(
        parent,
        seed=seed,
        config=scoped,
        metric_catalog=metric_catalog,
        snapshot_max_lag_seconds=snapshot_max_lag_seconds,
    )
    if not children:
        raise MutationError(f"no valid {operator.value} mutation available")
    return children[0]


def _parse_parent(parent: HypothesisSpec | Mapping[str, Any], metric_catalog: Any | None) -> HypothesisSpec:
    try:
        if isinstance(parent, HypothesisSpec):
            return parent
        return HypothesisSpec.model_validate(
            parent,
            context={"metric_catalog": metric_catalog} if metric_catalog is not None else None,
        )
    except ValidationError as exc:
        raise MutationError("parent hypothesis is not contract-valid") from exc


def _raise_for_leakage(
    parent: HypothesisSpec,
    metric_catalog: Any | None,
    snapshot_max_lag_seconds: int | None,
) -> None:
    review = validate_hypothesis(
        parent,
        metric_catalog=metric_catalog or _catalog_from_hypothesis(parent),
        snapshot_max_lag_seconds=snapshot_max_lag_seconds,
    )
    if not review.accepted:
        messages = "; ".join(issue.message for issue in review.issues)
        raise MutationError(messages)


def _operator_expressions(
    operator: MutationOperator,
    parent: HypothesisSpec,
    config: MutationConfig,
) -> Iterable[dict[str, Any]]:
    expression = parent.expression.model_dump(mode="json")
    if operator is MutationOperator.ADD_LAG:
        yield from _add_lag(expression, parent, config)
    elif operator is MutationOperator.REMOVE_LAG:
        yield from _replace_matching(expression, _remove_lag_at)
    elif operator is MutationOperator.CHANGE_LAG:
        yield from _change_lag(expression, parent, config)
    elif operator is MutationOperator.ADD_ROLLING:
        yield from _add_rolling(expression, config)
    elif operator is MutationOperator.REMOVE_ROLLING:
        yield from _replace_matching(expression, _remove_rolling_at)
    elif operator is MutationOperator.CHANGE_ROLLING_WINDOW:
        yield from _change_rolling_window(expression, config)
    elif operator is MutationOperator.ADD_INTERACTION:
        yield from _add_interaction(expression, parent, config)
    elif operator is MutationOperator.REMOVE_BRANCH:
        yield from _replace_matching(expression, _remove_branch_at)
    elif operator is MutationOperator.REPLACE_OPERATOR:
        yield from _replace_operator(expression)
    elif operator is MutationOperator.ADD_COEFFICIENT:
        yield from _add_coefficient(expression, parent)
    elif operator is MutationOperator.REMOVE_COEFFICIENT:
        yield from _replace_matching(expression, _remove_coefficient_at)
    elif operator is MutationOperator.SIMPLIFY:
        simplified = _simplified_expression(expression)
        if simplified != expression:
            yield simplified


def _child_payload(
    parent_payload: Mapping[str, Any],
    *,
    expression: Mapping[str, Any],
    parent_hash: str,
    seed: int,
) -> dict[str, Any]:
    payload = dict(parent_payload)
    parent_ids = tuple(parent_payload.get("parent_hypothesis_ids", ()) or ())
    lineage = tuple(dict.fromkeys((*parent_ids, parent_hash)))
    payload.update(
        {
            "title": f"Mutation of {parent_payload['title']}",
            "concise_claim": f"Mutated variant of: {parent_payload['concise_claim']}",
            "rationale": f"Deterministic bounded mutation of parent hypothesis {parent_hash}.",
            "origin": Origin.MUTATION.value,
            "parent_hypothesis_ids": list(lineage),
            "expression": _simplified_expression(dict(expression)),
            "random_seed": int(seed),
        }
    )
    return payload


def _child_seed(seed: int, operator: str, index: int, expression: Mapping[str, Any]) -> int:
    digest = stable_hash({"seed": seed, "operator": operator, "index": index, "expression": expression})
    return int(digest[:15], 16)


def _with_expression_derived_bounds(payload: Mapping[str, Any], config: MutationConfig) -> dict[str, Any]:
    value = dict(payload)
    expression_parameters = _raw_parameters(value["expression"])
    existing = dict(value.get("parameter_bounds", {}))
    value["parameter_bounds"] = {
        parameter: existing.get(parameter, config.coefficient_bounds.model_dump(mode="json"))
        for parameter in sorted(expression_parameters)
    }
    value["input_metrics"] = sorted(_raw_metrics(value["expression"]))
    max_lag = max(_raw_lags(value["expression"]), default=0)
    value["maximum_lag_seconds"] = max(int(value.get("maximum_lag_seconds", 0)), max_lag)
    return value


def _add_lag(
    expression: Mapping[str, Any],
    parent: HypothesisSpec,
    config: MutationConfig,
) -> Iterable[dict[str, Any]]:
    max_lag = _effective_max_lag(parent, config)
    for lag_seconds in _bounded_lags(max_lag, config):
        if lag_seconds == 0 and max_lag > 0:
            continue
        yield from _replace_matching(
            expression,
            lambda node, lag_seconds=lag_seconds: (
                {"node": "lag", "input": dict(node), "lag_seconds": lag_seconds}
                if node.get("node") != "lag"
                else None
            ),
        )


def _change_lag(
    expression: Mapping[str, Any],
    parent: HypothesisSpec,
    config: MutationConfig,
) -> Iterable[dict[str, Any]]:
    max_lag = _effective_max_lag(parent, config)
    for lag_seconds in _bounded_lags(max_lag, config):
        yield from _replace_matching(
            expression,
            lambda node, lag_seconds=lag_seconds: (
                {**node, "lag_seconds": lag_seconds}
                if node.get("node") == "lag" and int(node.get("lag_seconds", -1)) != lag_seconds
                else None
            ),
        )


def _add_rolling(expression: Mapping[str, Any], config: MutationConfig) -> Iterable[dict[str, Any]]:
    for window_seconds in _positive_values(config.rolling_windows_seconds):
        for node_name, min_periods in (
            ("rolling_mean", 1),
            ("rolling_std", 2),
            ("robust_zscore", 5),
        ):
            yield from _replace_matching(
                expression,
                lambda node, node_name=node_name, window_seconds=window_seconds, min_periods=min_periods: (
                    {
                        "node": node_name,
                        "input": dict(node),
                        "window_seconds": window_seconds,
                        "min_periods": min_periods,
                    }
                    if node.get("node") not in _ROLLING_NODES
                    else None
                ),
            )


def _change_rolling_window(
    expression: Mapping[str, Any],
    config: MutationConfig,
) -> Iterable[dict[str, Any]]:
    for window_seconds in _positive_values(config.rolling_windows_seconds):
        yield from _replace_matching(
            expression,
            lambda node, window_seconds=window_seconds: (
                {**node, "window_seconds": window_seconds}
                if node.get("node") in _ROLLING_NODES and int(node.get("window_seconds", -1)) != window_seconds
                else None
            ),
        )


def _add_interaction(
    expression: Mapping[str, Any],
    parent: HypothesisSpec,
    config: MutationConfig,
) -> Iterable[dict[str, Any]]:
    source_metrics = sorted(set(parent.input_metrics) - {parent.target_metric})
    expression_key = stable_hash(expression)
    for metric in source_metrics:
        metric_node = {"node": "metric", "metric": metric}
        if stable_hash(metric_node) == expression_key:
            continue
        if "multiply" in config.permitted_interactions:
            yield {"node": "multiply", "left": dict(expression), "right": metric_node}


def _replace_operator(expression: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    def replace(node: Mapping[str, Any]) -> dict[str, Any] | None:
        node_name = node.get("node")
        if node_name in {"add", "subtract", "multiply"}:
            replacements = {
                "add": "subtract",
                "subtract": "add",
                "multiply": "add",
            }
            return {**node, "node": replacements[str(node_name)]}
        if node_name in _ROLLING_NODES:
            replacements = {
                "rolling_mean": ("rolling_std", 2),
                "rolling_std": ("rolling_mean", 1),
                "robust_zscore": ("rolling_mean", 1),
            }
            new_name, min_periods = replacements[str(node_name)]
            return {**node, "node": new_name, "min_periods": min_periods}
        return None

    yield from _replace_matching(expression, replace)


def _add_coefficient(expression: Mapping[str, Any], parent: HypothesisSpec) -> Iterable[dict[str, Any]]:
    existing = set(parent.parameter_bounds)
    index = 1
    while f"coef_{index}" in existing:
        index += 1
    yield {
        "node": "multiply",
        "left": {"node": "fitted_parameter", "parameter": f"coef_{index}"},
        "right": dict(expression),
    }


def _remove_lag_at(node: Mapping[str, Any]) -> dict[str, Any] | None:
    if node.get("node") == "lag":
        return dict(node["input"])
    return None


def _remove_rolling_at(node: Mapping[str, Any]) -> dict[str, Any] | None:
    if node.get("node") in _ROLLING_NODES:
        return dict(node["input"])
    return None


def _remove_branch_at(node: Mapping[str, Any]) -> dict[str, Any] | None:
    node_name = node.get("node")
    if node_name in {"add", "subtract", "multiply"}:
        return dict(node["left"])
    if node_name == "safe_divide":
        return dict(node["numerator"])
    return None


def _remove_coefficient_at(node: Mapping[str, Any]) -> dict[str, Any] | None:
    if node.get("node") == "multiply":
        left = node["left"]
        right = node["right"]
        if isinstance(left, Mapping) and left.get("node") == "fitted_parameter":
            return dict(right)
        if isinstance(right, Mapping) and right.get("node") == "fitted_parameter":
            return dict(left)
    if node.get("node") in {"add", "subtract"}:
        left = node["left"]
        right = node["right"]
        if isinstance(right, Mapping) and right.get("node") == "fitted_parameter":
            return dict(left)
        if node.get("node") == "add" and isinstance(left, Mapping) and left.get("node") == "fitted_parameter":
            return dict(right)
    return None


def _replace_matching(
    expression: Mapping[str, Any],
    replacement: Any,
) -> Iterable[dict[str, Any]]:
    replaced = replacement(expression)
    if replaced is not None:
        yield _simplified_expression(replaced)
    for child_name, child in _child_items(expression):
        for child_replacement in _replace_matching(child, replacement):
            updated = dict(expression)
            updated[child_name] = child_replacement
            yield _simplified_expression(updated)


def _child_items(expression: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    node = expression.get("node")
    if node in {"add", "subtract", "multiply"}:
        return (("left", expression["left"]), ("right", expression["right"]))
    if node == "safe_divide":
        return (("numerator", expression["numerator"]), ("denominator", expression["denominator"]))
    if node in {"absolute_value", "clip", "difference", "lag", *_ROLLING_NODES}:
        return (("input", expression["input"]),)
    return ()


def _simplified_expression(expression: Mapping[str, Any]) -> dict[str, Any]:
    node = expression.get("node")
    if node in {"add", "subtract", "multiply"}:
        left = _simplified_expression(expression["left"])
        right = _simplified_expression(expression["right"])
        if node == "add":
            if _is_constant(left, 0.0):
                return right
            if _is_constant(right, 0.0):
                return left
            left, right = _ordered_commutative(left, right)
        elif node == "subtract":
            if _is_constant(right, 0.0):
                return left
        elif node == "multiply":
            if _is_constant(left, 0.0) or _is_constant(right, 0.0):
                return {"node": "numeric_constant", "value": 0.0}
            if _is_constant(left, 1.0):
                return right
            if _is_constant(right, 1.0):
                return left
            left, right = _ordered_commutative(left, right)
        return {"node": node, "left": left, "right": right}
    if node == "safe_divide":
        numerator = _simplified_expression(expression["numerator"])
        denominator = _simplified_expression(expression["denominator"])
        if _is_constant(numerator, 0.0):
            return {"node": "numeric_constant", "value": 0.0}
        if _is_constant(denominator, 1.0):
            return numerator
        return {
            "node": "safe_divide",
            "numerator": numerator,
            "denominator": denominator,
            "epsilon": expression.get("epsilon", 0.000001),
            "near_zero_behavior": expression.get("near_zero_behavior", NearZeroBehavior.RETURN_NULL.value),
        }
    if node in {"absolute_value", "clip", "difference", "lag", *_ROLLING_NODES}:
        updated = dict(expression)
        updated["input"] = _simplified_expression(expression["input"])
        if node == "absolute_value" and updated["input"].get("node") == "absolute_value":
            return dict(updated["input"])
        return updated
    return dict(expression)


def _ordered_commutative(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if stable_hash(left) <= stable_hash(right):
        return left, right
    return right, left


def _is_constant(expression: Mapping[str, Any], value: float) -> bool:
    return expression.get("node") == "numeric_constant" and float(expression.get("value", 0.0)) == value


def _effective_max_lag(parent: HypothesisSpec, config: MutationConfig) -> int:
    configured = max((lag for lag in config.lag_seconds if lag >= 0), default=0)
    return max(parent.maximum_lag_seconds, configured, *expression_lag_seconds(parent.expression))


def _bounded_lags(max_lag: int, config: MutationConfig) -> tuple[int, ...]:
    values = {0, max_lag}
    values.update(lag for lag in config.lag_seconds if 0 <= lag <= max_lag)
    return tuple(sorted(values))


def _positive_values(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values if int(value) > 0}))


def _raw_metrics(expression: Mapping[str, Any]) -> set[str]:
    metrics = {str(expression["metric"])} if expression.get("node") == "metric" else set()
    for _, child in _child_items(expression):
        metrics.update(_raw_metrics(child))
    return metrics


def _raw_parameters(expression: Mapping[str, Any]) -> set[str]:
    parameters = {str(expression["parameter"])} if expression.get("node") == "fitted_parameter" else set()
    for _, child in _child_items(expression):
        parameters.update(_raw_parameters(child))
    return parameters


def _raw_lags(expression: Mapping[str, Any]) -> tuple[int, ...]:
    current = (int(expression["lag_seconds"]),) if expression.get("node") == "lag" else ()
    child_lags: tuple[int, ...] = ()
    for _, child in _child_items(expression):
        child_lags = (*child_lags, *_raw_lags(child))
    return (*current, *child_lags)


def _catalog_from_hypothesis(hypothesis: HypothesisSpec) -> set[str]:
    return {
        hypothesis.target_metric,
        *hypothesis.input_metrics,
        *expression_metrics(hypothesis.expression),
        *(control.metric for control in hypothesis.negative_controls),
    }


_ROLLING_NODES = frozenset({"rolling_mean", "rolling_std", "robust_zscore"})


__all__ = [
    "MutationConfig",
    "MutationError",
    "MutationOperator",
    "mutate_hypothesis",
    "mutate_one",
]
