from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator


SCHEMA_VERSION = "1.0"
DEFAULT_MAX_AST_NODES = 15
DEFAULT_MAX_SOURCE_METRICS = 3


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HypothesisType(str, Enum):
    OBSERVATIONAL_PREDICTION = "observational_prediction"


class Direction(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONZERO = "nonzero"


class MetricName(str, Enum):
    RMSE = "rmse"
    MAE = "mae"
    PEARSON_R = "pearson_r"
    SPEARMAN_R = "spearman_r"


class Origin(str, Enum):
    MANUAL = "manual"
    LLM = "llm"
    MUTATION = "mutation"
    BASELINE = "baseline"


class NearZeroBehavior(str, Enum):
    RETURN_NULL = "return_null"
    RETURN_ZERO = "return_zero"
    USE_EPSILON_SIGN = "use_epsilon_sign"


class TargetTransform(str, Enum):
    IDENTITY = "identity"
    DIFFERENCE = "difference"
    ROBUST_ZSCORE = "robust_zscore"


MetricId = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]*$")]
ParameterId = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class MetricNode(StrictModel):
    node: Literal["metric"]
    metric: MetricId


class NumericConstantNode(StrictModel):
    node: Literal["numeric_constant"]
    value: float


class FittedParameterNode(StrictModel):
    node: Literal["fitted_parameter"]
    parameter: ParameterId


class AddNode(StrictModel):
    node: Literal["add"]
    left: Expression
    right: Expression


class SubtractNode(StrictModel):
    node: Literal["subtract"]
    left: Expression
    right: Expression


class MultiplyNode(StrictModel):
    node: Literal["multiply"]
    left: Expression
    right: Expression


class SafeDivideNode(StrictModel):
    node: Literal["safe_divide"]
    numerator: Expression
    denominator: Expression
    epsilon: Annotated[float, Field(gt=0.0)]
    near_zero_behavior: NearZeroBehavior


class AbsoluteValueNode(StrictModel):
    node: Literal["absolute_value"]
    input: Expression


class ClipNode(StrictModel):
    node: Literal["clip"]
    input: Expression
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ClipNode:
        if self.minimum > self.maximum:
            raise ValueError("clip minimum must be less than or equal to maximum")
        return self


class DifferenceNode(StrictModel):
    node: Literal["difference"]
    input: Expression
    period_seconds: Annotated[int, Field(gt=0)]


class LagNode(StrictModel):
    node: Literal["lag"]
    input: Expression
    lag_seconds: Annotated[int, Field(ge=0)]


class RollingMeanNode(StrictModel):
    node: Literal["rolling_mean"]
    input: Expression
    window_seconds: Annotated[int, Field(gt=0)]
    min_periods: Annotated[int, Field(gt=0)] = 1


class RollingStdNode(StrictModel):
    node: Literal["rolling_std"]
    input: Expression
    window_seconds: Annotated[int, Field(gt=0)]
    min_periods: Annotated[int, Field(gt=0)] = 2


class RobustZscoreNode(StrictModel):
    node: Literal["robust_zscore"]
    input: Expression
    window_seconds: Annotated[int, Field(gt=0)]
    min_periods: Annotated[int, Field(gt=0)] = 5


Expression: TypeAlias = Annotated[
    MetricNode
    | NumericConstantNode
    | FittedParameterNode
    | AddNode
    | SubtractNode
    | MultiplyNode
    | SafeDivideNode
    | AbsoluteValueNode
    | ClipNode
    | DifferenceNode
    | LagNode
    | RollingMeanNode
    | RollingStdNode
    | RobustZscoreNode,
    Field(discriminator="node"),
]


class ParameterBounds(StrictModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterBounds:
        if self.lower > self.upper:
            raise ValueError("parameter lower bound must be less than or equal to upper bound")
        return self


class NegativeControl(StrictModel):
    metric: MetricId
    rationale: str = Field(min_length=1)


class FalsificationCondition(StrictModel):
    description: str = Field(min_length=1)


class ComplexityBudget(StrictModel):
    max_ast_nodes: Annotated[int, Field(gt=0)] = DEFAULT_MAX_AST_NODES
    max_source_metrics: Annotated[int, Field(gt=0, le=DEFAULT_MAX_SOURCE_METRICS)] = DEFAULT_MAX_SOURCE_METRICS


class HypothesisSpec(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    hypothesis_type: Literal[HypothesisType.OBSERVATIONAL_PREDICTION] = HypothesisType.OBSERVATIONAL_PREDICTION
    title: str = Field(min_length=1)
    concise_claim: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    target_metric: MetricId
    input_metrics: Annotated[tuple[MetricId, ...], Field(min_length=1, max_length=DEFAULT_MAX_SOURCE_METRICS)]
    target_transform: TargetTransform
    expression: Expression
    parameter_bounds: dict[ParameterId, ParameterBounds] = Field(default_factory=dict)
    expected_direction: Direction
    maximum_lag_seconds: Annotated[int, Field(ge=0)]
    fitting_metric: MetricName
    tuning_metric: MetricName
    blind_metrics: Annotated[tuple[MetricName, ...], Field(min_length=1)]
    minimum_blind_effect: Annotated[float, Field(gt=0.0)]
    minimum_baseline_improvement: Annotated[float, Field(ge=0.0)]
    negative_controls: tuple[NegativeControl, ...] = Field(default_factory=tuple)
    falsification_conditions: Annotated[tuple[FalsificationCondition, ...], Field(min_length=1)]
    complexity_budget: ComplexityBudget = Field(default_factory=ComplexityBudget)
    origin: Origin
    parent_hypothesis_ids: tuple[str, ...] = Field(default_factory=tuple)
    random_seed: int

    @model_validator(mode="after")
    def validate_scientific_contract(self, info: ValidationInfo) -> HypothesisSpec:
        if len(set(self.input_metrics)) != len(self.input_metrics):
            raise ValueError("input metrics must be unique")
        if self.target_metric in self.input_metrics:
            raise ValueError("target metric cannot also be an input metric")
        source_metrics = expression_metrics(self.expression)
        declared_inputs = set(self.input_metrics)
        undeclared_metrics = source_metrics - declared_inputs
        if undeclared_metrics:
            names = ", ".join(sorted(undeclared_metrics))
            raise ValueError(f"expression references undeclared input metrics: {names}")
        if len(source_metrics) > self.complexity_budget.max_source_metrics:
            raise ValueError("expression exceeds source metric budget")
        node_count = expression_node_count(self.expression)
        if node_count > self.complexity_budget.max_ast_nodes:
            raise ValueError("expression exceeds AST complexity budget")
        lag_seconds = expression_lag_seconds(self.expression)
        if any(lag > self.maximum_lag_seconds for lag in lag_seconds):
            raise ValueError("expression lag exceeds declared maximum lag")
        parameters = expression_parameters(self.expression)
        missing_bounds = parameters - set(self.parameter_bounds)
        if missing_bounds:
            names = ", ".join(sorted(missing_bounds))
            raise ValueError(f"fitted parameters lack bounds: {names}")
        unknown_bounds = set(self.parameter_bounds) - parameters
        if unknown_bounds:
            names = ", ".join(sorted(unknown_bounds))
            raise ValueError(f"parameter bounds have no fitted parameter reference: {names}")
        catalog = (info.context or {}).get("metric_catalog")
        if catalog is not None:
            self.validate_metric_catalog(catalog)
        return self

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", exclude_none=True))

    def scientific_content(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={
                "title",
                "concise_claim",
                "rationale",
            },
            exclude_none=True,
        )

    def hypothesis_hash(self) -> str:
        return stable_hash(self.scientific_content())

    def validate_metric_catalog(self, metric_catalog: set[str] | frozenset[str] | tuple[str, ...] | list[str]) -> None:
        catalog = set(metric_catalog)
        referenced = {self.target_metric, *self.input_metrics}
        referenced.update(control.metric for control in self.negative_controls)
        unknown = referenced - catalog
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown metrics: {names}")


def expression_node_count(expression: Expression) -> int:
    return 1 + sum(expression_node_count(child) for child in expression_children(expression))


def expression_metrics(expression: Expression) -> set[str]:
    if isinstance(expression, MetricNode):
        return {expression.metric}
    metrics: set[str] = set()
    for child in expression_children(expression):
        metrics.update(expression_metrics(child))
    return metrics


def expression_parameters(expression: Expression) -> set[str]:
    if isinstance(expression, FittedParameterNode):
        return {expression.parameter}
    parameters: set[str] = set()
    for child in expression_children(expression):
        parameters.update(expression_parameters(child))
    return parameters


def expression_lag_seconds(expression: Expression) -> tuple[int, ...]:
    lag_values = (expression.lag_seconds,) if isinstance(expression, LagNode) else ()
    child_lags: tuple[int, ...] = ()
    for child in expression_children(expression):
        child_lags = (*child_lags, *expression_lag_seconds(child))
    return (*lag_values, *child_lags)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def expression_children(expression: Expression) -> tuple[Expression, ...]:
    if isinstance(expression, (AddNode, SubtractNode, MultiplyNode)):
        return (expression.left, expression.right)
    if isinstance(expression, SafeDivideNode):
        return (expression.numerator, expression.denominator)
    if isinstance(
        expression,
        (
            AbsoluteValueNode,
            ClipNode,
            DifferenceNode,
            LagNode,
            RollingMeanNode,
            RollingStdNode,
            RobustZscoreNode,
        ),
    ):
        return (expression.input,)
    return ()


for _model in (
    AddNode,
    SubtractNode,
    MultiplyNode,
    SafeDivideNode,
    AbsoluteValueNode,
    ClipNode,
    DifferenceNode,
    LagNode,
    RollingMeanNode,
    RollingStdNode,
    RobustZscoreNode,
    HypothesisSpec,
):
    _model.model_rebuild()


__all__ = [
    "ComplexityBudget",
    "Direction",
    "Expression",
    "FalsificationCondition",
    "HypothesisSpec",
    "HypothesisType",
    "MetricName",
    "NegativeControl",
    "NearZeroBehavior",
    "Origin",
    "ParameterBounds",
    "SCHEMA_VERSION",
    "TargetTransform",
    "ValidationError",
    "canonical_json",
    "expression_metrics",
    "expression_node_count",
    "expression_parameters",
    "expression_lag_seconds",
    "stable_hash",
]
