from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from pydantic import Field, ValidationError

from resonance.science.contracts import (
    DEFAULT_MAX_AST_NODES,
    HypothesisSpec,
    StrictModel,
    expression_lag_seconds,
    expression_metrics,
    expression_node_count,
    metric_catalog_names,
    stable_hash,
)


class ReviewRecommendation(str, Enum):
    REJECT = "reject"
    REVISE = "revise"
    PREREGISTRATION_ELIGIBLE = "preregistration-eligible"


class ReviewSpec(StrictModel):
    confounders: tuple[str, ...] = Field(min_length=1)
    simpler_explanation: str = Field(min_length=1)
    leakage_risk: str = Field(min_length=1)
    mechanical_correlation_risk: str = Field(min_length=1)
    suggested_controls_or_falsifications: tuple[str, ...] = Field(min_length=1)
    executable: bool
    distinct_from_prior: bool
    recommendation: ReviewRecommendation


class ReviewIssue(StrictModel):
    code: str
    message: str


class DeterministicReview(StrictModel):
    accepted: bool
    issues: tuple[ReviewIssue, ...] = Field(default_factory=tuple)

    def raise_for_issues(self) -> None:
        if self.issues:
            raise DeterministicReviewError("; ".join(issue.message for issue in self.issues))


class DeterministicReviewError(ValueError):
    """Raised when deterministic hypothesis validation rejects a proposal."""


def validate_hypothesis(
    hypothesis: HypothesisSpec | Mapping[str, Any],
    *,
    metric_catalog: Any,
    snapshot_max_lag_seconds: int | None = None,
    prior_hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]] = (),
) -> DeterministicReview:
    issues = list(_raw_hypothesis_issues(hypothesis, metric_catalog, snapshot_max_lag_seconds))
    parsed = _parse_hypothesis(hypothesis, metric_catalog)
    if parsed is None:
        return DeterministicReview(accepted=False, issues=tuple(_unique_issues(issues)))

    issues.extend(_parsed_hypothesis_issues(parsed, metric_catalog, snapshot_max_lag_seconds))
    if _is_duplicate(parsed, prior_hypotheses):
        issues.append(
            ReviewIssue(
                code="duplicate_hypothesis",
                message="hypothesis duplicates a prior hypothesis",
            )
        )
    return DeterministicReview(accepted=not issues, issues=tuple(_unique_issues(issues)))


def validate_hypotheses(
    hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]],
    *,
    metric_catalog: Any,
    snapshot_max_lag_seconds: int | None = None,
    prior_hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]] = (),
) -> tuple[DeterministicReview, ...]:
    accepted_so_far: list[HypothesisSpec] = [
        prior
        for prior in (_parse_hypothesis(item, metric_catalog) for item in prior_hypotheses)
        if prior is not None
    ]
    reviews: list[DeterministicReview] = []
    for hypothesis in hypotheses:
        review = validate_hypothesis(
            hypothesis,
            metric_catalog=metric_catalog,
            snapshot_max_lag_seconds=snapshot_max_lag_seconds,
            prior_hypotheses=accepted_so_far,
        )
        reviews.append(review)
        parsed = _parse_hypothesis(hypothesis, metric_catalog)
        if review.accepted and parsed is not None:
            accepted_so_far.append(parsed)
    return tuple(reviews)


def assert_hypothesis_valid(
    hypothesis: HypothesisSpec | Mapping[str, Any],
    *,
    metric_catalog: Any,
    snapshot_max_lag_seconds: int | None = None,
    prior_hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]] = (),
) -> HypothesisSpec:
    review = validate_hypothesis(
        hypothesis,
        metric_catalog=metric_catalog,
        snapshot_max_lag_seconds=snapshot_max_lag_seconds,
        prior_hypotheses=prior_hypotheses,
    )
    review.raise_for_issues()
    parsed = _parse_hypothesis(hypothesis, metric_catalog)
    if parsed is None:
        raise DeterministicReviewError("hypothesis failed schema validation")
    return parsed


def _raw_hypothesis_issues(
    hypothesis: HypothesisSpec | Mapping[str, Any],
    metric_catalog: Any,
    snapshot_max_lag_seconds: int | None,
) -> tuple[ReviewIssue, ...]:
    if isinstance(hypothesis, HypothesisSpec):
        value = hypothesis.model_dump(mode="json", exclude_none=True)
    else:
        value = dict(hypothesis)

    issues: list[ReviewIssue] = []
    expression = value.get("expression")
    target_metric = value.get("target_metric")
    input_metrics = tuple(str(metric) for metric in value.get("input_metrics", ()) or ())

    if not value.get("negative_controls"):
        issues.append(ReviewIssue(code="missing_negative_controls", message="negative controls are required"))

    if target_metric is not None and str(target_metric) in input_metrics:
        issues.append(
            ReviewIssue(
                code="direct_target_leakage",
                message="target metric cannot also be an input metric",
            )
        )

    expression_metric_names = _raw_expression_metrics(expression)
    if target_metric is not None and str(target_metric) in expression_metric_names:
        issues.append(
            ReviewIssue(
                code="future_target_values",
                message="expression must not use target metric values",
            )
        )

    raw_lags = _raw_lags(expression)
    if any(lag is None for lag in raw_lags):
        issues.append(
            ReviewIssue(
                code="unbounded_lag",
                message="all lag nodes must declare non-negative lag_seconds",
            )
        )
    if any(isinstance(lag, int) and lag < 0 for lag in raw_lags):
        issues.append(
            ReviewIssue(
                code="future_target_values",
                message="negative lags would use future values",
            )
        )

    maximum_lag = value.get("maximum_lag_seconds")
    if not isinstance(maximum_lag, int) or maximum_lag < 0:
        issues.append(
            ReviewIssue(
                code="unbounded_lag",
                message="maximum_lag_seconds must be declared as a non-negative integer",
            )
        )
    if snapshot_max_lag_seconds is not None and (
        not isinstance(maximum_lag, int) or maximum_lag > snapshot_max_lag_seconds
    ):
        issues.append(
            ReviewIssue(
                code="lag_exceeds_snapshot",
                message="declared maximum lag exceeds snapshot maximum lag",
            )
        )
    if isinstance(maximum_lag, int) and any(
        isinstance(lag, int) and lag > maximum_lag for lag in raw_lags
    ):
        issues.append(
            ReviewIssue(
                code="lag_exceeds_declared_maximum",
                message="expression lag exceeds declared maximum lag",
            )
        )

    supported = metric_catalog_names(metric_catalog)
    negative_control_metrics = {
        str(control.get("metric"))
        for control in value.get("negative_controls", ()) or ()
        if isinstance(control, Mapping) and control.get("metric") is not None
    }
    referenced = {str(target_metric), *input_metrics, *expression_metric_names, *negative_control_metrics}
    referenced.discard("None")
    unsupported = referenced - supported
    if unsupported:
        issues.append(
            ReviewIssue(
                code="unsupported_metrics",
                message=f"unsupported metrics: {', '.join(sorted(unsupported))}",
            )
        )

    complexity_budget = value.get("complexity_budget") or {}
    if isinstance(complexity_budget, Mapping):
        max_ast_nodes = int(complexity_budget.get("max_ast_nodes", DEFAULT_MAX_AST_NODES))
    else:
        max_ast_nodes = DEFAULT_MAX_AST_NODES
    if _raw_node_count(expression) > max_ast_nodes:
        issues.append(
            ReviewIssue(
                code="excessive_complexity",
                message="expression exceeds AST complexity budget",
            )
        )
    return tuple(issues)


def _parsed_hypothesis_issues(
    hypothesis: HypothesisSpec,
    metric_catalog: Any,
    snapshot_max_lag_seconds: int | None,
) -> tuple[ReviewIssue, ...]:
    issues: list[ReviewIssue] = []
    supported = metric_catalog_names(metric_catalog)
    referenced = {hypothesis.target_metric, *hypothesis.input_metrics}
    referenced.update(expression_metrics(hypothesis.expression))
    referenced.update(control.metric for control in hypothesis.negative_controls)
    unsupported = referenced - supported
    if unsupported:
        issues.append(
            ReviewIssue(
                code="unsupported_metrics",
                message=f"unsupported metrics: {', '.join(sorted(unsupported))}",
            )
        )

    if not hypothesis.negative_controls:
        issues.append(ReviewIssue(code="missing_negative_controls", message="negative controls are required"))
    if hypothesis.target_metric in hypothesis.input_metrics:
        issues.append(
            ReviewIssue(
                code="direct_target_leakage",
                message="target metric cannot also be an input metric",
            )
        )
    if hypothesis.target_metric in expression_metrics(hypothesis.expression):
        issues.append(
            ReviewIssue(
                code="future_target_values",
                message="expression must not use target metric values",
            )
        )

    if snapshot_max_lag_seconds is not None and hypothesis.maximum_lag_seconds > snapshot_max_lag_seconds:
        issues.append(
            ReviewIssue(
                code="lag_exceeds_snapshot",
                message="declared maximum lag exceeds snapshot maximum lag",
            )
        )
    if snapshot_max_lag_seconds is not None and any(
        lag > snapshot_max_lag_seconds for lag in expression_lag_seconds(hypothesis.expression)
    ):
        issues.append(
            ReviewIssue(
                code="lag_exceeds_snapshot",
                message="expression lag exceeds snapshot maximum lag",
            )
        )
    if any(lag > hypothesis.maximum_lag_seconds for lag in expression_lag_seconds(hypothesis.expression)):
        issues.append(
            ReviewIssue(
                code="lag_exceeds_declared_maximum",
                message="expression lag exceeds declared maximum lag",
            )
        )
    if expression_node_count(hypothesis.expression) > hypothesis.complexity_budget.max_ast_nodes:
        issues.append(
            ReviewIssue(
                code="excessive_complexity",
                message="expression exceeds AST complexity budget",
            )
        )
    return tuple(issues)


def _parse_hypothesis(hypothesis: HypothesisSpec | Mapping[str, Any], metric_catalog: Any) -> HypothesisSpec | None:
    if isinstance(hypothesis, HypothesisSpec):
        return hypothesis
    try:
        return HypothesisSpec.model_validate(hypothesis, context={"metric_catalog": metric_catalog})
    except ValidationError:
        return None


def _is_duplicate(
    hypothesis: HypothesisSpec,
    prior_hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]],
) -> bool:
    current_hash = hypothesis.hypothesis_hash()
    for prior in prior_hypotheses:
        parsed = prior if isinstance(prior, HypothesisSpec) else _parse_hypothesis(prior, ())
        if parsed is not None and parsed.hypothesis_hash() == current_hash:
            return True
        if isinstance(prior, Mapping):
            try:
                if stable_hash(HypothesisSpec.model_validate(prior).scientific_content()) == current_hash:
                    return True
            except ValidationError:
                continue
    return False


def _unique_issues(issues: Sequence[ReviewIssue]) -> list[ReviewIssue]:
    seen: set[str] = set()
    unique: list[ReviewIssue] = []
    for issue in issues:
        key = f"{issue.code}:{issue.message}"
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def _raw_expression_metrics(expression: Any) -> set[str]:
    if not isinstance(expression, Mapping):
        return set()
    metrics = {str(expression["metric"])} if expression.get("node") == "metric" and "metric" in expression else set()
    for child in _raw_expression_children(expression):
        metrics.update(_raw_expression_metrics(child))
    return metrics


def _raw_lags(expression: Any) -> tuple[int | None, ...]:
    if not isinstance(expression, Mapping):
        return ()
    current: tuple[int | None, ...] = ()
    if expression.get("node") == "lag":
        value = expression.get("lag_seconds")
        current = (value if isinstance(value, int) else None,)
    child_lags: tuple[int | None, ...] = ()
    for child in _raw_expression_children(expression):
        child_lags = (*child_lags, *_raw_lags(child))
    return (*current, *child_lags)


def _raw_node_count(expression: Any) -> int:
    if not isinstance(expression, Mapping):
        return 0
    return 1 + sum(_raw_node_count(child) for child in _raw_expression_children(expression))


def _raw_expression_children(expression: Mapping[str, Any]) -> tuple[Any, ...]:
    node = expression.get("node")
    if node in {"add", "subtract", "multiply"}:
        return (expression.get("left"), expression.get("right"))
    if node == "safe_divide":
        return (expression.get("numerator"), expression.get("denominator"))
    if node in {"absolute_value", "clip", "difference", "lag", "rolling_mean", "rolling_std", "robust_zscore"}:
        return (expression.get("input"),)
    return ()


__all__ = [
    "DeterministicReview",
    "DeterministicReviewError",
    "ReviewIssue",
    "ReviewRecommendation",
    "ReviewSpec",
    "assert_hypothesis_valid",
    "validate_hypotheses",
    "validate_hypothesis",
]
