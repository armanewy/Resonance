from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from resonance.science.contracts import (
    HypothesisSpec,
    ParameterBounds,
    canonical_json,
    expression_metrics,
    expression_node_count,
    expression_parameters,
    stable_hash,
)
from resonance.science.evaluation import (
    baseline_metric_bundles,
    metric_bundle,
    normalize_transform_config,
    transform_target,
)
from resonance.science.interpreter import ExecutionLimits, evaluate_expression, to_time_series_frame


EVALUATOR_VERSION = "science-fitting-v2"
DEFAULT_COMPLEXITY_WEIGHT = 0.001


class FittingError(ValueError):
    """Raised when a hypothesis cannot be fit on exploration data."""


@dataclass(frozen=True)
class FitResult:
    hypothesis_hash: str
    fitted_parameters: dict[str, float]
    exploration_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    complexity: dict[str, Any]
    convergence_status: dict[str, Any]
    warnings: list[str]
    target_transform_config: dict[str, Any]
    deterministic_fit_artifact: dict[str, Any]

    def artifact_hash(self) -> str:
        payload = {
            key: value
            for key, value in self.deterministic_fit_artifact.items()
            if key != "artifact_hash"
        }
        return stable_hash(payload)


@dataclass
class _FitContext:
    hypothesis: HypothesisSpec
    frame: pd.DataFrame
    target: pd.Series
    parameter_names: list[str]
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def fit_hypothesis(
    hypothesis: HypothesisSpec,
    exploration_data: pd.DataFrame | pd.Series | np.ndarray | Mapping[str, Any],
    *,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
    transform_config: Mapping[str, Any] | None = None,
) -> FitResult:
    """Fit declared parameters using exploration data only."""

    if complexity_weight < 0:
        raise FittingError("complexity weight must be non-negative")
    frame = to_time_series_frame(exploration_data)
    if hypothesis.target_metric not in frame.columns:
        raise FittingError(f"target metric missing from exploration data: {hypothesis.target_metric}")
    missing_inputs = set(hypothesis.input_metrics) - set(str(column) for column in frame.columns)
    if missing_inputs:
        names = ", ".join(sorted(missing_inputs))
        raise FittingError(f"input metrics missing from exploration data: {names}")

    target_transform_config = normalize_transform_config(
        hypothesis.target_transform,
        frame.index,
        transform_config,
    )
    target = transform_target(
        frame[hypothesis.target_metric],
        hypothesis.target_transform,
        target_transform_config,
    )
    parameter_names = sorted(expression_parameters(hypothesis.expression))
    _validate_parameter_bounds(parameter_names, hypothesis.parameter_bounds)
    fixed_parameters, variable_names, lower_bounds, upper_bounds, initial_values = _initial_parameters(
        parameter_names,
        hypothesis.parameter_bounds,
    )
    context = _FitContext(
        hypothesis=hypothesis,
        frame=frame,
        target=target,
        parameter_names=parameter_names,
        fixed_parameters=fixed_parameters,
    )

    if variable_names:
        result = least_squares(
            lambda values: _residuals(context, variable_names, values),
            x0=np.array(initial_values, dtype="float64"),
            bounds=(np.array(lower_bounds, dtype="float64"), np.array(upper_bounds, dtype="float64")),
            method="trf",
            max_nfev=2000,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        fitted_parameters = {
            **fixed_parameters,
            **{name: float(value) for name, value in zip(variable_names, result.x, strict=True)},
        }
        convergence_status = {
            "optimizer": "scipy.optimize.least_squares",
            "seed": int(hypothesis.random_seed),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        }
    else:
        fitted_parameters = dict(fixed_parameters)
        convergence_status = {
            "optimizer": "none",
            "seed": int(hypothesis.random_seed),
            "success": True,
            "status": 0,
            "message": "no variable fitted parameters",
            "nfev": 0,
            "cost": 0.0,
            "optimality": 0.0,
        }

    prediction = evaluate_expression(
        hypothesis.expression,
        frame,
        parameters=fitted_parameters,
        limits=ExecutionLimits(
            max_ast_nodes=hypothesis.complexity_budget.max_ast_nodes,
            max_source_metrics=hypothesis.complexity_budget.max_source_metrics,
        ),
    )
    valid_target, valid_prediction = _valid_pair(target, prediction)
    if valid_target.empty:
        raise FittingError("no finite aligned exploration observations after expression evaluation")

    exploration_metrics = _metrics(valid_target, valid_prediction)
    exploration_metrics["blocked_diagnostics"] = _blocked_diagnostics(valid_target, valid_prediction)
    complexity = _complexity(hypothesis, complexity_weight)
    exploration_metrics["complexity_penalized_rmse"] = _penalized_rmse(
        exploration_metrics["rmse"],
        complexity["penalty"],
        complexity_weight,
    )

    baseline_metrics = _baseline_metrics(target)
    warnings = [*context.warnings]
    _append_metric_warnings(warnings, "exploration", exploration_metrics)
    for name, metrics in baseline_metrics.items():
        if isinstance(metrics, dict):
            _append_metric_warnings(warnings, f"baseline.{name}", metrics)
    if not convergence_status["success"]:
        warnings.append("optimizer did not converge")

    deterministic_fit_artifact = _fit_artifact(
        hypothesis,
        target_transform_config,
        fitted_parameters,
        convergence_status,
        exploration_metrics,
        baseline_metrics,
        complexity,
        warnings,
        variable_names,
        initial_values,
    )
    deterministic_fit_artifact["artifact_hash"] = stable_hash(deterministic_fit_artifact)

    return FitResult(
        hypothesis_hash=hypothesis.hypothesis_hash(),
        fitted_parameters={name: fitted_parameters[name] for name in parameter_names},
        exploration_metrics=exploration_metrics,
        baseline_metrics=baseline_metrics,
        complexity=complexity,
        convergence_status=convergence_status,
        warnings=warnings,
        target_transform_config=target_transform_config,
        deterministic_fit_artifact=deterministic_fit_artifact,
    )


def _validate_parameter_bounds(
    parameter_names: list[str],
    parameter_bounds: Mapping[str, ParameterBounds],
) -> None:
    if set(parameter_names) != set(parameter_bounds):
        raise FittingError("parameter bounds must exactly match fitted_parameter nodes")
    for name in parameter_names:
        bounds = parameter_bounds[name]
        if not math.isfinite(bounds.lower) or not math.isfinite(bounds.upper):
            raise FittingError(f"parameter bounds must be finite for {name}")
        if bounds.lower > bounds.upper:
            raise FittingError(f"invalid parameter bounds for {name}")


def _initial_parameters(
    parameter_names: list[str],
    parameter_bounds: Mapping[str, ParameterBounds],
) -> tuple[dict[str, float], list[str], list[float], list[float], list[float]]:
    fixed_parameters: dict[str, float] = {}
    variable_names: list[str] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    initial_values: list[float] = []
    for name in parameter_names:
        bounds = parameter_bounds[name]
        if bounds.lower == bounds.upper:
            fixed_parameters[name] = float(bounds.lower)
            continue
        variable_names.append(name)
        lower_bounds.append(float(bounds.lower))
        upper_bounds.append(float(bounds.upper))
        initial_values.append(float((bounds.lower + bounds.upper) / 2.0))
    return fixed_parameters, variable_names, lower_bounds, upper_bounds, initial_values


def _residuals(context: _FitContext, variable_names: list[str], values: np.ndarray) -> np.ndarray:
    parameters = {
        **context.fixed_parameters,
        **{name: float(value) for name, value in zip(variable_names, values, strict=True)},
    }
    prediction = evaluate_expression(
        context.hypothesis.expression,
        context.frame,
        parameters=parameters,
        limits=ExecutionLimits(
            max_ast_nodes=context.hypothesis.complexity_budget.max_ast_nodes,
            max_source_metrics=context.hypothesis.complexity_budget.max_source_metrics,
        ),
    )
    valid_target, valid_prediction = _valid_pair(context.target, prediction)
    if valid_target.empty:
        raise FittingError("no finite aligned exploration observations during fitting")
    return (valid_prediction - valid_target).to_numpy(dtype="float64")


def _valid_pair(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat([left.rename("target"), right.rename("prediction")], axis=1)
    valid = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    return valid["target"], valid["prediction"]


def _metrics(target: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    return metric_bundle(target, prediction).to_dict()


def _blocked_diagnostics(target: pd.Series, prediction: pd.Series, blocks: int = 4) -> list[dict[str, Any]]:
    if len(target) < 4:
        return []
    block_ids = np.array_split(np.arange(len(target)), min(blocks, len(target)))
    diagnostics: list[dict[str, Any]] = []
    for block_number, indices in enumerate(block_ids, start=1):
        if len(indices) == 0:
            continue
        block_target = target.iloc[indices]
        block_prediction = prediction.iloc[indices]
        metrics = _metrics(block_target, block_prediction)
        diagnostics.append(
            {
                "block": block_number,
                "start": _index_value(block_target.index[0]),
                "end": _index_value(block_target.index[-1]),
                **metrics,
            }
        )
    return diagnostics


def _baseline_metrics(target: pd.Series) -> dict[str, Any]:
    return {
        name: bundle.to_dict()
        for name, bundle in baseline_metric_bundles(target).items()
    }


def _complexity(hypothesis: HypothesisSpec, complexity_weight: float) -> dict[str, Any]:
    node_count = expression_node_count(hypothesis.expression)
    parameter_count = len(expression_parameters(hypothesis.expression))
    source_metric_count = len(expression_metrics(hypothesis.expression))
    penalty = float(node_count + (2 * parameter_count) + source_metric_count)
    return {
        "node_count": node_count,
        "parameter_count": parameter_count,
        "source_metric_count": source_metric_count,
        "penalty": penalty,
        "weight": float(complexity_weight),
    }


def _penalized_rmse(rmse: float, penalty: float, complexity_weight: float) -> float:
    return float(rmse + (complexity_weight * penalty))


def _append_metric_warnings(warnings: list[str], prefix: str, metrics: dict[str, Any]) -> None:
    if metrics.get("spearman_rho") is None:
        warnings.append(f"{prefix} spearman_rho is undefined")


def _fit_artifact(
    hypothesis: HypothesisSpec,
    target_transform_config: dict[str, Any],
    fitted_parameters: dict[str, float],
    convergence_status: dict[str, Any],
    exploration_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    complexity: dict[str, Any],
    warnings: list[str],
    variable_names: list[str],
    initial_values: list[float],
) -> dict[str, Any]:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "target_transform_config": dict(target_transform_config),
        "optimizer_seed": int(hypothesis.random_seed),
        "parameter_order": sorted(fitted_parameters),
        "optimized_parameter_order": list(variable_names),
        "initial_values": {name: value for name, value in zip(variable_names, initial_values, strict=True)},
        "fitted_parameters": {name: fitted_parameters[name] for name in sorted(fitted_parameters)},
        "convergence_status": convergence_status,
        "exploration_metrics": exploration_metrics,
        "baseline_metrics": baseline_metrics,
        "complexity": complexity,
        "warnings": list(warnings),
        "canonical_hypothesis": hypothesis.scientific_content(),
    }


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _index_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def fit_result_json(result: FitResult) -> str:
    return canonical_json(asdict(result))


__all__ = [
    "EVALUATOR_VERSION",
    "FitResult",
    "FittingError",
    "fit_hypothesis",
    "fit_result_json",
]
