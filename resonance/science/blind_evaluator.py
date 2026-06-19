from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from pydantic import TypeAdapter

from resonance.science.contracts import Expression
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit, read_entries
from resonance.science.preregistration import (
    DEFAULT_EVALUATION_BUDGET,
    EVALUATOR_VERSION,
    ScientificPreregistration,
    current_evaluator_identity_hash,
    load_preregistration,
)
from resonance.science.snapshots import (
    DEFAULT_ARTIFACT_ROOT,
    create_blind_evaluator_capability,
    load_blind_view,
)
from resonance.time_utils import parse_utc, to_utc_iso


_EXPRESSION_ADAPTER = TypeAdapter(Expression)


class BlindEvaluationError(RuntimeError):
    """Base class for blind evaluator failures."""


class PreregistrationHashError(BlindEvaluationError):
    """Raised when the supplied preregistration hash is invalid."""


class BlindEvaluationAlreadyCompletedError(BlindEvaluationError):
    """Raised when a preregistered hypothesis has already spent its blind budget."""


class EvaluatorIdentityError(BlindEvaluationError):
    """Raised when evaluator code/config identity no longer matches preregistration."""


@dataclass(frozen=True)
class BlindEvaluationResult:
    status: str
    preregistration_hash: str
    hypothesis_hash: str
    snapshot_id: str
    evaluator_version: str
    metrics: dict[str, Any]
    warnings: tuple[str, ...]
    artifact: dict[str, str]
    ledger_entry_hash: str
    evaluation_budget_consumed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "preregistration_hash": self.preregistration_hash,
            "hypothesis_hash": self.hypothesis_hash,
            "snapshot_id": self.snapshot_id,
            "evaluator_version": self.evaluator_version,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "artifact": self.artifact,
            "ledger_entry_hash": self.ledger_entry_hash,
            "evaluation_budget_consumed": self.evaluation_budget_consumed,
            "raw_blind_values_exposed": False,
        }


def evaluate_preregistration(
    preregistration: ScientificPreregistration | Mapping[str, Any],
    preregistration_hash: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> BlindEvaluationResult:
    preregistration = load_preregistration(preregistration)
    _validate_preregistration_hash(preregistration, preregistration_hash)
    _validate_evaluator_identity(preregistration)
    _refuse_if_already_evaluated(preregistration, preregistration_hash, ledger_path)

    blind = load_blind_view(
        preregistration.snapshot_id,
        create_blind_evaluator_capability(),
        artifact_root=artifact_root,
    )
    metrics, warnings = _evaluate_metrics(preregistration, blind["rows"])
    status = _verdict(preregistration, metrics, warnings)
    metrics_artifact = _store_metrics_artifact(
        artifact_root,
        {
            "schema_version": 1,
            "preregistration_hash": preregistration_hash,
            "hypothesis_hash": preregistration.hypothesis_hash,
            "snapshot_id": preregistration.snapshot_id,
            "evaluator_version": EVALUATOR_VERSION,
            "status": status,
            "metrics": metrics,
            "warnings": list(warnings),
            "raw_blind_values_exposed": False,
        },
    )
    ledger_entry = append_event(
        "blind_evaluation_completed",
        {
            "preregistration_hash": preregistration_hash,
            "dataset_snapshot_id": preregistration.snapshot_id,
            "hypothesis_hash": preregistration.hypothesis_hash,
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_identity_hash": preregistration.evaluator_identity_hash,
            "random_seed": preregistration.random_seed,
            "evaluation_budget_consumed": DEFAULT_EVALUATION_BUDGET,
            "status": status,
            "metrics": metrics,
            "warnings": list(warnings),
            "artifact_root": str(Path(artifact_root).resolve()),
            "artifacts": {"blind_evaluation_metrics": metrics_artifact},
        },
        artifact_hashes={"blind_evaluation_metrics": metrics_artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return BlindEvaluationResult(
        status=status,
        preregistration_hash=preregistration_hash,
        hypothesis_hash=preregistration.hypothesis_hash,
        snapshot_id=preregistration.snapshot_id,
        evaluator_version=EVALUATOR_VERSION,
        metrics=metrics,
        warnings=warnings,
        artifact=metrics_artifact,
        ledger_entry_hash=ledger_entry["entry_hash"],
        evaluation_budget_consumed=DEFAULT_EVALUATION_BUDGET,
    )


def _validate_preregistration_hash(
    preregistration: ScientificPreregistration,
    supplied_hash: str,
) -> None:
    if not _is_hash(supplied_hash):
        raise PreregistrationHashError("preregistration hash must be a 64-character hex digest")
    actual = preregistration.preregistration_hash()
    if supplied_hash != actual:
        raise PreregistrationHashError("preregistration hash does not match frozen content")


def _validate_evaluator_identity(preregistration: ScientificPreregistration) -> None:
    if preregistration.evaluator_version != EVALUATOR_VERSION:
        raise EvaluatorIdentityError("evaluator version differs from preregistration")
    if preregistration.evaluator_identity_hash != current_evaluator_identity_hash():
        raise EvaluatorIdentityError("evaluator identity differs from preregistration")
    if preregistration.evaluator_code_commit != current_code_commit():
        raise EvaluatorIdentityError("code commit differs from preregistration")
    if preregistration.evaluation_budget != DEFAULT_EVALUATION_BUDGET:
        raise EvaluatorIdentityError("evaluation budget differs from sealed evaluator contract")


def _refuse_if_already_evaluated(
    preregistration: ScientificPreregistration,
    preregistration_hash: str,
    ledger_path: str | Path,
) -> None:
    for entry in read_entries(ledger_path):
        if entry["event_type"] != "blind_evaluation_completed":
            continue
        payload = entry["payload"]
        if payload.get("preregistration_hash") == preregistration_hash:
            raise BlindEvaluationAlreadyCompletedError(
                "blind evaluation already completed for this preregistration"
            )
        if (
            payload.get("dataset_snapshot_id") == preregistration.snapshot_id
            and payload.get("hypothesis_hash") == preregistration.hypothesis_hash
        ):
            raise BlindEvaluationAlreadyCompletedError(
                "blind evaluation already completed for this snapshot and hypothesis"
            )


def _evaluate_metrics(
    preregistration: ScientificPreregistration,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    timestamps = [parse_utc(str(row["timestamp_utc"])) for row in rows]
    series = _series_by_metric(rows)
    expression = _EXPRESSION_ADAPTER.validate_python(preregistration.exact_expression)
    predictions = _evaluate_expression(expression, timestamps, series, preregistration.fitted_parameters)
    target = _transform_target(
        series.get(preregistration.target_metric, [None] * len(rows)),
        timestamps,
        preregistration.target_transform,
        preregistration.transform_config,
    )
    pairs = [
        (float(prediction), float(actual))
        for prediction, actual in zip(predictions, target, strict=True)
        if prediction is not None and actual is not None and math.isfinite(prediction) and math.isfinite(actual)
    ]
    warnings: list[str] = []
    if not rows:
        warnings.append("blind partition is empty")
    if not pairs:
        warnings.append("no aligned blind observations after evaluating frozen expression")

    baseline_mae = _baseline_value(preregistration.baseline_metrics, "mae")
    baseline_rmse = _baseline_value(preregistration.baseline_metrics, "rmse")
    mae = _mae(pairs)
    rmse = _rmse(pairs)
    spearman = _spearman(pairs)
    direction_agreement = _direction_agreement(
        pairs,
        preregistration.transform_config,
        preregistration.expected_direction,
    )
    window_stability = _window_stability(
        pairs,
        preregistration.minimum_baseline_improvement,
        baseline_mae,
        preregistration.transform_config,
    )
    negative_controls = _negative_control_metrics(
        preregistration,
        timestamps,
        series,
        predictions,
    )

    coverage = round(len(pairs) / len(rows), 6) if rows else 0.0
    metrics = {
        "observation_count": len(pairs),
        "blind_row_count": len(rows),
        "coverage": coverage,
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": _improvement(baseline_mae, mae),
        "mae_improvement_fraction": _improvement_fraction(baseline_mae, mae),
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "rmse_improvement": _improvement(baseline_rmse, rmse),
        "rmse_improvement_fraction": _improvement_fraction(baseline_rmse, rmse),
        "spearman_rho": spearman,
        "direction_agreement": direction_agreement,
        "window_stability": window_stability,
        "negative_controls": negative_controls,
        "negative_control_performance": {
            "max_abs_spearman_rho": max(
                (abs(control["spearman_rho"]) for control in negative_controls),
                default=0.0,
            ),
            "passed": all(
                abs(control["spearman_rho"]) < preregistration.minimum_blind_effect
                for control in negative_controls
            ),
        },
    }
    if len(pairs) < int(preregistration.transform_config.get("minimum_observations", 4)):
        warnings.append("too few aligned observations for a decisive blind evaluation")
    if coverage < float(preregistration.transform_config.get("minimum_coverage", 0.5)):
        warnings.append("blind observation coverage is below the preregistered minimum")
    if baseline_mae is None or baseline_mae <= 0 or baseline_rmse is None or baseline_rmse <= 0:
        warnings.append("frozen baseline metrics are missing or non-positive")
    return metrics, tuple(warnings)


def _verdict(
    preregistration: ScientificPreregistration,
    metrics: Mapping[str, Any],
    warnings: Sequence[str],
) -> str:
    if warnings:
        return "inconclusive"
    if not metrics["negative_control_performance"]["passed"]:
        return "fail"
    mae_improvement = metrics["mae_improvement_fraction"]
    rmse_improvement = metrics["rmse_improvement_fraction"]
    spearman = metrics["spearman_rho"]
    if mae_improvement is None or rmse_improvement is None or spearman is None:
        return "inconclusive"
    if not (
        mae_improvement >= preregistration.minimum_baseline_improvement
        and rmse_improvement >= preregistration.minimum_baseline_improvement
    ):
        return "fail"
    if not _effect_direction_passes(
        spearman,
        preregistration.minimum_blind_effect,
        preregistration.expected_direction,
    ):
        return "fail"
    if metrics["direction_agreement"] is None or metrics["direction_agreement"] < 0.5:
        return "fail"
    if metrics["window_stability"]["stable_fraction"] < 0.5:
        return "fail"
    return "pass"


def _series_by_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    metrics = sorted(
        {
            str(metric)
            for row in rows
            for metric in dict(row.get("metrics", {})).keys()
        }
    )
    series: dict[str, list[float | None]] = {metric: [] for metric in metrics}
    for row in rows:
        row_metrics = dict(row.get("metrics", {}))
        for metric in metrics:
            observations = row_metrics.get(metric)
            if not observations:
                series[metric].append(None)
                continue
            values = [
                float(observation["value"])
                for observation in observations
                if observation.get("value") is not None
            ]
            series[metric].append(sum(values) / len(values) if values else None)
    return series


def _evaluate_expression(
    expression: Any,
    timestamps: Sequence[Any],
    series: Mapping[str, Sequence[float | None]],
    parameters: Mapping[str, float],
) -> list[float | None]:
    node = expression.node
    count = len(timestamps)
    if node == "metric":
        return list(series.get(expression.metric, [None] * count))
    if node == "numeric_constant":
        return [float(expression.value)] * count
    if node == "fitted_parameter":
        return [float(parameters[expression.parameter])] * count
    if node in {"add", "subtract", "multiply"}:
        left = _evaluate_expression(expression.left, timestamps, series, parameters)
        right = _evaluate_expression(expression.right, timestamps, series, parameters)
        return [_binary(node, a, b) for a, b in zip(left, right, strict=True)]
    if node == "safe_divide":
        numerator = _evaluate_expression(expression.numerator, timestamps, series, parameters)
        denominator = _evaluate_expression(expression.denominator, timestamps, series, parameters)
        return [
            _safe_divide(a, b, expression.epsilon, expression.near_zero_behavior.value)
            for a, b in zip(numerator, denominator, strict=True)
        ]
    if node == "absolute_value":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return [abs(value) if value is not None else None for value in values]
    if node == "clip":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return [
            min(max(value, expression.minimum), expression.maximum) if value is not None else None
            for value in values
        ]
    if node == "lag":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return _lag(values, timestamps, expression.lag_seconds)
    if node == "difference":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return _difference(values, timestamps, expression.period_seconds)
    if node == "rolling_mean":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return _rolling(values, timestamps, expression.window_seconds, expression.min_periods, "mean")
    if node == "rolling_std":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return _rolling(values, timestamps, expression.window_seconds, expression.min_periods, "std")
    if node == "robust_zscore":
        values = _evaluate_expression(expression.input, timestamps, series, parameters)
        return _rolling(values, timestamps, expression.window_seconds, expression.min_periods, "robust_zscore")
    raise BlindEvaluationError(f"unsupported expression node: {node}")


def _binary(node: str, left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if node == "add":
        return left + right
    if node == "subtract":
        return left - right
    if node == "multiply":
        return left * right
    raise BlindEvaluationError(f"unsupported binary node: {node}")


def _safe_divide(
    numerator: float | None,
    denominator: float | None,
    epsilon: float,
    near_zero_behavior: str,
) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) >= epsilon:
        return numerator / denominator
    if near_zero_behavior == "return_null":
        return None
    if near_zero_behavior == "return_zero":
        return 0.0
    if near_zero_behavior == "use_epsilon_sign":
        sign = -1.0 if denominator < 0 else 1.0
        return numerator / (sign * epsilon)
    raise BlindEvaluationError(f"unsupported near-zero behavior: {near_zero_behavior}")


def _lag(
    values: Sequence[float | None],
    timestamps: Sequence[Any],
    lag_seconds: int,
) -> list[float | None]:
    by_timestamp = {to_utc_iso(timestamp): value for timestamp, value in zip(timestamps, values, strict=True)}
    return [
        by_timestamp.get(to_utc_iso(timestamp - timedelta(seconds=lag_seconds)))
        for timestamp in timestamps
    ]


def _difference(
    values: Sequence[float | None],
    timestamps: Sequence[Any],
    period_seconds: int,
) -> list[float | None]:
    lagged = _lag(values, timestamps, period_seconds)
    return [
        (value - prior) if value is not None and prior is not None else None
        for value, prior in zip(values, lagged, strict=True)
    ]


def _rolling(
    values: Sequence[float | None],
    timestamps: Sequence[Any],
    window_seconds: int,
    min_periods: int,
    kind: str,
) -> list[float | None]:
    output: list[float | None] = []
    for index, timestamp in enumerate(timestamps):
        start = timestamp - timedelta(seconds=window_seconds)
        window = [
            float(value)
            for candidate_ts, value in zip(timestamps[: index + 1], values[: index + 1], strict=True)
            if start <= candidate_ts <= timestamp and value is not None
        ]
        if len(window) < min_periods:
            output.append(None)
        elif kind == "mean":
            output.append(sum(window) / len(window))
        elif kind == "std":
            mean = sum(window) / len(window)
            output.append(math.sqrt(sum((value - mean) ** 2 for value in window) / len(window)))
        elif kind == "robust_zscore":
            center = median(window)
            deviations = [abs(value - center) for value in window]
            mad = median(deviations)
            current = values[index]
            if current is None:
                output.append(None)
            elif mad == 0:
                output.append(0.0 if current == center else None)
            else:
                output.append((current - center) / (1.4826 * mad))
        else:
            raise BlindEvaluationError(f"unsupported rolling kind: {kind}")
    return output


def _transform_target(
    values: Sequence[float | None],
    timestamps: Sequence[Any],
    target_transform: str,
    config: Mapping[str, Any],
) -> list[float | None]:
    if target_transform == "identity":
        return list(values)
    if target_transform == "difference":
        period = config.get("target_difference_period_seconds")
        if period is not None:
            return _difference(values, timestamps, int(period))
        return [
            None if index == 0 or values[index] is None or values[index - 1] is None else values[index] - values[index - 1]
            for index in range(len(values))
        ]
    if target_transform == "robust_zscore":
        return _rolling(
            values,
            timestamps,
            int(config.get("target_window_seconds", 3600)),
            int(config.get("target_min_periods", 5)),
            "robust_zscore",
        )
    raise BlindEvaluationError(f"unsupported target transform: {target_transform}")


def _negative_control_metrics(
    preregistration: ScientificPreregistration,
    timestamps: Sequence[Any],
    series: Mapping[str, Sequence[float | None]],
    predictions: Sequence[float | None],
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for control in preregistration.negative_controls:
        metric = str(control["metric"])
        control_target = _transform_target(
            series.get(metric, [None] * len(timestamps)),
            timestamps,
            str(preregistration.transform_config.get("negative_control_transform", "identity")),
            preregistration.transform_config,
        )
        pairs = [
            (float(prediction), float(actual))
            for prediction, actual in zip(predictions, control_target, strict=True)
            if prediction is not None and actual is not None and math.isfinite(prediction) and math.isfinite(actual)
        ]
        rho = _spearman(pairs)
        controls.append(
            {
                "metric": metric,
                "observation_count": len(pairs),
                "spearman_rho": rho if rho is not None else 0.0,
                "passed": (rho is None) or abs(rho) < preregistration.minimum_blind_effect,
            }
        )
    return controls


def _mae(pairs: Sequence[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return sum(abs(prediction - actual) for prediction, actual in pairs) / len(pairs)


def _rmse(pairs: Sequence[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return math.sqrt(sum((prediction - actual) ** 2 for prediction, actual in pairs) / len(pairs))


def _spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    x_ranks = _ranks([pair[0] for pair in pairs])
    y_ranks = _ranks([pair[1] for pair in pairs])
    return _pearson(x_ranks, y_ranks)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_den == 0 or right_den == 0:
        return None
    return numerator / (left_den * right_den)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _direction_agreement(
    pairs: Sequence[tuple[float, float]],
    config: Mapping[str, Any],
    expected_direction: str,
) -> float | None:
    if not pairs:
        return None
    direction = expected_direction
    comparable = []
    for prediction, actual in pairs:
        prediction_sign = _sign(prediction)
        actual_sign = _sign(actual)
        if prediction_sign == 0 or actual_sign == 0:
            continue
        if direction == "negative":
            prediction_sign *= -1
        comparable.append(prediction_sign == actual_sign)
    if not comparable:
        return None
    return sum(1 for value in comparable if value) / len(comparable)


def _window_stability(
    pairs: Sequence[tuple[float, float]],
    minimum_improvement: float,
    baseline_mae: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not pairs:
        return {"window_count": 0, "stable_count": 0, "stable_fraction": 0.0}
    requested = int(config.get("window_count", 3))
    window_count = max(1, min(requested, len(pairs)))
    stable = 0
    details = []
    for index in range(window_count):
        start = (len(pairs) * index) // window_count
        end = (len(pairs) * (index + 1)) // window_count
        window_pairs = pairs[start:end]
        rho = _spearman(window_pairs)
        mae = _mae(window_pairs)
        improvement = _improvement_fraction(baseline_mae, mae)
        is_stable = bool(
            rho is not None
            and rho > 0
            and (improvement is None or improvement >= minimum_improvement)
        )
        if is_stable:
            stable += 1
        details.append(
            {
                "index": index,
                "observation_count": len(window_pairs),
                "spearman_rho": rho,
                "mae_improvement_fraction": improvement,
                "stable": is_stable,
            }
        )
    return {
        "window_count": window_count,
        "stable_count": stable,
        "stable_fraction": stable / window_count,
        "windows": details,
    }


def _effect_direction_passes(rho: float, minimum_effect: float, direction: str) -> bool:
    if direction == "negative":
        return rho <= -minimum_effect
    if direction == "nonzero":
        return abs(rho) >= minimum_effect
    return rho >= minimum_effect


def _baseline_value(baseline_metrics: Mapping[str, float], metric: str) -> float | None:
    for key in (metric, f"baseline_{metric}"):
        if key in baseline_metrics:
            return float(baseline_metrics[key])
    return None


def _improvement(baseline: float | None, observed: float | None) -> float | None:
    if baseline is None or observed is None:
        return None
    return baseline - observed


def _improvement_fraction(baseline: float | None, observed: float | None) -> float | None:
    if baseline is None or observed is None or baseline <= 0:
        return None
    return (baseline - observed) / baseline


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _store_metrics_artifact(root: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise BlindEvaluationError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "BlindEvaluationAlreadyCompletedError",
    "BlindEvaluationError",
    "BlindEvaluationResult",
    "EvaluatorIdentityError",
    "PreregistrationHashError",
    "evaluate_preregistration",
]
