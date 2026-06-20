from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from resonance.science.contracts import Expression, HypothesisSpec, MetricName, TargetTransform
from resonance.science.interpreter import ExecutionLimits, evaluate_expression, to_time_series_frame


DEFAULT_ROBUST_WINDOW_SECONDS = 3600
DEFAULT_ROBUST_MIN_PERIODS = 5
BASELINE_ZERO = "zero_residual"
BASELINE_PERSISTENCE = "persistence"
BASELINE_STRATEGIES = (BASELINE_ZERO, BASELINE_PERSISTENCE)


class ScientificEvaluationError(ValueError):
    """Raised when a frozen scientific program cannot be evaluated consistently."""


@dataclass(frozen=True)
class EvaluatedProgram:
    frame: pd.DataFrame
    target: pd.Series
    prediction: pd.Series
    aligned: pd.DataFrame
    transform_config: dict[str, Any]


@dataclass(frozen=True)
class MetricBundle:
    observations: int
    mae: float | None
    rmse: float | None
    pearson_r: float | None
    spearman_r: float | None
    direction_agreement: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.observations,
            "mae": self.mae,
            "rmse": self.rmse,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_r,
            "direction_agreement": self.direction_agreement,
        }


def evaluate_program(
    hypothesis: HypothesisSpec,
    data: pd.DataFrame | pd.Series | np.ndarray | Mapping[str, Any],
    *,
    parameters: Mapping[str, float],
    transform_config: Mapping[str, Any] | None = None,
) -> EvaluatedProgram:
    """Evaluate the same frozen program semantics used by fitting, tuning, and blind tests."""

    return evaluate_frozen_program(
        expression=hypothesis.expression,
        target_metric=str(hypothesis.target_metric),
        input_metrics=tuple(str(metric) for metric in hypothesis.input_metrics),
        target_transform=hypothesis.target_transform,
        data=data,
        parameters=parameters,
        transform_config=transform_config,
        max_ast_nodes=hypothesis.complexity_budget.max_ast_nodes,
        max_source_metrics=hypothesis.complexity_budget.max_source_metrics,
    )


def evaluate_frozen_program(
    *,
    expression: Expression,
    target_metric: str,
    input_metrics: Sequence[str],
    target_transform: TargetTransform | str,
    data: pd.DataFrame | pd.Series | np.ndarray | Mapping[str, Any],
    parameters: Mapping[str, float],
    transform_config: Mapping[str, Any] | None = None,
    max_ast_nodes: int = 15,
    max_source_metrics: int = 3,
) -> EvaluatedProgram:
    frame = to_time_series_frame(data)
    if target_metric not in frame.columns:
        raise ScientificEvaluationError(f"target metric is absent: {target_metric}")
    missing_inputs = set(input_metrics) - set(str(column) for column in frame.columns)
    if missing_inputs:
        names = ", ".join(sorted(missing_inputs))
        raise ScientificEvaluationError(f"input metrics are absent: {names}")
    normalized_config = normalize_transform_config(target_transform, frame.index, transform_config)
    target = transform_target(frame[target_metric], target_transform, normalized_config)
    prediction = evaluate_expression(
        expression,
        frame,
        parameters=parameters,
        limits=ExecutionLimits(
            max_ast_nodes=max_ast_nodes,
            max_source_metrics=max_source_metrics,
        ),
    )
    aligned = finite_pair_frame(target, prediction)
    return EvaluatedProgram(
        frame=frame,
        target=target,
        prediction=prediction,
        aligned=aligned,
        transform_config=normalized_config,
    )


def normalize_transform_config(
    transform: TargetTransform | str,
    index: pd.Index,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit, reproducible target-transform configuration."""

    transform_value = TargetTransform(transform)
    normalized = dict(config or {})
    if transform_value == TargetTransform.DIFFERENCE:
        if isinstance(index, pd.DatetimeIndex):
            period = normalized.get("target_difference_period_seconds")
            if period is None:
                period = infer_cadence_seconds(index)
            if int(period) <= 0:
                raise ScientificEvaluationError("target difference period must be positive")
            normalized["target_difference_period_seconds"] = int(period)
        else:
            periods = int(normalized.get("target_difference_periods", 1))
            if periods <= 0:
                raise ScientificEvaluationError("target difference periods must be positive")
            normalized["target_difference_periods"] = periods
    elif transform_value == TargetTransform.ROBUST_ZSCORE:
        minimum = int(normalized.get("target_min_periods", DEFAULT_ROBUST_MIN_PERIODS))
        if minimum < 2:
            raise ScientificEvaluationError("target_min_periods must be at least two")
        normalized["target_min_periods"] = minimum
        if isinstance(index, pd.DatetimeIndex):
            cadence = infer_cadence_seconds(index)
            default_window = max(DEFAULT_ROBUST_WINDOW_SECONDS, cadence * max(10, minimum))
            window = int(normalized.get("target_window_seconds", default_window))
            if window <= 0:
                raise ScientificEvaluationError("target_window_seconds must be positive")
            normalized["target_window_seconds"] = window
        else:
            window_points = int(normalized.get("target_window_points", max(10, minimum)))
            if window_points <= 0:
                raise ScientificEvaluationError("target_window_points must be positive")
            normalized["target_window_points"] = window_points
    return normalized


def transform_target(
    series: pd.Series,
    transform: TargetTransform | str,
    config: Mapping[str, Any] | None = None,
) -> pd.Series:
    source = pd.to_numeric(series, errors="coerce").astype("float64")
    transform_value = TargetTransform(transform)
    normalized = normalize_transform_config(transform_value, source.index, config)
    if transform_value == TargetTransform.IDENTITY:
        return source
    if transform_value == TargetTransform.DIFFERENCE:
        if isinstance(source.index, pd.DatetimeIndex):
            prior = _lag_series(source, int(normalized["target_difference_period_seconds"]))
            return _finite_series(source - prior)
        return _finite_series(source - source.shift(int(normalized["target_difference_periods"])))
    if transform_value == TargetTransform.ROBUST_ZSCORE:
        minimum = int(normalized["target_min_periods"])
        if isinstance(source.index, pd.DatetimeIndex):
            rolling = source.rolling(
                f"{int(normalized['target_window_seconds'])}s",
                min_periods=minimum,
            )
        else:
            rolling = source.rolling(
                int(normalized["target_window_points"]),
                min_periods=minimum,
            )
        center = rolling.median()
        mad = rolling.apply(_median_absolute_deviation, raw=False)
        scale = 1.4826 * mad
        result = (source - center) / scale.mask(scale.abs() <= 1.0e-12)
        return _finite_series(result)
    raise ScientificEvaluationError(f"unsupported target transform: {transform_value}")


def finite_pair_frame(target: pd.Series, prediction: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [target.rename("target"), prediction.rename("prediction")],
        axis=1,
    )
    return aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def metric_bundle(target: pd.Series, prediction: pd.Series) -> MetricBundle:
    aligned = finite_pair_frame(target, prediction)
    if aligned.empty:
        return MetricBundle(0, None, None, None, None, None)
    actual = aligned["target"]
    predicted = aligned["prediction"]
    residual = predicted - actual
    actual_values = actual.to_numpy(dtype="float64")
    predicted_values = predicted.to_numpy(dtype="float64")
    has_variance = np.ptp(actual_values) > 0 and np.ptp(predicted_values) > 0
    pearson = float(np.corrcoef(actual_values, predicted_values)[0, 1]) if has_variance else None
    if has_variance:
        from scipy.stats import rankdata

        spearman = float(
            np.corrcoef(
                rankdata(actual_values, method="average"),
                rankdata(predicted_values, method="average"),
            )[0, 1]
        )
    else:
        spearman = None
    return MetricBundle(
        observations=int(len(aligned)),
        mae=_finite_float(residual.abs().mean()),
        rmse=_finite_float(np.sqrt(np.mean(np.square(residual.to_numpy(dtype="float64"))))),
        pearson_r=_finite_float(pearson),
        spearman_r=_finite_float(spearman),
        direction_agreement=movement_direction_agreement(actual, predicted),
    )


def baseline_predictions(target: pd.Series) -> dict[str, pd.Series]:
    return {
        BASELINE_ZERO: pd.Series(0.0, index=target.index, dtype="float64"),
        BASELINE_PERSISTENCE: target.shift(1),
    }


def baseline_metric_bundles(
    target: pd.Series,
    *,
    evaluation_index: pd.Index | None = None,
) -> dict[str, MetricBundle]:
    selected_target = target if evaluation_index is None else target.reindex(evaluation_index)
    return {
        name: metric_bundle(selected_target, prediction.reindex(selected_target.index))
        for name, prediction in baseline_predictions(target).items()
    }


def choose_baseline_strategy(
    baselines: Mapping[str, MetricBundle | Mapping[str, Any]],
    metric: MetricName | str,
) -> str:
    metric_name = MetricName(metric)
    error_key = "mae" if metric_name == MetricName.MAE else "rmse"
    if metric_name not in {MetricName.MAE, MetricName.RMSE}:
        error_key = "rmse"
    candidates: list[tuple[float, str]] = []
    for strategy in BASELINE_STRATEGIES:
        raw = baselines.get(strategy)
        if raw is None:
            continue
        value = getattr(raw, error_key, None) if isinstance(raw, MetricBundle) else raw.get(error_key)
        numeric = _finite_float(value)
        if numeric is not None:
            candidates.append((numeric, strategy))
    if not candidates:
        raise ScientificEvaluationError("no valid baseline strategy is available")
    return min(candidates, key=lambda item: (item[0], BASELINE_STRATEGIES.index(item[1])))[1]


def metric_improvement_fraction(
    baseline: MetricBundle | Mapping[str, Any],
    model: MetricBundle | Mapping[str, Any],
    metric: MetricName | str,
) -> float | None:
    metric_name = MetricName(metric)
    key = {
        MetricName.MAE: "mae",
        MetricName.RMSE: "rmse",
        MetricName.PEARSON_R: "pearson_r",
        MetricName.SPEARMAN_R: "spearman_r",
    }[metric_name]
    baseline_value = getattr(baseline, key, None) if isinstance(baseline, MetricBundle) else baseline.get(key)
    model_value = getattr(model, key, None) if isinstance(model, MetricBundle) else model.get(key)
    baseline_numeric = _finite_float(baseline_value)
    model_numeric = _finite_float(model_value)
    if baseline_numeric is None or model_numeric is None:
        return None
    if metric_name in {MetricName.MAE, MetricName.RMSE}:
        if baseline_numeric <= 0:
            return None
        return (baseline_numeric - model_numeric) / baseline_numeric
    return abs(model_numeric) - abs(baseline_numeric)


def movement_direction_agreement(target: pd.Series, prediction: pd.Series) -> float | None:
    aligned = finite_pair_frame(target, prediction)
    if len(aligned) < 2:
        return None
    target_change = aligned["target"].diff()
    prediction_change = aligned["prediction"].diff()
    comparable = pd.concat([target_change, prediction_change], axis=1).dropna()
    if comparable.empty:
        return None
    left = np.sign(comparable.iloc[:, 0].to_numpy(dtype="float64"))
    right = np.sign(comparable.iloc[:, 1].to_numpy(dtype="float64"))
    nonzero = (left != 0) & (right != 0)
    if not nonzero.any():
        return None
    return float(np.mean(left[nonzero] == right[nonzero]))


def chronological_window_metrics(
    target: pd.Series,
    prediction: pd.Series,
    *,
    baseline_strategy: str,
    window_count: int,
) -> tuple[dict[str, Any], ...]:
    if baseline_strategy not in BASELINE_STRATEGIES:
        raise ScientificEvaluationError(f"unsupported baseline strategy: {baseline_strategy}")
    aligned = finite_pair_frame(target, prediction)
    if aligned.empty:
        return ()
    requested = max(1, int(window_count))
    windows: list[dict[str, Any]] = []
    baseline_full = baseline_predictions(target)[baseline_strategy]
    for index, indices in enumerate(np.array_split(np.arange(len(aligned)), min(requested, len(aligned)))):
        if len(indices) == 0:
            continue
        window = aligned.iloc[indices]
        model = metric_bundle(window["target"], window["prediction"])
        baseline = metric_bundle(window["target"], baseline_full.reindex(window.index))
        windows.append(
            {
                "index": index,
                "start": _index_value(window.index[0]),
                "end": _index_value(window.index[-1]),
                "observation_count": model.observations,
                "spearman_rho": model.spearman_r,
                "mae_improvement_fraction": metric_improvement_fraction(baseline, model, MetricName.MAE),
                "rmse_improvement_fraction": metric_improvement_fraction(baseline, model, MetricName.RMSE),
                "direction_agreement": model.direction_agreement,
            }
        )
    return tuple(windows)


def infer_cadence_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    values = index.view("int64")
    deltas = np.diff(values) / 1_000_000_000
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        return 1
    return max(1, int(np.median(positive)))


def _lag_series(series: pd.Series, seconds: int) -> pd.Series:
    if seconds == 0:
        return series.copy()
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(seconds=seconds)
    return shifted.reindex(series.index)


def _median_absolute_deviation(values: pd.Series) -> float:
    finite = values.dropna()
    if finite.empty:
        return np.nan
    center = float(finite.median())
    return float((finite - center).abs().median())


def _finite_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, index=series.index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _index_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


__all__ = [
    "BASELINE_PERSISTENCE",
    "BASELINE_STRATEGIES",
    "BASELINE_ZERO",
    "EvaluatedProgram",
    "MetricBundle",
    "ScientificEvaluationError",
    "baseline_metric_bundles",
    "baseline_predictions",
    "choose_baseline_strategy",
    "chronological_window_metrics",
    "evaluate_frozen_program",
    "evaluate_program",
    "finite_pair_frame",
    "infer_cadence_seconds",
    "metric_bundle",
    "metric_improvement_fraction",
    "movement_direction_agreement",
    "normalize_transform_config",
    "transform_target",
]
