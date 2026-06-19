from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from resonance.science.contracts import (
    DEFAULT_MAX_AST_NODES,
    DEFAULT_MAX_SOURCE_METRICS,
    AbsoluteValueNode,
    AddNode,
    ClipNode,
    DifferenceNode,
    Expression,
    FittedParameterNode,
    LagNode,
    MetricNode,
    NearZeroBehavior,
    NumericConstantNode,
    RobustZscoreNode,
    RollingMeanNode,
    RollingStdNode,
    SafeDivideNode,
    SubtractNode,
    MultiplyNode,
    expression_metrics,
    expression_node_count,
)


class ExpressionExecutionError(ValueError):
    """Raised when a restricted expression cannot be evaluated safely."""


@dataclass(frozen=True)
class ExecutionLimits:
    max_ast_nodes: int = DEFAULT_MAX_AST_NODES
    max_source_metrics: int = DEFAULT_MAX_SOURCE_METRICS


def evaluate_expression(
    expression: Expression,
    data: pd.DataFrame | pd.Series | np.ndarray | Mapping[str, Any],
    *,
    parameters: Mapping[str, float] | None = None,
    limits: ExecutionLimits | None = None,
) -> pd.Series:
    """Evaluate a restricted expression against aligned time-series data.

    Positive ``lag_seconds`` means "known only after time has advanced":
    an observation at timestamp ``s`` contributes to the lagged output at
    ``s + lag_seconds``. The output is reindexed to the original timestamps,
    so no interpolation or forward-fill can introduce future information.
    Rolling windows are trailing windows ending at the current observation,
    and therefore use only present and past values.
    """

    frame = to_time_series_frame(data)
    checked_limits = limits or ExecutionLimits()
    _validate_expression_limits(expression, frame, checked_limits)
    result = _evaluate_node(expression, frame, dict(parameters or {}))
    return _finite_series(result.reindex(frame.index), frame.index)


def to_time_series_frame(data: pd.DataFrame | pd.Series | np.ndarray | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if "timestamp_utc" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame.pop("timestamp_utc"), utc=True))
    elif isinstance(data, pd.Series):
        name = data.name or "value"
        frame = data.rename(name).to_frame()
    elif isinstance(data, np.ndarray):
        array = np.asarray(data)
        if array.ndim == 1:
            frame = pd.DataFrame({"value": array})
        elif array.ndim == 2:
            frame = pd.DataFrame(array, columns=[f"metric_{index}" for index in range(array.shape[1])])
        else:
            raise ExpressionExecutionError("numpy time-series input must be one or two dimensional")
    elif isinstance(data, Mapping):
        if "rows" in data and isinstance(data["rows"], list):
            frame = frame_from_snapshot_rows(data["rows"])
        else:
            frame = pd.DataFrame(data)
    else:
        raise ExpressionExecutionError("data must be a pandas, numpy, or mapping time-series")

    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
    elif not isinstance(frame.index, pd.RangeIndex):
        try:
            frame.index = pd.to_datetime(frame.index, utc=True)
        except (TypeError, ValueError):
            frame.index = pd.RangeIndex(len(frame))

    if not frame.index.is_monotonic_increasing:
        raise ExpressionExecutionError("time-series index must be monotonically increasing")
    if frame.index.has_duplicates:
        raise ExpressionExecutionError("time-series index must not contain duplicate timestamps")
    return frame.apply(pd.to_numeric, errors="coerce")


def frame_from_snapshot_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"timestamp_utc": row["timestamp_utc"]}
        metrics = row.get("metrics", {})
        for metric, observations in metrics.items():
            values = [
                float(observation["value"])
                for observation in observations
                if observation.get("value") is not None
            ]
            record[metric] = float(np.mean(values)) if values else np.nan
        records.append(record)
    return to_time_series_frame(pd.DataFrame(records))


def _validate_expression_limits(
    expression: Expression,
    frame: pd.DataFrame,
    limits: ExecutionLimits,
) -> None:
    node_count = expression_node_count(expression)
    if node_count > limits.max_ast_nodes:
        raise ExpressionExecutionError("expression exceeds AST complexity budget")
    metrics = expression_metrics(expression)
    if len(metrics) > limits.max_source_metrics:
        raise ExpressionExecutionError("expression exceeds source metric budget")
    unknown = metrics - set(str(column) for column in frame.columns)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ExpressionExecutionError(f"expression references unknown metrics: {names}")


def _evaluate_node(expression: Expression, frame: pd.DataFrame, parameters: dict[str, float]) -> pd.Series:
    if isinstance(expression, MetricNode):
        return _metric(frame, expression.metric)
    if isinstance(expression, NumericConstantNode):
        return pd.Series(float(expression.value), index=frame.index, dtype="float64")
    if isinstance(expression, FittedParameterNode):
        if expression.parameter not in parameters:
            raise ExpressionExecutionError(f"missing fitted parameter: {expression.parameter}")
        return pd.Series(float(parameters[expression.parameter]), index=frame.index, dtype="float64")
    if isinstance(expression, AddNode):
        return _evaluate_node(expression.left, frame, parameters) + _evaluate_node(expression.right, frame, parameters)
    if isinstance(expression, SubtractNode):
        return _evaluate_node(expression.left, frame, parameters) - _evaluate_node(expression.right, frame, parameters)
    if isinstance(expression, MultiplyNode):
        return _evaluate_node(expression.left, frame, parameters) * _evaluate_node(expression.right, frame, parameters)
    if isinstance(expression, SafeDivideNode):
        return _safe_divide(
            _evaluate_node(expression.numerator, frame, parameters),
            _evaluate_node(expression.denominator, frame, parameters),
            epsilon=float(expression.epsilon),
            near_zero_behavior=expression.near_zero_behavior,
        )
    if isinstance(expression, AbsoluteValueNode):
        return _evaluate_node(expression.input, frame, parameters).abs()
    if isinstance(expression, ClipNode):
        return _evaluate_node(expression.input, frame, parameters).clip(
            lower=float(expression.minimum),
            upper=float(expression.maximum),
        )
    if isinstance(expression, DifferenceNode):
        current = _evaluate_node(expression.input, frame, parameters)
        return current - _lag_series(current, expression.period_seconds)
    if isinstance(expression, LagNode):
        return _lag_series(_evaluate_node(expression.input, frame, parameters), expression.lag_seconds)
    if isinstance(expression, RollingMeanNode):
        return _rolling(_evaluate_node(expression.input, frame, parameters), expression.window_seconds, expression.min_periods).mean()
    if isinstance(expression, RollingStdNode):
        return _rolling(_evaluate_node(expression.input, frame, parameters), expression.window_seconds, expression.min_periods).std()
    if isinstance(expression, RobustZscoreNode):
        source = _evaluate_node(expression.input, frame, parameters)
        rolling = _rolling(source, expression.window_seconds, expression.min_periods)
        median = rolling.median()
        mad = rolling.apply(_median_absolute_deviation, raw=False)
        scale = 1.4826 * mad
        return _safe_divide(source - median, scale, epsilon=1.0e-12, near_zero_behavior=NearZeroBehavior.RETURN_NULL)
    raise ExpressionExecutionError(f"unsupported expression node: {type(expression).__name__}")


def _metric(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in frame.columns:
        raise ExpressionExecutionError(f"expression references unknown metric: {metric}")
    return frame[metric].astype("float64")


def _lag_series(series: pd.Series, lag_seconds: int) -> pd.Series:
    if lag_seconds == 0:
        return series.copy()
    if isinstance(series.index, pd.DatetimeIndex):
        lagged = series.copy()
        lagged.index = lagged.index + pd.Timedelta(seconds=lag_seconds)
        return lagged.reindex(series.index)
    periods = int(lag_seconds)
    return series.shift(periods=periods)


def _rolling(series: pd.Series, window_seconds: int, min_periods: int) -> Any:
    if isinstance(series.index, pd.DatetimeIndex):
        return series.rolling(f"{int(window_seconds)}s", min_periods=int(min_periods))
    return series.rolling(window=int(window_seconds), min_periods=int(min_periods))


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float,
    near_zero_behavior: NearZeroBehavior,
) -> pd.Series:
    near_zero = denominator.abs() <= epsilon
    result = numerator / denominator.mask(near_zero)
    if near_zero_behavior == NearZeroBehavior.RETURN_NULL:
        result = result.mask(near_zero, np.nan)
    elif near_zero_behavior == NearZeroBehavior.RETURN_ZERO:
        result = result.mask(near_zero, 0.0)
    elif near_zero_behavior == NearZeroBehavior.USE_EPSILON_SIGN:
        sign = np.sign(denominator).replace(0.0, 1.0)
        adjusted = denominator.where(~near_zero, epsilon * sign)
        result = numerator / adjusted
    else:
        raise ExpressionExecutionError(f"unsupported near-zero behavior: {near_zero_behavior}")
    return _finite_series(result, numerator.index)


def _finite_series(series: pd.Series, index: pd.Index) -> pd.Series:
    result = pd.Series(series, index=index, dtype="float64")
    return result.replace([np.inf, -np.inf], np.nan)


def _median_absolute_deviation(values: pd.Series) -> float:
    finite = values.dropna()
    if finite.empty:
        return np.nan
    median = float(finite.median())
    return float((finite - median).abs().median())


__all__ = [
    "ExecutionLimits",
    "ExpressionExecutionError",
    "evaluate_expression",
    "frame_from_snapshot_rows",
    "to_time_series_frame",
]
