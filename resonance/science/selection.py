from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from resonance.science.contracts import (
    Direction,
    HypothesisSpec,
    TargetTransform,
    expression_metrics,
    expression_node_count,
)
from resonance.science.ledger import current_code_commit
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT, load_snapshot_manifest
from resonance.time_utils import parse_utc


EVALUATOR_VERSION = "candidate-selection-v1"
DEFAULT_WINDOW_COUNT = 4
DEFAULT_MIN_WINDOW_OBSERVATIONS = 8
DEFAULT_ROBUST_WINDOW_SECONDS = 3600
DEFAULT_ROBUST_MIN_PERIODS = 5


class SelectionError(ValueError):
    """Raised when candidate selection cannot evaluate the supplied inputs."""


@dataclass(frozen=True)
class SelectionGates:
    min_tuning_observations: int = 20
    min_abs_spearman: float = 0.35
    min_baseline_improvement: float = 0.05
    min_sign_consistency: float = 0.60
    min_window_stability: float = 0.60
    min_window_observations: int = DEFAULT_MIN_WINDOW_OBSERVATIONS
    window_count: int = DEFAULT_WINDOW_COUNT


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    hypothesis_hash: str
    title: str
    fit_result_id: str | None
    tuning_observations: int
    tuning_mae: float | None
    tuning_rmse: float | None
    tuning_spearman_rho: float | None
    zero_baseline_mae: float | None
    zero_baseline_rmse: float | None
    persistence_baseline_mae: float | None
    persistence_baseline_rmse: float | None
    baseline_improvement: float | None
    sign_consistency: float | None
    window_stability: float | None
    window_scores: tuple[float | None, ...]
    complexity: dict[str, int]
    complexity_penalty: float
    score: float
    passed_gates: dict[str, bool]
    default_winner: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(self.passed_gates.values()) if self.passed_gates else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis_hash": self.hypothesis_hash,
            "title": self.title,
            "fit_result_id": self.fit_result_id,
            "tuning_observations": self.tuning_observations,
            "tuning_mae": _clean_optional_number(self.tuning_mae),
            "tuning_rmse": _clean_optional_number(self.tuning_rmse),
            "tuning_spearman_rho": _clean_optional_number(self.tuning_spearman_rho),
            "zero_baseline_mae": _clean_optional_number(self.zero_baseline_mae),
            "zero_baseline_rmse": _clean_optional_number(self.zero_baseline_rmse),
            "persistence_baseline_mae": _clean_optional_number(self.persistence_baseline_mae),
            "persistence_baseline_rmse": _clean_optional_number(self.persistence_baseline_rmse),
            "baseline_improvement": _clean_optional_number(self.baseline_improvement),
            "sign_consistency": _clean_optional_number(self.sign_consistency),
            "window_stability": _clean_optional_number(self.window_stability),
            "window_scores": [_clean_optional_number(score) for score in self.window_scores],
            "complexity": dict(self.complexity),
            "complexity_penalty": _clean_number(self.complexity_penalty),
            "score": _clean_number(self.score),
            "passed_gates": dict(self.passed_gates),
            "default_winner": self.default_winner,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SelectionResult:
    snapshot_id: str
    evaluator_version: str
    selected_candidate_id: str | None
    selected_hypothesis_hash: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    ranking: tuple[str, ...]
    pareto_front: tuple[str, ...]
    artifact: dict[str, str] | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "evaluator_version": self.evaluator_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_hypothesis_hash": self.selected_hypothesis_hash,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "ranking": list(self.ranking),
            "pareto_front": list(self.pareto_front),
            "artifact": None if self.artifact is None else dict(self.artifact),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    hypothesis: HypothesisSpec
    fitted_parameters: dict[str, float]
    fit_result_id: str | None


def select_candidate(
    snapshot_id: str,
    candidates: Sequence[Any],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    gates: SelectionGates | Mapping[str, Any] | None = None,
    record_artifact: bool = True,
) -> SelectionResult:
    """Evaluate fitted hypotheses on tuning data and select at most one winner.

    Candidate inputs may be mappings, dataclasses, or small protocol-like
    objects. Each candidate should expose a hypothesis/spec and an exploration
    fit result or fitted parameter mapping. The selector never refits.
    """

    resolved_gates = _selection_gates(gates)
    normalized = tuple(_normalize_candidate(candidate) for candidate in candidates)
    if not normalized:
        raise SelectionError("at least one fitted candidate is required")

    root = Path(artifact_root)
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=root)
    tuning_rows = _load_partition_rows(manifest, root, "tuning")
    frame = _frame_from_rows(tuning_rows)
    warnings = _snapshot_warnings(manifest, normalized)

    evaluations = tuple(
        _evaluate_candidate(candidate, frame, resolved_gates)
        for candidate in normalized
    )
    ranked = tuple(sorted(evaluations, key=_ranking_key))
    winner = next((evaluation for evaluation in ranked if evaluation.passed), None)
    evaluations_with_winner = tuple(
        CandidateEvaluation(
            **{
                **evaluation.__dict__,
                "default_winner": winner is not None and evaluation.candidate_id == winner.candidate_id,
            }
        )
        for evaluation in evaluations
    )
    ranked_ids = tuple(evaluation.candidate_id for evaluation in ranked)
    pareto_front = _pareto_front(evaluations_with_winner)
    selected_hash = winner.hypothesis_hash if winner is not None else None
    result = SelectionResult(
        snapshot_id=snapshot_id,
        evaluator_version=EVALUATOR_VERSION,
        selected_candidate_id=winner.candidate_id if winner is not None else None,
        selected_hypothesis_hash=selected_hash,
        evaluations=tuple(sorted(evaluations_with_winner, key=lambda item: item.candidate_id)),
        ranking=ranked_ids,
        pareto_front=pareto_front,
        artifact=None,
        warnings=tuple(warnings),
    )
    if not record_artifact:
        return result

    artifact_payload = {
        **result.to_dict(),
        "artifact": None,
        "code_commit": current_code_commit(),
    }
    artifact = _store_json_artifact(root, artifact_payload)
    return SelectionResult(
        snapshot_id=result.snapshot_id,
        evaluator_version=result.evaluator_version,
        selected_candidate_id=result.selected_candidate_id,
        selected_hypothesis_hash=result.selected_hypothesis_hash,
        evaluations=result.evaluations,
        ranking=result.ranking,
        pareto_front=result.pareto_front,
        artifact=artifact,
        warnings=result.warnings,
    )


def _evaluate_candidate(
    candidate: _Candidate,
    frame: pd.DataFrame,
    gates: SelectionGates,
) -> CandidateEvaluation:
    hypothesis = candidate.hypothesis
    complexity = {
        "ast_nodes": expression_node_count(hypothesis.expression),
        "source_metrics": len(expression_metrics(hypothesis.expression)),
    }
    complexity_penalty = round(0.02 * complexity["ast_nodes"] + 0.05 * max(0, complexity["source_metrics"] - 1), 6)
    base_kwargs = {
        "candidate_id": candidate.candidate_id,
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "title": hypothesis.title,
        "fit_result_id": candidate.fit_result_id,
        "complexity": complexity,
        "complexity_penalty": complexity_penalty,
    }
    try:
        actual = _target_series(frame, hypothesis)
        predicted = _eval_expression(hypothesis.expression, frame, candidate.fitted_parameters)
        aligned = pd.concat(
            {
                "actual": actual,
                "predicted": predicted,
                "zero": pd.Series(0.0, index=actual.index),
                "persistence": actual.shift(1),
            },
            axis=1,
        ).dropna(how="any")
    except Exception as exc:
        return CandidateEvaluation(
            **base_kwargs,
            tuning_observations=0,
            tuning_mae=None,
            tuning_rmse=None,
            tuning_spearman_rho=None,
            zero_baseline_mae=None,
            zero_baseline_rmse=None,
            persistence_baseline_mae=None,
            persistence_baseline_rmse=None,
            baseline_improvement=None,
            sign_consistency=None,
            window_stability=None,
            window_scores=(),
            score=-math.inf,
            passed_gates=_empty_gates(),
            warnings=(f"evaluation failed: {exc}",),
        )

    tuning_observations = len(aligned)
    if tuning_observations == 0:
        return CandidateEvaluation(
            **base_kwargs,
            tuning_observations=0,
            tuning_mae=None,
            tuning_rmse=None,
            tuning_spearman_rho=None,
            zero_baseline_mae=None,
            zero_baseline_rmse=None,
            persistence_baseline_mae=None,
            persistence_baseline_rmse=None,
            baseline_improvement=None,
            sign_consistency=None,
            window_stability=None,
            window_scores=(),
            score=-math.inf,
            passed_gates=_empty_gates(),
            warnings=("no complete tuning observations after alignment",),
        )

    actual_values = aligned["actual"]
    predicted_values = aligned["predicted"]
    tuning_mae = _mae(actual_values, predicted_values)
    tuning_rmse = _rmse(actual_values, predicted_values)
    zero_mae = _mae(actual_values, aligned["zero"])
    zero_rmse = _rmse(actual_values, aligned["zero"])
    persistence_mae = _mae(actual_values, aligned["persistence"])
    persistence_rmse = _rmse(actual_values, aligned["persistence"])
    baseline_mae = min(zero_mae, persistence_mae)
    baseline_improvement = _relative_improvement(baseline_mae, tuning_mae)
    spearman = _spearman(actual_values, predicted_values)
    sign_consistency = _sign_consistency(actual_values, predicted_values, hypothesis.expected_direction)
    window_scores, window_stability = _window_stability(
        aligned,
        hypothesis.expected_direction,
        window_count=gates.window_count,
        min_window_observations=gates.min_window_observations,
    )
    passed_gates = {
        "tuning_observations": tuning_observations >= gates.min_tuning_observations,
        "spearman": _direction_passes(spearman, hypothesis.expected_direction, gates.min_abs_spearman),
        "baseline_improvement": baseline_improvement
        >= max(gates.min_baseline_improvement, float(hypothesis.minimum_baseline_improvement)),
        "sign_consistency": sign_consistency >= gates.min_sign_consistency,
        "window_stability": window_stability >= gates.min_window_stability,
    }
    score = (
        baseline_improvement * 100.0
        + abs(spearman) * 20.0
        + sign_consistency * 10.0
        + window_stability * 10.0
        - complexity_penalty
    )
    return CandidateEvaluation(
        **base_kwargs,
        tuning_observations=tuning_observations,
        tuning_mae=tuning_mae,
        tuning_rmse=tuning_rmse,
        tuning_spearman_rho=spearman,
        zero_baseline_mae=zero_mae,
        zero_baseline_rmse=zero_rmse,
        persistence_baseline_mae=persistence_mae,
        persistence_baseline_rmse=persistence_rmse,
        baseline_improvement=baseline_improvement,
        sign_consistency=sign_consistency,
        window_stability=window_stability,
        window_scores=window_scores,
        score=score,
        passed_gates=passed_gates,
    )


def _normalize_candidate(candidate: Any) -> _Candidate:
    data = _object_mapping(candidate)
    hypothesis_value = _first_present(data, "hypothesis", "spec", "hypothesis_spec") or candidate
    hypothesis = (
        hypothesis_value
        if isinstance(hypothesis_value, HypothesisSpec)
        else HypothesisSpec.model_validate(_object_mapping(hypothesis_value))
    )
    fit_result = _first_present(data, "fit_result", "exploration_fit_result", "fit")
    fit_data = _object_mapping(fit_result) if fit_result is not None else {}
    parameters_value = (
        _first_present(data, "fitted_parameters", "parameters")
        or _first_present(fit_data, "fitted_parameters", "parameters", "fit_parameters", "coefficients")
        or {}
    )
    parameters = {
        str(key): float(value)
        for key, value in _object_mapping(parameters_value).items()
    }
    candidate_id = str(
        _first_present(data, "candidate_id", "id", "name")
        or _first_present(fit_data, "candidate_id", "hypothesis_id")
        or hypothesis.hypothesis_hash()[:16]
    )
    fit_result_id = _first_present(fit_data, "fit_result_id", "id", "artifact_id")
    return _Candidate(
        candidate_id=candidate_id,
        hypothesis=hypothesis,
        fitted_parameters=parameters,
        fit_result_id=None if fit_result_id is None else str(fit_result_id),
    )


def _selection_gates(gates: SelectionGates | Mapping[str, Any] | None) -> SelectionGates:
    if gates is None:
        return SelectionGates()
    if isinstance(gates, SelectionGates):
        return gates
    return SelectionGates(**dict(gates))


def _object_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, HypothesisSpec):
        return value.model_dump(mode="json", exclude_none=True)
    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _first_present(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _snapshot_warnings(manifest: Mapping[str, Any], candidates: Sequence[_Candidate]) -> list[str]:
    warnings = []
    embargo_seconds = int(manifest.get("embargo_seconds", 0))
    max_candidate_lag = max((candidate.hypothesis.maximum_lag_seconds for candidate in candidates), default=0)
    if embargo_seconds < max_candidate_lag:
        warnings.append(
            f"snapshot embargo ({embargo_seconds}s) is smaller than candidate maximum lag ({max_candidate_lag}s)"
        )
    return warnings


def _load_partition_rows(manifest: Mapping[str, Any], root: Path, partition: str) -> list[dict[str, Any]]:
    if partition == "blind":
        raise PermissionError("candidate selection must not load blind data")
    artifact = manifest["artifacts"][partition]
    path = root / artifact["path"]
    content = path.read_bytes()
    if _sha256(content) != artifact["sha256"]:
        raise SelectionError(f"{partition} artifact hash mismatch for {path}")
    payload = json.loads(gzip.decompress(content).decode("utf-8"))
    return list(payload["rows"])


def _frame_from_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"timestamp_utc": parse_utc(str(row["timestamp_utc"]))}
        for metric, observations in row.get("metrics", {}).items():
            values = [
                float(observation["value"])
                for observation in observations
                if _is_finite_number(observation.get("value"))
            ]
            if values:
                record[str(metric)] = sum(values) / len(values)
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records).set_index("timestamp_utc").sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def _target_series(frame: pd.DataFrame, hypothesis: HypothesisSpec) -> pd.Series:
    if hypothesis.target_metric not in frame:
        raise SelectionError(f"target metric {hypothesis.target_metric!r} is absent from tuning data")
    target = frame[hypothesis.target_metric].astype(float)
    if hypothesis.target_transform == TargetTransform.IDENTITY:
        return target
    if hypothesis.target_transform == TargetTransform.DIFFERENCE:
        return target.diff()
    if hypothesis.target_transform == TargetTransform.ROBUST_ZSCORE:
        return _robust_zscore(target, DEFAULT_ROBUST_WINDOW_SECONDS, DEFAULT_ROBUST_MIN_PERIODS)
    raise SelectionError(f"unsupported target transform {hypothesis.target_transform!r}")


def _eval_expression(expression: Any, frame: pd.DataFrame, parameters: Mapping[str, float]) -> pd.Series:
    node = _node_type(expression)
    if node == "metric":
        metric = _field(expression, "metric")
        if metric not in frame:
            raise SelectionError(f"metric {metric!r} is absent from tuning data")
        return frame[str(metric)].astype(float)
    if node == "numeric_constant":
        return pd.Series(float(_field(expression, "value")), index=frame.index, dtype=float)
    if node == "fitted_parameter":
        parameter = str(_field(expression, "parameter"))
        if parameter not in parameters:
            raise SelectionError(f"missing fitted parameter {parameter!r}")
        return pd.Series(float(parameters[parameter]), index=frame.index, dtype=float)
    if node == "add":
        return _eval_expression(_field(expression, "left"), frame, parameters) + _eval_expression(
            _field(expression, "right"), frame, parameters
        )
    if node == "subtract":
        return _eval_expression(_field(expression, "left"), frame, parameters) - _eval_expression(
            _field(expression, "right"), frame, parameters
        )
    if node == "multiply":
        return _eval_expression(_field(expression, "left"), frame, parameters) * _eval_expression(
            _field(expression, "right"), frame, parameters
        )
    if node == "safe_divide":
        numerator = _eval_expression(_field(expression, "numerator"), frame, parameters)
        denominator = _eval_expression(_field(expression, "denominator"), frame, parameters)
        epsilon = float(_field(expression, "epsilon"))
        near_zero = str(_field(expression, "near_zero_behavior"))
        safe = denominator.abs() >= epsilon
        result = numerator / denominator.where(safe)
        if near_zero == "return_zero":
            return result.where(safe, 0.0)
        if near_zero == "use_epsilon_sign":
            signs = denominator.apply(lambda value: 1.0 if value >= 0 else -1.0)
            return result.where(safe, numerator / (epsilon * signs))
        return result
    if node == "absolute_value":
        return _eval_expression(_field(expression, "input"), frame, parameters).abs()
    if node == "clip":
        return _eval_expression(_field(expression, "input"), frame, parameters).clip(
            lower=float(_field(expression, "minimum")),
            upper=float(_field(expression, "maximum")),
        )
    if node == "difference":
        series = _eval_expression(_field(expression, "input"), frame, parameters)
        return series - _lag_series(series, int(_field(expression, "period_seconds")))
    if node == "lag":
        return _lag_series(
            _eval_expression(_field(expression, "input"), frame, parameters),
            int(_field(expression, "lag_seconds")),
        )
    if node == "rolling_mean":
        return _eval_expression(_field(expression, "input"), frame, parameters).rolling(
            f"{int(_field(expression, 'window_seconds'))}s",
            min_periods=int(_field(expression, "min_periods")),
        ).mean()
    if node == "rolling_std":
        return _eval_expression(_field(expression, "input"), frame, parameters).rolling(
            f"{int(_field(expression, 'window_seconds'))}s",
            min_periods=int(_field(expression, "min_periods")),
        ).std()
    if node == "robust_zscore":
        return _robust_zscore(
            _eval_expression(_field(expression, "input"), frame, parameters),
            int(_field(expression, "window_seconds")),
            int(_field(expression, "min_periods")),
        )
    raise SelectionError(f"unsupported expression node {node!r}")


def _node_type(expression: Any) -> str:
    return str(_field(expression, "node"))


def _field(expression: Any, name: str) -> Any:
    if isinstance(expression, Mapping):
        return expression[name]
    return getattr(expression, name)


def _lag_series(series: pd.Series, lag_seconds: int) -> pd.Series:
    if lag_seconds < 0:
        raise SelectionError("lag_seconds must be non-negative")
    if lag_seconds == 0:
        return series
    cadence = _median_cadence_seconds(series.index)
    if cadence is None or cadence <= 0:
        raise SelectionError("cannot apply lag without at least two timestamps")
    periods = max(1, round(lag_seconds / cadence))
    return series.shift(periods)


def _robust_zscore(series: pd.Series, window_seconds: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(f"{window_seconds}s", min_periods=min_periods)
    median = rolling.median()
    absolute_deviation = (series - median).abs()
    mad = absolute_deviation.rolling(f"{window_seconds}s", min_periods=min_periods).median()
    return (series - median) / (1.4826 * mad.replace(0.0, math.nan))


def _median_cadence_seconds(index: pd.Index) -> float | None:
    if len(index) < 2:
        return None
    deltas = pd.Series(index).diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    return float(deltas.median())


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(math.sqrt(((actual - predicted) ** 2).mean()))


def _relative_improvement(baseline_error: float, candidate_error: float) -> float:
    if baseline_error <= 0:
        return 0.0 if candidate_error <= 0 else -math.inf
    return float((baseline_error - candidate_error) / baseline_error)


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    frame = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna(how="any")
    if len(frame) < 2:
        return 0.0
    left = frame["actual"].rank(method="average")
    right = frame["predicted"].rank(method="average")
    if left.nunique() < 2 or right.nunique() < 2:
        return 0.0
    value = left.corr(right)
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _sign_consistency(actual: pd.Series, predicted: pd.Series, direction: Direction) -> float:
    frame = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna(how="any")
    if frame.empty:
        return 0.0
    actual_centered = frame["actual"] - frame["actual"].median()
    predicted_centered = frame["predicted"] - frame["predicted"].median()
    pairs = [
        (_sign(a), _sign(p))
        for a, p in zip(actual_centered, predicted_centered, strict=False)
        if _sign(a) != 0 and _sign(p) != 0
    ]
    if not pairs:
        return 0.0
    if direction == Direction.NEGATIVE:
        matches = sum(1 for actual_sign, predicted_sign in pairs if actual_sign == -predicted_sign)
    else:
        matches = sum(1 for actual_sign, predicted_sign in pairs if actual_sign == predicted_sign)
    return matches / len(pairs)


def _window_stability(
    aligned: pd.DataFrame,
    direction: Direction,
    *,
    window_count: int,
    min_window_observations: int,
) -> tuple[tuple[float | None, ...], float]:
    if window_count <= 0:
        return (), 0.0
    windows = []
    size = math.ceil(len(aligned) / window_count)
    for start in range(0, len(aligned), size):
        windows.append(aligned.iloc[start : start + size])
    scores: list[float | None] = []
    stable = 0
    usable = 0
    for window in windows[:window_count]:
        if len(window) < min_window_observations:
            scores.append(None)
            continue
        rho = _spearman(window["actual"], window["predicted"])
        scores.append(rho)
        usable += 1
        if _direction_passes(rho, direction, 0.20):
            stable += 1
    if usable == 0:
        return tuple(scores), 0.0
    return tuple(scores), stable / usable


def _direction_passes(value: float, direction: Direction, minimum_abs: float) -> bool:
    if abs(value) < minimum_abs:
        return False
    if direction == Direction.POSITIVE:
        return value > 0
    if direction == Direction.NEGATIVE:
        return value < 0
    return True


def _ranking_key(evaluation: CandidateEvaluation) -> tuple[Any, ...]:
    score = evaluation.score if math.isfinite(evaluation.score) else -math.inf
    mae = evaluation.tuning_mae if evaluation.tuning_mae is not None else math.inf
    rmse = evaluation.tuning_rmse if evaluation.tuning_rmse is not None else math.inf
    return (
        not evaluation.passed,
        -score,
        mae,
        rmse,
        evaluation.complexity["ast_nodes"],
        evaluation.hypothesis_hash,
        evaluation.candidate_id,
    )


def _pareto_front(evaluations: Sequence[CandidateEvaluation]) -> tuple[str, ...]:
    front = []
    for candidate in evaluations:
        candidate_mae = candidate.tuning_mae
        if candidate_mae is None:
            continue
        candidate_complexity = candidate.complexity["ast_nodes"]
        dominated = False
        for other in evaluations:
            if other.candidate_id == candidate.candidate_id or other.tuning_mae is None:
                continue
            other_complexity = other.complexity["ast_nodes"]
            if (
                other.tuning_mae <= candidate_mae
                and other_complexity <= candidate_complexity
                and (other.tuning_mae < candidate_mae or other_complexity < candidate_complexity)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(
        key=lambda item: (
            item.tuning_mae if item.tuning_mae is not None else math.inf,
            item.complexity["ast_nodes"],
            item.hypothesis_hash,
            item.candidate_id,
        )
    )
    return tuple(item.candidate_id for item in front)


def _empty_gates() -> dict[str, bool]:
    return {
        "tuning_observations": False,
        "spearman": False,
        "baseline_improvement": False,
        "sign_consistency": False,
        "window_stability": False,
    }


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = _sha256(content)
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise SelectionError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": f"sha256/{digest[:2]}/{digest}.json", "format": "json"}


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clean_optional_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    return _clean_number(value)


def _clean_number(value: float | int) -> float | int:
    numeric = float(value)
    if not math.isfinite(numeric):
        return numeric
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 6)


__all__ = [
    "CandidateEvaluation",
    "SelectionError",
    "SelectionGates",
    "SelectionResult",
    "select_candidate",
]
