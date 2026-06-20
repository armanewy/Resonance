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

from resonance.science.contracts import HypothesisSpec, MetricName, expression_metrics, expression_node_count
from resonance.science.evaluation import (
    BASELINE_PERSISTENCE,
    BASELINE_ZERO,
    baseline_metric_bundles,
    choose_baseline_strategy,
    chronological_window_metrics,
    evaluate_program,
    metric_bundle,
    metric_improvement_fraction,
)
from resonance.science.interpreter import frame_from_snapshot_rows
from resonance.science.ledger import current_code_commit
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT, load_snapshot_manifest


EVALUATOR_VERSION = "candidate-selection-v2"
DEFAULT_WINDOW_COUNT = 4
DEFAULT_MIN_WINDOW_OBSERVATIONS = 8


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
    baseline_strategy: str | None
    baseline_improvement: float | None
    sign_consistency: float | None
    window_stability: float | None
    window_scores: tuple[float | None, ...]
    target_transform_config: dict[str, Any]
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
            "baseline_strategy": self.baseline_strategy,
            "baseline_improvement": _clean_optional_number(self.baseline_improvement),
            "sign_consistency": _clean_optional_number(self.sign_consistency),
            "window_stability": _clean_optional_number(self.window_stability),
            "window_scores": [_clean_optional_number(score) for score in self.window_scores],
            "target_transform_config": dict(self.target_transform_config),
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
    target_transform_config: dict[str, Any]


def select_candidate(
    snapshot_id: str,
    candidates: Sequence[Any],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    gates: SelectionGates | Mapping[str, Any] | None = None,
    record_artifact: bool = True,
) -> SelectionResult:
    """Evaluate already-fitted hypotheses on tuning data and select at most one winner."""

    resolved_gates = _selection_gates(gates)
    normalized = tuple(_normalize_candidate(candidate) for candidate in candidates)
    if not normalized:
        raise SelectionError("at least one fitted candidate is required")

    root = Path(artifact_root)
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=root)
    tuning_rows = _load_partition_rows(manifest, root, "tuning")
    frame = frame_from_snapshot_rows(tuning_rows)
    warnings = _snapshot_warnings(manifest, normalized)

    evaluations = tuple(_evaluate_candidate(candidate, frame, resolved_gates) for candidate in normalized)
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
    result = SelectionResult(
        snapshot_id=snapshot_id,
        evaluator_version=EVALUATOR_VERSION,
        selected_candidate_id=winner.candidate_id if winner is not None else None,
        selected_hypothesis_hash=winner.hypothesis_hash if winner is not None else None,
        evaluations=tuple(sorted(evaluations_with_winner, key=lambda item: item.candidate_id)),
        ranking=ranked_ids,
        pareto_front=pareto_front,
        artifact=None,
        warnings=tuple(warnings),
    )
    if not record_artifact:
        return result

    artifact_payload = {**result.to_dict(), "artifact": None, "code_commit": current_code_commit()}
    artifact = _store_json_artifact(root, artifact_payload)
    return SelectionResult(**{**result.__dict__, "artifact": artifact})


def _evaluate_candidate(candidate: _Candidate, frame: pd.DataFrame, gates: SelectionGates) -> CandidateEvaluation:
    hypothesis = candidate.hypothesis
    complexity = {
        "ast_nodes": expression_node_count(hypothesis.expression),
        "source_metrics": len(expression_metrics(hypothesis.expression)),
    }
    complexity_penalty = round(
        0.02 * complexity["ast_nodes"] + 0.05 * max(0, complexity["source_metrics"] - 1),
        6,
    )
    base_kwargs = {
        "candidate_id": candidate.candidate_id,
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "title": hypothesis.title,
        "fit_result_id": candidate.fit_result_id,
        "complexity": complexity,
        "complexity_penalty": complexity_penalty,
    }
    try:
        evaluated = evaluate_program(
            hypothesis,
            frame,
            parameters=candidate.fitted_parameters,
            transform_config=candidate.target_transform_config,
        )
        model = metric_bundle(evaluated.aligned["target"], evaluated.aligned["prediction"])
        baselines = baseline_metric_bundles(
            evaluated.target,
            evaluation_index=evaluated.aligned.index,
        )
        baseline_strategy = choose_baseline_strategy(baselines, hypothesis.tuning_metric)
        chosen_baseline = baselines[baseline_strategy]
    except Exception as exc:
        return _failed_evaluation(
            base_kwargs,
            target_transform_config=candidate.target_transform_config,
            warning=f"evaluation failed: {exc}",
        )

    if model.observations == 0:
        return _failed_evaluation(
            base_kwargs,
            target_transform_config=evaluated.transform_config,
            warning="no complete tuning observations after alignment",
        )

    baseline_improvement = _selection_baseline_improvement(hypothesis, chosen_baseline, model)
    windows = chronological_window_metrics(
        evaluated.target,
        evaluated.prediction,
        baseline_strategy=baseline_strategy,
        window_count=gates.window_count,
    )
    usable_windows = [window for window in windows if window["observation_count"] >= gates.min_window_observations]
    window_scores = tuple(window["spearman_rho"] for window in usable_windows)
    stable_windows = [
        window
        for window in usable_windows
        if window["spearman_rho"] is not None
        and float(window["spearman_rho"]) >= 0.20
        and (
            window["mae_improvement_fraction"] is None
            or float(window["mae_improvement_fraction"]) >= 0.0
        )
    ]
    window_stability = len(stable_windows) / len(usable_windows) if usable_windows else 0.0
    sign_consistency = model.direction_agreement or 0.0
    spearman = model.spearman_r or 0.0
    required_improvement = max(
        gates.min_baseline_improvement,
        float(hypothesis.minimum_baseline_improvement),
    )
    passed_gates = {
        "tuning_observations": model.observations >= gates.min_tuning_observations,
        "spearman": spearman >= gates.min_abs_spearman,
        "baseline_improvement": baseline_improvement is not None
        and baseline_improvement >= required_improvement,
        "sign_consistency": sign_consistency >= gates.min_sign_consistency,
        "window_stability": window_stability >= gates.min_window_stability,
    }
    score = (
        (baseline_improvement if baseline_improvement is not None else -1.0) * 100.0
        + spearman * 20.0
        + sign_consistency * 10.0
        + window_stability * 10.0
        - complexity_penalty
    )
    return CandidateEvaluation(
        **base_kwargs,
        tuning_observations=model.observations,
        tuning_mae=model.mae,
        tuning_rmse=model.rmse,
        tuning_spearman_rho=model.spearman_r,
        zero_baseline_mae=baselines[BASELINE_ZERO].mae,
        zero_baseline_rmse=baselines[BASELINE_ZERO].rmse,
        persistence_baseline_mae=baselines[BASELINE_PERSISTENCE].mae,
        persistence_baseline_rmse=baselines[BASELINE_PERSISTENCE].rmse,
        baseline_strategy=baseline_strategy,
        baseline_improvement=baseline_improvement,
        sign_consistency=sign_consistency,
        window_stability=window_stability,
        window_scores=window_scores,
        target_transform_config=evaluated.transform_config,
        score=score,
        passed_gates=passed_gates,
    )


def _selection_baseline_improvement(hypothesis: HypothesisSpec, baseline: Any, model: Any) -> float | None:
    metric = hypothesis.tuning_metric
    if metric in {MetricName.MAE, MetricName.RMSE}:
        return metric_improvement_fraction(baseline, model, metric)
    # Correlation objectives still need a scale-sensitive guard against trivial predictions.
    return metric_improvement_fraction(baseline, model, MetricName.MAE)


def _failed_evaluation(
    base_kwargs: Mapping[str, Any],
    *,
    target_transform_config: Mapping[str, Any],
    warning: str,
) -> CandidateEvaluation:
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
        baseline_strategy=None,
        baseline_improvement=None,
        sign_consistency=None,
        window_stability=None,
        window_scores=(),
        target_transform_config=dict(target_transform_config),
        score=-math.inf,
        passed_gates=_empty_gates(),
        warnings=(warning,),
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
    deterministic_artifact = _object_mapping(fit_data.get("deterministic_fit_artifact"))
    parameters_value = (
        _first_present(data, "fitted_parameters", "parameters")
        or _first_present(fit_data, "fitted_parameters", "parameters", "fit_parameters", "coefficients")
        or {}
    )
    parameters = {str(key): float(value) for key, value in _object_mapping(parameters_value).items()}
    target_transform_config = (
        _first_present(data, "target_transform_config", "transform_config")
        or _first_present(fit_data, "target_transform_config", "transform_config")
        or _first_present(deterministic_artifact, "target_transform_config", "transform_config")
        or {}
    )
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
        target_transform_config=dict(target_transform_config),
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
    path = _resolve_under(root, artifact["path"])
    content = path.read_bytes()
    if _sha256(content) != artifact["sha256"]:
        raise SelectionError(f"{partition} artifact hash mismatch for {path}")
    payload = json.loads(gzip.decompress(content).decode("utf-8"))
    if payload.get("snapshot_id") != manifest.get("snapshot_id") or payload.get("partition") != partition:
        raise SelectionError(f"{partition} artifact identity does not match snapshot manifest")
    return list(payload["rows"])


def _resolve_under(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise SelectionError("artifact path escapes artifact root")
    return path


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
