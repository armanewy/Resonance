from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from resonance.science.contracts import HypothesisSpec, TargetTransform, canonical_json, expression_parameters, stable_hash
from resonance.science.evaluation import BASELINE_STRATEGIES, BASELINE_ZERO
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


PREREGISTRATION_SCHEMA_VERSION = 2
EVALUATOR_VERSION = "sealed-blind-evaluator-2"
EVALUATOR_RULESET = {
    "name": "one-shot-blind-evaluator",
    "verdict_source": "deterministic_numeric_metrics",
    "raw_blind_observations_exposed": False,
    "one_evaluation_per_preregistration": True,
    "budget_consumed_before_blind_access": True,
    "baseline_recomputed_on_blind_partition": True,
    "shared_program_semantics": True,
}
DEFAULT_EVALUATION_BUDGET = 1


class PreregistrationError(ValueError):
    """Raised when a preregistration cannot be created or loaded."""


@dataclass(frozen=True)
class ScientificPreregistration:
    schema_version: int
    snapshot_id: str
    snapshot_artifacts: dict[str, Any]
    snapshot_split: dict[str, Any]
    hypothesis_hash: str
    exact_expression: dict[str, Any]
    fitted_parameters: dict[str, float]
    target_metric: str
    input_metrics: tuple[str, ...]
    target_transform: str
    expected_direction: str
    transform_config: dict[str, Any]
    blind_metrics: tuple[str, ...]
    baseline_strategy: str
    baseline_metrics: dict[str, float]
    minimum_blind_effect: float
    minimum_baseline_improvement: float
    negative_controls: tuple[dict[str, Any], ...]
    falsification_conditions: tuple[dict[str, Any], ...]
    evaluator_version: str
    evaluator_identity_hash: str
    evaluator_code_commit: str
    random_seed: int
    evaluation_budget: int
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifacts": self.snapshot_artifacts,
            "snapshot_split": self.snapshot_split,
            "hypothesis_hash": self.hypothesis_hash,
            "exact_expression": self.exact_expression,
            "fitted_parameters": self.fitted_parameters,
            "target_metric": self.target_metric,
            "input_metrics": list(self.input_metrics),
            "target_transform": self.target_transform,
            "expected_direction": self.expected_direction,
            "transform_config": self.transform_config,
            "blind_metrics": list(self.blind_metrics),
            "baseline_strategy": self.baseline_strategy,
            "baseline_metrics": self.baseline_metrics,
            "minimum_blind_effect": self.minimum_blind_effect,
            "minimum_baseline_improvement": self.minimum_baseline_improvement,
            "negative_controls": list(self.negative_controls),
            "falsification_conditions": list(self.falsification_conditions),
            "evaluator_version": self.evaluator_version,
            "evaluator_identity_hash": self.evaluator_identity_hash,
            "evaluator_code_commit": self.evaluator_code_commit,
            "random_seed": self.random_seed,
            "evaluation_budget": self.evaluation_budget,
            "created_at_utc": self.created_at_utc,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def preregistration_hash(self) -> str:
        return stable_hash(self.to_dict())


def create_preregistration(
    *,
    hypothesis: HypothesisSpec | Mapping[str, Any],
    snapshot_manifest: Mapping[str, Any],
    fitted_parameters: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    baseline_strategy: str = BASELINE_ZERO,
    transform_config: Mapping[str, Any] | None = None,
    evaluation_budget: int = DEFAULT_EVALUATION_BUDGET,
    created_at_utc: datetime | str | None = None,
) -> ScientificPreregistration:
    """Freeze the exact program, data identity, evaluator identity, and decision gates."""

    if not isinstance(hypothesis, HypothesisSpec):
        hypothesis = HypothesisSpec.model_validate(
            hypothesis,
            context={"metric_catalog": snapshot_manifest.get("metric_catalog")},
        )
    else:
        hypothesis.validate_metric_catalog(snapshot_manifest.get("metric_catalog", ()))

    if evaluation_budget != DEFAULT_EVALUATION_BUDGET:
        raise PreregistrationError("sealed blind evaluation budget must be exactly one")
    if baseline_strategy not in BASELINE_STRATEGIES:
        raise PreregistrationError(f"unsupported baseline strategy: {baseline_strategy}")

    snapshot_id = str(snapshot_manifest.get("snapshot_id") or "")
    if not _is_hash(snapshot_id):
        raise PreregistrationError("snapshot_manifest must include a valid snapshot_id")
    snapshot_artifacts = dict(snapshot_manifest.get("artifacts") or {})
    if "blind" not in snapshot_artifacts:
        raise PreregistrationError("snapshot_manifest must include a blind artifact")

    expected_parameters = expression_parameters(hypothesis.expression)
    normalized_parameters = {str(name): float(value) for name, value in fitted_parameters.items()}
    if set(normalized_parameters) != expected_parameters:
        missing = expected_parameters - set(normalized_parameters)
        extra = set(normalized_parameters) - expected_parameters
        details = []
        if missing:
            details.append(f"missing fitted parameters: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected fitted parameters: {', '.join(sorted(extra))}")
        raise PreregistrationError("; ".join(details))

    normalized_baselines = _finite_mapping(baseline_metrics, "baseline_metrics")
    normalized_transform = dict(transform_config or {})
    _validate_frozen_transform(hypothesis.target_transform, normalized_transform)

    created = _normalize_created_at(created_at_utc)
    return ScientificPreregistration(
        schema_version=PREREGISTRATION_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        snapshot_artifacts=snapshot_artifacts,
        snapshot_split=dict(snapshot_manifest.get("split_boundaries") or {}),
        hypothesis_hash=hypothesis.hypothesis_hash(),
        exact_expression=hypothesis.expression.model_dump(mode="json"),
        fitted_parameters=normalized_parameters,
        target_metric=str(hypothesis.target_metric),
        input_metrics=tuple(str(metric) for metric in hypothesis.input_metrics),
        target_transform=str(hypothesis.target_transform.value),
        expected_direction=str(hypothesis.expected_direction.value),
        transform_config=normalized_transform,
        blind_metrics=tuple(str(metric.value) for metric in hypothesis.blind_metrics),
        baseline_strategy=baseline_strategy,
        baseline_metrics=normalized_baselines,
        minimum_blind_effect=float(hypothesis.minimum_blind_effect),
        minimum_baseline_improvement=float(hypothesis.minimum_baseline_improvement),
        negative_controls=tuple(control.model_dump(mode="json") for control in hypothesis.negative_controls),
        falsification_conditions=tuple(
            condition.model_dump(mode="json") for condition in hypothesis.falsification_conditions
        ),
        evaluator_version=EVALUATOR_VERSION,
        evaluator_identity_hash=current_evaluator_identity_hash(),
        evaluator_code_commit=current_code_commit(),
        random_seed=int(hypothesis.random_seed),
        evaluation_budget=DEFAULT_EVALUATION_BUDGET,
        created_at_utc=created,
    )


def load_preregistration(value: ScientificPreregistration | Mapping[str, Any]) -> ScientificPreregistration:
    if isinstance(value, ScientificPreregistration):
        return value
    try:
        return ScientificPreregistration(
            schema_version=int(value["schema_version"]),
            snapshot_id=str(value["snapshot_id"]),
            snapshot_artifacts=dict(value["snapshot_artifacts"]),
            snapshot_split=dict(value["snapshot_split"]),
            hypothesis_hash=str(value["hypothesis_hash"]),
            exact_expression=dict(value["exact_expression"]),
            fitted_parameters={str(name): float(parameter) for name, parameter in dict(value["fitted_parameters"]).items()},
            target_metric=str(value["target_metric"]),
            input_metrics=tuple(str(metric) for metric in value["input_metrics"]),
            target_transform=str(value["target_transform"]),
            expected_direction=str(value["expected_direction"]),
            transform_config=dict(value["transform_config"]),
            blind_metrics=tuple(str(metric) for metric in value["blind_metrics"]),
            baseline_strategy=str(value.get("baseline_strategy", BASELINE_ZERO)),
            baseline_metrics={str(name): float(metric) for name, metric in dict(value.get("baseline_metrics", {})).items()},
            minimum_blind_effect=float(value["minimum_blind_effect"]),
            minimum_baseline_improvement=float(value["minimum_baseline_improvement"]),
            negative_controls=tuple(dict(control) for control in value["negative_controls"]),
            falsification_conditions=tuple(dict(condition) for condition in value["falsification_conditions"]),
            evaluator_version=str(value["evaluator_version"]),
            evaluator_identity_hash=str(value["evaluator_identity_hash"]),
            evaluator_code_commit=str(value["evaluator_code_commit"]),
            random_seed=int(value["random_seed"]),
            evaluation_budget=int(value["evaluation_budget"]),
            created_at_utc=to_utc_iso(parse_utc(str(value["created_at_utc"]))),
        )
    except KeyError as exc:
        raise PreregistrationError(f"missing preregistration field: {exc.args[0]}") from exc


def append_preregistration_event(
    preregistration: ScientificPreregistration,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    payload = preregistration.to_dict()
    payload["preregistration_hash"] = preregistration.preregistration_hash()
    return append_event(
        "hypothesis_preregistered",
        payload,
        artifact_hashes={
            "preregistration": preregistration.preregistration_hash(),
            "snapshot_blind": preregistration.snapshot_artifacts["blind"]["sha256"],
        },
        code_commit=preregistration.evaluator_code_commit,
        ledger_path=ledger_path,
    )


def current_evaluator_identity_hash() -> str:
    package_root = Path(__file__).resolve().parent
    critical_files = (
        "contracts.py",
        "evaluation.py",
        "interpreter.py",
        "preregistration.py",
        "blind_evaluator.py",
    )
    source_hashes = {
        name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        for name in critical_files
    }
    dependency_versions = {}
    for distribution in ("numpy", "pandas", "pydantic", "scipy"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = "missing"
    return stable_hash(
        {
            "evaluator_version": EVALUATOR_VERSION,
            "ruleset": EVALUATOR_RULESET,
            "source_hashes": source_hashes,
            "dependency_versions": dependency_versions,
        }
    )


def _validate_frozen_transform(transform: TargetTransform, config: Mapping[str, Any]) -> None:
    if transform == TargetTransform.DIFFERENCE and not (
        "target_difference_period_seconds" in config or "target_difference_periods" in config
    ):
        raise PreregistrationError("difference target transform requires a frozen period")
    if transform == TargetTransform.ROBUST_ZSCORE:
        if "target_min_periods" not in config:
            raise PreregistrationError("robust_zscore target transform requires target_min_periods")
        if not ("target_window_seconds" in config or "target_window_points" in config):
            raise PreregistrationError("robust_zscore target transform requires a frozen window")


def _finite_mapping(value: Mapping[str, float], name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, raw in value.items():
        numeric = float(raw)
        if not math_is_finite(numeric):
            raise PreregistrationError(f"{name}.{key} must be finite")
        normalized[str(key)] = numeric
    return normalized


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _normalize_created_at(value: datetime | str | None) -> str:
    if value is None:
        return to_utc_iso(utc_now())
    if isinstance(value, datetime):
        return to_utc_iso(ensure_utc(value))
    return to_utc_iso(parse_utc(value))


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "DEFAULT_EVALUATION_BUDGET",
    "EVALUATOR_VERSION",
    "PreregistrationError",
    "ScientificPreregistration",
    "append_preregistration_event",
    "create_preregistration",
    "current_evaluator_identity_hash",
    "load_preregistration",
]
