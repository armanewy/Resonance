from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from resonance.science.contracts import (
    HypothesisSpec,
    canonical_json,
    expression_parameters,
    stable_hash,
)
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


PREREGISTRATION_SCHEMA_VERSION = 1
EVALUATOR_VERSION = "sealed-blind-evaluator-1"
EVALUATOR_RULESET = {
    "name": "one-shot-blind-evaluator",
    "verdict_source": "deterministic_numeric_metrics",
    "raw_blind_observations_exposed": False,
    "one_evaluation_per_preregistration": True,
    "metrics": (
        "mae_improvement_vs_baseline",
        "rmse_improvement_vs_baseline",
        "spearman_rho",
        "direction_agreement",
        "window_stability",
        "negative_control_performance",
        "observation_count",
        "coverage",
    ),
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
    transform_config: Mapping[str, Any] | None = None,
    evaluation_budget: int = DEFAULT_EVALUATION_BUDGET,
    created_at_utc: datetime | str | None = None,
) -> ScientificPreregistration:
    """Freeze the exact scientific and evaluator identity before blind access."""

    if not isinstance(hypothesis, HypothesisSpec):
        hypothesis = HypothesisSpec.model_validate(
            hypothesis,
            context={"metric_catalog": snapshot_manifest.get("metric_catalog")},
        )
    else:
        hypothesis.validate_metric_catalog(snapshot_manifest.get("metric_catalog", ()))

    if evaluation_budget != DEFAULT_EVALUATION_BUDGET:
        raise PreregistrationError("sealed blind evaluation budget must be exactly one")

    snapshot_id = str(snapshot_manifest.get("snapshot_id") or "")
    if not snapshot_id:
        raise PreregistrationError("snapshot_manifest must include snapshot_id")
    snapshot_artifacts = dict(snapshot_manifest.get("artifacts") or {})
    if "blind" not in snapshot_artifacts:
        raise PreregistrationError("snapshot_manifest must include a blind artifact")

    expected_parameters = expression_parameters(hypothesis.expression)
    normalized_parameters = {
        str(name): float(value) for name, value in fitted_parameters.items()
    }
    if set(normalized_parameters) != expected_parameters:
        missing = expected_parameters - set(normalized_parameters)
        extra = set(normalized_parameters) - expected_parameters
        details = []
        if missing:
            details.append(f"missing fitted parameters: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected fitted parameters: {', '.join(sorted(extra))}")
        raise PreregistrationError("; ".join(details))

    normalized_baselines = {
        str(name): float(value) for name, value in baseline_metrics.items()
    }
    if _baseline_value(normalized_baselines, "mae") is None:
        raise PreregistrationError("baseline_metrics must include mae or baseline_mae")
    if _baseline_value(normalized_baselines, "rmse") is None:
        raise PreregistrationError("baseline_metrics must include rmse or baseline_rmse")

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
        transform_config=dict(transform_config or {}),
        blind_metrics=tuple(str(metric.value) for metric in hypothesis.blind_metrics),
        baseline_metrics=normalized_baselines,
        minimum_blind_effect=float(hypothesis.minimum_blind_effect),
        minimum_baseline_improvement=float(hypothesis.minimum_baseline_improvement),
        negative_controls=tuple(
            control.model_dump(mode="json") for control in hypothesis.negative_controls
        ),
        falsification_conditions=tuple(
            condition.model_dump(mode="json")
            for condition in hypothesis.falsification_conditions
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
            fitted_parameters={
                str(name): float(parameter)
                for name, parameter in dict(value["fitted_parameters"]).items()
            },
            target_metric=str(value["target_metric"]),
            input_metrics=tuple(str(metric) for metric in value["input_metrics"]),
            target_transform=str(value["target_transform"]),
            expected_direction=str(value["expected_direction"]),
            transform_config=dict(value["transform_config"]),
            blind_metrics=tuple(str(metric) for metric in value["blind_metrics"]),
            baseline_metrics={
                str(name): float(metric)
                for name, metric in dict(value["baseline_metrics"]).items()
            },
            minimum_blind_effect=float(value["minimum_blind_effect"]),
            minimum_baseline_improvement=float(value["minimum_baseline_improvement"]),
            negative_controls=tuple(dict(control) for control in value["negative_controls"]),
            falsification_conditions=tuple(
                dict(condition) for condition in value["falsification_conditions"]
            ),
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
    return stable_hash(
        {
            "evaluator_version": EVALUATOR_VERSION,
            "ruleset": EVALUATOR_RULESET,
        }
    )


def _normalize_created_at(value: datetime | str | None) -> str:
    if value is None:
        return to_utc_iso(utc_now())
    if isinstance(value, datetime):
        return to_utc_iso(ensure_utc(value))
    return to_utc_iso(parse_utc(value))


def _baseline_value(baseline_metrics: Mapping[str, float], metric: str) -> float | None:
    for key in (metric, f"baseline_{metric}"):
        if key in baseline_metrics:
            return float(baseline_metrics[key])
    return None


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
