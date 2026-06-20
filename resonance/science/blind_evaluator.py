from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from pydantic import TypeAdapter

from resonance.science.contracts import Expression, MetricName, TargetTransform
from resonance.science.evaluation import (
    BASELINE_STRATEGIES,
    baseline_metric_bundles,
    chronological_window_metrics,
    evaluate_frozen_program,
    metric_bundle,
    metric_improvement_fraction,
    transform_target,
)
from resonance.science.interpreter import frame_from_snapshot_rows
from resonance.science.ledger import (
    DEFAULT_LEDGER_PATH,
    LedgerError,
    append_event,
    claim_blind_evaluation,
    current_code_commit,
)
from resonance.science.preregistration import (
    DEFAULT_EVALUATION_BUDGET,
    EVALUATOR_VERSION,
    ScientificPreregistration,
    current_evaluator_identity_hash,
    load_preregistration,
)
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT, create_blind_evaluator_capability, load_blind_view


_EXPRESSION_ADAPTER = TypeAdapter(Expression)


class BlindEvaluationError(RuntimeError):
    """Base class for blind evaluator failures."""


class PreregistrationHashError(BlindEvaluationError):
    """Raised when the supplied preregistration hash is invalid."""


class BlindEvaluationAlreadyCompletedError(BlindEvaluationError):
    """Raised after the one-shot blind budget has already been claimed."""


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
    """Consume one blind budget, evaluate once, and expose aggregate evidence only."""

    preregistration = load_preregistration(preregistration)
    _validate_preregistration_hash(preregistration, preregistration_hash)
    _validate_evaluator_identity(preregistration)
    try:
        claim = claim_blind_evaluation(
            preregistration_hash=preregistration_hash,
            snapshot_id=preregistration.snapshot_id,
            hypothesis_hash=preregistration.hypothesis_hash,
            payload={
                "evaluator_version": EVALUATOR_VERSION,
                "evaluator_identity_hash": preregistration.evaluator_identity_hash,
                "random_seed": preregistration.random_seed,
                "evaluation_budget_consumed": DEFAULT_EVALUATION_BUDGET,
            },
            code_commit=current_code_commit(),
            ledger_path=ledger_path,
        )
    except LedgerError as exc:
        if "already been consumed" in str(exc):
            raise BlindEvaluationAlreadyCompletedError("blind evaluation already completed; budget has been consumed") from exc
        raise BlindEvaluationError(str(exc)) from exc

    try:
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
                "schema_version": 2,
                "preregistration_hash": preregistration_hash,
                "hypothesis_hash": preregistration.hypothesis_hash,
                "snapshot_id": preregistration.snapshot_id,
                "evaluator_version": EVALUATOR_VERSION,
                "claim_entry_hash": claim["entry_hash"],
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
                "claim_entry_hash": claim["entry_hash"],
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
    except Exception as exc:
        _record_failed_evaluation(
            preregistration,
            preregistration_hash,
            claim["entry_hash"],
            exc,
            ledger_path,
        )
        raise

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


def _validate_preregistration_hash(preregistration: ScientificPreregistration, supplied_hash: str) -> None:
    if not _is_hash(supplied_hash):
        raise PreregistrationHashError("preregistration hash must be a 64-character hex digest")
    if supplied_hash != preregistration.preregistration_hash():
        raise PreregistrationHashError("preregistration hash does not match frozen content")


def _validate_evaluator_identity(preregistration: ScientificPreregistration) -> None:
    if preregistration.evaluator_version != EVALUATOR_VERSION:
        raise EvaluatorIdentityError("evaluator version differs from preregistration")
    if preregistration.evaluator_identity_hash != current_evaluator_identity_hash():
        raise EvaluatorIdentityError("evaluator source/dependency fingerprint differs from preregistration")
    if preregistration.evaluation_budget != DEFAULT_EVALUATION_BUDGET:
        raise EvaluatorIdentityError("evaluation budget differs from sealed evaluator contract")
    if preregistration.baseline_strategy not in BASELINE_STRATEGIES:
        raise EvaluatorIdentityError("preregistered baseline strategy is unsupported")


def _evaluate_metrics(
    preregistration: ScientificPreregistration,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []
    if not rows:
        return _empty_metrics(preregistration), ("blind partition is empty",)

    frame = frame_from_snapshot_rows(list(rows))
    expression = _EXPRESSION_ADAPTER.validate_python(preregistration.exact_expression)
    evaluated = evaluate_frozen_program(
        expression=expression,
        target_metric=preregistration.target_metric,
        input_metrics=preregistration.input_metrics,
        target_transform=preregistration.target_transform,
        data=frame,
        parameters=preregistration.fitted_parameters,
        transform_config=preregistration.transform_config,
        max_ast_nodes=100,
        max_source_metrics=3,
    )
    model = metric_bundle(evaluated.aligned["target"], evaluated.aligned["prediction"])
    baselines = baseline_metric_bundles(
        evaluated.target,
        evaluation_index=evaluated.aligned.index,
    )
    baseline = baselines[preregistration.baseline_strategy]
    windows = chronological_window_metrics(
        evaluated.target,
        evaluated.prediction,
        baseline_strategy=preregistration.baseline_strategy,
        window_count=int(preregistration.transform_config.get("window_count", 3)),
    )
    stable_windows = [
        window
        for window in windows
        if window["spearman_rho"] is not None
        and float(window["spearman_rho"]) >= min(0.20, preregistration.minimum_blind_effect)
        and (
            window["mae_improvement_fraction"] is None
            or float(window["mae_improvement_fraction"]) >= preregistration.minimum_baseline_improvement
        )
    ]
    window_stability = {
        "window_count": len(windows),
        "stable_count": len(stable_windows),
        "stable_fraction": len(stable_windows) / len(windows) if windows else 0.0,
        "windows": list(windows),
    }
    negative_controls = _negative_control_metrics(preregistration, evaluated.frame, evaluated.prediction)
    max_control = max(
        (abs(float(control["spearman_rho"])) for control in negative_controls if control["spearman_rho"] is not None),
        default=0.0,
    )
    coverage = model.observations / len(rows) if rows else 0.0
    metrics = {
        "observation_count": model.observations,
        "blind_row_count": len(rows),
        "coverage": round(coverage, 6),
        "baseline_strategy": preregistration.baseline_strategy,
        "tuning_baseline_metrics_provenance": dict(preregistration.baseline_metrics),
        "mae": model.mae,
        "baseline_mae": baseline.mae,
        "mae_improvement": _difference(baseline.mae, model.mae),
        "mae_improvement_fraction": metric_improvement_fraction(baseline, model, MetricName.MAE),
        "rmse": model.rmse,
        "baseline_rmse": baseline.rmse,
        "rmse_improvement": _difference(baseline.rmse, model.rmse),
        "rmse_improvement_fraction": metric_improvement_fraction(baseline, model, MetricName.RMSE),
        "pearson_r": model.pearson_r,
        "spearman_rho": model.spearman_r,
        "direction_agreement": model.direction_agreement,
        "window_stability": window_stability,
        "negative_controls": negative_controls,
        "negative_control_performance": {
            "max_abs_spearman_rho": max_control,
            "passed": all(control["passed"] for control in negative_controls),
        },
    }

    if model.observations < int(preregistration.transform_config.get("minimum_observations", 4)):
        warnings.append("too few aligned observations for a decisive blind evaluation")
    if coverage < float(preregistration.transform_config.get("minimum_coverage", 0.5)):
        warnings.append("blind observation coverage is below the preregistered minimum")
    if model.spearman_r is None:
        warnings.append("blind Spearman association is undefined")
    if model.direction_agreement is None:
        warnings.append("blind movement-direction agreement is undefined")
    for required_metric in preregistration.blind_metrics:
        if required_metric == MetricName.MAE.value and (model.mae is None or baseline.mae is None):
            warnings.append("preregistered MAE metric is unavailable")
        elif required_metric == MetricName.RMSE.value and (model.rmse is None or baseline.rmse is None):
            warnings.append("preregistered RMSE metric is unavailable")
        elif required_metric == MetricName.PEARSON_R.value and model.pearson_r is None:
            warnings.append("preregistered Pearson metric is unavailable")
        elif required_metric == MetricName.SPEARMAN_R.value and model.spearman_r is None:
            warnings.append("preregistered Spearman metric is unavailable")
    return metrics, tuple(dict.fromkeys(warnings))


def _verdict(
    preregistration: ScientificPreregistration,
    metrics: Mapping[str, Any],
    warnings: Sequence[str],
) -> str:
    if warnings:
        return "inconclusive"
    if not metrics["negative_control_performance"]["passed"]:
        return "fail"

    checks: list[bool] = []
    selected = set(preregistration.blind_metrics)
    if MetricName.MAE.value in selected:
        checks.append(
            metrics["mae_improvement_fraction"] is not None
            and metrics["mae_improvement_fraction"] >= preregistration.minimum_baseline_improvement
        )
    if MetricName.RMSE.value in selected:
        checks.append(
            metrics["rmse_improvement_fraction"] is not None
            and metrics["rmse_improvement_fraction"] >= preregistration.minimum_baseline_improvement
        )
    if MetricName.SPEARMAN_R.value in selected:
        checks.append(
            metrics["spearman_rho"] is not None
            and metrics["spearman_rho"] >= preregistration.minimum_blind_effect
        )
    if MetricName.PEARSON_R.value in selected:
        checks.append(
            metrics["pearson_r"] is not None
            and metrics["pearson_r"] >= preregistration.minimum_blind_effect
        )
    checks.extend(
        [
            metrics["direction_agreement"] is not None
            and metrics["direction_agreement"] >= float(preregistration.transform_config.get("minimum_direction_agreement", 0.5)),
            metrics["window_stability"]["stable_fraction"]
            >= float(preregistration.transform_config.get("minimum_window_stability", 0.5)),
        ]
    )
    return "pass" if all(checks) else "fail"


def _negative_control_metrics(
    preregistration: ScientificPreregistration,
    frame: pd.DataFrame,
    predictions: pd.Series,
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    transform = str(preregistration.transform_config.get("negative_control_transform", "identity"))
    for control in preregistration.negative_controls:
        metric = str(control["metric"])
        if metric not in frame:
            controls.append(
                {
                    "metric": metric,
                    "observation_count": 0,
                    "spearman_rho": None,
                    "passed": False,
                    "warning": "negative-control metric is absent",
                }
            )
            continue
        target = transform_target(frame[metric], TargetTransform(transform), preregistration.transform_config)
        bundle = metric_bundle(target, predictions)
        rho = bundle.spearman_r
        controls.append(
            {
                "metric": metric,
                "observation_count": bundle.observations,
                "spearman_rho": rho,
                "passed": rho is not None and abs(rho) < preregistration.minimum_blind_effect,
            }
        )
    return controls


def _empty_metrics(preregistration: ScientificPreregistration) -> dict[str, Any]:
    return {
        "observation_count": 0,
        "blind_row_count": 0,
        "coverage": 0.0,
        "baseline_strategy": preregistration.baseline_strategy,
        "mae": None,
        "baseline_mae": None,
        "mae_improvement": None,
        "mae_improvement_fraction": None,
        "rmse": None,
        "baseline_rmse": None,
        "rmse_improvement": None,
        "rmse_improvement_fraction": None,
        "pearson_r": None,
        "spearman_rho": None,
        "direction_agreement": None,
        "window_stability": {"window_count": 0, "stable_count": 0, "stable_fraction": 0.0, "windows": []},
        "negative_controls": [],
        "negative_control_performance": {"max_abs_spearman_rho": 0.0, "passed": False},
    }


def _record_failed_evaluation(
    preregistration: ScientificPreregistration,
    preregistration_hash: str,
    claim_entry_hash: str,
    error: Exception,
    ledger_path: str | Path,
) -> None:
    try:
        append_event(
            "blind_evaluation_completed",
            {
                "preregistration_hash": preregistration_hash,
                "dataset_snapshot_id": preregistration.snapshot_id,
                "hypothesis_hash": preregistration.hypothesis_hash,
                "evaluator_version": EVALUATOR_VERSION,
                "evaluator_identity_hash": preregistration.evaluator_identity_hash,
                "claim_entry_hash": claim_entry_hash,
                "random_seed": preregistration.random_seed,
                "evaluation_budget_consumed": DEFAULT_EVALUATION_BUDGET,
                "status": "error",
                "metrics": {},
                "warnings": [f"blind evaluator failed after budget claim: {type(error).__name__}"],
                "raw_blind_values_exposed": False,
            },
            code_commit=current_code_commit(),
            ledger_path=ledger_path,
        )
    except Exception:
        # The started claim remains in the ledger, which is enough to fail closed.
        pass


def _difference(baseline: float | None, observed: float | None) -> float | None:
    if baseline is None or observed is None:
        return None
    return baseline - observed


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
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "BlindEvaluationAlreadyCompletedError",
    "BlindEvaluationError",
    "BlindEvaluationResult",
    "EvaluatorIdentityError",
    "PreregistrationHashError",
    "evaluate_preregistration",
]
