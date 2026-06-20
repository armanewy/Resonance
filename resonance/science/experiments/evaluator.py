from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from resonance.science.contracts import stable_hash
from resonance.science.experiments.contracts import AnalysisMethod, ConditionName
from resonance.science.experiments.runner import (
    EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
    EXPERIMENT_RUNNER_VERSION,
    ExperimentRunnerError,
    experiment_status,
)
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT
from resonance.storage import DEFAULT_DB_PATH, connect, fetch_measurements
from resonance.time_utils import parse_utc, to_utc_iso, utc_now


EXPERIMENT_EVALUATOR_VERSION = "manual-experiment-evaluator-v1"


class ExperimentEvaluationError(RuntimeError):
    """Raised when prospective experiment evaluation cannot proceed."""


def evaluate_experiment(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate the preregistered primary outcome for a completed manual experiment."""

    root = Path(artifact_root)
    status = experiment_status(experiment_id, artifact_root=root)
    if status["status"] != "completed":
        raise ExperimentEvaluationError(f"experiment must be completed before evaluation: {status['status']}")
    state = _load_state_from_status(root, status)
    spec = state["spec"]
    metric = spec["primary_outcome_metric"]
    blocks = _extract_block_outcomes(state, metric, db_path)
    included = [block for block in blocks if block["included"]]
    method = AnalysisMethod(spec["analysis_method"])
    analysis = _analyze_blocks(included, method)
    balance = _condition_balance(blocks, included)
    exclusions = [block["exclusion"] for block in blocks if block["exclusion"] is not None]
    failures = list(state.get("failures", []))
    warnings = _warnings(analysis, balance, exclusions, failures)
    completed_at = to_utc_iso(now_utc or utc_now())
    result_payload = {
        "schema_version": EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_experiment_evaluation",
        "runner_version": EXPERIMENT_RUNNER_VERSION,
        "evaluator_version": EXPERIMENT_EVALUATOR_VERSION,
        "experiment_id": experiment_id,
        "experiment_hash": state["experiment_hash"],
        "status": "completed" if not warnings else "completed_with_warnings",
        "evaluated_at_utc": completed_at,
        "primary_outcome_metric": metric,
        "analysis_method": method.value,
        "minimum_effect": spec["minimum_effect"],
        "effect_size": analysis["effect_size"],
        "uncertainty": analysis["uncertainty"],
        "condition_balance": balance,
        "block_outcomes": blocks,
        "included_block_count": len(included),
        "exclusions": exclusions,
        "failures": failures,
        "warnings": warnings,
        "source_database_path": _display_db_path(db_path),
        "code_commit": current_code_commit(),
        "automatic_intervention_applied": False,
    }
    artifact = _store_json_artifact(root, result_payload)
    append_event(
        "experiment_completed",
        {
            "experiment_id": experiment_id,
            "experiment_hash": state["experiment_hash"],
            "observation_type": "experiment_evaluated",
            "status": result_payload["status"],
            "evaluator_version": EXPERIMENT_EVALUATOR_VERSION,
            "primary_outcome_metric": metric,
            "analysis_method": method.value,
            "effect_size": analysis["effect_size"],
            "uncertainty": analysis["uncertainty"],
            "condition_balance": balance,
            "exclusion_count": len(exclusions),
            "failure_count": len(failures),
            "warnings": warnings,
            "artifact_root": str(root.resolve()),
            "artifacts": {"evaluation": artifact, "state": state["artifacts"]["state"]},
        },
        artifact_hashes={"evaluation": artifact["sha256"], "state": state["artifacts"]["state"]["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
        timestamp_utc=completed_at,
    )
    return {**result_payload, "evaluation_id": artifact["sha256"], "artifact": artifact}


def _extract_block_outcomes(
    state: Mapping[str, Any],
    metric: str,
    db_path: str | Path,
) -> list[dict[str, Any]]:
    conn = _connect_read_only(db_path)
    try:
        outcomes = [_block_outcome(conn, block, metric) for block in state["blocks"]]
    finally:
        conn.close()
    return outcomes


def _block_outcome(conn: sqlite3.Connection, block: Mapping[str, Any], metric: str) -> dict[str, Any]:
    base = {
        "block_index": block["block_index"],
        "condition": block["condition"],
        "block_status": block["status"],
        "planned_start_utc": block["planned_start_utc"],
        "planned_end_utc": block["planned_end_utc"],
        "actual_start_utc": block["actual_start_utc"],
        "actual_end_utc": block["actual_end_utc"],
        "condition_confirmed": block["condition_confirmed"],
        "compliant": block["compliant"],
        "sample_count": 0,
        "mean_value": None,
        "included": False,
        "exclusion": None,
    }
    if block["status"] != "completed" or not block["condition_confirmed"] or block["compliant"] is not True:
        return {**base, "exclusion": _exclusion(block, "block was missed, noncompliant, or unconfirmed")}
    if not block["actual_start_utc"] or not block["actual_end_utc"]:
        return {**base, "exclusion": _exclusion(block, "block lacks actual start or end time")}
    rows = fetch_measurements(
        conn,
        parse_utc(block["actual_start_utc"]),
        parse_utc(block["actual_end_utc"]),
        metrics=[metric],
    )
    values = [float(row["value"]) for row in rows if math.isfinite(float(row["value"]))]
    if not values:
        return {**base, "exclusion": _exclusion(block, f"no measurements for primary metric {metric!r}")}
    return {
        **base,
        "sample_count": len(values),
        "mean_value": mean(values),
        "included": True,
    }


def _analyze_blocks(blocks: Sequence[Mapping[str, Any]], method: AnalysisMethod) -> dict[str, Any]:
    intervention = [float(block["mean_value"]) for block in blocks if block["condition"] == ConditionName.INTERVENTION.value]
    control = [float(block["mean_value"]) for block in blocks if block["condition"] == ConditionName.CONTROL.value]
    if not intervention or not control:
        return _empty_analysis("both conditions need at least one included block")

    if method == AnalysisMethod.PAIRED_BLOCK_DIFFERENCE:
        pair_count = min(len(intervention), len(control))
        if pair_count < 1:
            return _empty_analysis("no matched condition pairs available")
        effects = [intervention[index] - control[index] for index in range(pair_count)]
        effect = mean(effects)
        return {
            "effect_size": effect,
            "uncertainty": _uncertainty(effects),
            "paired_differences": effects,
            "analysis_failure": None,
        }
    if method == AnalysisMethod.NONPARAMETRIC_SIGN_TEST:
        pair_count = min(len(intervention), len(control))
        effects = [intervention[index] - control[index] for index in range(pair_count)]
        positives = sum(1 for value in effects if value > 0)
        negatives = sum(1 for value in effects if value < 0)
        return {
            "effect_size": mean(effects) if effects else None,
            "uncertainty": {
                **_uncertainty(effects),
                "positive_pairs": positives,
                "negative_pairs": negatives,
                "ties": pair_count - positives - negatives,
            },
            "paired_differences": effects,
            "analysis_failure": None if effects else "no matched condition pairs available",
        }

    effects = [value - mean(control) for value in intervention] + [mean(intervention) - value for value in control]
    return {
        "effect_size": mean(intervention) - mean(control),
        "uncertainty": _uncertainty(effects),
        "paired_differences": None,
        "analysis_failure": None,
    }


def _uncertainty(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"standard_error": None, "confidence_interval_95": None, "sample_count": 0}
    if len(values) == 1:
        return {"standard_error": None, "confidence_interval_95": None, "sample_count": 1}
    standard_error = stdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    center = mean(values)
    return {
        "standard_error": standard_error,
        "confidence_interval_95": [center - margin, center + margin],
        "sample_count": len(values),
    }


def _empty_analysis(reason: str) -> dict[str, Any]:
    return {
        "effect_size": None,
        "uncertainty": {"standard_error": None, "confidence_interval_95": None, "sample_count": 0},
        "paired_differences": None,
        "analysis_failure": reason,
    }


def _condition_balance(
    blocks: Sequence[Mapping[str, Any]],
    included: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "planned": _counts(blocks),
        "included": _counts(included),
        "excluded": _counts([block for block in blocks if not block["included"]]),
        "balanced_included": _counts(included).get("intervention", 0) == _counts(included).get("control", 0),
    }


def _counts(blocks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        ConditionName.INTERVENTION.value: sum(1 for block in blocks if block["condition"] == ConditionName.INTERVENTION.value),
        ConditionName.CONTROL.value: sum(1 for block in blocks if block["condition"] == ConditionName.CONTROL.value),
    }


def _warnings(
    analysis: Mapping[str, Any],
    balance: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if analysis.get("analysis_failure"):
        warnings.append(str(analysis["analysis_failure"]))
    if balance["included"][ConditionName.INTERVENTION.value] != balance["included"][ConditionName.CONTROL.value]:
        warnings.append("included blocks are not condition-balanced")
    if exclusions:
        warnings.append(f"{len(exclusions)} block(s) excluded")
    if failures:
        warnings.append(f"{len(failures)} failure(s) recorded during the experiment")
    return warnings


def _exclusion(block: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "block_index": block["block_index"],
        "condition": block["condition"],
        "reason": reason,
    }


def _load_state_from_status(root: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    artifact = status["artifacts"].get("state")
    if not artifact:
        raise ExperimentRunnerError("experiment status lacks state artifact")
    path = root / artifact["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_hash(payload) != artifact["sha256"]:
        raise ExperimentRunnerError("state artifact hash mismatch")
    return payload


def _connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) == ":memory:":
        return connect(":memory:")
    db = Path(db_path)
    if not db.exists():
        raise ExperimentEvaluationError(f"database not found: {db}")
    uri_path = db.resolve().as_posix()
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ExperimentEvaluationError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _display_db_path(db_path: str | Path) -> str:
    if str(db_path) == ":memory:":
        return ":memory:"
    return str(Path(db_path).resolve())


__all__ = [
    "EXPERIMENT_EVALUATOR_VERSION",
    "ExperimentEvaluationError",
    "evaluate_experiment",
]
