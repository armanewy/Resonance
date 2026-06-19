from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from resonance.science.blind_evaluator import (
    BlindEvaluationAlreadyCompletedError,
    EvaluatorIdentityError,
    PreregistrationHashError,
    evaluate_preregistration,
)
from resonance.science.contracts import HypothesisSpec
from resonance.science.ledger import read_entries, verify_ledger
from resonance.science.preregistration import create_preregistration
from resonance.science.snapshots import create_snapshot
from resonance.storage import Measurement, init_db, insert_measurements


START = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_blind_data_is_not_accessible_through_result_objects(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(
        tmp_path,
        "strong",
        sentinel=12345.6789,
    )

    result = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    artifact_payload = json.loads((artifact_root / result.artifact["path"]).read_text(encoding="utf-8"))
    assert result.status == "pass"
    assert "rows" not in serialized
    assert "observations" not in serialized
    assert "12345.6789" not in serialized
    assert artifact_payload["raw_blind_values_exposed"] is False
    assert "rows" not in json.dumps(artifact_payload, sort_keys=True)


def test_preregistration_cannot_be_evaluated_twice(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "strong")
    first = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    with pytest.raises(BlindEvaluationAlreadyCompletedError):
        evaluate_preregistration(
            preregistration,
            preregistration_hash,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )

    entries = read_entries(ledger_path)
    assert first.status == "pass"
    assert len([entry for entry in entries if entry["event_type"] == "blind_evaluation_completed"]) == 1


def test_modified_hypothesis_is_rejected_by_preregistration_hash(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "strong")
    modified = replace(
        preregistration,
        exact_expression={
            "node": "add",
            "left": preregistration.exact_expression,
            "right": {"node": "numeric_constant", "value": 0.5},
        },
    )

    with pytest.raises(PreregistrationHashError):
        evaluate_preregistration(
            modified,
            preregistration_hash,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )


def test_changed_evaluator_identity_requires_new_preregistration(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "strong")
    changed = replace(preregistration, evaluator_identity_hash="0" * 64)

    with pytest.raises(EvaluatorIdentityError):
        evaluate_preregistration(
            changed,
            changed.preregistration_hash(),
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )


def test_strong_synthetic_relationship_can_pass(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "strong")

    result = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    assert result.status == "pass"
    assert result.metrics["mae_improvement_fraction"] >= 0.2
    assert result.metrics["rmse_improvement_fraction"] >= 0.2
    assert result.metrics["spearman_rho"] >= 0.5
    assert result.metrics["negative_control_performance"]["passed"] is True
    assert _artifact_hash_matches(artifact_root / result.artifact["path"], result.artifact["sha256"])


def test_null_scenario_fails_or_is_inconclusive(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "null")

    result = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    assert result.status in {"fail", "inconclusive"}
    assert result.status != "pass"


def test_seasonality_only_scenario_fails_or_is_inconclusive(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(
        tmp_path,
        "seasonality",
    )

    result = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    assert result.status in {"fail", "inconclusive"}
    assert result.status != "pass"
    assert result.metrics["negative_control_performance"]["passed"] is False


def test_ledger_records_both_passes_and_failures(tmp_path: Path) -> None:
    pass_case = _case(tmp_path / "pass", "strong")
    fail_case = _case(tmp_path / "fail", "seasonality", ledger_path=pass_case[3])

    pass_result = evaluate_preregistration(
        pass_case[0],
        pass_case[1],
        artifact_root=pass_case[2],
        ledger_path=pass_case[3],
    )
    fail_result = evaluate_preregistration(
        fail_case[0],
        fail_case[1],
        artifact_root=fail_case[2],
        ledger_path=fail_case[3],
    )

    entries = read_entries(pass_case[3])
    statuses = [
        entry["payload"]["status"]
        for entry in entries
        if entry["event_type"] == "blind_evaluation_completed"
    ]
    assert pass_result.status == "pass"
    assert fail_result.status == "fail"
    assert statuses == ["pass", "fail"]
    assert verify_ledger(pass_case[3]).valid is True


def test_failure_cannot_be_overwritten(tmp_path: Path) -> None:
    preregistration, preregistration_hash, artifact_root, ledger_path = _case(tmp_path, "seasonality")
    result = evaluate_preregistration(
        preregistration,
        preregistration_hash,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    improved_baseline = replace(preregistration, baseline_metrics={"mae": 1000.0, "rmse": 1000.0})
    with pytest.raises(BlindEvaluationAlreadyCompletedError):
        evaluate_preregistration(
            improved_baseline,
            improved_baseline.preregistration_hash(),
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )

    entries = read_entries(ledger_path)
    assert result.status == "fail"
    assert len([entry for entry in entries if entry["event_type"] == "blind_evaluation_completed"]) == 1
    assert entries[-1]["payload"]["status"] == "fail"


def _case(
    tmp_path: Path,
    scenario: str,
    *,
    sentinel: float | None = None,
    ledger_path: Path | None = None,
) -> tuple:
    artifact_root = tmp_path / "artifacts"
    ledger = ledger_path or (tmp_path / "ledger.jsonl")
    measurements = _measurements(scenario, sentinel=sentinel)
    db_path = _create_db(tmp_path, measurements)
    manifest = create_snapshot(
        db_path=db_path,
        hours=120,
        metrics=["control", "target", "x"],
        max_lag_seconds=0,
        artifact_root=artifact_root,
    )
    baseline = _blind_baseline(manifest, artifact_root, "target")
    preregistration = create_preregistration(
        hypothesis=_hypothesis(),
        snapshot_manifest=manifest,
        fitted_parameters={"scale": 2.0, "offset": 1.0},
        baseline_metrics=baseline,
        transform_config={
            "expected_direction": "positive",
            "minimum_observations": 4,
            "minimum_coverage": 0.8,
            "window_count": 3,
        },
        created_at_utc=NOW,
    )
    return preregistration, preregistration.preregistration_hash(), artifact_root, ledger


def _hypothesis() -> HypothesisSpec:
    return HypothesisSpec.model_validate(
        {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": "x predicts target",
            "concise_claim": "x predicts target in this dataset.",
            "rationale": "Synthetic deterministic fixture for sealed evaluator tests.",
            "target_metric": "target",
            "input_metrics": ["x"],
            "target_transform": "identity",
            "expression": {
                "node": "add",
                "left": {
                    "node": "multiply",
                    "left": {"node": "fitted_parameter", "parameter": "scale"},
                    "right": {"node": "metric", "metric": "x"},
                },
                "right": {"node": "fitted_parameter", "parameter": "offset"},
            },
            "parameter_bounds": {
                "scale": {"lower": -5.0, "upper": 5.0},
                "offset": {"lower": -10.0, "upper": 10.0},
            },
            "expected_direction": "positive",
            "maximum_lag_seconds": 0,
            "fitting_metric": "rmse",
            "tuning_metric": "mae",
            "blind_metrics": ["mae", "rmse", "spearman_r"],
            "minimum_blind_effect": 0.3,
            "minimum_baseline_improvement": 0.1,
            "negative_controls": [
                {"metric": "control", "rationale": "Control should not move with the prediction."}
            ],
            "falsification_conditions": [
                {"description": "Blind error does not improve over baseline."},
                {"description": "Negative control performs like the target."},
            ],
            "origin": "manual",
            "random_seed": 1234,
        }
    )


def _measurements(scenario: str, *, sentinel: float | None = None) -> list[Measurement]:
    rows: list[Measurement] = []
    for index in range(80):
        timestamp = START + timedelta(hours=index)
        seasonal = math.sin(index / 3.0)
        x = (index - 40) / 10.0
        if sentinel is not None and index == 70:
            x = sentinel
        if scenario == "strong":
            target = 2.0 * x + 1.0 + (0.03 * math.sin(index))
            control = math.cos(index * 1.7)
        elif scenario == "seasonality":
            x = seasonal
            target = 2.0 * seasonal + 1.0
            control = seasonal
        elif scenario == "null":
            target = math.sin(index * 2.1) * 3.0
            control = math.cos(index * 1.7)
        else:
            raise AssertionError(f"unknown scenario: {scenario}")
        rows.extend(
            [
                Measurement(timestamp, "x", x, "unit", "test"),
                Measurement(timestamp, "target", target, "unit", "test"),
                Measurement(timestamp, "control", control, "unit", "test"),
            ]
        )
    return rows


def _create_db(tmp_path: Path, measurements: list[Measurement]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "resonance.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    insert_measurements(conn, measurements)
    conn.close()
    return db_path


def _blind_baseline(manifest: dict, artifact_root: Path, metric: str) -> dict[str, float]:
    import gzip

    artifact = manifest["artifacts"]["blind"]
    blind = json.loads(gzip.decompress((artifact_root / artifact["path"]).read_bytes()).decode("utf-8"))
    values = [
        row["metrics"][metric][0]["value"]
        for row in blind["rows"]
        if metric in row["metrics"]
    ]
    mean = sum(values) / len(values)
    mae = sum(abs(value - mean) for value in values) / len(values)
    rmse = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {"mae": mae, "rmse": rmse}


def _artifact_hash_matches(path: Path, expected: str) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected
