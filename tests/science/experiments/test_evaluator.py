from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from resonance.science.experiments.cli import main as experiment_main
from resonance.science.experiments.contracts import generate_randomized_schedule
from resonance.science.experiments.evaluator import evaluate_experiment
from resonance.science.experiments.runner import (
    begin_block,
    confirm_condition,
    end_block,
    experiment_status,
    preregister_experiment,
    start_experiment,
)
from resonance.science.ledger import read_entries, verify_ledger
from resonance.storage import Measurement, connect, init_db, insert_measurements


def test_evaluator_scores_preregistered_primary_outcome_from_measurements(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    db_path = tmp_path / "resonance.db"
    spec_path = _write_spec(tmp_path)
    experiment_id = _complete_experiment(spec_path, artifact_root, ledger_path)
    status = experiment_status(experiment_id, artifact_root=artifact_root)
    _write_measurements(db_path, status["blocks"])

    result = evaluate_experiment(
        experiment_id,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        db_path=db_path,
        now_utc=_utc(2026, 1, 1, 12, 0),
    )

    assert result["primary_outcome_metric"] == "focus_score"
    assert result["analysis_method"] == "paired_block_difference"
    assert result["effect_size"] == 5.0
    assert result["uncertainty"]["sample_count"] == 2
    assert result["condition_balance"]["included"] == {"intervention": 2, "control": 2}
    assert result["condition_balance"]["balanced_included"] is True
    assert result["exclusions"] == []
    assert result["failures"] == []
    assert result["automatic_intervention_applied"] is False
    assert (artifact_root / result["artifact"]["path"]).exists()

    entries = read_entries(ledger_path)
    assert entries[-1]["event_type"] == "experiment_completed"
    assert entries[-1]["payload"]["observation_type"] == "experiment_evaluated"
    assert entries[-1]["payload"]["effect_size"] == 5.0
    assert verify_ledger(ledger_path).valid is True


def test_evaluator_preserves_noncompliant_blocks_as_exclusions(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    db_path = tmp_path / "resonance.db"
    spec_path = _write_spec(tmp_path)
    planned = preregister_experiment(spec_path, artifact_root=artifact_root, ledger_path=ledger_path)
    start_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)

    for index, block in enumerate(planned["blocks"]):
        begin_block(
            planned["experiment_id"],
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            now_utc=block["planned_start_utc"],
        )
        if index != 0:
            confirm_condition(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)
        end_block(
            planned["experiment_id"],
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            now_utc=block["planned_end_utc"],
        )
    status = experiment_status(planned["experiment_id"], artifact_root=artifact_root)
    _write_measurements(db_path, status["blocks"])

    result = evaluate_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path, db_path=db_path)

    assert len(result["exclusions"]) == 1
    assert result["exclusions"][0]["reason"] == "block was missed, noncompliant, or unconfirmed"
    assert result["failures"][0]["failure_type"] == "noncompliant"
    assert "included blocks are not condition-balanced" in result["warnings"]
    assert result["status"] == "completed_with_warnings"


def test_experiment_cli_runs_manual_flow(tmp_path: Path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    db_path = tmp_path / "resonance.db"
    spec_path = _write_spec(tmp_path)
    base = ["--artifact-root", str(artifact_root), "--ledger", str(ledger_path), "--db", str(db_path)]

    planned = _run_ok(capsys, [*base, "preregister", str(spec_path)])
    _run_ok(capsys, [*base, "start", planned["experiment_id"]])
    for _block in planned["blocks"]:
        _run_ok(capsys, [*base, "begin-block", planned["experiment_id"]])
        _run_ok(capsys, [*base, "confirm-condition", planned["experiment_id"]])
        _run_ok(capsys, [*base, "end-block", planned["experiment_id"]])

    status = _run_ok(capsys, [*base, "status", planned["experiment_id"]])
    _write_measurements(db_path, status["blocks"])
    evaluation = _run_ok(capsys, [*base, "evaluate", planned["experiment_id"]])

    assert status["status"] == "completed"
    assert evaluation["primary_outcome_metric"] == "focus_score"
    assert evaluation["effect_size"] is not None
    assert evaluation["source_database_path"] == str(db_path.resolve())


def _complete_experiment(spec_path: Path, artifact_root: Path, ledger_path: Path) -> str:
    planned = preregister_experiment(spec_path, artifact_root=artifact_root, ledger_path=ledger_path)
    start_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)
    for block in planned["blocks"]:
        begin_block(
            planned["experiment_id"],
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            now_utc=block["planned_start_utc"],
        )
        confirm_condition(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)
        end_block(
            planned["experiment_id"],
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            now_utc=block["planned_end_utc"],
        )
    return planned["experiment_id"]


def _write_measurements(db_path: Path, blocks: list[dict]) -> None:
    conn = connect(db_path)
    init_db(conn)
    measurements: list[Measurement] = []
    condition_counts = {"intervention": 0, "control": 0}
    for block in blocks:
        if not block["actual_start_utc"] or not block["actual_end_utc"]:
            continue
        condition_counts[block["condition"]] += 1
        occurrence = condition_counts[block["condition"]]
        base_value = 15.0 if block["condition"] == "intervention" else 10.0
        value = base_value + occurrence
        start = datetime.fromisoformat(block["actual_start_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(block["actual_end_utc"].replace("Z", "+00:00"))
        timestamp = start + min((end - start) / 2, timedelta(minutes=5))
        measurements.append(Measurement(timestamp, "focus_score", value, "score", "synthetic-experiment"))
        if timestamp < end:
            measurements.append(Measurement(timestamp, "focus_score", value, "score", "synthetic-experiment"))
    insert_measurements(conn, measurements)
    conn.close()


def _run_ok(capsys, args: list[str]) -> dict:
    assert experiment_main(args) == 0
    return json.loads(capsys.readouterr().out)


def _write_spec(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(_spec(), indent=2), encoding="utf-8")
    return path


def _spec() -> dict:
    planned_start = _utc(2026, 1, 1, 9, 0)
    schedule = generate_randomized_schedule(
        planned_start=planned_start,
        block_duration_seconds=1800,
        number_of_blocks=4,
        washout_duration_seconds=300,
        randomization_seed=23,
    )
    return {
        "schema_version": "1.0",
        "title": "Synthetic manual experiment",
        "hypothesis_id": "hypothesis-controlled",
        "intervention_condition": {
            "name": "quiet desk",
            "instructions": "Use the quiet desk setup.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "control_condition": {
            "name": "usual desk",
            "instructions": "Use the usual desk setup.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "primary_outcome_metric": "focus_score",
        "secondary_outcome_metrics": [],
        "block_duration_seconds": 1800,
        "number_of_blocks": 4,
        "washout_duration_seconds": 300,
        "randomization_seed": 23,
        "randomized_schedule": [block.model_dump(mode="json") for block in schedule],
        "planned_start": planned_start.isoformat(),
        "inclusion_rules": [{"description": "Use completed confirmed blocks."}],
        "exclusion_rules": [{"description": "Exclude missed and noncompliant blocks."}],
        "stopping_rules": [{"description": "Stop after all blocks are ended."}],
        "abort_conditions": [{"description": "Abort if the manual condition cannot be applied safely."}],
        "minimum_effect": 0.5,
        "analysis_method": "paired_block_difference",
        "safety_notes": "Low-risk reversible setup only.",
        "requires_manual_confirmation": True,
        "prohibited_automatic_actions": [
            "medical_interventions",
            "hazardous_physical_actions",
            "automatic_router_or_os_setting_changes",
            "blocking_emergency_communication",
        ],
        "maximum_experiment_duration_seconds": 86400,
    }


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
