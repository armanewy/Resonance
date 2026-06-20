from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from resonance.science.experiments.contracts import generate_randomized_schedule
from resonance.science.experiments.runner import (
    ExperimentRunnerError,
    begin_block,
    confirm_condition,
    end_block,
    experiment_status,
    preregister_experiment,
    start_experiment,
)
from resonance.science.ledger import read_entries, verify_ledger


def test_runner_freezes_spec_and_records_confirmed_blocks(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    spec_path = _write_spec(tmp_path)

    planned = preregister_experiment(
        spec_path,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=_utc(2026, 1, 1, 8, 55),
    )
    started = start_experiment(
        planned["experiment_id"],
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=_utc(2026, 1, 1, 8, 59),
    )
    first = begin_block(
        planned["experiment_id"],
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=_utc(2026, 1, 1, 9, 0),
    )
    confirm_condition(
        planned["experiment_id"],
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=_utc(2026, 1, 1, 9, 1),
    )
    ended = end_block(
        planned["experiment_id"],
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=_utc(2026, 1, 1, 9, 30),
    )

    assert started["status"] == "started"
    assert first["current_condition"]["instructions"]
    assert first["current_condition"]["condition"] == planned["blocks"][0]["condition"]
    assert ended["block"]["actual_start_utc"] == "2026-01-01T09:00:00Z"
    assert ended["block"]["actual_end_utc"] == "2026-01-01T09:30:00Z"
    assert ended["block"]["condition_confirmed"] is True
    assert ended["block"]["status"] == "completed"

    status = experiment_status(planned["experiment_id"], artifact_root=artifact_root)
    assert status["status"] == "started"
    assert status["artifacts"]["spec"]["sha256"]
    assert (artifact_root / "experiments" / f"{planned['experiment_id']}.json").exists()

    entries = read_entries(ledger_path)
    assert [entry["event_type"] for entry in entries][:2] == [
        "experiment_planned",
        "experiment_started",
    ]
    assert entries[0]["payload"]["schedule_frozen"] is True
    assert entries[0]["payload"]["analysis_frozen"] is True
    assert entries[-1]["payload"]["automatic_intervention_applied"] is False
    assert verify_ledger(ledger_path).valid is True


def test_runner_requires_confirmation_or_records_noncompliance(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    spec_path = _write_spec(tmp_path)
    planned = preregister_experiment(spec_path, artifact_root=artifact_root, ledger_path=ledger_path)
    start_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)

    begin_block(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)
    ended = end_block(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)

    assert ended["block"]["status"] == "noncompliant"
    assert ended["block"]["compliant"] is False
    assert ended["failures"][0]["failure_type"] == "noncompliant"


def test_runner_prevents_double_start_and_nested_blocks(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    spec_path = _write_spec(tmp_path)
    planned = preregister_experiment(spec_path, artifact_root=artifact_root, ledger_path=ledger_path)
    start_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)

    with pytest.raises(ExperimentRunnerError, match="cannot be started"):
        start_experiment(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)

    begin_block(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)
    with pytest.raises(ExperimentRunnerError, match="already in progress"):
        begin_block(planned["experiment_id"], artifact_root=artifact_root, ledger_path=ledger_path)


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
