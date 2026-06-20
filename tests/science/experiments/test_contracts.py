from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from resonance.science.experiments import ExperimentSpec, generate_randomized_schedule


def test_valid_experiment_round_trips_through_json() -> None:
    spec = ExperimentSpec.model_validate(_valid_experiment())

    parsed = ExperimentSpec.model_validate_json(spec.canonical_json())

    assert parsed == spec
    assert json.loads(parsed.canonical_json())["requires_manual_confirmation"] is True


def test_randomized_schedule_is_deterministic_for_seed() -> None:
    payload = _valid_experiment()

    first = generate_randomized_schedule(
        planned_start=datetime.fromisoformat(payload["planned_start"]),
        block_duration_seconds=payload["block_duration_seconds"],
        number_of_blocks=payload["number_of_blocks"],
        washout_duration_seconds=payload["washout_duration_seconds"],
        randomization_seed=payload["randomization_seed"],
    )
    second = generate_randomized_schedule(
        planned_start=datetime.fromisoformat(payload["planned_start"]),
        block_duration_seconds=payload["block_duration_seconds"],
        number_of_blocks=payload["number_of_blocks"],
        washout_duration_seconds=payload["washout_duration_seconds"],
        randomization_seed=payload["randomization_seed"],
    )

    assert first == second
    assert [block.condition.value for block in first] == [
        block["condition"] for block in payload["randomized_schedule"]
    ]


def test_schedule_must_be_frozen_from_seed_before_analysis() -> None:
    payload = _valid_experiment()
    payload["randomized_schedule"] = list(reversed(payload["randomized_schedule"]))

    with pytest.raises(ValidationError, match="deterministic seed schedule"):
        ExperimentSpec.model_validate(payload)


def test_every_block_requires_user_confirmation() -> None:
    payload = _valid_experiment()
    payload["randomized_schedule"][0]["requires_user_confirmation"] = False

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_unsafe_condition_attestations_fail_closed() -> None:
    payload = _valid_experiment()
    payload["intervention_condition"]["is_medical_intervention"] = True

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_automated_router_or_os_changes_are_rejected() -> None:
    payload = _valid_experiment()
    payload["control_condition"]["changes_router_or_os_settings_automatically"] = True

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_emergency_communication_blocking_is_rejected() -> None:
    payload = _valid_experiment()
    payload["intervention_condition"]["prevents_emergency_communication"] = True

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_number_of_blocks_must_be_balanced() -> None:
    payload = _valid_experiment()
    payload["number_of_blocks"] = 3

    with pytest.raises(ValidationError, match="even"):
        ExperimentSpec.model_validate(payload)


def test_washout_must_be_adequate() -> None:
    payload = _valid_experiment()
    payload["washout_duration_seconds"] = 0

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(payload)


def test_duration_must_respect_configured_bound() -> None:
    payload = _valid_experiment()

    with pytest.raises(ValidationError, match="duration exceeds"):
        ExperimentSpec.model_validate(payload, context={"max_experiment_duration_seconds": 1800})


def test_primary_outcome_must_be_distinct_and_preregistered() -> None:
    payload = _valid_experiment()
    payload["secondary_outcome_metrics"] = ["focus_score"]

    with pytest.raises(ValidationError, match="primary outcome"):
        ExperimentSpec.model_validate(payload)


def test_missing_prohibited_action_attestation_fails() -> None:
    payload = _valid_experiment()
    payload["prohibited_automatic_actions"] = ["medical_interventions"]

    with pytest.raises(ValidationError, match="prohibited automatic actions"):
        ExperimentSpec.model_validate(payload)


def test_checked_in_json_schema_matches_model_schema() -> None:
    schema_path = Path(__file__).parents[3] / "resonance" / "science" / "experiments" / "schema.json"
    checked_in = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = ExperimentSpec.model_json_schema()

    assert checked_in == generated


def _valid_experiment() -> dict:
    planned_start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    schedule = generate_randomized_schedule(
        planned_start=planned_start,
        block_duration_seconds=1800,
        number_of_blocks=4,
        washout_duration_seconds=300,
        randomization_seed=8675309,
    )
    return {
        "schema_version": "1.0",
        "title": "Manual focus condition comparison",
        "hypothesis_id": "hypothesis-123",
        "intervention_condition": {
            "name": "quiet desk",
            "instructions": "Use the quiet desk setup selected before the experiment.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "control_condition": {
            "name": "usual desk",
            "instructions": "Use the usual desk setup selected before the experiment.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "primary_outcome_metric": "focus_score",
        "secondary_outcome_metrics": ["self_reported_distraction"],
        "block_duration_seconds": 1800,
        "number_of_blocks": 4,
        "washout_duration_seconds": 300,
        "randomization_seed": 8675309,
        "randomized_schedule": [block.model_dump(mode="json") for block in schedule],
        "planned_start": planned_start.isoformat(),
        "inclusion_rules": [{"description": "Only start blocks during preselected work hours."}],
        "exclusion_rules": [{"description": "Exclude blocks interrupted by unavoidable external events."}],
        "stopping_rules": [{"description": "Stop after all confirmed blocks are complete."}],
        "abort_conditions": [{"description": "Abort immediately if safety or communication access is uncertain."}],
        "minimum_effect": 0.5,
        "analysis_method": "paired_block_difference",
        "safety_notes": "Low-risk, reversible desk setup only; no medical, hazardous, or automated setting changes.",
        "requires_manual_confirmation": True,
        "prohibited_automatic_actions": [
            "medical_interventions",
            "hazardous_physical_actions",
            "automatic_router_or_os_setting_changes",
            "blocking_emergency_communication",
        ],
        "maximum_experiment_duration_seconds": 86400,
    }
