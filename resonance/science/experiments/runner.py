from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from resonance.science.contracts import stable_hash
from resonance.science.experiments.contracts import ConditionName, ExperimentSpec, ScheduledBlock
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


EXPERIMENT_RUNNER_VERSION = "manual-experiment-runner-v1"
EXPERIMENT_ARTIFACT_SCHEMA_VERSION = 1


class ExperimentRunnerError(RuntimeError):
    """Raised when a manual experiment transition cannot proceed."""


def preregister_experiment(
    spec_path: str | Path,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate and freeze a manual experiment spec before any run starts."""

    root = Path(artifact_root)
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    try:
        spec = ExperimentSpec.model_validate(payload)
    except ValidationError:
        raise
    experiment_id = spec.experiment_hash()
    existing = _load_state_or_none(root, experiment_id)
    if existing is not None:
        return _public_state(existing)

    preregistered_at = _timestamp(now_utc)
    spec_payload = {
        "schema_version": EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_experiment_spec",
        "runner_version": EXPERIMENT_RUNNER_VERSION,
        "experiment_id": experiment_id,
        "experiment_hash": experiment_id,
        "spec": spec.frozen_content(),
        "frozen_schedule": [block.model_dump(mode="json") for block in spec.randomized_schedule],
        "frozen_analysis": {
            "primary_outcome_metric": spec.primary_outcome_metric,
            "secondary_outcome_metrics": list(spec.secondary_outcome_metrics),
            "analysis_method": spec.analysis_method.value,
            "minimum_effect": spec.minimum_effect,
            "inclusion_rules": [rule.model_dump(mode="json") for rule in spec.inclusion_rules],
            "exclusion_rules": [rule.model_dump(mode="json") for rule in spec.exclusion_rules],
            "stopping_rules": [rule.model_dump(mode="json") for rule in spec.stopping_rules],
            "abort_conditions": [rule.model_dump(mode="json") for rule in spec.abort_conditions],
        },
        "preregistered_at_utc": preregistered_at,
        "code_commit": current_code_commit(),
    }
    spec_artifact = _store_json_artifact(root, spec_payload)
    state = {
        "schema_version": EXPERIMENT_ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_experiment_state",
        "runner_version": EXPERIMENT_RUNNER_VERSION,
        "experiment_id": experiment_id,
        "experiment_hash": experiment_id,
        "status": "preregistered",
        "created_at_utc": preregistered_at,
        "updated_at_utc": preregistered_at,
        "started_at_utc": None,
        "completed_at_utc": None,
        "spec": spec.frozen_content(),
        "spec_artifact": spec_artifact,
        "blocks": [_initial_block_state(block) for block in spec.randomized_schedule],
        "failures": [],
        "artifacts": {"spec": spec_artifact},
        "code_commit": current_code_commit(),
    }
    state_artifact = _record_state(root, state)
    append_event(
        "experiment_planned",
        {
            "experiment_id": experiment_id,
            "experiment_hash": experiment_id,
            "title": spec.title,
            "hypothesis_id": spec.hypothesis_id,
            "status": "preregistered",
            "runner_version": EXPERIMENT_RUNNER_VERSION,
            "schedule_frozen": True,
            "analysis_frozen": True,
            "manual_only": True,
            "automatic_intervention_applied": False,
            "artifact_root": str(root.resolve()),
            "artifacts": {"spec": spec_artifact, "state": state_artifact},
        },
        artifact_hashes={"spec": spec_artifact["sha256"], "state": state_artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
        timestamp_utc=preregistered_at,
    )
    return _public_state(state)


def start_experiment(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    state = _load_state(Path(artifact_root), experiment_id)
    if state["status"] != "preregistered":
        raise ExperimentRunnerError(f"experiment cannot be started from status {state['status']!r}")
    timestamp = _timestamp(now_utc)
    state["status"] = "started"
    state["started_at_utc"] = timestamp
    state["updated_at_utc"] = timestamp
    return _transition(
        state,
        "experiment_started",
        "experiment_started",
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        timestamp_utc=timestamp,
        extra_payload={"schedule_frozen": True, "analysis_frozen": True},
    )


def begin_block(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    state = _load_state(Path(artifact_root), experiment_id)
    if state["status"] not in {"started", "in_progress"}:
        raise ExperimentRunnerError(f"experiment is not active: {state['status']}")
    if _active_block(state) is not None:
        raise ExperimentRunnerError("a block is already in progress")
    block = _next_open_block(state)
    if block is None:
        raise ExperimentRunnerError("no remaining block to begin")
    timestamp = _timestamp(now_utc)
    block["status"] = "in_progress"
    block["actual_start_utc"] = timestamp
    block["confirmation_required"] = True
    state["status"] = "in_progress"
    state["updated_at_utc"] = timestamp
    condition = _condition_details(state, block["condition"])
    return _transition(
        state,
        "experiment_observation",
        "block_started",
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        timestamp_utc=timestamp,
        extra_payload={
            "block_index": block["block_index"],
            "condition": block["condition"],
            "condition_name": condition["name"],
            "condition_instructions": condition["instructions"],
            "requires_user_confirmation": True,
            "automatic_intervention_applied": False,
        },
        public_extra={"current_condition": condition, "block": deepcopy(block)},
    )


def confirm_condition(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    state = _load_state(Path(artifact_root), experiment_id)
    block = _active_block(state)
    if block is None:
        raise ExperimentRunnerError("no block is in progress")
    timestamp = _timestamp(now_utc)
    block["condition_confirmed"] = True
    block["condition_confirmed_at_utc"] = timestamp
    state["updated_at_utc"] = timestamp
    return _transition(
        state,
        "experiment_observation",
        "condition_confirmed",
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        timestamp_utc=timestamp,
        extra_payload={
            "block_index": block["block_index"],
            "condition": block["condition"],
            "automatic_intervention_applied": False,
        },
        public_extra={"block": deepcopy(block)},
    )


def end_block(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | None = None,
    missed: bool = False,
    compliant: bool = True,
    note: str = "",
) -> dict[str, Any]:
    state = _load_state(Path(artifact_root), experiment_id)
    block = _active_block(state)
    if block is None:
        raise ExperimentRunnerError("no block is in progress")
    timestamp = _timestamp(now_utc)
    block["actual_end_utc"] = timestamp
    block["note"] = note
    block["missed"] = bool(missed)
    block["compliant"] = bool(compliant) and not missed
    if missed:
        block["status"] = "missed"
    elif not block["condition_confirmed"]:
        block["status"] = "noncompliant"
        block["compliant"] = False
        block["note"] = note or "Condition was not confirmed before block end."
    elif not compliant:
        block["status"] = "noncompliant"
    else:
        block["status"] = "completed"
    if block["status"] in {"missed", "noncompliant"}:
        state["failures"].append(
            {
                "block_index": block["block_index"],
                "failure_type": block["status"],
                "recorded_at_utc": timestamp,
                "note": block["note"],
            }
        )
    if _next_open_block(state) is None:
        state["status"] = "completed"
        state["completed_at_utc"] = timestamp
    else:
        state["status"] = "started"
    state["updated_at_utc"] = timestamp
    event_type = "experiment_completed" if state["status"] == "completed" else "experiment_observation"
    return _transition(
        state,
        event_type,
        "block_ended",
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        timestamp_utc=timestamp,
        extra_payload={
            "block_index": block["block_index"],
            "condition": block["condition"],
            "block_status": block["status"],
            "actual_start_utc": block["actual_start_utc"],
            "actual_end_utc": block["actual_end_utc"],
            "missed": block["missed"],
            "compliant": block["compliant"],
            "automatic_intervention_applied": False,
        },
        public_extra={"block": deepcopy(block)},
    )


def experiment_status(
    experiment_id: str,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    return _public_state(_load_state(Path(artifact_root), experiment_id))


def _transition(
    state: dict[str, Any],
    event_type: str,
    observation_type: str,
    *,
    artifact_root: str | Path,
    ledger_path: str | Path,
    timestamp_utc: str,
    extra_payload: Mapping[str, Any] | None = None,
    public_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    state_artifact = _record_state(root, state)
    payload = {
        "experiment_id": state["experiment_id"],
        "experiment_hash": state["experiment_hash"],
        "observation_type": observation_type,
        "status": state["status"],
        "runner_version": EXPERIMENT_RUNNER_VERSION,
        "artifact_root": str(root.resolve()),
        "artifacts": {"state": state_artifact, "spec": state["spec_artifact"]},
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    append_event(
        event_type,
        payload,
        artifact_hashes={"state": state_artifact["sha256"], "spec": state["spec_artifact"]["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
        timestamp_utc=timestamp_utc,
    )
    public = _public_state(state)
    public["state_artifact"] = state_artifact
    if public_extra:
        public.update(dict(public_extra))
    return public


def _initial_block_state(block: ScheduledBlock) -> dict[str, Any]:
    return {
        "block_index": block.block_index,
        "condition": block.condition.value,
        "planned_start_utc": to_utc_iso(block.planned_start),
        "planned_end_utc": to_utc_iso(block.planned_end),
        "status": "planned",
        "confirmation_required": block.requires_user_confirmation,
        "condition_confirmed": False,
        "condition_confirmed_at_utc": None,
        "actual_start_utc": None,
        "actual_end_utc": None,
        "missed": False,
        "compliant": None,
        "note": "",
    }


def _condition_details(state: Mapping[str, Any], condition: str) -> dict[str, str]:
    key = "intervention_condition" if condition == ConditionName.INTERVENTION.value else "control_condition"
    value = state["spec"][key]
    return {"condition": condition, "name": value["name"], "instructions": value["instructions"]}


def _active_block(state: Mapping[str, Any]) -> dict[str, Any] | None:
    return next((block for block in state["blocks"] if block["status"] == "in_progress"), None)


def _next_open_block(state: Mapping[str, Any]) -> dict[str, Any] | None:
    return next((block for block in state["blocks"] if block["status"] == "planned"), None)


def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": state["experiment_id"],
        "experiment_hash": state["experiment_hash"],
        "status": state["status"],
        "title": state["spec"]["title"],
        "primary_outcome_metric": state["spec"]["primary_outcome_metric"],
        "analysis_method": state["spec"]["analysis_method"],
        "started_at_utc": state.get("started_at_utc"),
        "completed_at_utc": state.get("completed_at_utc"),
        "blocks": deepcopy(state["blocks"]),
        "failures": deepcopy(state.get("failures", [])),
        "artifacts": deepcopy(state.get("artifacts", {})),
    }


def _record_state(root: Path, state: dict[str, Any]) -> dict[str, str]:
    state_payload = deepcopy(state)
    state_payload["record_type"] = "manual_experiment_state"
    artifact = _store_json_artifact(root, state_payload)
    state["artifacts"]["state"] = artifact
    _write_index(root, state["experiment_id"], artifact)
    return artifact


def _write_index(root: Path, experiment_id: str, state_artifact: Mapping[str, str]) -> None:
    path = _index_path(root, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        {"experiment_id": experiment_id, "state": dict(state_artifact)},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    path.write_text(content, encoding="utf-8")


def _load_state(root: Path, experiment_id: str) -> dict[str, Any]:
    state = _load_state_or_none(root, experiment_id)
    if state is None:
        raise ExperimentRunnerError(f"unknown experiment id: {experiment_id}")
    return state


def _load_state_or_none(root: Path, experiment_id: str) -> dict[str, Any] | None:
    path = _index_path(root, experiment_id)
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    state = _load_artifact(root, index["state"]["sha256"], "manual_experiment_state")
    state.setdefault("artifacts", {})["state"] = index["state"]
    return state


def _index_path(root: Path, experiment_id: str) -> Path:
    return root / "experiments" / f"{experiment_id}.json"


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ExperimentRunnerError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _load_artifact(root: Path, digest: str, expected_record_type: str) -> dict[str, Any]:
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_hash(payload) != digest:
        raise ExperimentRunnerError(f"artifact hash mismatch: {path}")
    if payload.get("record_type") != expected_record_type:
        raise ExperimentRunnerError(f"expected {expected_record_type}, found {payload.get('record_type')!r}")
    return payload


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        return to_utc_iso(utc_now())
    if isinstance(value, str):
        return to_utc_iso(parse_utc(value))
    return to_utc_iso(ensure_utc(value))


__all__ = [
    "EXPERIMENT_ARTIFACT_SCHEMA_VERSION",
    "EXPERIMENT_RUNNER_VERSION",
    "ExperimentRunnerError",
    "begin_block",
    "confirm_condition",
    "end_block",
    "experiment_status",
    "preregister_experiment",
    "start_experiment",
]
