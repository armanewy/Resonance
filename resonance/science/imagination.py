from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from resonance.science.contracts import HypothesisSpec, Origin, stable_hash
from resonance.science.discovery_brief import discovery_brief_from_exploration_view
from resonance.science.fitting import EVALUATOR_VERSION as FITTING_VERSION
from resonance.science.fitting import fit_hypothesis
from resonance.science.interpreter import frame_from_snapshot_rows
from resonance.science.ledger import append_event, current_code_commit, read_entries
from resonance.science.providers import FileProvider, MockProvider, ProviderError, run_provider
from resonance.science.review import (
    ReviewRecommendation,
    ReviewSpec,
    validate_hypotheses,
)
from resonance.science.selection import EVALUATOR_VERSION as SELECTION_VERSION
from resonance.science.selection import select_candidate
from resonance.science.snapshots import load_exploration_view, load_snapshot_manifest


ARTIFACT_SCHEMA_VERSION = 1
IMAGINATION_VERSION = "llm-hypothesis-imagination-v1"
DEFAULT_IMAGINATION_SEED = 20260619


class ImaginationError(RuntimeError):
    """Raised when the hypothesis imagination flow cannot proceed."""


def imagine_hypotheses(
    *,
    snapshot_id: str,
    provider_name: str,
    max_hypotheses: int,
    artifact_root: Path,
    ledger_path: Path,
    seed: int = DEFAULT_IMAGINATION_SEED,
    provider_file: str | Path | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=artifact_root)
    exploration = load_exploration_view(snapshot_id, artifact_root=artifact_root)
    brief = discovery_brief_from_exploration_view(
        exploration,
        metric_catalog=manifest["metric_catalog"],
    )
    brief_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "discovery_brief",
            "imagination_version": IMAGINATION_VERSION,
            "brief": brief.model_dump(mode="json", exclude_none=True),
        },
    )
    prompt_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "imagination_prompt_versions",
            "imagination_version": IMAGINATION_VERSION,
            "prompts": _prompt_versions(),
        },
    )

    resolved_provider = provider or _provider_for_name(
        provider_name,
        manifest=manifest,
        provider_file=provider_file,
    )
    provider_run = run_provider(
        resolved_provider,
        brief,
        max_hypotheses=max_hypotheses,
        seed=seed,
    )
    raw_proposals = tuple(getattr(resolved_provider, "last_raw_proposals", ()))
    raw_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "imagination_raw_proposals",
            "provider": provider_run.metadata.model_dump(mode="json"),
            "proposals": [_artifact_value(item) for item in raw_proposals],
        },
    )
    provider_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "imagination_provider_run",
            "provider_run": provider_run.artifact_payload(),
        },
    )
    validation_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "imagination_validation_failures",
            "failures": [
                failure.model_dump(mode="json")
                for failure in provider_run.rejected_proposals
            ],
        },
    )

    deterministic_reviews = validate_hypotheses(
        provider_run.hypotheses,
        metric_catalog=manifest["metric_catalog"],
        snapshot_max_lag_seconds=int(manifest["max_lag_seconds"]),
    )
    proposal_records: list[dict[str, Any]] = []
    reviewer_records: list[dict[str, Any]] = []
    for index, hypothesis in enumerate(provider_run.hypotheses):
        candidate_artifact = _store_json_artifact(
            artifact_root,
            _candidate_artifact_payload(hypothesis),
        )
        deterministic_review = deterministic_reviews[index]
        skeptical_review = _skeptical_review_spec(hypothesis, deterministic_review.accepted)
        reviewer_records.append(
            {
                "index": index,
                "candidate_id": candidate_artifact["sha256"],
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "deterministic_review": deterministic_review.model_dump(mode="json"),
                "skeptical_review": skeptical_review.model_dump(mode="json"),
                "status": "review_accepted" if deterministic_review.accepted else "review_rejected",
            }
        )
        proposal_records.append(
            {
                "index": index,
                "candidate_id": candidate_artifact["sha256"],
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "title": hypothesis.title,
                "status": "review_accepted" if deterministic_review.accepted else "review_rejected",
                "artifacts": {"hypothesis": candidate_artifact},
            }
        )

    reviewer_artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "imagination_reviewer_output",
            "reviewer_version": "deterministic-skeptical-reviewer-v1",
            "reviews": reviewer_records,
            "rejected_provider_proposals": [
                failure.model_dump(mode="json")
                for failure in provider_run.rejected_proposals
            ],
        },
    )
    run_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "imagination_run",
        "imagination_version": IMAGINATION_VERSION,
        "snapshot_id": snapshot_id,
        "seed": seed,
        "max_hypotheses": max_hypotheses,
        "provider": provider_run.metadata.model_dump(mode="json"),
        "proposal_count": len(proposal_records),
        "accepted_review_count": sum(1 for record in proposal_records if record["status"] == "review_accepted"),
        "rejected_provider_count": len(provider_run.rejected_proposals),
        "proposals": proposal_records,
        "artifacts": {
            "discovery_brief": brief_artifact,
            "prompt_versions": prompt_artifact,
            "provider_run": provider_artifact,
            "raw_structured_proposals": raw_artifact,
            "validation_failures": validation_artifact,
            "reviewer_output": reviewer_artifact,
        },
        "raw_blind_values_exposed": False,
        "loaded_partitions_before_provider": ["exploration"],
        "llm_used": provider_run.metadata.provider_name != "mock",
        "arbitrary_generated_code_executable": False,
        "code_commit": current_code_commit(),
    }
    run_artifact = _store_json_artifact(artifact_root, run_payload)
    append_event(
        "hypothesis_proposed",
        {
            "run_id": run_artifact["sha256"],
            "dataset_snapshot_id": snapshot_id,
            "imagination_version": IMAGINATION_VERSION,
            "provider": provider_run.metadata.model_dump(mode="json"),
            "max_hypotheses": max_hypotheses,
            "proposal_count": len(proposal_records),
            "accepted_review_count": run_payload["accepted_review_count"],
            "rejected_provider_count": len(provider_run.rejected_proposals),
            "llm_used": run_payload["llm_used"],
            "arbitrary_generated_code_executable": False,
            "raw_blind_values_exposed": False,
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {**run_payload["artifacts"], "imagination_run": run_artifact},
        },
        artifact_hashes={
            "discovery_brief": brief_artifact["sha256"],
            "prompt_versions": prompt_artifact["sha256"],
            "provider_run": provider_artifact["sha256"],
            "raw_structured_proposals": raw_artifact["sha256"],
            "validation_failures": validation_artifact["sha256"],
            "reviewer_output": reviewer_artifact["sha256"],
            "imagination_run": run_artifact["sha256"],
        },
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return {
        "run_id": run_artifact["sha256"],
        "snapshot_id": snapshot_id,
        "provider": provider_run.metadata.model_dump(mode="json"),
        "proposals": proposal_records,
        "accepted_review_count": run_payload["accepted_review_count"],
        "rejected_provider_count": len(provider_run.rejected_proposals),
        "artifact": run_artifact,
    }


def review_imagination_run(
    run_id: str,
    *,
    artifact_root: Path,
    ledger_path: Path,
    approve: str | None = None,
) -> dict[str, Any]:
    run = _load_artifact_by_hash(artifact_root, run_id, "imagination_run")
    approvals = _approval_records(run_id, ledger_path)
    if approve is not None:
        proposal = _select_proposal(run, approve)
        if proposal["status"] != "review_accepted":
            raise ImaginationError("only review-accepted proposals can be approved")
        if proposal["candidate_id"] not in {item["candidate_id"] for item in approvals}:
            approval_payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "record_type": "imagination_human_approval",
                "run_id": run_id,
                "snapshot_id": run["snapshot_id"],
                "proposal_index": proposal["index"],
                "candidate_id": proposal["candidate_id"],
                "hypothesis_hash": proposal["hypothesis_hash"],
                "approval_type": "explicit_human_cli_approval",
                "approved": True,
                "raw_blind_values_exposed": False,
                "code_commit": current_code_commit(),
            }
            approval_artifact = _store_json_artifact(artifact_root, approval_payload)
            append_event(
                "result_interpreted",
                {
                    "interpretation_type": "imagination_human_approval",
                    "run_id": run_id,
                    "dataset_snapshot_id": run["snapshot_id"],
                    "proposal_index": proposal["index"],
                    "candidate_id": proposal["candidate_id"],
                    "hypothesis_hash": proposal["hypothesis_hash"],
                    "approved": True,
                    "artifact_root": str(artifact_root.resolve()),
                    "artifacts": {"human_approval": approval_artifact},
                },
                artifact_hashes={"human_approval": approval_artifact["sha256"]},
                code_commit=current_code_commit(),
                ledger_path=ledger_path,
            )
            approvals = _approval_records(run_id, ledger_path)
    approved_ids = {item["candidate_id"] for item in approvals}
    return {
        "run_id": run_id,
        "snapshot_id": run["snapshot_id"],
        "provider": run["provider"],
        "proposals": [
            {
                **proposal,
                "approved": proposal["candidate_id"] in approved_ids,
            }
            for proposal in run["proposals"]
        ],
        "approval_count": len(approved_ids),
        "raw_blind_values_exposed": False,
    }


def fit_approved(
    run_id: str,
    *,
    artifact_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    run = _load_artifact_by_hash(artifact_root, run_id, "imagination_run")
    approved_ids = {record["candidate_id"] for record in _approval_records(run_id, ledger_path)}
    proposals = [
        proposal
        for proposal in run["proposals"]
        if proposal["candidate_id"] in approved_ids and proposal["status"] == "review_accepted"
    ]
    if not proposals:
        raise ImaginationError("no approved review-accepted proposals found")

    exploration = load_exploration_view(run["snapshot_id"], artifact_root=artifact_root)
    frame = frame_from_snapshot_rows(exploration["rows"])
    candidates: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    for proposal in proposals:
        candidate_record = _load_artifact_by_hash(
            artifact_root,
            proposal["candidate_id"],
            "manual_hypothesis_candidate",
        )
        hypothesis = HypothesisSpec.model_validate(candidate_record["hypothesis"])
        fit_result = fit_hypothesis(hypothesis, frame)
        fit_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "manual_fit_result",
            "cli_version": IMAGINATION_VERSION,
            "imagination_run_id": run_id,
            "evaluator_version": FITTING_VERSION,
            "snapshot_id": run["snapshot_id"],
            "candidate_id": proposal["candidate_id"],
            "hypothesis_hash": hypothesis.hypothesis_hash(),
            "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
            "fit_result": asdict(fit_result),
            "code_commit": current_code_commit(),
        }
        fit_artifact = _store_json_artifact(artifact_root, fit_payload)
        append_event(
            "fit_completed",
            {
                "run_id": fit_artifact["sha256"],
                "imagination_run_id": run_id,
                "candidate_id": proposal["candidate_id"],
                "dataset_snapshot_id": run["snapshot_id"],
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "evaluator_version": FITTING_VERSION,
                "random_seed": hypothesis.random_seed,
                "fitted_parameters": fit_result.fitted_parameters,
                "metrics": fit_result.exploration_metrics,
                "baseline_metrics": fit_result.baseline_metrics,
                "warnings": fit_result.warnings,
                "artifact_root": str(artifact_root.resolve()),
                "artifacts": {
                    "fit_result": fit_artifact,
                    "hypothesis": proposal["artifacts"]["hypothesis"],
                },
            },
            artifact_hashes={
                "fit_result": fit_artifact["sha256"],
                "hypothesis": proposal["candidate_id"],
            },
            code_commit=current_code_commit(),
            ledger_path=ledger_path,
        )
        candidates.append(
            {
                "candidate_id": proposal["candidate_id"],
                "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
                "fitted_parameters": fit_result.fitted_parameters,
                "fit_result": {"fit_result_id": fit_artifact["sha256"]},
            }
        )
        fit_summaries.append(
            {
                "run_id": fit_artifact["sha256"],
                "candidate_id": proposal["candidate_id"],
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "fitted_parameters": fit_result.fitted_parameters,
                "exploration_metrics": fit_result.exploration_metrics,
                "artifact": fit_artifact,
            }
        )

    selection = select_candidate(
        run["snapshot_id"],
        candidates,
        artifact_root=artifact_root,
        record_artifact=True,
    )
    if selection.artifact is None:
        raise ImaginationError("selection did not produce an artifact")
    selected_candidate_id = selection.selected_candidate_id
    append_event(
        "result_interpreted",
        {
            "interpretation_type": "imagination_tuning_selection",
            "run_id": run_id,
            "candidate_id": selected_candidate_id,
            "dataset_snapshot_id": run["snapshot_id"],
            "selected_candidate_id": selected_candidate_id,
            "selected_hypothesis_hash": selection.selected_hypothesis_hash,
            "evaluator_version": SELECTION_VERSION,
            "evaluations": [evaluation.to_dict() for evaluation in selection.evaluations],
            "warnings": list(selection.warnings),
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {"tuning_selection": selection.artifact},
        },
        artifact_hashes={"tuning_selection": selection.artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return {
        "run_id": run_id,
        "snapshot_id": run["snapshot_id"],
        "fit_results": fit_summaries,
        "tuning": selection.to_dict(),
        "selected_candidate_id": selected_candidate_id,
        "raw_blind_values_exposed": False,
    }


def _provider_for_name(
    provider_name: str,
    *,
    manifest: Mapping[str, Any],
    provider_file: str | Path | None,
) -> Any:
    if provider_name == "mock":
        return MockProvider([_default_mock_hypothesis(manifest)])
    if provider_name == "file":
        if provider_file is None:
            raise ProviderError("--provider-file is required for --provider file")
        return FileProvider(provider_file)
    raise ProviderError(f"unsupported provider: {provider_name}")


def _default_mock_hypothesis(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metrics = [str(metric["name"]) for metric in manifest["metric_catalog"]["metrics"]]
    catalog_id = str(manifest["metric_catalog"]["catalog_id"])
    if {"x", "y", "control"}.issubset(metrics):
        target = "y"
        source = "x"
        control = "control"
    elif len(metrics) >= 3:
        source, control, target = metrics[0], metrics[1], metrics[2]
    elif len(metrics) == 2:
        source, target = metrics[0], metrics[1]
        control = metrics[0]
    else:
        raise ProviderError("mock provider requires at least two metrics")
    lag = min(900, int(manifest["max_lag_seconds"]))
    return {
        "schema_version": "1.0",
        "hypothesis_type": "observational_prediction",
        "title": f"Mock lagged {source} predicts {target}",
        "concise_claim": f"Lagged {source} is associated with {target}.",
        "rationale": "Deterministic mock proposal used to exercise the LLM imagination workflow.",
        "target_metric": target,
        "input_metrics": [source],
        "target_transform": "identity",
        "expression": {
            "node": "add",
            "left": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {
                    "node": "lag",
                    "input": {"node": "metric", "metric": source},
                    "lag_seconds": lag,
                },
            },
            "right": {"node": "fitted_parameter", "parameter": "offset"},
        },
        "parameter_bounds": {
            "scale": {"lower": 0.0, "upper": 3.0},
            "offset": {"lower": -5.0, "upper": 5.0},
        },
        "expected_direction": "positive",
        "maximum_lag_seconds": lag,
        "fitting_metric": "rmse",
        "tuning_metric": "rmse",
        "blind_metrics": ["mae", "rmse", "spearman_r"],
        "minimum_blind_effect": 0.5,
        "minimum_baseline_improvement": 0.05,
        "negative_controls": [
            {
                "metric": control,
                "rationale": "Independent control should not track the fitted prediction.",
            }
        ],
        "falsification_conditions": [
            {"description": "Tuning performance does not improve over baseline."},
            {"description": "The negative control is associated with the prediction."},
        ],
        "complexity_budget": {"max_ast_nodes": 8, "max_source_metrics": 1},
        "origin": Origin.LLM.value,
        "parent_hypothesis_ids": [],
        "snapshot_metric_catalog_id": catalog_id,
        "random_seed": DEFAULT_IMAGINATION_SEED,
    }


def _skeptical_review_spec(hypothesis: HypothesisSpec, executable: bool) -> ReviewSpec:
    recommendation = (
        ReviewRecommendation.PREREGISTRATION_ELIGIBLE
        if executable
        else ReviewRecommendation.REVISE
    )
    return ReviewSpec(
        confounders=(
            "Shared seasonality, autocorrelation, operational schedules, or common upstream drivers may explain the association.",
        ),
        simpler_explanation=(
            "A persistence or shared-trend baseline may account for apparent predictive value."
        ),
        leakage_risk=(
            "The deterministic checks reject direct target leakage and future values, but human review should still inspect data lineage."
        ),
        mechanical_correlation_risk=(
            "The relationship may be mechanical if the source and target share instrumentation or preprocessing."
        ),
        suggested_controls_or_falsifications=tuple(
            control.rationale for control in hypothesis.negative_controls
        )
        or ("Run the declared negative controls before preregistration.",),
        executable=executable,
        distinct_from_prior=True,
        recommendation=recommendation,
    )


def _candidate_artifact_payload(hypothesis: HypothesisSpec) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_hypothesis_candidate",
        "cli_version": IMAGINATION_VERSION,
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
        "llm_used": True,
        "arbitrary_generated_code_executable": False,
    }


def _approval_records(run_id: str, ledger_path: Path) -> list[dict[str, Any]]:
    approvals: list[dict[str, Any]] = []
    for entry in read_entries(ledger_path):
        payload = entry["payload"]
        if (
            entry["event_type"] == "result_interpreted"
            and payload.get("interpretation_type") == "imagination_human_approval"
            and payload.get("run_id") == run_id
            and payload.get("approved") is True
        ):
            approvals.append(payload)
    return approvals


def _select_proposal(run: Mapping[str, Any], selector: str) -> dict[str, Any]:
    for proposal in run["proposals"]:
        if str(proposal["index"]) == selector:
            return proposal
        if selector in {proposal["candidate_id"], proposal["hypothesis_hash"]}:
            return proposal
    raise ImaginationError(f"unknown proposal selector for run: {selector}")


def _prompt_versions() -> dict[str, Any]:
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    prompts: dict[str, Any] = {}
    for name in ("proposer_v1.md", "reviewer_v1.md"):
        path = prompt_dir / name
        content = path.read_text(encoding="utf-8")
        prompts[name] = {
            "path": f"resonance/science/prompts/{name}",
            "sha256": stable_hash({"content": content}),
        }
    return prompts


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ImaginationError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _load_artifact_by_hash(root: Path, digest: str, expected_record_type: str) -> dict[str, Any]:
    if not _is_hash(digest):
        raise ImaginationError(f"artifact id must be a 64-character hash: {digest}")
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_hash(payload) != digest:
        raise ImaginationError(f"artifact hash mismatch: {path}")
    record_type = payload.get("record_type")
    if record_type != expected_record_type:
        raise ImaginationError(f"expected {expected_record_type}, found {record_type!r}")
    return payload


def _artifact_value(value: Any) -> Any:
    if isinstance(value, HypothesisSpec):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_value(item) for item in value]
    return value


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "DEFAULT_IMAGINATION_SEED",
    "IMAGINATION_VERSION",
    "ImaginationError",
    "fit_approved",
    "imagine_hypotheses",
    "review_imagination_run",
]
