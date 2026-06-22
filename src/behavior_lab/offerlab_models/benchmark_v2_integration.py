from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from behavior_lab import __version__
from behavior_lab.core import stable_hash, utc_now
from behavior_lab.datasets.nber_best_offer.real_normalize import verify_full_release_evidence
from behavior_lab.offerlab_models.benchmark_v2 import BenchmarkV2Paths as BenchmarkV2BuildPaths
from behavior_lab.offerlab_models.benchmark_v2 import build_offerlab_benchmark_v2
from behavior_lab.offerlab_models.benchmark_v2_runner import BenchmarkV2Paths as BenchmarkV2RunnerPaths
from behavior_lab.offerlab_models.benchmark_v2_runner import run_offerlab_benchmark_v2


DEFAULT_PROTOCOL = Path("datasets/manifests/offerlab_benchmark_v2.yaml")
DEFAULT_V1_FINAL = Path("reports/offerlab_benchmark_v1_final_manifest.json")
DEFAULT_OUTPUT = Path("reports/offerlab_benchmark_v2.json")
DEFAULT_PREREGISTRATION = Path("reports/offerlab_benchmark_v2_preregistration.json")
DEFAULT_PRE_HIDDEN = Path("reports/offerlab_benchmark_v2_pre_hidden.json")
DEFAULT_DOC = Path("docs/runs/OFFERLAB_BENCHMARK_V2_INTEGRATION.md")
DEFAULT_PRE_HIDDEN_DOC = Path("docs/runs/OFFERLAB_BENCHMARK_V2_PRE_HIDDEN.md")
DEFAULT_MODEL_CARDS = Path("docs/model_cards/offerlab_benchmark_v2")
TARGETS = (
    "seller_next_action",
    "buyer_response_to_counter",
    "agreement",
    "final_price_ratio",
    "response_latency",
)


@dataclass(frozen=True)
class BenchmarkV2IntegrationPaths:
    normalized_dir: Path
    benchmark_dir: Path
    output_path: Path = DEFAULT_OUTPUT
    preregistration_path: Path = DEFAULT_PREREGISTRATION
    pre_hidden_output_path: Path = DEFAULT_PRE_HIDDEN
    doc_path: Path = DEFAULT_DOC
    pre_hidden_doc_path: Path = DEFAULT_PRE_HIDDEN_DOC
    model_cards_dir: Path = DEFAULT_MODEL_CARDS
    protocol_path: Path = DEFAULT_PROTOCOL
    v1_final_manifest_path: Path = DEFAULT_V1_FINAL
    external_v1_hidden_tokens_path: Path | None = None


def run_offerlab_benchmark_v2_integration(
    paths: BenchmarkV2IntegrationPaths,
    *,
    batch_size: int = 50_000,
    partition_rows: int = 50_000,
    allow_bounded_test_input: bool = False,
    submit_hidden: bool = False,
) -> dict[str, Any]:
    """Run the Benchmark v2 integration gate without fabricating hidden evidence."""

    protocol = _read_json(paths.protocol_path)
    manifest = _read_json(paths.normalized_dir / "manifest.json")
    full_release_evidence = _full_release_evidence_summary(manifest)
    build_report: dict[str, Any] | None = None
    pre_hidden: dict[str, Any] | None = None
    errors: list[str] = []

    if full_release_evidence["passed"] or allow_bounded_test_input:
        try:
            build_report = build_offerlab_benchmark_v2(
                BenchmarkV2BuildPaths(
                    normalized_dir=paths.normalized_dir,
                    output_dir=paths.benchmark_dir,
                    protocol_path=paths.protocol_path,
                    v1_final_manifest_path=paths.v1_final_manifest_path,
                    external_v1_hidden_tokens_path=paths.external_v1_hidden_tokens_path,
                ),
                require_full_release=not allow_bounded_test_input,
                partition_rows=partition_rows,
            )
        except Exception as exc:  # pragma: no cover - exercised through returned fail-closed status.
            errors.append(_safe_error("build_failed", exc))
    else:
        errors.append("build_skipped:audited_full_release_evidence_failed")

    if build_report is not None:
        try:
            pre_hidden = run_offerlab_benchmark_v2(
                BenchmarkV2RunnerPaths(
                    normalized_dir=paths.normalized_dir,
                    output_path=paths.pre_hidden_output_path,
                    doc_path=paths.pre_hidden_doc_path,
                    model_cards_dir=paths.model_cards_dir,
                    protocol_path=paths.protocol_path,
                ),
                batch_size=batch_size,
                allow_hidden_submission=False,
            )
        except Exception as exc:  # pragma: no cover - exercised through returned fail-closed status.
            errors.append(_safe_error("pre_hidden_runner_failed", exc))

    preregistration = _preregistration_artifact(
        protocol=protocol,
        build_report=build_report,
        pre_hidden=pre_hidden,
        full_release_evidence=full_release_evidence,
        submit_hidden_requested=submit_hidden,
    )
    _write_atomic_json(paths.preregistration_path, preregistration)

    prerequisites = _prerequisites(build_report, pre_hidden, preregistration, full_release_evidence)
    hidden_submission = _hidden_submission_plan(
        prerequisites,
        submit_hidden=submit_hidden,
    )
    gate = _integration_gate(prerequisites, hidden_submission, errors)
    report = {
        "schema_version": "offerlab_benchmark_v2_integration.v1",
        "benchmark_id": "offerlab_benchmark_v2",
        "generated_at": utc_now(),
        "software_version": __version__,
        "git_commit": _git_commit(),
        "research_only": True,
        "production_export_allowed": False,
        "causal_claim": False,
        "seller_profit_claim": False,
        "hidden_results_used_for_selection": False,
        "hidden_submission_performed": hidden_submission["performed"],
        "source_dataset_ids": ["nber_ebay_best_offer"],
        "normalized_dir_hash": stable_hash(str(paths.normalized_dir.resolve())),
        "benchmark_dir_hash": stable_hash(str(paths.benchmark_dir.resolve())),
        "protocol": {
            "path": str(paths.protocol_path.as_posix()),
            "hash": stable_hash(protocol),
            "status": protocol.get("status"),
        },
        "full_release_evidence": full_release_evidence,
        "build": _build_summary(build_report),
        "pre_hidden": _pre_hidden_summary(pre_hidden),
        "preregistration": {
            "path_hash": stable_hash(str(paths.preregistration_path.resolve())),
            "filename": paths.preregistration_path.name,
            "hash": preregistration["preregistration_hash"],
            "target_count": len(preregistration.get("targets", {})),
        },
        "prerequisites": prerequisites,
        "hidden_submission": hidden_submission,
        "errors": errors,
        "gate": gate,
    }
    _write_atomic_json(paths.output_path, report)
    _write_markdown(paths.doc_path, report)
    return report


def _preregistration_artifact(
    *,
    protocol: dict[str, Any],
    build_report: dict[str, Any] | None,
    pre_hidden: dict[str, Any] | None,
    full_release_evidence: dict[str, Any],
    submit_hidden_requested: bool,
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    objectives = protocol.get("model_selection_rule", {}).get("target_objectives", {})
    if pre_hidden is not None:
        for target in protocol.get("targets", TARGETS):
            payload = pre_hidden.get("targets", {}).get(target, {})
            selected = payload.get("selected_model", {})
            objective = objectives.get(target, {})
            artifact = {
                "target": target,
                "selected_model_id": selected.get("model_id"),
                "selected_artifact_id": selected.get("artifact_id"),
                "preregistered_baseline": objective.get("preregistered_baseline"),
                "selection_split": selected.get("selection_split"),
                "selection_metric": selected.get("selection_metric") or objective.get("selection_metric"),
                "features_used": selected.get("features_used", []),
                "lineage_hash": stable_hash(selected.get("lineage", {})),
                "support_coverage": selected.get("coverage"),
                "hidden_results_used": False,
                "baseline_hidden_scoring_preregistered": True,
            }
            artifact["artifact_hash"] = stable_hash(artifact)
            targets[target] = artifact
    fresh_hidden = (build_report or {}).get("fresh_hidden_lockbox", {})
    build_preregistration_summary = _build_preregistration_summary(build_report)
    artifact = {
        "schema_version": "offerlab_benchmark_v2_preregistration.v1",
        "benchmark_id": "offerlab_benchmark_v2",
        "generated_at": utc_now(),
        "research_only": True,
        "production_export_allowed": False,
        "hidden_results_used_for_selection": False,
        "submit_hidden_requested": submit_hidden_requested,
        "full_release_evidence_passed": full_release_evidence["passed"],
        "build_artifact_hash": stable_hash(build_preregistration_summary),
        "pre_hidden_report_hash": stable_hash(_pre_hidden_summary(pre_hidden)) if pre_hidden else None,
        "fresh_hidden_lockbox_hashes": {
            target: payload.get("manifest_hash")
            for target, payload in fresh_hidden.items()
            if isinstance(payload, dict)
        },
        "targets": targets,
    }
    artifact["candidate_family_hash"] = stable_hash(targets)
    frozen_payload = {
        "schema_version": artifact["schema_version"],
        "benchmark_id": artifact["benchmark_id"],
        "full_release_evidence_passed": artifact["full_release_evidence_passed"],
        "build_artifact_hash": artifact["build_artifact_hash"],
        "pre_hidden_report_hash": artifact["pre_hidden_report_hash"],
        "fresh_hidden_lockbox_hashes": artifact["fresh_hidden_lockbox_hashes"],
        "candidate_family_hash": artifact["candidate_family_hash"],
        "targets": targets,
        "hidden_results_used_for_selection": False,
    }
    artifact["frozen_payload_hash"] = stable_hash(frozen_payload)
    artifact["preregistration_hash"] = artifact["frozen_payload_hash"]
    return artifact


def _prerequisites(
    build_report: dict[str, Any] | None,
    pre_hidden: dict[str, Any] | None,
    preregistration: dict[str, Any],
    full_release_evidence: dict[str, Any],
) -> dict[str, Any]:
    build_complete = build_report is not None
    pre_hidden_complete = pre_hidden is not None
    selected_targets = set(preregistration.get("targets", {}))
    fresh_hidden = (build_report or {}).get("fresh_hidden_lockbox", {})
    zero_v1_overlap = bool(fresh_hidden) and all(
        isinstance(payload, dict)
        and payload.get("labels_in_public_lockbox") is False
        and int(payload.get("v1_exclusion_cases", 0)) > 0
        for payload in fresh_hidden.values()
    )
    one_selected_artifact_per_target = selected_targets == set(TARGETS) and all(
        preregistration["targets"][target].get("selected_artifact_id")
        for target in TARGETS
    )
    readiness = (pre_hidden or {}).get("pre_hidden_readiness", {})
    runner_gate = (pre_hidden or {}).get("gate", {})
    return {
        "audited_full_release_evidence": full_release_evidence["passed"],
        "build_completed": build_complete,
        "pre_hidden_runner_completed": pre_hidden_complete,
        "pre_hidden_readiness_ready": readiness.get("status") == "ready_for_hidden",
        "runner_gate_research_signal": runner_gate.get("status") == "RESEARCH_SIGNAL",
        "zero_v1_hidden_overlap": zero_v1_overlap,
        "one_selected_artifact_per_target": one_selected_artifact_per_target,
        "preregistration_frozen": bool(preregistration.get("preregistration_hash")),
        "all_passed": all(
            [
                full_release_evidence["passed"],
                build_complete,
                pre_hidden_complete,
                readiness.get("status") == "ready_for_hidden",
                runner_gate.get("status") == "RESEARCH_SIGNAL",
                zero_v1_overlap,
                one_selected_artifact_per_target,
                bool(preregistration.get("preregistration_hash")),
            ]
        ),
    }


def _hidden_submission_plan(prerequisites: dict[str, Any], *, submit_hidden: bool) -> dict[str, Any]:
    if not submit_hidden:
        return {
            "performed": False,
            "status": "not_requested",
            "reason": "hidden submission requires explicit --submit-hidden after all preregistered prerequisites pass",
        }
    if not prerequisites["all_passed"]:
        return {
            "performed": False,
            "status": "blocked",
            "reason": "hidden submission blocked because one or more preregistered prerequisites failed",
        }
    return {
        "performed": False,
        "status": "blocked",
        "reason": "local integration froze the selected artifacts; hidden scoring requires the external one-shot Benchmark v2 evaluator",
    }


def _integration_gate(prerequisites: dict[str, Any], hidden_submission: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if errors or not prerequisites["all_passed"]:
        status = "STOP"
    elif hidden_submission["performed"]:
        status = "READY_FOR_SELLER_SHADOW_VALIDATION"
    else:
        status = "RESEARCH_SIGNAL"
    reasons = []
    if errors:
        reasons.extend(errors)
    for key, value in prerequisites.items():
        if key != "all_passed" and value is not True:
            if key == "audited_full_release_evidence":
                reasons.append("prerequisite_failed:audited_full_release_evidence:complete audited full-release NBER normalization is required")
            else:
                reasons.append(f"prerequisite_failed:{key}")
    if hidden_submission["status"] != "not_requested":
        reasons.append(f"hidden_submission_{hidden_submission['status']}:{hidden_submission['reason']}")
    if not reasons:
        reasons.append("pre-hidden Benchmark v2 integration prerequisites passed")
    return {
        "status": status,
        "allowed_statuses": ["STOP", "RESEARCH_SIGNAL", "READY_FOR_SELLER_SHADOW_VALIDATION"],
        "production_ready": False,
        "hidden_submission_performed": hidden_submission["performed"],
        "reasons": reasons,
    }


def _build_summary(build_report: dict[str, Any] | None) -> dict[str, Any]:
    if build_report is None:
        return {"completed": False}
    return {
        "completed": True,
        "manifest_hash": build_report.get("manifest_hash"),
        "model_training_executed": build_report.get("model_training_executed"),
        "task_manifests": build_report.get("task_manifests", {}),
        "fresh_hidden_lockbox": {
            target: {
                "manifest_hash": payload.get("manifest_hash"),
                "hidden_rows": payload.get("hidden_rows"),
                "excluded_overlap_rows": payload.get("excluded_overlap_rows"),
                "v1_exclusion_cases": payload.get("v1_exclusion_cases"),
                "labels_in_public_lockbox": payload.get("labels_in_public_lockbox"),
            }
            for target, payload in build_report.get("fresh_hidden_lockbox", {}).items()
            if isinstance(payload, dict)
        },
    }


def _build_preregistration_summary(build_report: dict[str, Any] | None) -> dict[str, Any]:
    if build_report is None:
        return {"completed": False}
    return {
        "completed": True,
        "benchmark_id": build_report.get("benchmark_id"),
        "protocol_hash": build_report.get("protocol", {}).get("hash"),
        "normalization_manifest_hash": build_report.get("normalization", {}).get("normalization_manifest_hash"),
        "task_manifests": build_report.get("task_manifests", {}),
        "fresh_hidden_lockbox_hashes": {
            target: payload.get("manifest_hash")
            for target, payload in build_report.get("fresh_hidden_lockbox", {}).items()
            if isinstance(payload, dict)
        },
    }


def _pre_hidden_summary(pre_hidden: dict[str, Any] | None) -> dict[str, Any]:
    if pre_hidden is None:
        return {"completed": False}
    selected = {
        target: {
            "model_id": payload.get("selected_model", {}).get("model_id"),
            "artifact_id": payload.get("selected_model", {}).get("artifact_id"),
            "selection_split": payload.get("selected_model", {}).get("selection_split"),
            "selection_metric": payload.get("selected_model", {}).get("selection_metric"),
        }
        for target, payload in pre_hidden.get("targets", {}).items()
    }
    summary = {
        "completed": True,
        "gate": pre_hidden.get("gate", {}),
        "pre_hidden_readiness": pre_hidden.get("pre_hidden_readiness", {}),
        "hidden_submission_performed": pre_hidden.get("hidden_submission_performed"),
        "selected_models": selected,
    }
    summary["report_hash"] = stable_hash(summary)
    return summary


def _full_release_evidence_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    report = verify_full_release_evidence(manifest)
    return {
        "schema_version": "offerlab_benchmark_v2_full_release_evidence_summary.v1",
        "passed": bool(report.get("passed") is True),
        "checks": dict(report.get("checks", {})),
        "failures": list(report.get("failures", [])),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_error(stage: str, exc: Exception) -> str:
    return f"{stage}:{type(exc).__name__}:{stable_hash(str(exc))}"


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OfferLab Benchmark v2 Integration",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Gate status: **{report['gate']['status']}**",
        "",
        "Hidden submission performed: `False`" if not report["hidden_submission_performed"] else "Hidden submission performed: `True`",
        "",
        "## Prerequisites",
        "",
    ]
    for key, value in report["prerequisites"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}" for reason in report["gate"]["reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
