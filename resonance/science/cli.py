from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from resonance.science.ablation import (
    DEFAULT_CANDIDATE_BUDGET as DEFAULT_ABLATION_CANDIDATE_BUDGET,
    DEFAULT_DURATION_HOURS as DEFAULT_ABLATION_DURATION_HOURS,
    DEFAULT_SCENARIOS as DEFAULT_ABLATION_SCENARIOS,
    DEFAULT_SNAPSHOT_HOURS as DEFAULT_ABLATION_SNAPSHOT_HOURS,
    AblationError,
    run_ablation,
)
from resonance.science.blind_evaluator import (
    BlindEvaluationError,
    evaluate_preregistration,
)
from resonance.science.contracts import HypothesisSpec, stable_hash
from resonance.science.fitting import EVALUATOR_VERSION as FITTING_VERSION
from resonance.science.fitting import FittingError, fit_hypothesis
from resonance.science.imagination import (
    DEFAULT_IMAGINATION_SEED,
    ImaginationError,
    fit_approved,
    imagine_hypotheses,
    review_imagination_run,
)
from resonance.science.interpreter import frame_from_snapshot_rows
from resonance.science.ledger import (
    DEFAULT_LEDGER_PATH,
    append_event,
    current_code_commit,
    read_entries,
)
from resonance.science.preregistration import (
    EVALUATOR_VERSION as BLIND_EVALUATOR_VERSION,
    create_preregistration,
    load_preregistration,
)
from resonance.science.selection import (
    EVALUATOR_VERSION as SELECTION_VERSION,
    SelectionError,
    select_candidate,
)
from resonance.science.snapshots import (
    DEFAULT_ARTIFACT_ROOT,
    create_snapshot,
    load_exploration_view,
    load_snapshot_manifest,
    parse_metric_csv,
    snapshot_summary,
)
from resonance.storage import Measurement, init_db, insert_measurements
from resonance.synthetic import DEFAULT_SEED, SCENARIO_DESCRIPTIONS, generate_synthetic_series


CLI_VERSION = "manual-science-cli-v1"
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_SYNTHETIC_HOURS = 96.0
DEFAULT_SYNTHETIC_MAX_LAG_SECONDS = 900
DEFAULT_SYNTHETIC_NOISE = 0.6
DEFAULT_SYNTHETIC_SAMPLE_INTERVAL_SECONDS = 300


class ScienceCliError(RuntimeError):
    """Raised when a manual science CLI operation cannot proceed."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except (
        BlindEvaluationError,
        AblationError,
        FileNotFoundError,
        FittingError,
        ImaginationError,
        ScienceCliError,
        SelectionError,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"Science loop command failed: {exc}")
        return 1
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the manual sealed scientific loop.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=f"content-addressed artifact root; defaults to {DEFAULT_ARTIFACT_ROOT}",
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help=f"scientific ledger path; defaults to {DEFAULT_LEDGER_PATH}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="create or inspect sealed snapshots")
    snapshot_subparsers = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_create = snapshot_subparsers.add_parser("create", help="create a snapshot from SQLite measurements")
    snapshot_create.add_argument("--db", required=True, help="source SQLite database path")
    snapshot_create.add_argument("--hours", type=int, required=True)
    snapshot_create.add_argument("--metrics", required=True, help="comma-separated metric names")
    snapshot_create.add_argument("--max-lag-seconds", type=int, required=True)

    snapshot_synthetic = snapshot_subparsers.add_parser(
        "synthetic",
        help="generate a deterministic synthetic scenario database and sealed snapshot",
    )
    snapshot_synthetic.add_argument("--scenario", required=True, choices=sorted(SCENARIO_DESCRIPTIONS))
    snapshot_synthetic.add_argument("--db", help="optional output database path")
    snapshot_synthetic.add_argument("--hours", type=int, default=240)
    snapshot_synthetic.add_argument(
        "--duration-hours",
        type=float,
        default=DEFAULT_SYNTHETIC_HOURS,
        help="synthetic generation duration",
    )
    snapshot_synthetic.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=DEFAULT_SYNTHETIC_SAMPLE_INTERVAL_SECONDS,
    )
    snapshot_synthetic.add_argument("--noise", type=float, default=0.6)
    snapshot_synthetic.add_argument("--seed", type=int, default=DEFAULT_SEED)
    snapshot_synthetic.add_argument(
        "--max-lag-seconds",
        type=int,
        default=DEFAULT_SYNTHETIC_MAX_LAG_SECONDS,
    )

    snapshot_inspect = snapshot_subparsers.add_parser("inspect", help="inspect a snapshot without blind values")
    snapshot_inspect.add_argument("snapshot_id")

    hypothesis = subparsers.add_parser("hypothesis", help="validate manual hypotheses")
    hypothesis_subparsers = hypothesis.add_subparsers(dest="hypothesis_command", required=True)
    hypothesis_validate = hypothesis_subparsers.add_parser("validate")
    hypothesis_validate.add_argument("hypothesis_file")
    hypothesis_validate.add_argument("--snapshot", help="optionally validate against a snapshot metric catalog")

    fit = subparsers.add_parser("fit", help="fit a validated hypothesis on exploration data")
    fit.add_argument("hypothesis_file")
    fit.add_argument("--snapshot", required=True)

    tune = subparsers.add_parser("tune", help="evaluate a fitted candidate on tuning data")
    tune.add_argument("--run", required=True, dest="run_id")

    imagine = subparsers.add_parser("imagine", help="propose and review hypotheses from exploration only")
    imagine.add_argument("--snapshot", required=True, dest="snapshot_id")
    imagine.add_argument("--provider", required=True, choices=("mock", "file"))
    imagine.add_argument("--provider-file", help="JSON proposal file for --provider file")
    imagine.add_argument("--max-hypotheses", type=int, default=8)
    imagine.add_argument("--seed", type=int, default=DEFAULT_IMAGINATION_SEED)

    review = subparsers.add_parser("review", help="show or approve an imagination run")
    review.add_argument("run_id")
    review.add_argument("--approve", help="proposal index, candidate id, or hypothesis hash to approve")

    fit_approved_parser = subparsers.add_parser(
        "fit-approved",
        help="fit approved imagination proposals on exploration and tune them",
    )
    fit_approved_parser.add_argument("run_id")

    ablate = subparsers.add_parser(
        "ablate",
        help="compare mock LLM hypotheses against baseline generators on synthetic scenarios",
    )
    ablate.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ABLATION_SCENARIOS),
        help="comma-separated synthetic scenarios to evaluate",
    )
    ablate.add_argument("--provider", default="mock", choices=("mock",))
    ablate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ablate.add_argument("--candidate-budget", type=int, default=DEFAULT_ABLATION_CANDIDATE_BUDGET)
    ablate.add_argument("--duration-hours", type=float, default=DEFAULT_ABLATION_DURATION_HOURS)
    ablate.add_argument("--snapshot-hours", type=int, default=DEFAULT_ABLATION_SNAPSHOT_HOURS)
    ablate.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=DEFAULT_SYNTHETIC_SAMPLE_INTERVAL_SECONDS,
    )
    ablate.add_argument("--noise", type=float, default=DEFAULT_SYNTHETIC_NOISE)
    ablate.add_argument("--max-lag-seconds", type=int, default=DEFAULT_SYNTHETIC_MAX_LAG_SECONDS)

    preregister = subparsers.add_parser("preregister", help="freeze a tuned candidate before blind access")
    preregister.add_argument("--candidate", required=True, dest="candidate_id")

    blind = subparsers.add_parser("blind-evaluate", help="spend one sealed blind evaluation")
    blind.add_argument("preregistration_id")

    report = subparsers.add_parser("report", help="write and record a verdict report")
    report.add_argument("preregistration_id")
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any] | None:
    artifact_root = Path(args.artifact_root)
    ledger_path = Path(args.ledger)
    if args.command == "snapshot":
        if args.snapshot_command == "create":
            manifest = create_snapshot(
                db_path=Path(args.db),
                hours=args.hours,
                metrics=parse_metric_csv(args.metrics),
                max_lag_seconds=args.max_lag_seconds,
                artifact_root=artifact_root,
                ledger_path=ledger_path,
            )
            return {"snapshot_id": manifest["snapshot_id"], "manifest": manifest}
        if args.snapshot_command == "synthetic":
            db_path = Path(args.db) if args.db else _default_synthetic_db_path(artifact_root, args.scenario)
            metadata = _write_synthetic_database(
                scenario=args.scenario,
                db_path=db_path,
                duration_hours=args.duration_hours,
                sample_interval_seconds=args.sample_interval_seconds,
                noise=args.noise,
                seed=args.seed,
            )
            manifest = create_snapshot(
                db_path=db_path,
                hours=args.hours,
                metrics=["control", "x", "y"],
                max_lag_seconds=args.max_lag_seconds,
                artifact_root=artifact_root,
                ledger_path=ledger_path,
            )
            return {
                "snapshot_id": manifest["snapshot_id"],
                "scenario": args.scenario,
                "source_database_path": str(db_path.resolve()),
                "synthetic_metadata": metadata,
                "manifest": manifest,
            }
        if args.snapshot_command == "inspect":
            return snapshot_summary(args.snapshot_id, artifact_root=artifact_root)
    if args.command == "hypothesis" and args.hypothesis_command == "validate":
        manifest = (
            load_snapshot_manifest(args.snapshot, artifact_root=artifact_root)
            if args.snapshot
            else None
        )
        hypothesis = _load_hypothesis(args.hypothesis_file, snapshot_manifest=manifest)
        return _record_hypothesis_proposal(hypothesis, artifact_root, ledger_path)
    if args.command == "fit":
        manifest = load_snapshot_manifest(args.snapshot, artifact_root=artifact_root)
        hypothesis = _load_hypothesis(args.hypothesis_file, snapshot_manifest=manifest)
        proposal = _candidate_artifact_payload(hypothesis)
        candidate_artifact = _store_json_artifact(artifact_root, proposal)
        exploration = load_exploration_view(args.snapshot, artifact_root=artifact_root)
        frame = frame_from_snapshot_rows(exploration["rows"])
        result = fit_hypothesis(hypothesis, frame)
        fit_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "manual_fit_result",
            "cli_version": CLI_VERSION,
            "evaluator_version": FITTING_VERSION,
            "snapshot_id": args.snapshot,
            "candidate_id": candidate_artifact["sha256"],
            "hypothesis_hash": hypothesis.hypothesis_hash(),
            "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
            "fit_result": asdict(result),
            "code_commit": current_code_commit(),
        }
        fit_artifact = _store_json_artifact(artifact_root, fit_payload)
        append_event(
            "fit_completed",
            {
                "run_id": fit_artifact["sha256"],
                "candidate_id": candidate_artifact["sha256"],
                "dataset_snapshot_id": args.snapshot,
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "evaluator_version": FITTING_VERSION,
                "random_seed": hypothesis.random_seed,
                "fitted_parameters": result.fitted_parameters,
                "metrics": result.exploration_metrics,
                "baseline_metrics": result.baseline_metrics,
                "warnings": result.warnings,
                "artifact_root": str(artifact_root.resolve()),
                "artifacts": {"fit_result": fit_artifact, "hypothesis": candidate_artifact},
            },
            artifact_hashes={
                "fit_result": fit_artifact["sha256"],
                "hypothesis": candidate_artifact["sha256"],
            },
            code_commit=current_code_commit(),
            ledger_path=ledger_path,
        )
        return {
            "run_id": fit_artifact["sha256"],
            "candidate_id": candidate_artifact["sha256"],
            "hypothesis_hash": hypothesis.hypothesis_hash(),
            "snapshot_id": args.snapshot,
            "fitted_parameters": result.fitted_parameters,
            "exploration_metrics": result.exploration_metrics,
        }
    if args.command == "tune":
        fit_record = _load_artifact_by_hash(artifact_root, args.run_id, "manual_fit_result")
        candidate = {
            "candidate_id": fit_record["candidate_id"],
            "hypothesis": fit_record["hypothesis"],
            "fitted_parameters": fit_record["fit_result"]["fitted_parameters"],
            "fit_result": {"fit_result_id": args.run_id},
        }
        selection = select_candidate(
            fit_record["snapshot_id"],
            [candidate],
            artifact_root=artifact_root,
            record_artifact=True,
        )
        if selection.artifact is None:
            raise ScienceCliError("selection did not produce an artifact")
        append_event(
            "result_interpreted",
            {
                "interpretation_type": "tuning_selection",
                "run_id": args.run_id,
                "candidate_id": fit_record["candidate_id"],
                "dataset_snapshot_id": fit_record["snapshot_id"],
                "hypothesis_hash": fit_record["hypothesis_hash"],
                "evaluator_version": SELECTION_VERSION,
                "random_seed": fit_record["hypothesis"]["random_seed"],
                "selected_candidate_id": selection.selected_candidate_id,
                "selected_hypothesis_hash": selection.selected_hypothesis_hash,
                "evaluations": [evaluation.to_dict() for evaluation in selection.evaluations],
                "warnings": list(selection.warnings),
                "artifact_root": str(artifact_root.resolve()),
                "artifacts": {"tuning_selection": selection.artifact},
            },
            artifact_hashes={"tuning_selection": selection.artifact["sha256"]},
            code_commit=current_code_commit(),
            ledger_path=ledger_path,
        )
        return selection.to_dict()
    if args.command == "imagine":
        return imagine_hypotheses(
            snapshot_id=args.snapshot_id,
            provider_name=args.provider,
            max_hypotheses=args.max_hypotheses,
            seed=args.seed,
            provider_file=args.provider_file,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )
    if args.command == "review":
        return review_imagination_run(
            args.run_id,
            approve=args.approve,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )
    if args.command == "fit-approved":
        return fit_approved(
            args.run_id,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )
    if args.command == "ablate":
        return run_ablation(
            scenarios=parse_metric_csv(args.scenarios),
            provider_name=args.provider,
            seed=args.seed,
            candidate_budget=args.candidate_budget,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
            duration_hours=args.duration_hours,
            snapshot_hours=args.snapshot_hours,
            sample_interval_seconds=args.sample_interval_seconds,
            max_lag_seconds=args.max_lag_seconds,
            noise=args.noise,
        )
    if args.command == "preregister":
        return _preregister_candidate(args.candidate_id, artifact_root, ledger_path)
    if args.command == "blind-evaluate":
        preregistration = _load_preregistration_from_ledger(
            args.preregistration_id,
            artifact_root,
            ledger_path,
        )
        result = evaluate_preregistration(
            preregistration,
            args.preregistration_id,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )
        return result.to_dict()
    if args.command == "report":
        return _report_result(args.preregistration_id, artifact_root, ledger_path)
    raise ScienceCliError(f"unknown command: {args.command}")


def _record_hypothesis_proposal(
    hypothesis: HypothesisSpec,
    artifact_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    payload = _candidate_artifact_payload(hypothesis)
    artifact = _store_json_artifact(artifact_root, payload)
    append_event(
        "hypothesis_proposed",
        {
            "candidate_id": artifact["sha256"],
            "hypothesis_hash": hypothesis.hypothesis_hash(),
            "origin": hypothesis.origin.value,
            "title": hypothesis.title,
            "target_metric": hypothesis.target_metric,
            "input_metrics": list(hypothesis.input_metrics),
            "maximum_lag_seconds": hypothesis.maximum_lag_seconds,
            "random_seed": hypothesis.random_seed,
            "llm_used": False,
            "arbitrary_generated_code_executable": False,
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {"hypothesis": artifact},
        },
        artifact_hashes={"hypothesis": artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return {
        "candidate_id": artifact["sha256"],
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "artifact": artifact,
        "llm_used": False,
        "arbitrary_generated_code_executable": False,
    }


def _preregister_candidate(
    candidate_id: str,
    artifact_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    fit_entry = _latest_fit_entry(candidate_id, ledger_path)
    if fit_entry is None:
        raise ScienceCliError(f"no fit_completed ledger entry found for candidate {candidate_id}")
    fit_artifact = fit_entry["payload"]["artifacts"]["fit_result"]
    fit_record = _load_artifact_by_hash(artifact_root, fit_artifact["sha256"], "manual_fit_result")
    manifest = load_snapshot_manifest(fit_record["snapshot_id"], artifact_root=artifact_root)
    baseline_metrics = _baseline_metrics_for_preregistration(candidate_id, ledger_path)
    preregistration = create_preregistration(
        hypothesis=fit_record["hypothesis"],
        snapshot_manifest=manifest,
        fitted_parameters=fit_record["fit_result"]["fitted_parameters"],
        baseline_metrics=baseline_metrics,
        transform_config={
            "minimum_observations": 20,
            "minimum_coverage": 0.8,
            "window_count": 3,
        },
    )
    prereg_payload = preregistration.to_dict()
    prereg_payload["preregistration_hash"] = preregistration.preregistration_hash()
    prereg_payload["record_type"] = "manual_preregistration"
    prereg_payload["cli_version"] = CLI_VERSION
    prereg_artifact = _store_json_artifact(artifact_root, prereg_payload)
    append_event(
        "hypothesis_preregistered",
        {
            **preregistration.to_dict(),
            "preregistration_hash": preregistration.preregistration_hash(),
            "candidate_id": candidate_id,
            "fit_run_id": fit_artifact["sha256"],
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {"preregistration": prereg_artifact},
        },
        artifact_hashes={
            "preregistration": prereg_artifact["sha256"],
            "snapshot_blind": preregistration.snapshot_artifacts["blind"]["sha256"],
        },
        code_commit=preregistration.evaluator_code_commit,
        ledger_path=ledger_path,
    )
    return {
        "preregistration_id": preregistration.preregistration_hash(),
        "candidate_id": candidate_id,
        "snapshot_id": preregistration.snapshot_id,
        "hypothesis_hash": preregistration.hypothesis_hash,
        "baseline_metrics": baseline_metrics,
        "artifact": prereg_artifact,
    }


def _report_result(
    preregistration_id: str,
    artifact_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    prereg_entry = _find_preregistration_entry(preregistration_id, ledger_path)
    if prereg_entry is None:
        raise ScienceCliError(f"unknown preregistration id: {preregistration_id}")
    evaluation_entry = _find_evaluation_entry(preregistration_id, ledger_path)
    if evaluation_entry is None:
        raise ScienceCliError("cannot report before blind evaluation is complete")
    snapshot_id = prereg_entry["payload"]["snapshot_id"]
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=artifact_root)
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_science_report",
        "cli_version": CLI_VERSION,
        "preregistration_id": preregistration_id,
        "status": evaluation_entry["payload"]["status"],
        "claim_language": _claim_language(evaluation_entry["payload"]["status"]),
        "snapshot_id": snapshot_id,
        "snapshot_time_range_utc": manifest["time_range_utc"],
        "snapshot_git_commit": manifest.get("git_commit"),
        "hypothesis_hash": prereg_entry["payload"]["hypothesis_hash"],
        "evaluator_version": evaluation_entry["payload"]["evaluator_version"],
        "preregistered_code_commit": prereg_entry["payload"]["evaluator_code_commit"],
        "evaluation_code_commit": evaluation_entry["code_commit"],
        "report_code_commit": current_code_commit(),
        "metrics": evaluation_entry["payload"]["metrics"],
        "warnings": evaluation_entry["payload"]["warnings"],
        "artifact_hashes": {
            "preregistration": prereg_entry["artifact_hashes"].get("preregistration"),
            "blind_evaluation": evaluation_entry["artifact_hashes"].get("blind_evaluation_metrics"),
        },
        "raw_blind_values_exposed": False,
        "llm_used": False,
        "arbitrary_generated_code_executable": False,
    }
    report_artifact = _store_json_artifact(artifact_root, report)
    append_event(
        "result_interpreted",
        {
            "interpretation_type": "blind_verdict_report",
            "preregistration_hash": preregistration_id,
            "dataset_snapshot_id": snapshot_id,
            "hypothesis_hash": report["hypothesis_hash"],
            "evaluator_version": BLIND_EVALUATOR_VERSION,
            "status": report["status"],
            "claim_language": report["claim_language"],
            "random_seed": prereg_entry["payload"]["random_seed"],
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {"report": report_artifact},
        },
        artifact_hashes={"report": report_artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return {**report, "report_id": report_artifact["sha256"], "artifact": report_artifact}


def _load_hypothesis(
    path: str | Path,
    *,
    snapshot_manifest: Mapping[str, Any] | None = None,
) -> HypothesisSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HypothesisSpec.model_validate(
        payload,
        context={"metric_catalog": snapshot_manifest.get("metric_catalog")} if snapshot_manifest else None,
    )


def _candidate_artifact_payload(hypothesis: HypothesisSpec) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "manual_hypothesis_candidate",
        "cli_version": CLI_VERSION,
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
        "llm_used": False,
        "arbitrary_generated_code_executable": False,
    }


def _baseline_metrics_for_preregistration(candidate_id: str, ledger_path: Path) -> dict[str, float]:
    tuning_entry = _latest_tuning_entry(candidate_id, ledger_path)
    if tuning_entry is None:
        raise ScienceCliError(f"no tuning result found for candidate {candidate_id}")
    evaluations = tuning_entry["payload"].get("evaluations") or []
    evaluation = next(
        (item for item in evaluations if item.get("candidate_id") == candidate_id),
        None,
    )
    if evaluation is None:
        raise ScienceCliError(f"candidate {candidate_id} not found in latest tuning result")
    mae = _first_number(evaluation, "persistence_baseline_mae", "zero_baseline_mae")
    rmse = _first_number(evaluation, "persistence_baseline_rmse", "zero_baseline_rmse")
    if mae is None or rmse is None:
        raise ScienceCliError("tuning result lacks baseline metrics for preregistration")
    return {"mae": mae, "rmse": rmse}


def _latest_fit_entry(candidate_id: str, ledger_path: Path) -> dict[str, Any] | None:
    for entry in reversed(read_entries(ledger_path)):
        if entry["event_type"] == "fit_completed" and entry["payload"].get("candidate_id") == candidate_id:
            return entry
    return None


def _latest_tuning_entry(candidate_id: str, ledger_path: Path) -> dict[str, Any] | None:
    for entry in reversed(read_entries(ledger_path)):
        if entry["event_type"] != "result_interpreted":
            continue
        payload = entry["payload"]
        if (
            payload.get("interpretation_type") in {"tuning_selection", "imagination_tuning_selection"}
            and payload.get("candidate_id") == candidate_id
        ):
            return entry
    return None


def _find_preregistration_entry(preregistration_id: str, ledger_path: Path) -> dict[str, Any] | None:
    for entry in read_entries(ledger_path):
        if (
            entry["event_type"] == "hypothesis_preregistered"
            and entry["payload"].get("preregistration_hash") == preregistration_id
        ):
            return entry
    return None


def _find_evaluation_entry(preregistration_id: str, ledger_path: Path) -> dict[str, Any] | None:
    for entry in read_entries(ledger_path):
        if (
            entry["event_type"] == "blind_evaluation_completed"
            and entry["payload"].get("preregistration_hash") == preregistration_id
        ):
            return entry
    return None


def _load_preregistration_from_ledger(
    preregistration_id: str,
    artifact_root: Path,
    ledger_path: Path,
):
    entry = _find_preregistration_entry(preregistration_id, ledger_path)
    if entry is None:
        raise ScienceCliError(f"unknown preregistration id: {preregistration_id}")
    artifact = entry["payload"]["artifacts"]["preregistration"]
    payload = _load_artifact_by_hash(artifact_root, artifact["sha256"], "manual_preregistration")
    payload.pop("record_type", None)
    payload.pop("cli_version", None)
    payload.pop("preregistration_hash", None)
    return load_preregistration(payload)


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ScienceCliError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _load_artifact_by_hash(root: Path, digest: str, expected_record_type: str) -> dict[str, Any]:
    if not _is_hash(digest):
        raise ScienceCliError(f"artifact id must be a 64-character hash: {digest}")
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_hash(payload) != digest:
        raise ScienceCliError(f"artifact hash mismatch: {path}")
    record_type = payload.get("record_type")
    if record_type != expected_record_type:
        raise ScienceCliError(f"expected {expected_record_type}, found {record_type!r}")
    return payload


def _write_synthetic_database(
    *,
    scenario: str,
    db_path: Path,
    duration_hours: float,
    sample_interval_seconds: int,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    dataset = generate_synthetic_series(
        scenario,
        sample_interval_seconds=sample_interval_seconds,
        duration_hours=duration_hours,
        noise=noise,
        seed=seed,
    )
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    control_rng = random.Random(seed + 9173)
    measurements: list[Measurement] = []
    for index, sample in enumerate(dataset.samples):
        if sample.x is not None and math.isfinite(sample.x):
            measurements.append(Measurement(sample.timestamp_utc, "x", sample.x, "synthetic", scenario))
        if sample.y is not None and math.isfinite(sample.y):
            measurements.append(Measurement(sample.timestamp_utc, "y", sample.y, "synthetic", scenario))
        control = 1.8 * math.sin(index * 1.913 + 0.2) + control_rng.gauss(0.0, max(noise, 0.2))
        measurements.append(Measurement(sample.timestamp_utc, "control", control, "synthetic", scenario))
    insert_measurements(conn, measurements)
    conn.close()
    return {
        **dataset.metadata,
        "database_path": str(db_path.resolve()),
        "inserted_measurement_count": len(measurements),
        "control_metric_seed": seed + 9173,
    }


def _default_synthetic_db_path(artifact_root: Path, scenario: str) -> Path:
    return artifact_root.parent / "synthetic" / f"{scenario}.db"


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _claim_language(status: str) -> str:
    if status == "pass":
        return "The preregistered predictor predicts in this dataset."
    if status == "fail":
        return "The preregistered predictor did not pass the blind evaluation."
    return "The preregistered blind evaluation is inconclusive."


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


if __name__ == "__main__":
    raise SystemExit(main())
