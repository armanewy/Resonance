from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from behavior_lab import __version__
from behavior_lab.benchmarks.splits import chronological_group_purged_split, group_disjoint_split
from behavior_lab.core import stable_hash, utc_now
from behavior_lab.data_sources.registry import default_registry
from behavior_lab.datasets.nber_best_offer.audit import audit as nber_audit
from behavior_lab.datasets.nber_best_offer.audit import benchmark as nber_benchmark
from behavior_lab.datasets.nber_best_offer.tasks import build_tasks
from behavior_lab.offerlab_models.formulas.formulas import evaluate_formula_candidates
from behavior_lab.offerlab_models.predictive.models import predictive_suite


DEFAULT_PROTOCOL = Path("datasets/manifests/offerlab_benchmark_v1.yaml")
DEFAULT_OUTPUT = Path("reports/offerlab_benchmark_v1.json")
DEFAULT_DOC = Path("docs/runs/OFFERLAB_BENCHMARK_V1_RESULTS.md")
DEFAULT_MODEL_CARDS = Path("docs/model_cards/offerlab_benchmark_v1")
DEFAULT_ROW_CAP = 500
EXECUTED_PROTOCOL_SPLITS = ["chronological_listing_purged", "seller_disjoint"]
OMITTED_PROTOCOL_SPLITS = ["buyer_disjoint", "category_disjoint_diagnostic", "thread_safe_nested_development"]
NEGATIVE_CONTROL_PROTOCOL_MAP = {
    "random_label_permutation": "random_labels",
    "random_row_split": "random_row_split_inflation",
    "same_timestamp_ordering": "same_timestamp_ordering_perturbation",
    "artifact_name_canary": "artifact_name_leakage_canary",
}


@dataclass(frozen=True)
class BenchmarkPaths:
    normalized_dir: Path
    output_path: Path = DEFAULT_OUTPUT
    doc_path: Path = DEFAULT_DOC
    model_cards_dir: Path = DEFAULT_MODEL_CARDS
    protocol_path: Path = DEFAULT_PROTOCOL
    lockbox_store_path: Path | None = None


def run_offerlab_benchmark_v1(
    paths: BenchmarkPaths,
    *,
    row_cap: int = DEFAULT_ROW_CAP,
    seed: int = 20240621,
) -> dict[str, Any]:
    if row_cap <= 0:
        raise ValueError("row_cap must be positive")
    normalized_dir = Path(paths.normalized_dir)
    manifest = _read_json(normalized_dir / "manifest.json")
    protocol = _read_json(paths.protocol_path)
    audit = nber_audit(normalized_dir)
    _require_nber_audit_passes(audit)
    if paths.lockbox_store_path is None:
        raise ValueError("lockbox_store_path is required for Benchmark v1 hidden submissions")
    lockbox_store = paths.lockbox_store_path
    _validate_lockbox_store(lockbox_store, manifest=manifest, protocol=protocol)
    lockbox_store.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(normalized_dir)
    baseline = nber_benchmark(normalized_dir)
    registry = default_registry()
    permission_report = {
        "production_export": registry.verify_lineage(["nber_ebay_best_offer"], "production_export"),
        "commercial_training": registry.verify_lineage(["nber_ebay_best_offer"], "commercial_training"),
    }
    targets: dict[str, Any] = {}
    model_cards: dict[str, str] = {}
    for target in protocol["targets"]:
        rows = list(tasks.get(target, []))
        sampled = _deterministic_sample(rows, cap=row_cap, seed=seed, target=target)
        chronological = chronological_group_purged_split(sampled, time_key="timestamp", group_key="listing_id") if sampled else None
        seller_disjoint = group_disjoint_split(sampled, group_key="seller_id") if sampled else None
        chronological_report = (
            predictive_suite(
                target,
                chronological.train,
                chronological.development,
                chronological.hidden,
                hidden_lockbox_id=f"offerlab_benchmark_v1:{target}",
                hidden_lockbox_store_path=lockbox_store,
            )
            if chronological and chronological.train and chronological.development
            else _empty_predictive_report(target, "chronological")
        )
        _attach_protocol_negative_control_summary(chronological_report, protocol)
        seller_report = (
            predictive_suite(
                target,
                seller_disjoint.train,
                seller_disjoint.development,
                seller_disjoint.hidden,
            )
            if seller_disjoint and seller_disjoint.train and seller_disjoint.development
            else _empty_predictive_report(target, "seller_disjoint")
        )
        formula_report = None
        if target == "seller_next_action" and chronological and chronological.train and chronological.development:
            formula_report = evaluate_formula_candidates(
                chronological.train,
                chronological.development,
                chronological.hidden,
                black_box_model_id=_selected_model_id(chronological_report),
                black_box_hidden_loss=None,
            )
        targets[target] = {
            "total_rows": len(rows),
            "model_row_cap": row_cap,
            "sampled_rows": len(sampled),
            "chronological_split": chronological.audit() if chronological else None,
            "seller_disjoint_split": seller_disjoint.audit() if seller_disjoint else None,
            "chronological": _public_report(chronological_report),
            "seller_disjoint": _public_report(seller_report),
            "formula_hypotheses": formula_report,
            "answers": _target_answers(chronological_report, seller_report, formula_report),
        }

    report = {
        "schema_version": "offerlab_benchmark_v1_result.v1",
        "benchmark_id": "offerlab_benchmark_v1",
        "generated_at": utc_now(),
        "software_version": __version__,
        "git_commit": _git_commit(),
        "research_only": True,
        "production_export_allowed": False,
        "causal_claim": False,
        "universal_aggregate_score": None,
        "source_dataset_ids": ["nber_ebay_best_offer"],
        "protocol": {
            "path": str(paths.protocol_path.as_posix()),
            "hash": stable_hash(protocol),
            "status": protocol.get("status"),
        },
        "data": _data_summary(normalized_dir, manifest),
        "scope": {
            "full_release_evidence": False,
            "evidence_scope": "bounded_smoke_or_semantics",
            "full_normalization_status": "blocked",
            "baseline_scope": "bounded_100k_normalized_sample",
            "model_scope": f"deterministic_row_cap_{row_cap}_per_target",
            "protocol_splits_complete": False,
            "executed_protocol_splits": EXECUTED_PROTOCOL_SPLITS,
            "omitted_protocol_splits": OMITTED_PROTOCOL_SPLITS,
            "hidden_lockbox_store_retained_outside_repo": _lockbox_store_retained_outside_repo(lockbox_store),
            "canonical_lockbox_store_name": _canonical_lockbox_store_name(manifest, protocol),
            "hidden_lockbox_store_event_count": _line_count(lockbox_store) if lockbox_store.exists() else 0,
        },
        "nber_audit": {
            "leakage_checks": audit["leakage_checks"],
            "split_checks": audit["split_checks"],
        },
        "full_bounded_baselines": baseline,
        "targets": targets,
        "permission_report": permission_report,
        "gate": _decision_gate(targets, baseline),
        "limitations": [
            "This run uses the 100,000-thread bounded normalization, not the full NBER release.",
            "Inspectable model runs use a deterministic per-target row cap because the current full model suite is not yet optimized for hundreds of thousands of rows.",
            "Hidden submissions are one-shot within the recorded lockbox store, but this is not a remote third-party lockbox.",
            "No production model is exported and no causal seller-profit claim is made.",
        ],
    }
    for target, payload in targets.items():
        card_path = paths.model_cards_dir / f"{target}.md"
        model_cards[target] = _write_model_card(
            card_path,
            target,
            payload,
            permission_report,
            gate=report["gate"],
            scope=report["scope"],
        )
    report["model_cards"] = model_cards
    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    paths.output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(paths.doc_path, report)
    return report


def _empty_predictive_report(target: str, reason: str) -> dict[str, Any]:
    return {
        "task": target,
        "submitted": False,
        "leaderboards": {"development": [], "hidden": []},
        "hidden_lockbox": {"submitted": False, "reason": f"insufficient rows for {reason}"},
    }


def _deterministic_sample(rows: list[dict[str, Any]], *, cap: int, seed: int, target: str) -> list[dict[str, Any]]:
    if len(rows) <= cap:
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: stable_hash(
            {
                "seed": seed,
                "target": target,
                "row_id": row.get("row_id"),
                "timestamp": row.get("timestamp"),
                "listing_id": row.get("listing_id"),
            }
        ),
    )
    return sorted(ranked[:cap], key=lambda row: (str(row.get("timestamp", "")), str(row.get("row_id", ""))))


def _target_answers(chronological: dict[str, Any], seller: dict[str, Any], formula: dict[str, Any] | None) -> dict[str, Any]:
    hidden = chronological.get("leaderboards", {}).get("hidden", [])
    development = chronological.get("leaderboards", {}).get("development", [])
    seller_dev = seller.get("leaderboards", {}).get("development", [])
    hidden_lockbox = chronological.get("hidden_lockbox", {})
    selected_hidden = _selected_hidden_row(chronological)
    support_coverage = selected_hidden.get("coverage") if selected_hidden else None
    return {
        "hidden_submission_succeeded": bool(hidden_lockbox.get("submitted")),
        "hidden_relative_improvement": _best_relative_improvement(hidden),
        "beats_strong_simple_baseline": _best_relative_improvement(hidden or development),
        "survives_seller_disjoint": _best_relative_improvement(seller_dev),
        "calibration_reported": _has_calibration(hidden or development),
        "calibration_quality_validated": False,
        "compact_formula_passed_development_falsification": bool(formula and formula.get("chosen_formula_id")),
        "abstention_reported": bool((hidden or development) and "abstention" in (hidden or development)[0]),
        "selected_hidden_support_coverage": support_coverage,
        "hidden_support_coverage_at_least_80pct": support_coverage is not None and float(support_coverage) >= 0.80,
        "negative_controls_present": bool(chronological.get("negative_controls")),
        "negative_controls_passed": _negative_controls_passed(chronological),
    }


def _attach_protocol_negative_control_summary(report: dict[str, Any], protocol: dict[str, Any]) -> None:
    required = list(protocol.get("negative_controls", []))
    controls = report.setdefault("negative_controls", {})
    executed_protocol_controls: list[str] = []
    for internal_name, protocol_name in NEGATIVE_CONTROL_PROTOCOL_MAP.items():
        payload = controls.get(internal_name)
        if isinstance(payload, dict):
            payload["protocol_control_name"] = protocol_name
            executed_protocol_controls.append(protocol_name)
    missing = [name for name in required if name not in executed_protocol_controls]
    for name in missing:
        controls[name] = {
            "executed": False,
            "passed": False,
            "threshold": "required by frozen Benchmark v1 protocol",
            "reason": "not implemented by this bounded smoke runner",
        }
    report["negative_control_protocol"] = {
        "required": required,
        "executed": executed_protocol_controls,
        "missing": missing,
        "all_required_passed": not missing and all(
            bool(payload.get("passed"))
            for payload in controls.values()
            if isinstance(payload, dict)
        ),
    }


def _public_report(value: Any) -> Any:
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key in {
            "artifact_id",
            "canonical_lockbox_id",
            "hidden_case_set_hash",
            "hidden_case_tokens",
            "hidden_case_tokens_hash",
            "lockbox_id",
            "reservation_event_id",
        }:
            continue
        if key == "abstained_rows" and isinstance(item, list):
            output["abstained_row_count"] = len(item)
            output["abstained_rows_hash"] = stable_hash(item)
        elif key == "features_used" and isinstance(item, list):
            output[key] = _summarize_features(item)
        elif key in {"training_rows_hash", "training_feature_values_hash"}:
            output[key] = item
        elif key in {"predictions", "predictions_redacted"}:
            output[f"{key}_count"] = len(item) if isinstance(item, list) else None
        else:
            output[key] = _public_report(item)
    return output


def _summarize_features(features: list[Any], *, limit: int = 20) -> list[str] | dict[str, Any]:
    cleaned = [str(feature) for feature in features]
    if len(cleaned) <= limit:
        return cleaned
    return {
        "count": len(cleaned),
        "first": cleaned[:limit],
        "hash": stable_hash(cleaned),
    }


def _best_relative_improvement(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return max(float(row.get("relative_improvement") or 0.0) for row in rows)


def _has_calibration(rows: list[dict[str, Any]]) -> bool:
    return any("calibration" in row for row in rows)


def _negative_controls_passed(report: dict[str, Any]) -> bool:
    summary = report.get("negative_control_protocol")
    if isinstance(summary, dict):
        return bool(summary.get("all_required_passed"))
    controls = report.get("negative_controls", {})
    if not isinstance(controls, dict) or not controls:
        return False
    return all(bool(payload.get("passed")) for payload in controls.values() if isinstance(payload, dict))


def _selected_hidden_row(report: dict[str, Any]) -> dict[str, Any] | None:
    selected = _selected_model_id(report)
    hidden = report.get("leaderboards", {}).get("hidden", [])
    return next((row for row in hidden if row.get("model_id") == selected), hidden[0] if hidden else None)


def _selected_model_id(report: dict[str, Any]) -> str | None:
    return report.get("hidden_lockbox", {}).get("selected_model_id")


def _hidden_loss(report: dict[str, Any]) -> float | None:
    hidden = report.get("leaderboards", {}).get("hidden", [])
    if not hidden:
        return None
    selected = _selected_model_id(report)
    row = next((item for item in hidden if item.get("model_id") == selected), hidden[0])
    return row.get("log_loss") or row.get("rmse")


def _decision_gate(targets: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    core = targets.get("seller_next_action", {}).get("answers", {})
    improvement = core.get("beats_strong_simple_baseline")
    predicates = {
        "full_release_evidence": {
            "passed": False,
            "value": False,
            "required_for_non_stop": True,
            "threshold": "full NBER release normalized and evaluated",
        },
        "row_cap_disabled": {
            "passed": False,
            "value": False,
            "required_for_non_stop": True,
            "threshold": "model evidence must cover the full eligible task rows",
        },
        "protocol_splits_complete": {
            "passed": False,
            "value": False,
            "required_for_non_stop": True,
            "threshold": "all frozen Benchmark v1 split diagnostics executed",
        },
        "hidden_submission_succeeded": {
            "passed": bool(core.get("hidden_submission_succeeded")),
            "value": bool(core.get("hidden_submission_succeeded")),
            "required_for_non_stop": True,
            "threshold": "one-shot hidden submission recorded",
        },
        "hidden_improvement_at_least_5pct": {
            "passed": improvement is not None and improvement >= 0.05,
            "value": improvement,
            "required_for_non_stop": True,
            "threshold": 0.05,
        },
        "seller_disjoint_improvement_positive": {
            "passed": core.get("survives_seller_disjoint") is not None and core.get("survives_seller_disjoint") > 0.0,
            "value": core.get("survives_seller_disjoint"),
            "required_for_non_stop": True,
            "threshold": "> 0.0 relative improvement",
        },
        "calibration_reported": {
            "passed": bool(core.get("calibration_reported")),
            "value": bool(core.get("calibration_reported")),
            "required_for_non_stop": False,
            "threshold": "classification calibration payload emitted",
        },
        "calibration_quality_validated": {
            "passed": bool(core.get("calibration_quality_validated")),
            "value": bool(core.get("calibration_quality_validated")),
            "required_for_non_stop": True,
            "threshold": "declared multiclass calibration target and acceptable ECE threshold validated",
        },
        "abstention_reported": {
            "passed": bool(core.get("abstention_reported")),
            "value": bool(core.get("abstention_reported")),
            "required_for_non_stop": False,
            "threshold": "support abstention payload emitted",
        },
        "hidden_support_coverage_at_least_80pct": {
            "passed": bool(core.get("hidden_support_coverage_at_least_80pct")),
            "value": core.get("selected_hidden_support_coverage"),
            "required_for_non_stop": True,
            "threshold": 0.80,
        },
        "negative_controls_passed": {
            "passed": bool(core.get("negative_controls_passed")),
            "value": bool(core.get("negative_controls_passed")),
            "required_for_non_stop": True,
            "threshold": "all negative-control diagnostics explicitly passed",
        },
    }
    failing = [name for name, payload in predicates.items() if payload["required_for_non_stop"] and not payload["passed"]]
    status = "MAYBE" if not failing else "STOP"
    reasons = [f"gate predicate failed: {name}" for name in failing]
    if improvement is not None and improvement >= 0.05:
        reasons.append("classification smoke improvement exists but non-stop predicates still require full-release evidence")
    return {
        "status": status,
        "engineering_threshold": "5% relative hidden log-loss improvement on core classification target",
        "seller_next_action_smoke_improvement": improvement,
        "predicates": predicates,
        "full_bounded_baseline_scope": baseline.get("scope", {}),
        "reasons": reasons,
    }


def _canonical_lockbox_store_name(manifest: dict[str, Any], protocol: dict[str, Any]) -> str:
    return f"hidden_lockbox_offerlab_benchmark_v1_{stable_hash(manifest)[:12]}.jsonl"


def _validate_lockbox_store(path: Path, *, manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = _canonical_lockbox_store_name(manifest, protocol)
    resolved = path.resolve()
    if not _lockbox_store_retained_outside_repo(resolved):
        raise ValueError("lockbox_store_path must be outside the repository worktree")
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved.relative_to(temp_root)
        return
    except ValueError:
        pass
    if path.name == expected:
        return
    raise ValueError(
        "lockbox_store_path must use the canonical benchmark store name "
        f"{expected!r} outside temporary test directories"
    )


def _require_nber_audit_passes(audit: dict[str, Any]) -> None:
    failed_leakage = [
        name for name, passed in audit.get("leakage_checks", {}).items()
        if not passed
    ]
    failed_splits = [
        name for name, passed in audit.get("split_checks", {}).items()
        if not passed
    ]
    if failed_leakage or failed_splits:
        raise ValueError(
            "NBER audit failed before Benchmark v1 model execution: "
            f"leakage={failed_leakage}, splits={failed_splits}"
        )


def _lockbox_store_retained_outside_repo(path: Path) -> bool:
    resolved = path.resolve()
    repo_root = _repo_root()
    try:
        resolved.relative_to(repo_root)
        return False
    except ValueError:
        return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_summary(normalized_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    quarantine = manifest.get("quarantine", {})
    if not isinstance(quarantine, dict):
        quarantine = {}
    return {
        "normalized_dir": _redacted_path(normalized_dir),
        "manifest_hash": stable_hash(manifest),
        "source_hashes": manifest.get("lineage", {}).get("raw_source_hashes", {}),
        "transformation_version": manifest.get("command_args", {}).get("transformation_version") or manifest.get("transformation_version"),
        "normalization_git_commit": manifest.get("git_commit"),
        "random_seed": manifest.get("random_seed"),
        "command_args": {
            key: value
            for key, value in manifest.get("command_args", {}).items()
            if key not in {"raw_dir"}
        },
        "tables": {
            name: {"rows": table.get("rows"), "format": table.get("format")}
            for name, table in manifest.get("tables", {}).items()
        },
        "quarantine_counts": quarantine.get("counts", {}),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OfferLab Benchmark v1 Results",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Git commit: `{report['git_commit']}`",
        "",
        "## Decision",
        "",
        f"Gate status: **{report['gate']['status']}**",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in report["gate"]["reasons"])
    lines.extend(
        [
            "",
            "## Scope",
            "",
            f"- Evidence scope: `{report['scope']['evidence_scope']}`",
            f"- Baseline scope: `{report['scope']['baseline_scope']}`",
            f"- Model scope: `{report['scope']['model_scope']}`",
            f"- Protocol splits complete: `{report['scope']['protocol_splits_complete']}`",
            f"- Production export allowed: `{report['production_export_allowed']}`",
            "",
            "## Target Summary",
            "",
            "| Target | Rows | Sampled | Hidden submitted | Best hidden improvement | Seller-disjoint improvement |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for target, payload in report["targets"].items():
        hidden_submitted = payload["chronological"].get("hidden_lockbox", {}).get("submitted", False)
        answers = payload["answers"]
        lines.append(
            "| {target} | {rows} | {sampled} | {hidden} | {hidden_imp} | {seller_imp} |".format(
                target=target,
                rows=payload["total_rows"],
                sampled=payload["sampled_rows"],
                hidden=hidden_submitted,
                hidden_imp=_fmt(answers.get("beats_strong_simple_baseline")),
                seller_imp=_fmt(answers.get("survives_seller_disjoint")),
            )
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            "- Does any model beat the strongest simple baseline? Bounded smoke only; see per-target relative improvement in the JSON report.",
            "- Does the gain survive seller-disjoint evaluation? Bounded smoke only; not a full evidence gate.",
            "- Does the gain survive a later time block? Chronological bounded baselines ran; full-release later-period evidence is not available.",
            "- Is calibration reported? Calibration payloads are emitted for classification rows; this is not a production calibration claim.",
            "- What variables carry the gain? Feature lists and lineage are in the JSON report and model cards.",
            "- Did compact formulas pass development falsification? Formula hypotheses ran for `seller_next_action` only without a hidden formula submission.",
            "- Where does the model abstain? Abstention reports are emitted per model row.",
            "- How much performance came from identifiers or history fields? Identifier fields are forbidden by the feature contract; frozen Benchmark v1 negative controls remain incomplete and are gate failures.",
            "- Does the result remain after all canary and negative controls? Negative-control diagnostics report pass/fail fields, but full-release confirmation remains blocked.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_card(
    path: Path,
    target: str,
    payload: dict[str, Any],
    permission_report: dict[str, Any],
    *,
    gate: dict[str, Any],
    scope: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = payload["chronological"].get("leaderboards", {}).get("hidden", [])
    selected_model_id = payload["chronological"].get("hidden_lockbox", {}).get("selected_model_id")
    selected = next((row for row in hidden if row.get("model_id") == selected_model_id), hidden[0] if hidden else None)
    lines = [
        f"# OfferLab Benchmark v1 Model Card: {target}",
        "",
        "Research-only NBER-derived artifact. Not production-exportable.",
        "",
        f"- Total target rows: `{payload['total_rows']}`",
        f"- Sampled model rows: `{payload['sampled_rows']}`",
        f"- Hidden submitted: `{payload['chronological'].get('hidden_lockbox', {}).get('submitted', False)}`",
        f"- Overall Benchmark v1 gate: `{gate.get('status')}`",
        f"- Protocol splits complete: `{scope.get('protocol_splits_complete')}`",
        f"- Omitted protocol splits: `{', '.join(scope.get('omitted_protocol_splits', []))}`",
        f"- Missing negative controls: `{', '.join(payload['chronological'].get('negative_control_protocol', {}).get('missing', []))}`",
        f"- Production export permission: `{permission_report['production_export']['allowed']}`",
    ]
    if selected:
        features = selected.get("features_used", [])
        if isinstance(features, dict):
            feature_text = f"{features.get('count')} features, hash {features.get('hash')}"
        else:
            feature_text = ", ".join(features)
        lines.extend(
            [
                f"- Selected model: `{selected.get('model_id')}`",
                f"- Features used: `{feature_text}`",
                f"- Hidden relative improvement vs development-selected baseline: `{_fmt(selected.get('relative_improvement'))}`",
                f"- Hidden support coverage: `{_fmt(selected.get('coverage'))}`",
                f"- Hidden abstention rate: `{_fmt(selected.get('abstention', {}).get('rate'))}`",
                f"- Lineage hash: `{stable_hash(selected.get('lineage', {}))}`",
            ]
        )
    lines.extend(
        [
            "",
            "Limitations:",
            "",
            "- Bounded 100k normalization, not full-release evidence.",
            "- Row-capped inspectable model run.",
            "- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.",
            "- No causal or seller-profit claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.as_posix())


def _redacted_path(path: Path) -> str:
    data_root = Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")).resolve()
    resolved = path.resolve()
    try:
        return "$OFFERLAB_DATA_ROOT/" + resolved.relative_to(data_root).as_posix()
    except ValueError:
        return f"local_path_hash:{stable_hash(str(resolved))[:16]}"


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
