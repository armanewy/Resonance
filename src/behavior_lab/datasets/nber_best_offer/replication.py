from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from behavior_lab.datasets.nber_best_offer.source_schema import default_mapping_path, load_real_mapping, repo_root, sha256_file


def default_targets_path() -> Path:
    return repo_root() / "datasets" / "manifests" / "nber_replication_targets.yaml"


def load_replication_targets(path: str | Path | None = None) -> dict[str, Any]:
    target_path = Path(path) if path is not None else default_targets_path()
    return json.loads(target_path.read_text(encoding="utf-8"))


def validate_replication_targets(path: str | Path | None = None) -> dict[str, Any]:
    targets = load_replication_targets(path)
    all_targets = _flatten_targets(targets)
    errors = []
    ids = set()
    for target in all_targets:
        target_id = target.get("id")
        if not target_id:
            errors.append("target missing id")
            continue
        if target_id in ids:
            errors.append(f"duplicate target id {target_id}")
        ids.add(target_id)
        for key in ["formula", "tolerance"]:
            if key not in target:
                errors.append(f"{target_id} missing {key}")
        if "fatal" not in target and "status" not in target:
            errors.append(f"{target_id} missing fatal/status")
        if "source" not in target and "source_refs" not in target:
            errors.append(f"{target_id} missing source/source_refs")
    level_counts: dict[str, int] = {}
    for target in all_targets:
        level = str(target.get("level", "unknown"))
        level_counts[level] = level_counts.get(level, 0) + 1
    return {
        "valid": not errors,
        "errors": errors,
        "target_count": len(all_targets),
        "level_counts": level_counts,
        "targets_hash": sha256_file(Path(path) if path is not None else default_targets_path()),
    }


def replication_check(normalized_dir: str | Path, targets_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(normalized_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing normalized manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = load_replication_targets(targets_path)
    mapping = load_real_mapping(default_mapping_path())
    results = []
    structural_summary = _structural_summary(manifest)
    for target in _flatten_targets(targets):
        target_id = target["id"]
        if target_id == "headers_lists_exact":
            passed = manifest.get("header_validation", {}).get("files", {}).get("anon_bo_lists.csv", {}).get("valid") is True
            results.append(_result(target, passed=passed, observed=passed))
        elif target_id == "headers_threads_exact":
            passed = manifest.get("header_validation", {}).get("files", {}).get("anon_bo_threads.csv", {}).get("valid") is True
            results.append(_result(target, passed=passed, observed=passed))
        elif target_id == "status_codes_known":
            known = set(mapping["code_maps"]["status_id"])
            observed = set(manifest.get("source_thread_pass", {}).get("status_counts", {}).keys())
            passed = not observed or observed <= known
            results.append(_result(target, passed=passed, observed=sorted(observed)))
        elif target_id == "offer_type_codes_known":
            known = set(mapping["code_maps"]["offr_type_id"])
            observed = set(manifest.get("source_thread_pass", {}).get("offer_type_counts", {}).keys())
            passed = not observed or observed <= known
            results.append(_result(target, passed=passed, observed=sorted(observed)))
        elif target_id == "thread_rows_have_thread_listing_buyer_seller":
            missing = manifest.get("quarantine", {}).get("counts", {}).get("missing_required_thread_identifier", 0)
            results.append(_result(target, passed=missing == 0, observed=missing))
        elif target_id.startswith("struct_"):
            observed = structural_summary.get(target_id, "not_evaluated_on_current_sample")
            results.append(_result(target, passed=None, observed=observed))
        else:
            results.append(_result(target, passed=None, observed="not_evaluated_on_current_sample"))
    fatal_failures = [item for item in results if item["fatal"] and item["passed"] is False]
    fatal_unevaluated = [item for item in results if item["fatal"] and item["passed"] is None]
    bounded_structure_passed = not fatal_failures
    full_replication_passed = not fatal_failures and not fatal_unevaluated
    return {
        "schema_version": "nber_replication_check.v1",
        "normalized_dir": str(root.resolve()),
        "manifest_hash": sha256_file(manifest_path),
        "targets_hash": sha256_file(Path(targets_path) if targets_path is not None else default_targets_path()),
        "results": results,
        "fatal_failures": fatal_failures,
        "fatal_unevaluated": fatal_unevaluated,
        "bounded_structure_passed": bounded_structure_passed,
        "full_replication_passed": full_replication_passed,
        "passed": full_replication_passed,
        "limitations": [
            "Published descriptive moments require the full official source and authors' sample restrictions.",
            "Sample-limited runs can validate structure and lineage but not published aggregate values.",
        ],
    }


def _flatten_targets(targets: dict[str, Any]) -> list[dict[str, Any]]:
    if "targets" in targets:
        return [dict(item) for item in targets["targets"]]
    rows = []
    for level, items in targets.get("levels", {}).items():
        for item in items:
            row = dict(item)
            row["level"] = level
            rows.append(row)
    return rows


def _result(target: dict[str, Any], *, passed: bool | None, observed: Any) -> dict[str, Any]:
    if passed is True:
        evaluation_status = "passed"
    elif passed is False:
        evaluation_status = "failed"
    else:
        evaluation_status = "not_evaluated"
    return {
        "id": target["id"],
        "level": target.get("level"),
        "fatal": _is_fatal(target),
        "passed": passed,
        "evaluation_status": evaluation_status,
        "observed": observed,
        "expected": target.get("expected", target.get("expected_value", target.get("formula"))),
        "tolerance": target.get("tolerance"),
    }


def _is_fatal(target: dict[str, Any]) -> bool:
    if "fatal" in target:
        return bool(target["fatal"])
    return str(target.get("status", "")).lower() == "fatal"


def _structural_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "struct_raw_listings_before_restrictions": manifest.get("source_inventory", {}).get("anon_bo_lists", {}).get("rows"),
        "struct_main_sample_listings_after_restrictions": "requires paper_sample restrictions on full source",
        "struct_l1_price_over_1000_exclusions": "requires full listing source",
        "struct_l2_sale_price_above_listing_exclusions": "requires full listing source",
        "struct_t1_offer_above_listing_exclusions": "requires full listing-thread join",
        "struct_t2_offer_limit_exclusions": "requires full thread grouping",
        "struct_t3_t4_sequence_integrity_exclusions": "requires full thread grouping",
    }
