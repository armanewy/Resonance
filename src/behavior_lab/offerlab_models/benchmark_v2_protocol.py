from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class V2ProtocolError(ValueError):
    """Raised when Benchmark v2 protocol gates are not satisfied."""


@dataclass(frozen=True)
class HiddenExclusionReport:
    candidate_hidden_cases: int
    v1_exclusion_cases: int
    status: str


@dataclass(frozen=True)
class PreHiddenReadinessReport:
    targets_checked: int
    negative_controls_checked: int
    status: str


def validate_v2_hidden_exclusion(
    *,
    v2_manifest: dict[str, Any],
    v1_final_manifest: dict[str, Any],
    candidate_hidden_case_tokens: Iterable[str],
    external_v1_hidden_case_tokens: Iterable[str] | None = None,
) -> HiddenExclusionReport:
    """Validate that a proposed v2 hidden set cannot reuse v1 hidden cases."""

    hidden_policy = v2_manifest.get("hidden_policy", {})
    required_policy = {
        "fresh_hidden_lockbox_required",
        "exclude_all_v1_hidden_case_tokens",
        "external_v1_hidden_case_token_artifact_required_if_manifest_tokens_unavailable",
        "block_hidden_creation_if_v1_tokens_unavailable",
        "exclude_v1_hidden_case_tokens_before_sampling",
    }
    missing_or_false = sorted(name for name in required_policy if hidden_policy.get(name) is not True)
    if missing_or_false:
        raise V2ProtocolError(f"Benchmark v2 hidden policy is not fail-closed: {', '.join(missing_or_false)}")
    if hidden_policy.get("protocol_changes_after_hidden_access_allowed") is not False:
        raise V2ProtocolError("Benchmark v2 must forbid protocol changes after hidden access")
    if not hidden_policy.get("exclude_all_v1_hidden_case_tokens", False):
        raise V2ProtocolError("Benchmark v2 must require exclusion of all v1 hidden case tokens")

    candidate_tokens = {str(token) for token in candidate_hidden_case_tokens if str(token).strip()}
    if not candidate_tokens:
        raise V2ProtocolError("candidate hidden case token set is empty")

    manifest_token_block = (
        v1_final_manifest.get("hidden_lockbox", {})
        .get("case_tokens", {})
    )
    manifest_tokens = {
        str(token)
        for token in manifest_token_block.get("tokens", [])
        if str(token).strip()
    }
    external_tokens = {
        str(token)
        for token in (external_v1_hidden_case_tokens or [])
        if str(token).strip()
    }
    exclusion_tokens = manifest_tokens | external_tokens

    if not exclusion_tokens:
        raise V2ProtocolError(
            "v1 hidden exclusion tokens are unavailable; v2 hidden creation must remain blocked"
        )

    overlap = candidate_tokens & exclusion_tokens
    if overlap:
        raise V2ProtocolError(f"v2 hidden case set overlaps v1 hidden cases: {len(overlap)} token(s)")

    return HiddenExclusionReport(
        candidate_hidden_cases=len(candidate_tokens),
        v1_exclusion_cases=len(exclusion_tokens),
        status="ready",
    )


def validate_v2_pre_hidden_readiness(
    *,
    v2_manifest: dict[str, Any],
    readiness_report: dict[str, Any],
) -> PreHiddenReadinessReport:
    """Validate the Benchmark v2 development-stage gates before hidden access."""

    _validate_split_specs(v2_manifest, readiness_report)
    _validate_task_manifest_counts(v2_manifest, readiness_report)
    _validate_negative_controls(v2_manifest, readiness_report)
    _validate_calibration(v2_manifest, readiness_report)
    _validate_model_selection(v2_manifest, readiness_report)
    return PreHiddenReadinessReport(
        targets_checked=len(v2_manifest.get("targets", [])),
        negative_controls_checked=len(v2_manifest.get("negative_controls", [])),
        status="ready_for_hidden",
    )


def _validate_split_specs(v2_manifest: dict[str, Any], readiness_report: dict[str, Any]) -> None:
    expected = {split["name"]: split for split in v2_manifest.get("splits", [])}
    observed = readiness_report.get("splits", {})
    if set(observed) != set(expected):
        raise V2ProtocolError("readiness split names do not exactly match v2 manifest")
    allowed_extra_keys = {"passed", "manifest_hash", "row_counts_hash", "case_set_hash", "purged_rows"}
    for name, expected_spec in expected.items():
        actual = observed.get(name)
        if not isinstance(actual, dict):
            raise V2ProtocolError(f"missing v2 split report: {name}")
        extra_keys = set(actual) - set(expected_spec) - allowed_extra_keys
        if extra_keys:
            raise V2ProtocolError(f"split {name} includes unregistered fields: {', '.join(sorted(extra_keys))}")
        for key, expected_value in expected_spec.items():
            if actual.get(key) != expected_value:
                raise V2ProtocolError(f"split {name} does not match required {key}")
        if actual.get("passed") is not True:
            raise V2ProtocolError(f"split {name} did not pass")


def _validate_task_manifest_counts(v2_manifest: dict[str, Any], readiness_report: dict[str, Any]) -> None:
    required_counts = v2_manifest.get("task_manifest_requirements", {}).get("per_target_counts_required", [])
    task_manifests = readiness_report.get("task_manifests", {})
    for target in v2_manifest.get("targets", []):
        counts = task_manifests.get(target, {})
        if not isinstance(counts, dict):
            raise V2ProtocolError(f"missing task manifest counts for {target}")
        for key in required_counts:
            value = counts.get(key)
            if not isinstance(value, int) or value < 0:
                raise V2ProtocolError(f"invalid {key} count for {target}")
        if counts.get("unknown_and_censored_labeled_as_rejection", False):
            raise V2ProtocolError(f"unknown or censored outcomes were labeled as rejection for {target}")


def _validate_negative_controls(v2_manifest: dict[str, Any], readiness_report: dict[str, Any]) -> None:
    gates = v2_manifest.get("negative_control_gates", {})
    controls = readiness_report.get("negative_controls", {})
    if gates.get("all_controls_must_pass_before_hidden_access") is not True:
        raise V2ProtocolError("v2 manifest must require negative controls to pass")
    for name in v2_manifest.get("negative_controls", []):
        actual = controls.get(name)
        if not isinstance(actual, dict):
            raise V2ProtocolError(f"missing negative control report: {name}")
        if actual.get("executed") is not True or actual.get("passed") is not True:
            raise V2ProtocolError(f"negative control did not pass: {name}")
        expected_gate = gates.get(name, {}).get("pass_condition")
        if actual.get("pass_condition") != expected_gate:
            raise V2ProtocolError(f"negative control pass condition mismatch: {name}")


def _validate_calibration(v2_manifest: dict[str, Any], readiness_report: dict[str, Any]) -> None:
    calibration_spec = v2_manifest.get("calibration_acceptance", {})
    classification_spec = calibration_spec.get("classification", {})
    regression_spec = calibration_spec.get("regression", {})
    calibration = readiness_report.get("calibration", {})
    target_objectives = v2_manifest.get("model_selection_rule", {}).get("target_objectives", {})
    for target in v2_manifest.get("targets", []):
        report = calibration.get(target)
        if not isinstance(report, dict):
            raise V2ProtocolError(f"missing calibration report for {target}")
        metric = target_objectives.get(target, {}).get("selection_metric", "")
        if "log_loss" in metric:
            if report.get("ece_definition") != classification_spec.get("ece_definition"):
                raise V2ProtocolError(f"wrong ECE definition for {target}")
            if report.get("expected_calibration_error", 1.0) > classification_spec.get("expected_calibration_error_max", 0.0):
                raise V2ProtocolError(f"ECE threshold failed for {target}")
            if report.get("reliability_bin_count", 0) < classification_spec.get("minimum_reliability_bin_count", 0):
                raise V2ProtocolError(f"not enough reliability bins for {target}")
            if report.get("nonempty_reliability_bins", 0) < classification_spec.get("minimum_nonempty_reliability_bins", 0):
                raise V2ProtocolError(f"not enough reliability bins for {target}")
            if report.get("classwise_ece_definition") != classification_spec.get("classwise_ece_definition"):
                raise V2ProtocolError(f"wrong classwise ECE definition for {target}")
            classwise = report.get("classwise_expected_calibration_error", {})
            if not isinstance(classwise, dict) or not classwise:
                raise V2ProtocolError(f"missing classwise calibration for {target}")
            for class_name, ece in classwise.items():
                if ece > classification_spec.get("classwise_expected_calibration_error_max", 0.0):
                    raise V2ProtocolError(f"classwise calibration threshold failed for {target}: {class_name}")
            class_rows = report.get("class_row_counts", {})
            if not isinstance(class_rows, dict) or not class_rows:
                raise V2ProtocolError(f"missing class row counts for {target}")
            if any(count < classification_spec.get("minimum_rows_per_reported_class", 0) for count in class_rows.values()):
                raise V2ProtocolError(f"class row support threshold failed for {target}")
            if report.get("macro_classwise_expected_calibration_error", 1.0) > classification_spec.get("macro_classwise_expected_calibration_error_max", 0.0):
                raise V2ProtocolError(f"classwise calibration threshold failed for {target}")
        else:
            if report.get("central_interval_nominal_coverage") != regression_spec.get("central_interval_nominal_coverage"):
                raise V2ProtocolError(f"wrong interval coverage target for {target}")
            if report.get("central_interval_absolute_error", 1.0) > regression_spec.get("central_interval_allowed_absolute_error", 0.0):
                raise V2ProtocolError(f"interval calibration threshold failed for {target}")
            if report.get("interval_width_to_median_target_iqr", float("inf")) > regression_spec.get("maximum_interval_width_to_median_target_iqr", 0.0):
                raise V2ProtocolError(f"interval width threshold failed for {target}")
            if report.get("quantile_levels") != regression_spec.get("quantile_levels"):
                raise V2ProtocolError(f"wrong quantile levels for {target}")
            if report.get("quantile_pinball_loss_ratio_to_median_baseline", float("inf")) > regression_spec.get("maximum_quantile_pinball_loss_ratio_to_median_baseline", 0.0):
                raise V2ProtocolError(f"quantile loss threshold failed for {target}")


def _validate_model_selection(v2_manifest: dict[str, Any], readiness_report: dict[str, Any]) -> None:
    objectives = v2_manifest.get("model_selection_rule", {}).get("target_objectives", {})
    selections = readiness_report.get("model_selection", {})
    for target in v2_manifest.get("targets", []):
        objective = objectives.get(target, {})
        selection = selections.get(target)
        if not isinstance(selection, dict):
            raise V2ProtocolError(f"missing model selection report for {target}")
        for key in ("selection_metric", "preregistered_baseline"):
            if selection.get(key) != objective.get(key):
                raise V2ProtocolError(f"model selection {key} mismatch for {target}")
        if selection.get("fit_on_training_only") is not True:
            raise V2ProtocolError(f"model was not fit on training only for {target}")
        if selection.get("hidden_results_used") is True:
            raise V2ProtocolError(f"hidden results used for model selection on {target}")
        if selection.get("primary_split_survival") != objective.get("required_primary_split_survival"):
            raise V2ProtocolError(f"primary split survival mismatch for {target}")
        if "minimum_relative_improvement" in objective:
            if selection.get("relative_improvement", -1.0) < objective["minimum_relative_improvement"]:
                raise V2ProtocolError(f"minimum improvement failed for {target}")
        if "maximum_error_ratio_to_baseline" in objective:
            if selection.get("error_ratio_to_baseline", 2.0) > objective["maximum_error_ratio_to_baseline"]:
                raise V2ProtocolError(f"baseline error-ratio gate failed for {target}")
        if "minimum_support_coverage" in objective:
            if selection.get("support_coverage", 0.0) < objective["minimum_support_coverage"]:
                raise V2ProtocolError(f"support coverage gate failed for {target}")
