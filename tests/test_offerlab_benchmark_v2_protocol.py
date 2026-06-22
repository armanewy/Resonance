from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import _bootstrap  # noqa: F401,E402

from behavior_lab.offerlab_models.benchmark_v2_protocol import (
    V2ProtocolError,
    validate_v2_hidden_exclusion,
    validate_v2_pre_hidden_readiness,
)
from behavior_lab.cli import main as cli_main


V1_FINAL = ROOT / "reports" / "offerlab_benchmark_v1_final_manifest.json"
V2_MANIFEST = ROOT / "datasets" / "manifests" / "offerlab_benchmark_v2.yaml"
V2_DOC = ROOT / "docs" / "research" / "OFFERLAB_BENCHMARK_V2.md"


class OfferLabBenchmarkV2ProtocolTests(unittest.TestCase):
    def test_v1_final_manifest_permanently_freezes_hidden_spent_benchmark(self) -> None:
        manifest = json.loads(V1_FINAL.read_text(encoding="utf-8"))

        self.assertEqual(manifest["benchmark_id"], "offerlab_benchmark_v1")
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["hidden_status"], "hidden_spent")
        self.assertEqual(manifest["reuse_policy"], "never_reusable_for_model_selection")
        self.assertEqual(manifest["final_decision"]["status"], "STOP")
        self.assertFalse(manifest["v2_implications"]["repeat_v1_allowed"])
        self.assertTrue(manifest["v2_implications"]["fresh_hidden_cases_required"])

    def test_v1_manifest_records_all_hidden_queries_and_token_availability(self) -> None:
        manifest = json.loads(V1_FINAL.read_text(encoding="utf-8"))
        lockbox = manifest["hidden_lockbox"]

        self.assertEqual(lockbox["canonical_store_name"], "hidden_lockbox_offerlab_benchmark_v1_4717f92cdb18.jsonl")
        self.assertEqual(lockbox["event_count_reported"], 5)
        self.assertEqual(len(lockbox["queries"]), 5)
        self.assertEqual({query["hidden_submission_count"] for query in lockbox["queries"]}, {1})
        self.assertFalse(lockbox["case_tokens"]["tokens"])
        self.assertIn("block v2 hidden lockbox creation", lockbox["case_tokens"]["required_v2_behavior"])

    def test_v2_requires_full_release_and_all_protocol_splits(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(manifest["benchmark_id"], "offerlab_benchmark_v2")
        self.assertEqual(manifest["v1_relationship"], "new_benchmark_not_repeat")
        self.assertTrue(manifest["research_only"])
        self.assertFalse(manifest["production_export_allowed"])
        self.assertTrue(manifest["required_normalization"]["full_release_required"])
        self.assertTrue(manifest["required_normalization"]["streaming_required"])
        self.assertFalse(manifest["required_normalization"]["model_row_cap_allowed"])

        expected_splits = {
            "chronological_listing_purged": {"group_key": "listing_id", "primary": True, "required": True},
            "seller_disjoint": {"group_key": "seller_id", "primary": True, "required": True},
            "buyer_disjoint": {"group_key": "buyer_id", "primary": False, "required": "where_identifiers_permit"},
            "category_disjoint_diagnostic": {"group_key": "category", "primary": False, "required": True},
            "thread_safe_nested_development": {"group_key": "thread_id", "primary": True, "required": True},
            "fresh_hidden_lockbox": {"query_budget_per_target": 1, "primary": True, "required": True},
        }
        split_specs = {split["name"]: {key: value for key, value in split.items() if key != "name"} for split in manifest["splits"]}
        self.assertEqual(split_specs, expected_splits)

    def test_v2_hidden_policy_blocks_reuse_or_overlap_with_v1(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        hidden = manifest["hidden_policy"]

        self.assertEqual(hidden["hidden_queries_per_target"], 1)
        self.assertTrue(hidden["fresh_hidden_lockbox_required"])
        self.assertTrue(hidden["exclude_all_v1_hidden_case_tokens"])
        self.assertTrue(hidden["block_hidden_creation_if_v1_tokens_unavailable"])
        self.assertTrue(hidden["external_v1_hidden_case_token_artifact_required_if_manifest_tokens_unavailable"])
        self.assertNotIn("all_source_exclusion_proof_allowed", hidden)
        self.assertFalse(hidden["protocol_changes_after_hidden_access_allowed"])

    def test_v2_hidden_validator_blocks_when_v1_tokens_are_unavailable(self) -> None:
        v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        v1 = json.loads(V1_FINAL.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(V2ProtocolError, "unavailable"):
            validate_v2_hidden_exclusion(
                v2_manifest=v2,
                v1_final_manifest=v1,
                candidate_hidden_case_tokens=["candidate-1"],
            )

    def test_v2_hidden_validator_rejects_overlap_and_accepts_external_exclusion(self) -> None:
        v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        v1 = json.loads(V1_FINAL.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(V2ProtocolError, "overlaps"):
            validate_v2_hidden_exclusion(
                v2_manifest=v2,
                v1_final_manifest=v1,
                candidate_hidden_case_tokens=["candidate-1", "spent-v1"],
                external_v1_hidden_case_tokens=["spent-v1"],
            )

        report = validate_v2_hidden_exclusion(
            v2_manifest=v2,
            v1_final_manifest=v1,
            candidate_hidden_case_tokens=["candidate-1"],
            external_v1_hidden_case_tokens=["spent-v1"],
        )
        self.assertEqual(report.status, "ready")
        self.assertEqual(report.v1_exclusion_cases, 1)

    def test_v2_pre_hidden_validator_requires_all_development_gates(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        report = _valid_v2_readiness_report(manifest)

        readiness = validate_v2_pre_hidden_readiness(v2_manifest=manifest, readiness_report=report)

        self.assertEqual(readiness.status, "ready_for_hidden")
        self.assertEqual(readiness.targets_checked, len(manifest["targets"]))
        self.assertEqual(readiness.negative_controls_checked, len(manifest["negative_controls"]))

    def test_v2_pre_hidden_validator_rejects_failed_controls_and_missing_censored_counts(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        report = _valid_v2_readiness_report(manifest)
        report["negative_controls"]["random_labels"]["passed"] = False

        with self.assertRaisesRegex(V2ProtocolError, "negative control did not pass"):
            validate_v2_pre_hidden_readiness(v2_manifest=manifest, readiness_report=report)

        report = _valid_v2_readiness_report(manifest)
        del report["task_manifests"]["seller_next_action"]["censored_outcome_rows"]
        with self.assertRaisesRegex(V2ProtocolError, "censored_outcome_rows"):
            validate_v2_pre_hidden_readiness(v2_manifest=manifest, readiness_report=report)

    def test_v2_pre_hidden_validator_rejects_calibration_and_selection_failures(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        report = _valid_v2_readiness_report(manifest)
        report["calibration"]["seller_next_action"]["expected_calibration_error"] = 0.5

        with self.assertRaisesRegex(V2ProtocolError, "ECE threshold failed"):
            validate_v2_pre_hidden_readiness(v2_manifest=manifest, readiness_report=report)

        report = _valid_v2_readiness_report(manifest)
        report["model_selection"]["seller_next_action"]["hidden_results_used"] = True
        with self.assertRaisesRegex(V2ProtocolError, "hidden results used"):
            validate_v2_pre_hidden_readiness(v2_manifest=manifest, readiness_report=report)

    def test_v2_requires_calibration_coverage_controls_and_censored_label_handling(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))

        self.assertTrue(manifest["calibration_acceptance"]["must_be_declared_before_hidden_access"])
        self.assertEqual(
            manifest["calibration_acceptance"]["classification"]["ece_definition"],
            "top_label_expected_calibration_error_weighted_by_bin_count",
        )
        self.assertLessEqual(manifest["calibration_acceptance"]["classification"]["expected_calibration_error_max"], 0.08)
        self.assertLessEqual(
            manifest["calibration_acceptance"]["classification"]["classwise_expected_calibration_error_max"],
            0.12,
        )
        self.assertGreaterEqual(manifest["support_coverage"]["primary_candidate_minimum"], 0.8)
        self.assertTrue(manifest["missing_and_censored_label_policy"]["preserve_unknown_outcomes"])
        self.assertTrue(manifest["missing_and_censored_label_policy"]["preserve_censored_outcomes"])
        self.assertFalse(manifest["missing_and_censored_label_policy"]["convert_censored_to_rejection_allowed"])

        controls = set(manifest["negative_controls"])
        for name in {
            "random_labels",
            "future_status_canary",
            "accepted_price_canary",
            "identifier_memorization_canary",
            "random_row_split_inflation",
            "same_timestamp_ordering_perturbation",
            "censoring_as_rejection_canary",
            "artifact_name_leakage_canary",
        }:
            self.assertIn(name, controls)
            self.assertIn(name, manifest["negative_control_gates"])
        self.assertTrue(manifest["negative_control_gates"]["all_controls_must_pass_before_hidden_access"])

        task_counts = set(manifest["task_manifest_requirements"]["per_target_counts_required"])
        self.assertTrue({"eligible_rows", "supervised_rows", "unknown_outcome_rows", "censored_outcome_rows"}.issubset(task_counts))
        self.assertTrue(manifest["task_manifest_requirements"]["unknown_and_censored_rows_must_not_be_labeled_as_rejection"])

    def test_v2_preregisters_model_selection_objectives_and_primary_split_survival(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        objectives = manifest["model_selection_rule"]["target_objectives"]

        self.assertEqual(set(objectives), set(manifest["targets"]))
        self.assertEqual(objectives["seller_next_action"]["selection_metric"], "multiclass_log_loss")
        self.assertEqual(objectives["seller_next_action"]["minimum_relative_improvement"], 0.05)
        self.assertEqual(objectives["seller_next_action"]["minimum_support_coverage"], 0.8)
        for objective in objectives.values():
            self.assertIn("preregistered_baseline", objective)
            self.assertEqual(
                objective["required_primary_split_survival"],
                ["chronological_listing_purged", "seller_disjoint"],
            )

    def test_v2_doc_refuses_production_and_v1_repeat_claims(self) -> None:
        text = V2_DOC.read_text(encoding="utf-8")

        self.assertIn("It is not a repeat of Benchmark v1", text)
        self.assertIn("never_reusable_for_model_selection", text)
        self.assertIn("It may never return production-ready based on NBER data", text)
        self.assertIn("one hidden submission per target", text)
        self.assertIn("passed its preregistered falsification", text)

    def test_cli_benchmark_v1_is_retired_after_hidden_spent_manifest(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Benchmark v1 is frozen and hidden-spent"):
            cli_main(
                [
                    "offerlab-models",
                    "benchmark-v1",
                    "--normalized-dir",
                    "does-not-matter",
                    "--lockbox-store",
                    "outside-repo.jsonl",
                ]
            )


def _valid_v2_readiness_report(manifest: dict) -> dict:
    splits = {}
    for split in manifest["splits"]:
        splits[split["name"]] = {**split, "passed": True}

    task_manifests = {}
    for target in manifest["targets"]:
        task_manifests[target] = {
            "eligible_rows": 100,
            "supervised_rows": 80,
            "unknown_outcome_rows": 10,
            "censored_outcome_rows": 10,
            "excluded_rows": 0,
            "unknown_and_censored_labeled_as_rejection": False,
        }

    negative_controls = {
        name: {"executed": True, "passed": True}
        for name in manifest["negative_controls"]
    }

    objectives = manifest["model_selection_rule"]["target_objectives"]
    calibration = {}
    model_selection = {}
    for target, objective in objectives.items():
        if "log_loss" in objective["selection_metric"]:
            calibration[target] = {
                "ece_definition": manifest["calibration_acceptance"]["classification"]["ece_definition"],
                "expected_calibration_error": 0.03,
                "nonempty_reliability_bins": 6,
                "macro_classwise_expected_calibration_error": 0.05,
            }
            selection = {
                "relative_improvement": objective.get("minimum_relative_improvement", 0.05),
            }
        else:
            calibration[target] = {
                "central_interval_nominal_coverage": manifest["calibration_acceptance"]["regression"]["central_interval_nominal_coverage"],
                "central_interval_absolute_error": 0.03,
                "quantile_levels": manifest["calibration_acceptance"]["regression"]["quantile_levels"],
            }
            selection = {
                "error_ratio_to_baseline": objective.get("maximum_error_ratio_to_baseline", 0.98),
            }
        selection.update(
            {
                "selection_metric": objective["selection_metric"],
                "preregistered_baseline": objective["preregistered_baseline"],
                "fit_on_training_only": True,
                "hidden_results_used": False,
                "primary_split_survival": objective["required_primary_split_survival"],
                "support_coverage": objective.get("minimum_support_coverage", 1.0),
            }
        )
        model_selection[target] = selection

    return {
        "splits": splits,
        "task_manifests": task_manifests,
        "negative_controls": negative_controls,
        "calibration": calibration,
        "model_selection": model_selection,
    }


if __name__ == "__main__":
    unittest.main()
