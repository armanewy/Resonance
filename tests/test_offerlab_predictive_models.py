from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from behavior_lab.benchmarks.splits import chronological_split
from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.datasets.nber_best_offer.tasks import build_tasks
from behavior_lab.offerlab_models import build_research_leaderboards
from behavior_lab.offerlab_models.common import validate_feature_contract
from behavior_lab.offerlab_models.predictive import EmpiricalQuantileRegressor, predictive_suite


def _tasks() -> dict[str, list[dict[str, object]]]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    build_sample_dataset(root / "raw")
    normalize_dataset(root / "raw", root / "normalized")
    tasks = build_tasks(root / "normalized")
    tmp.cleanup()
    return tasks


class OfferLabPredictiveModelTests(unittest.TestCase):
    def test_classification_suite_contains_requested_research_models_and_baselines(self) -> None:
        rows = _tasks()["seller_next_action"]
        split = chronological_split(rows, time_key="timestamp")
        report = predictive_suite("seller_next_action", split.train, split.development, split.hidden)
        development_models = {row["model_id"] for row in report["leaderboards"]["development"]}
        self.assertIn("majority", development_models)
        self.assertIn("offer_ratio_threshold", development_models)
        self.assertIn("regularized_glm", development_models)
        self.assertIn("smoothed_offer_histogram", development_models)
        self.assertIn("deterministic_stump_ensemble", development_models)
        self.assertIn("monotonic_offer_model", development_models)
        self.assertEqual(report["leaderboards"]["hidden"], [])
        self.assertFalse(report["hidden_lockbox"]["submitted"])
        self.assertTrue(report["research_only"])
        self.assertEqual(report["scope"]["evidence_scope"], "bounded_smoke_or_semantics")
        self.assertTrue(report["negative_controls"]["random_label_permutation"]["executed"])
        self.assertTrue(report["negative_controls"]["random_row_split"]["executed"])
        self.assertTrue(report["negative_controls"]["same_timestamp_ordering"]["executed"])
        self.assertTrue(report["negative_controls"]["artifact_name_canary"]["rejected"])
        for control in report["negative_controls"].values():
            self.assertTrue(control["passed"])
            self.assertIn("threshold", control)
        self.assertFalse(report["production_export_allowed"])
        self.assertFalse(report["participant_id_features_used"])
        for board in report["leaderboards"].values():
            for row in board:
                self.assertNotIn("seller_id", row["features_used"])
                self.assertNotIn("buyer_id", row["features_used"])
                self.assertIn("lineage", row)
                self.assertIn("abstention", row)
                self.assertIn("relative_improvement", row)
                self.assertIn("subgroup_counts", row)
                self.assertIn("negative_control_references", row)

    def test_quantile_regression_reports_intervals_for_final_price(self) -> None:
        rows = [
            _final_price_row("r1", "2020-01-01T00:00:00", "cameras", 0.70),
            _final_price_row("r2", "2020-01-02T00:00:00", "cameras", 0.75),
            _final_price_row("r3", "2020-01-03T00:00:00", "cameras", 0.80),
            _final_price_row("r4", "2020-01-04T00:00:00", "parts", 0.85),
        ]
        split = chronological_split(rows, time_key="timestamp")
        report = predictive_suite("final_price_ratio", split.train, split.development, split.hidden)
        development = {row["model_id"]: row for row in report["leaderboards"]["development"]}
        self.assertIn("empirical_category_quantiles", development)
        self.assertIn("interval_coverage", development["empirical_category_quantiles"])
        model = EmpiricalQuantileRegressor().fit(split.train)
        prediction = model.predict(split.hidden).predictions[0]
        self.assertLessEqual(prediction["lower"], prediction["prediction"])
        self.assertGreaterEqual(prediction["upper"], prediction["prediction"])

    def test_regression_hidden_selection_rationale_does_not_claim_calibration_tiebreaker(self) -> None:
        rows = [
            _final_price_row("r1", "2020-01-01T00:00:00", "cameras", 0.70),
            _final_price_row("r2", "2020-01-02T00:00:00", "cameras", 0.75),
            _final_price_row("r3", "2020-01-03T00:00:00", "cameras", 0.80),
            _final_price_row("r4", "2020-01-04T00:00:00", "parts", 0.85),
            _final_price_row("r5", "2020-01-05T00:00:00", "parts", 0.90),
        ]
        split = chronological_split(rows, time_key="timestamp")
        with tempfile.TemporaryDirectory() as tmp:
            report = predictive_suite(
                "final_price_ratio",
                split.train,
                split.development,
                split.hidden,
                hidden_lockbox_id="regression-selection-test",
                hidden_lockbox_store_path=Path(tmp) / "hidden.jsonl",
            )
        rationale = report["hidden_lockbox"]["selection_rationale"]
        self.assertNotIn("better_calibration", rationale["tie_breakers"])
        self.assertIn("higher_support_coverage", rationale["tie_breakers"])

    def test_hidden_predictive_evaluation_is_explicit_and_one_shot(self) -> None:
        rows = _tasks()["seller_next_action"]
        split = chronological_split(rows, time_key="timestamp")
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "hidden.jsonl"
            report = predictive_suite(
                "seller_next_action",
                split.train,
                split.development,
                split.hidden,
                hidden_lockbox_id="predictive-test",
                hidden_lockbox_store_path=store,
            )
            self.assertTrue(report["hidden_lockbox"]["submitted"])
            self.assertGreaterEqual(len(report["leaderboards"]["hidden"]), 1)
            baseline_model_id = report["hidden_lockbox"]["baseline_model_id"]
            hidden_models = {row["model_id"] for row in report["leaderboards"]["hidden"]}
            self.assertIn(baseline_model_id, hidden_models)
            self.assertTrue(report["hidden_lockbox"]["baseline_hidden_scoring_preregistered"])
            self.assertIn("selection_rationale", report["hidden_lockbox"])
            self.assertIn("evaluation_bundle_model_ids", report["hidden_lockbox"])
            for row in report["leaderboards"]["hidden"]:
                self.assertIn("relative_improvement", row)
            self.assertTrue(store.exists())
            with self.assertRaises(RuntimeError):
                predictive_suite(
                    "seller_next_action",
                    split.train,
                    split.development,
                    split.hidden,
                    hidden_lockbox_id="predictive-test",
                    hidden_lockbox_store_path=store,
                )
            with self.assertRaises(RuntimeError):
                predictive_suite(
                    "seller_next_action",
                    split.train,
                    split.development,
                    list(reversed(split.hidden)),
                    hidden_lockbox_id="predictive-test-reordered",
                    hidden_lockbox_store_path=store,
                )
            altered_hidden = [dict(row) for row in split.hidden]
            altered_hidden[0] = dict(altered_hidden[0])
            altered_hidden[0]["row_id"] = "different-hidden-row"
            with self.assertRaises(RuntimeError):
                predictive_suite(
                    "seller_next_action",
                    split.train,
                    split.development,
                    altered_hidden,
                    hidden_lockbox_id="predictive-test-renamed-row",
                    hidden_lockbox_store_path=store,
                )

    def test_hidden_predictive_evaluation_requires_persistent_store(self) -> None:
        rows = _tasks()["seller_next_action"]
        split = chronological_split(rows, time_key="timestamp")
        with self.assertRaises(ValueError):
            predictive_suite("seller_next_action", split.train, split.development, split.hidden, hidden_lockbox_id="missing-store")

    def test_label_is_not_allowed_as_a_model_feature(self) -> None:
        self.assertFalse(validate_feature_contract([{"features": {"label": "accept"}}]))

    def test_research_leaderboards_are_separate_and_non_production(self) -> None:
        report = build_research_leaderboards(_tasks())
        self.assertFalse(report["production_export_allowed"])
        self.assertIsNone(report["universal_winner"])
        for task in ["seller_next_action", "buyer_response_to_counter", "agreement", "final_price_ratio"]:
            self.assertIn(task, report["leaderboards"])
            self.assertIn("chronological", report["leaderboards"][task])
            self.assertIn("seller_disjoint", report["leaderboards"][task])
        self.assertFalse(report["artifact_lineage"]["production_export"]["allowed"])

    def test_cli_sample_model_suite_smoke(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        output = subprocess.run(
            [sys.executable, "-m", "behavior_lab", "offerlab-models", "sample"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(output.stdout)
        self.assertEqual(payload["evidence_role"], "OFFERLAB_RESEARCH_MODEL_SUITE")
        self.assertTrue(payload["research_only"])
        self.assertEqual(payload["scope"]["evidence_scope"], "bounded_smoke_or_semantics")
        self.assertFalse(payload["production_export_allowed"])


if __name__ == "__main__":
    unittest.main()


def _final_price_row(row_id: str, timestamp: str, category: str, label: float) -> dict[str, object]:
    return {
        "row_id": row_id,
        "label": label,
        "timestamp": timestamp,
        "seller_id": f"seller-{row_id}",
            "features": {
                "category": category,
                "condition": "used",
                "listing_price": 100.0,
                "current_actor": "buyer",
                "current_action": "offer",
                "current_amount": label * 100.0,
                "offer_to_asking_ratio": label,
                "round_number": 1,
                "prior_turn_count": 0,
                "prior_counter_count": 0,
            },
        "observed_history": [],
    }
