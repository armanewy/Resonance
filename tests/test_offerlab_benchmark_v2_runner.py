from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.offerlab_models.benchmark_v2 import BenchmarkV2Paths, _leaderboard, run_offerlab_benchmark_v2


class OfferLabBenchmarkV2RunnerTests(unittest.TestCase):
    def test_runner_writes_pre_hidden_report_without_hidden_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_sample_dataset(root / "raw")
            normalize_dataset(root / "raw", root / "normalized")

            report = run_offerlab_benchmark_v2(
                BenchmarkV2Paths(
                    normalized_dir=root / "normalized",
                    output_path=root / "v2.json",
                    doc_path=root / "v2.md",
                    model_cards_dir=root / "cards",
                ),
                batch_size=2,
            )

            self.assertTrue((root / "v2.json").exists())
            self.assertTrue((root / "v2.md").exists())
            self.assertTrue((root / "cards" / "seller_next_action.md").exists())
            self.assertEqual(report["benchmark_id"], "offerlab_benchmark_v2")
            self.assertFalse(report["hidden_submission_performed"])
            self.assertFalse(report["hidden_results_used_for_selection"])
            self.assertEqual(report["gate"]["status"], "STOP")
            self.assertEqual(report["pre_hidden_readiness"]["status"], "blocked")
            self.assertIsNone(report["targets"]["seller_next_action"]["row_cap"])
            self.assertTrue(report["scope"]["streaming_or_batch_inputs"])
            self.assertFalse(report["scope"]["model_row_cap_used"])
            for target, payload in report["targets"].items():
                self.assertFalse(payload["hidden_lockbox"]["submitted"], target)
                self.assertIn("artifact_id", payload["selected_model"])
                self.assertFalse(payload["selected_model"]["hidden_results_used"])
                self.assertIn("support", payload)
                self.assertIn("calibration", payload)
            persisted = json.loads((root / "v2.json").read_text(encoding="utf-8"))
            self.assertFalse(persisted["hidden_submission_performed"])

    def test_runner_executes_all_v2_negative_controls_with_manifest_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_sample_dataset(root / "raw")
            normalize_dataset(root / "raw", root / "normalized")

            report = run_offerlab_benchmark_v2(
                BenchmarkV2Paths(
                    normalized_dir=root / "normalized",
                    output_path=root / "v2.json",
                    doc_path=root / "v2.md",
                    model_cards_dir=root / "cards",
                ),
                batch_size=2,
            )

            manifest = json.loads(Path("datasets/manifests/offerlab_benchmark_v2.yaml").read_text(encoding="utf-8"))
            self.assertEqual(set(report["readiness_report"]["negative_controls"]), set(manifest["negative_controls"]))
            for name in manifest["negative_controls"]:
                aggregate = report["readiness_report"]["negative_controls"][name]
                self.assertTrue(aggregate["executed"], name)
                self.assertEqual(aggregate["pass_condition"], manifest["negative_control_gates"][name]["pass_condition"])
            controls = report["targets"]["seller_next_action"]["negative_controls"]
            self.assertTrue(controls["future_status_canary"]["rejected"])
            self.assertTrue(controls["accepted_price_canary"]["rejected"])
            self.assertTrue(controls["identifier_memorization_canary"]["identifier_features_rejected"])
            self.assertTrue(controls["artifact_name_leakage_canary"]["rejected"])

    def test_leaderboard_includes_required_models_and_scores_more_than_500_rows(self) -> None:
        rows = [_seller_row(index) for index in range(720)]
        train = rows[:540]
        development = rows[540:]
        board = _leaderboard("seller_next_action", train, development, batch_size=37)
        model_ids = {row["model_id"] for row in board}
        families = {row["model_family"] for row in board}

        self.assertIn("majority", model_ids)
        self.assertIn("category_majority", model_ids)
        self.assertIn("offer_ratio_threshold", model_ids)
        self.assertIn("prior_concession_heuristic", model_ids)
        self.assertIn("split_the_difference_heuristic", model_ids)
        self.assertIn("regularized_glm", model_ids)
        self.assertIn("deterministic_stump_ensemble", model_ids)
        self.assertIn("compact_formula_candidate", families)
        for row in board:
            self.assertEqual(row["prediction_count"], 180)
            self.assertIn("brier_score", row)
            self.assertIn("calibration_report", row)
            self.assertIn("lineage", row)

    def test_cli_benchmark_v2_smoke_does_not_submit_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_sample_dataset(root / "raw")
            normalize_dataset(root / "raw", root / "normalized")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "behavior_lab",
                    "offerlab-models",
                    "benchmark-v2",
                    "--normalized-dir",
                    str(root / "normalized"),
                    "--output",
                    str(root / "v2.json"),
                    "--doc",
                    str(root / "v2.md"),
                    "--model-cards-dir",
                    str(root / "cards"),
                    "--batch-size",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            payload = json.loads(output.stdout)
            self.assertFalse(payload["hidden_submission_performed"])
            self.assertFalse(payload["hidden_results_used_for_selection"])
            self.assertEqual(payload["gate"]["status"], "STOP")

    def test_explicit_hidden_submission_is_blocked_until_readiness_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_sample_dataset(root / "raw")
            normalize_dataset(root / "raw", root / "normalized")

            with self.assertRaisesRegex(Exception, "hidden submission blocked"):
                run_offerlab_benchmark_v2(
                    BenchmarkV2Paths(
                        normalized_dir=root / "normalized",
                        output_path=root / "v2.json",
                        doc_path=root / "v2.md",
                        model_cards_dir=root / "cards",
                    ),
                    batch_size=2,
                    allow_hidden_submission=True,
                )


def _seller_row(index: int) -> dict[str, object]:
    ratio = 0.55 + (index % 40) / 100.0
    if ratio >= 0.86:
        label = "accept"
    elif ratio >= 0.72:
        label = "counter"
    else:
        label = "decline"
    return {
        "row_id": f"r{index}",
        "task": "seller_next_action",
        "label": label,
        "timestamp": f"2020-01-{(index % 28) + 1:02d}T{(index % 24):02d}:00:00",
        "thread_id": f"thread-{index}",
        "listing_id": f"listing-{index}",
        "seller_id": f"seller-{index % 90}",
        "buyer_id": f"buyer-{index % 120}",
        "features": {
            "category": "parts" if index % 2 else "electronics",
            "condition": "used",
            "listing_price": 100.0,
            "current_actor": "buyer",
            "current_action": "offer",
            "current_amount": round(ratio * 100.0, 2),
            "offer_to_asking_ratio": ratio,
            "round_number": (index % 4) + 1,
            "prior_turn_count": index % 4,
            "prior_counter_count": 1 if index % 5 == 0 else 0,
        },
        "observed_history": [],
    }


if __name__ == "__main__":
    unittest.main()
