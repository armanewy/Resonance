from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path
import tempfile
import unittest

from behavior_lab.core import stable_hash
from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.offerlab_models import benchmark_v1
from behavior_lab.offerlab_models.benchmark_v1 import BenchmarkPaths, run_offerlab_benchmark_v1


class OfferLabBenchmarkV1RunnerTests(unittest.TestCase):
    def test_runner_writes_scoped_report_docs_and_model_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            normalized = root / "normalized"
            build_sample_dataset(raw)
            normalize_dataset(raw, normalized)

            report = run_offerlab_benchmark_v1(
                BenchmarkPaths(
                    normalized_dir=normalized,
                    output_path=root / "report.json",
                    doc_path=root / "report.md",
                    model_cards_dir=root / "cards",
                    lockbox_store_path=root / "lockbox.jsonl",
                ),
                row_cap=50,
            )

            self.assertTrue((root / "report.json").exists())
            self.assertTrue((root / "report.md").exists())
            self.assertTrue((root / "cards" / "seller_next_action.md").exists())
            report_md = (root / "report.md").read_text(encoding="utf-8")
            model_card = (root / "cards" / "seller_next_action.md").read_text(encoding="utf-8")
            self.assertTrue(report["research_only"])
            self.assertFalse(report["production_export_allowed"])
            self.assertEqual(report["scope"]["evidence_scope"], "bounded_smoke_or_semantics")
            self.assertIn("canonical_lockbox_store_name", report["scope"])
            self.assertIn("hidden_lockbox_store_event_count", report["scope"])
            self.assertNotIn("hidden_lockbox_store_hash", report["scope"])
            self.assertFalse(report["scope"]["protocol_splits_complete"])
            self.assertIn("buyer_disjoint", report["scope"]["omitted_protocol_splits"])
            self.assertIn("seller_next_action", report["targets"])
            self.assertEqual(report["gate"]["status"], "STOP")
            self.assertIn("predicates", report["gate"])
            self.assertFalse(report["gate"]["predicates"]["full_release_evidence"]["passed"])
            self.assertFalse(report["gate"]["predicates"]["protocol_splits_complete"]["passed"])
            self.assertFalse(report["gate"]["predicates"]["negative_controls_passed"]["passed"])
            answers = report["targets"]["seller_next_action"]["answers"]
            self.assertIn("calibration_reported", answers)
            self.assertIn("calibration_quality_validated", answers)
            self.assertFalse(answers["calibration_quality_validated"])
            self.assertIn("abstention_reported", answers)
            self.assertIn("hidden_support_coverage_at_least_80pct", answers)
            self.assertIn("compact_formula_passed_development_falsification", answers)
            self.assertIn("negative_controls_passed", answers)
            self.assertFalse(answers["negative_controls_passed"])
            self.assertNotIn("calibrated", answers)
            self.assertNotIn("compact_formula_retains_gain", answers)
            controls = report["targets"]["seller_next_action"]["chronological"]["negative_control_protocol"]
            self.assertIn("future_status_canary", controls["missing"])
            hidden_lockbox = report["targets"]["seller_next_action"]["chronological"]["hidden_lockbox"]
            self.assertNotIn("artifact_id", hidden_lockbox)
            self.assertNotIn("reservation_event_id", hidden_lockbox)
            self.assertNotIn("hidden_case_set_hash", hidden_lockbox)
            self.assertIn("Overall Benchmark v1 gate: `STOP`", model_card)
            self.assertIn("Protocol splits complete: `False`", model_card)
            self.assertIn("Missing negative controls:", model_card)
            self.assertIn("Standalone card inherits the overall STOP gate", model_card)
            self.assertIn("frozen Benchmark v1 negative controls remain incomplete", report_md)
            self.assertNotIn("red-team canaries passed before this run", report_md)
            self.assertIn("permission_report", report)
            self.assertFalse(report["permission_report"]["production_export"]["allowed"])

    def test_runner_rejects_in_repo_lockbox_even_with_canonical_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            normalized = root / "normalized"
            build_sample_dataset(raw)
            normalize_dataset(raw, normalized)
            manifest = json.loads((normalized / "manifest.json").read_text(encoding="utf-8"))
            lockbox_name = f"hidden_lockbox_offerlab_benchmark_v1_{stable_hash(manifest)[:12]}.jsonl"

            with self.assertRaises(ValueError):
                run_offerlab_benchmark_v1(
                    BenchmarkPaths(
                        normalized_dir=normalized,
                        output_path=root / "report.json",
                        doc_path=root / "report.md",
                        model_cards_dir=root / "cards",
                        lockbox_store_path=Path.cwd() / lockbox_name,
                    ),
                    row_cap=50,
                )

    def test_lockbox_validator_rejects_temp_path_inside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_repo = root / "repo"
            fake_repo.mkdir()
            manifest = {"manifest": "test"}
            protocol = {"protocol": "test"}
            canonical = benchmark_v1._canonical_lockbox_store_name(manifest, protocol)
            original_repo_root = benchmark_v1._repo_root
            try:
                benchmark_v1._repo_root = lambda: fake_repo
                with self.assertRaises(ValueError):
                    benchmark_v1._validate_lockbox_store(
                        fake_repo / canonical,
                        manifest=manifest,
                        protocol=protocol,
                    )
            finally:
                benchmark_v1._repo_root = original_repo_root


if __name__ == "__main__":
    unittest.main()
