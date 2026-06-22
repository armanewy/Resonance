from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from behavior_lab.offerlab_models.benchmark_v2_integration import (
    BenchmarkV2IntegrationPaths,
    run_offerlab_benchmark_v2_integration,
)
from test_offerlab_benchmark_v2_build import _write_normalized
from test_offerlab_benchmark_v2_runner import _mark_as_bounded_fixture


class OfferLabBenchmarkV2IntegrationTests(unittest.TestCase):
    def test_integration_freezes_preregistration_and_stops_without_audited_full_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = _write_normalized(root / "normalized")
            _mark_as_bounded_fixture(normalized)
            tokens = root / "v1_tokens.json"
            tokens.write_text(json.dumps({"tokens": ["external-v1-token"]}), encoding="utf-8")

            report = run_offerlab_benchmark_v2_integration(
                BenchmarkV2IntegrationPaths(
                    normalized_dir=normalized,
                    benchmark_dir=root / "benchmark_v2",
                    output_path=root / "reports" / "v2.json",
                    preregistration_path=root / "reports" / "preregistration.json",
                    pre_hidden_output_path=root / "reports" / "pre_hidden.json",
                    doc_path=root / "docs" / "v2.md",
                    pre_hidden_doc_path=root / "docs" / "pre_hidden.md",
                    model_cards_dir=root / "cards",
                    external_v1_hidden_tokens_path=tokens,
                ),
                batch_size=2,
                partition_rows=3,
                allow_bounded_test_input=True,
            )

            self.assertEqual(report["gate"]["status"], "STOP")
            self.assertFalse(report["hidden_submission_performed"])
            self.assertFalse(report["full_release_evidence"]["passed"])
            self.assertTrue(report["build"]["completed"])
            self.assertTrue(report["pre_hidden"]["completed"])
            self.assertTrue(report["prerequisites"]["build_completed"])
            self.assertTrue(report["prerequisites"]["zero_v1_hidden_overlap"])
            self.assertIn("full-release", " ".join(report["gate"]["reasons"]))
            self.assertNotIn("path", report["preregistration"])
            preregistration = json.loads((root / "reports" / "preregistration.json").read_text(encoding="utf-8"))
            self.assertEqual(set(preregistration["targets"]), {
                "seller_next_action",
                "buyer_response_to_counter",
                "agreement",
                "final_price_ratio",
                "response_latency",
            })
            self.assertFalse(preregistration["hidden_results_used_for_selection"])
            persisted = json.loads((root / "reports" / "v2.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["gate"]["status"], "STOP")

    def test_integration_does_not_build_from_forged_full_release_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = _write_normalized(root / "normalized")
            manifest_path = normalized / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["command_args"] = {"full": True, "limit_threads": None}
            manifest["audited_full_release_evidence"] = {"passed": True}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

            report = run_offerlab_benchmark_v2_integration(
                BenchmarkV2IntegrationPaths(
                    normalized_dir=normalized,
                    benchmark_dir=root / "benchmark_v2",
                    output_path=root / "reports" / "v2.json",
                    preregistration_path=root / "reports" / "preregistration.json",
                    pre_hidden_output_path=root / "reports" / "pre_hidden.json",
                    doc_path=root / "docs" / "v2.md",
                    pre_hidden_doc_path=root / "docs" / "pre_hidden.md",
                    model_cards_dir=root / "cards",
                ),
                batch_size=2,
                partition_rows=3,
            )

            self.assertEqual(report["gate"]["status"], "STOP")
            self.assertFalse(report["build"]["completed"])
            self.assertFalse(report["full_release_evidence"]["passed"])
            self.assertIn("audited_full_release_evidence_failed", " ".join(report["errors"]))

    def test_preregistration_hash_is_stable_across_repeat_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = _write_normalized(root / "normalized")
            _mark_as_bounded_fixture(normalized)
            tokens = root / "v1_tokens.json"
            tokens.write_text(json.dumps({"tokens": ["external-v1-token"]}), encoding="utf-8")

            first = run_offerlab_benchmark_v2_integration(
                BenchmarkV2IntegrationPaths(
                    normalized_dir=normalized,
                    benchmark_dir=root / "benchmark_v2",
                    output_path=root / "reports" / "v2_first.json",
                    preregistration_path=root / "reports" / "preregistration_first.json",
                    pre_hidden_output_path=root / "reports" / "pre_hidden_first.json",
                    doc_path=root / "docs" / "v2_first.md",
                    pre_hidden_doc_path=root / "docs" / "pre_hidden_first.md",
                    model_cards_dir=root / "cards_first",
                    external_v1_hidden_tokens_path=tokens,
                ),
                batch_size=2,
                partition_rows=3,
                allow_bounded_test_input=True,
            )
            second = run_offerlab_benchmark_v2_integration(
                BenchmarkV2IntegrationPaths(
                    normalized_dir=normalized,
                    benchmark_dir=root / "benchmark_v2",
                    output_path=root / "reports" / "v2_second.json",
                    preregistration_path=root / "reports" / "preregistration_second.json",
                    pre_hidden_output_path=root / "reports" / "pre_hidden_second.json",
                    doc_path=root / "docs" / "v2_second.md",
                    pre_hidden_doc_path=root / "docs" / "pre_hidden_second.md",
                    model_cards_dir=root / "cards_second",
                    external_v1_hidden_tokens_path=tokens,
                ),
                batch_size=2,
                partition_rows=3,
                allow_bounded_test_input=True,
            )

            self.assertEqual(first["preregistration"]["hash"], second["preregistration"]["hash"])
            first_prereg = json.loads((root / "reports" / "preregistration_first.json").read_text(encoding="utf-8"))
            second_prereg = json.loads((root / "reports" / "preregistration_second.json").read_text(encoding="utf-8"))
            self.assertNotEqual(first_prereg["generated_at"], second_prereg["generated_at"])
            self.assertEqual(first_prereg["preregistration_hash"], second_prereg["preregistration_hash"])
            self.assertEqual(first_prereg["candidate_family_hash"], second_prereg["candidate_family_hash"])

    def test_cli_benchmark_v2_integrate_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = _write_normalized(root / "normalized")
            _mark_as_bounded_fixture(normalized)
            tokens = root / "v1_tokens.json"
            tokens.write_text(json.dumps({"tokens": ["external-v1-token"]}), encoding="utf-8")
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "behavior_lab",
                    "offerlab-models",
                    "benchmark-v2-integrate",
                    "--normalized-dir",
                    str(normalized),
                    "--benchmark-dir",
                    str(root / "benchmark_v2"),
                    "--output",
                    str(root / "reports" / "v2.json"),
                    "--preregistration",
                    str(root / "reports" / "preregistration.json"),
                    "--pre-hidden-output",
                    str(root / "reports" / "pre_hidden.json"),
                    "--doc",
                    str(root / "docs" / "v2.md"),
                    "--pre-hidden-doc",
                    str(root / "docs" / "pre_hidden.md"),
                    "--model-cards-dir",
                    str(root / "cards"),
                    "--external-v1-hidden-tokens",
                    str(tokens),
                    "--partition-rows",
                    "3",
                    "--batch-size",
                    "2",
                    "--allow-bounded-test-input",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(output.stdout)
            self.assertEqual(payload["gate"]["status"], "STOP")
            self.assertTrue((root / "reports" / "preregistration.json").exists())


if __name__ == "__main__":
    unittest.main()
