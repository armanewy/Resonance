from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.cli import main
from behavior_lab.money.operations import DEFAULT_RELEASE_COMMIT, MoneyOperations, MoneyOperationsError


class MoneyOperationsTests(unittest.TestCase):
    def test_start_writes_immutable_release_manifest_and_prevents_duplicate_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)

            self.assertEqual(started["status"], "running")
            self.assertTrue(Path(started["state_dir"]).is_absolute())
            self.assertEqual(started["manifest"]["release_commit"], DEFAULT_RELEASE_COMMIT)
            self.assertTrue(started["manifest"]["release_hash"])
            self.assertFalse(any(started["production_state"].values()))
            self.assertIn("weather_edge", started["manifest"]["canary_hashes"])
            self.assertIn("etf_risk", started["manifest"]["canary_hashes"])
            self.assertFalse(started["manifest"]["seller_readiness"]["canary_start_allowed"])

            with self.assertRaises(MoneyOperationsError):
                operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)

    def test_recover_resumes_missed_cycles_without_repeating_blind_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            operations.stop()

            recovered = operations.recover(as_of="2026-07-09T12:00:00+00:00")

            resumed_labs = {item["snapshot"]["lab"] for item in recovered["resumed"]}
            self.assertTrue({"weather_edge", "etf_risk"} <= resumed_labs)
            self.assertFalse(recovered["blind_evaluation_repeated"])
            status = operations.status()
            self.assertEqual(status["status"], "running")
            self.assertTrue(status["ledger_valid"])

    def test_stale_source_blocks_decision_recovery_and_doctor_reports_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            operations.record_source_health("weather_edge", healthy=True, stale=True, reason="source_outage")

            recovered = operations.recover(as_of="2026-07-03T12:00:00+00:00")
            doctor = operations.doctor()

            self.assertIn({"lab": "weather_edge", "reason": "stale_or_unhealthy_source"}, recovered["skipped"])
            self.assertTrue(any(issue["code"] == "source_stale" and issue["lab"] == "weather_edge" for issue in doctor["issues"]))
            weather = operations.status()["canaries"]["weather_edge"]
            self.assertEqual(weather["metrics"]["snapshot_count"], 1)

    def test_canary_hash_mismatch_and_frozen_strategy_mutation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            manifest_path = Path(tmp) / "ops" / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["canary_hashes"]["weather_edge"]["protocol_hash"] = "tampered"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            doctor = operations.doctor()
            recovered = operations.recover(
                as_of="2026-07-03T12:00:00+00:00",
                strategy_versions={"weather_edge": "mutated_strategy"},
            )

            self.assertTrue(any(issue["code"] == "canary_hash_mismatch" for issue in doctor["issues"]))
            self.assertTrue(any(item["reason"] == "frozen_canary_rejected_change" for item in recovered["skipped"]))
            self.assertFalse(any(started["production_state"].values()))

    def test_weekly_report_leads_with_values_and_distinguishes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            report = operations.weekly_report()

            expected_leading_keys = [
                "schema_version",
                "notice",
                "paper_or_shadow_value",
                "paper_value",
                "seller_shadow_value",
                "resolved_decisions",
            ]
            self.assertEqual(list(report)[:6], expected_leading_keys)
            self.assertIn("canary_comparability", report)
            self.assertIn("seller_data_readiness", report)
            self.assertFalse(any(report["production_state"].values()))

    def test_cli_operations_start_status_recover_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = str(Path(tmp) / "ops")
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "operations", "start", "--state-dir", state_dir, "--as-of", "2026-07-01T12:00:00+00:00"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "running")

            for command in ("status", "doctor", "weekly-report"):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    main(["money", "operations", command, "--state-dir", state_dir])
                self.assertTrue(json.loads(stream.getvalue())["paper_only"])

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "operations", "recover", "--state-dir", state_dir, "--as-of", "2026-07-09T12:00:00+00:00"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "recovered")

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "operations", "stop", "--state-dir", state_dir])
            self.assertEqual(json.loads(stream.getvalue())["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
