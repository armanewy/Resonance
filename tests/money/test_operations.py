from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
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
from behavior_lab.money.canary import MoneyCanaryManager
from behavior_lab.money.operations import DEFAULT_RELEASE_COMMIT, MoneyOperations, MoneyOperationsError
from test_offerlab_pilot_onboard import _write_many_complete_rows
from behavior_lab.offerlab_pilot import onboard_input


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

    def test_doctor_allows_intentionally_blocked_seller_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)

            doctor = operations.doctor()

            self.assertEqual(started["manifest"]["canary_hashes"]["offerlab_seller_pilot"].get("status"), "blocked")
            self.assertTrue(doctor["healthy"])
            self.assertFalse(any(issue["code"] == "missing_canary" for issue in doctor["issues"]))

    def test_doctor_does_not_allow_blocked_public_canary_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            operations.start(as_of="2026-07-01T12:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            manifest_path = Path(tmp) / "ops" / "release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["canary_hashes"]["weather_edge"] = {"status": "blocked", "reason": "forged_public_canary_block"}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            doctor = operations.doctor()

            self.assertFalse(doctor["healthy"])
            self.assertTrue(any(issue["code"] == "missing_canary" and issue["lab"] == "weather_edge" for issue in doctor["issues"]))

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

    def test_operations_does_not_start_seller_canary_from_blank_cost_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_many_complete_rows(source, rows=30, blank_cost_basis=True)
            readiness_path = Path(tmp) / "readiness.json"
            readiness = onboard_input(source, output_path=readiness_path)

            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(
                as_of="2026-07-01T12:00:00+00:00",
                release_commit=DEFAULT_RELEASE_COMMIT,
                seller_readiness_report=readiness_path,
            )

            self.assertFalse(readiness["data_readiness"]["canary_start_allowed"])
            seller = started["manifest"]["canary_hashes"]["offerlab_seller_pilot"]
            self.assertEqual(seller.get("status"), "blocked")

    def test_operations_rejects_forged_blank_cost_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readiness = {
                "schema_version": "offerlab_seller_pilot_onboarding.v1",
                "data_readiness": {
                    "readiness_gate": {"passed": True},
                    "canary_start_allowed": True,
                    "never_silently_impute_material_costs": True,
                    "material_value_summary": {
                        "cost_basis": {"unit_cost_amount": {"row_count": 30, "valid_count": 0, "blank_or_invalid_count": 30}},
                        "fees": {"fee_amount": {"row_count": 30, "valid_count": 30, "blank_or_invalid_count": 0}},
                        "shipping_costs": {"shipping_cost_amount": {"row_count": 30, "valid_count": 30, "blank_or_invalid_count": 0}},
                        "orders": {"sale_price_amount": {"row_count": 30, "valid_count": 30, "blank_or_invalid_count": 0}},
                    },
                },
                "mapping_approval": {"human_approval_required": False, "material_ambiguities": []},
            }
            readiness_path = Path(tmp) / "forged.json"
            readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

            started = MoneyOperations(Path(tmp) / "ops").start(
                as_of="2026-07-01T12:00:00+00:00",
                release_commit=DEFAULT_RELEASE_COMMIT,
                seller_readiness_report=readiness_path,
            )

            self.assertFalse(started["manifest"]["seller_readiness"]["passed"])
            self.assertEqual(started["manifest"]["canary_hashes"]["offerlab_seller_pilot"].get("status"), "blocked")

    def test_invalidated_canary_remains_behind_final_evidence_gate_after_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(as_of="2026-07-01T00:00:00+00:00", release_commit=DEFAULT_RELEASE_COMMIT)
            weather_id = started["manifest"]["canary_hashes"]["weather_edge"]["canary_id"]
            manager = MoneyCanaryManager(Path(tmp) / "ops" / "canaries")
            start = datetime(2026, 7, 1, tzinfo=timezone.utc)
            for offset in range(1, 60):
                manager.resume(weather_id, as_of=(start + timedelta(days=offset)).isoformat())

            before_invalidation = manager.report(weather_id)
            self.assertTrue(before_invalidation["final_evidence_report"]["available"])

            manager.invalidate(weather_id, reason="test invalidation", as_of="2026-08-30T00:00:00+00:00")
            report = manager.report(weather_id)

            self.assertTrue(report["invalidated"])
            self.assertTrue(report["metrics"]["minimum_duration_elapsed"])
            self.assertFalse(report["final_evidence_report"]["available"])

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
