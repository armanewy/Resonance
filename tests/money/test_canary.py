from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.cli import main
import behavior_lab.money.canary as canary_module
from behavior_lab.money.canary import CanaryOptions, MoneyCanaryError, MoneyCanaryManager, start_fixture_canaries


class MoneyCanaryTests(unittest.TestCase):
    def test_weather_canary_records_immutable_protocol_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            started = manager.start(
                "weather-edge-contract",
                CanaryOptions(lab="weather_edge", as_of="2026-07-01T12:00:00+00:00"),
            )
            canary_id = started["canary_id"]
            resumed = manager.resume(canary_id, as_of="2026-07-02T12:00:00+00:00")
            status = manager.status(canary_id)
            report = manager.report(canary_id)

            self.assertEqual(started["protocol"]["minimum_duration_days"], 60)
            self.assertEqual(started["protocol"]["cadence"], "daily")
            self.assertTrue(started["protocol"]["data_cutoff_policy"]["executable_prices_only"])
            self.assertFalse(started["protocol"]["data_cutoff_policy"]["midpoint_allowed"])
            self.assertFalse(started["protocol"]["notifications_allowed"])
            self.assertEqual(resumed["snapshot"]["snapshot_index"], 2)
            self.assertEqual(status["snapshot_count"], 2)
            self.assertEqual(len(report["source_health_history"]), 2)
            self.assertEqual(len(report["prediction_history"]), 2)
            self.assertEqual(len(report["decision_history"]), 2)
            self.assertEqual(report["final_evidence_report"]["real_money_allowed"], False)
            self.assertFalse(any(report["production_state"].values()))
            self.assertTrue(manager.verify())

    def test_weather_canary_completion_requires_elapsed_consecutive_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            started = manager.start(
                "weather-edge-contract",
                CanaryOptions(lab="weather_edge", as_of="2026-07-01T12:00:00+00:00"),
            )
            canary_id = started["canary_id"]
            for _ in range(59):
                manager.resume(canary_id, as_of="2026-07-01T12:00:00+00:00")

            report = manager.report(canary_id)
            self.assertEqual(report["metrics"]["snapshot_count"], 60)
            self.assertEqual(report["metrics"]["elapsed_days"], 1)
            self.assertEqual(report["metrics"]["distinct_observation_periods"], 1)
            self.assertEqual(report["metrics"]["consecutive_observation_periods"], 1)
            self.assertFalse(report["metrics"]["minimum_duration_elapsed"])
            self.assertFalse(report["final_evidence_report"]["available"])

            manager.resume(canary_id, as_of="2026-08-29T12:00:00+00:00")
            elapsed_report = manager.report(canary_id)
            self.assertEqual(elapsed_report["metrics"]["elapsed_days"], 60)
            self.assertEqual(elapsed_report["metrics"]["distinct_observation_periods"], 2)
            self.assertEqual(elapsed_report["metrics"]["consecutive_observation_periods"], 1)
            self.assertFalse(elapsed_report["metrics"]["minimum_duration_elapsed"])
            self.assertFalse(elapsed_report["final_evidence_report"]["available"])

    def test_weather_canary_completion_rejects_sparse_elapsed_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            start = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
            started = manager.start(
                "weather-edge-contract",
                CanaryOptions(lab="weather_edge", as_of=start.isoformat()),
            )
            canary_id = started["canary_id"]
            for offset in range(2, 120, 2):
                manager.resume(canary_id, as_of=(start + timedelta(days=offset)).isoformat())

            report = manager.report(canary_id)
            self.assertEqual(report["metrics"]["snapshot_count"], 60)
            self.assertEqual(report["metrics"]["distinct_observation_periods"], 60)
            self.assertEqual(report["metrics"]["consecutive_observation_periods"], 1)
            self.assertGreater(report["metrics"]["elapsed_days"], 60)
            self.assertFalse(report["metrics"]["minimum_duration_elapsed"])
            self.assertFalse(report["final_evidence_report"]["available"])

    def test_weather_canary_completion_accepts_distinct_elapsed_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            started = manager.start(
                "weather-edge-contract",
                CanaryOptions(lab="weather_edge", as_of="2026-07-01T12:00:00+00:00"),
            )
            canary_id = started["canary_id"]
            for day in range(1, 60):
                manager.resume(canary_id, as_of=f"2026-07-{day + 1:02d}T12:00:00+00:00" if day < 31 else f"2026-08-{day - 30:02d}T12:00:00+00:00")

            report = manager.report(canary_id)
            self.assertEqual(report["metrics"]["elapsed_days"], 60)
            self.assertEqual(report["metrics"]["distinct_observation_periods"], 60)
            self.assertEqual(report["metrics"]["consecutive_observation_periods"], 60)
            self.assertTrue(report["metrics"]["minimum_duration_elapsed"])
            self.assertTrue(report["final_evidence_report"]["available"])

    def test_strategy_change_is_rejected_on_resume_and_creates_new_canary_on_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            first = manager.start("etf-risk-contract", CanaryOptions(lab="etf_risk", strategy_version="frozen_a"))

            with self.assertRaises(MoneyCanaryError):
                manager.resume(first["canary_id"], strategy_version="mutated_b")

            second = manager.start("etf-risk-contract", CanaryOptions(lab="etf_risk", strategy_version="mutated_b"))
            self.assertNotEqual(first["canary_id"], second["canary_id"])
            self.assertEqual(first["protocol"]["minimum_duration_days"], 183)
            self.assertEqual(first["protocol"]["cadence"], "weekly")
            self.assertTrue(first["protocol"]["cost_assumptions"]["transaction_costs_included"])
            self.assertFalse(first["protocol"]["cost_assumptions"]["leverage_allowed"])

    def test_live_material_policy_change_is_rejected_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            started = manager.start(
                "weather-edge-contract",
                CanaryOptions(lab="weather_edge", as_of="2026-07-01T12:00:00+00:00"),
            )
            original_lab_protocol = canary_module._lab_protocol

            def changed_lab_protocol(lab: str) -> dict[str, object]:
                protocol = json.loads(json.dumps(original_lab_protocol(lab)))
                if lab == "weather_edge":
                    protocol["cost_assumptions"]["slippage_included"] = False
                return protocol

            with mock.patch.object(canary_module, "_lab_protocol", side_effect=changed_lab_protocol):
                with self.assertRaises(MoneyCanaryError):
                    manager.resume(started["canary_id"], as_of="2026-07-02T12:00:00+00:00")

    def test_offerlab_canary_requires_seller_readiness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            manager = MoneyCanaryManager(tmp_name)
            blocked = manager.start("offerlab-seller-contract", CanaryOptions(lab="offerlab_seller_pilot"))
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["reason"], "seller_pilot_readiness_gate_not_passed")
            self.assertEqual([], manager._events("canary_started"))

            started = manager.start(
                "offerlab-seller-contract",
                CanaryOptions(lab="offerlab_seller_pilot", seller_pilot_ready=True),
            )
            self.assertEqual(started["status"], "started")
            self.assertEqual(started["protocol"]["minimum_duration_days"], 30)
            self.assertTrue(started["protocol"]["prospective_gates"]["seller_pilot_readiness_required"])
            self.assertFalse(started["protocol"]["prospective_gates"]["causal_claim_allowed"])

    def test_cli_start_status_report_resume_and_invalidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(
                    [
                        "money",
                        "canary",
                        "start",
                        "weather-cli",
                        "--state-dir",
                        tmp_name,
                        "--lab",
                        "weather_edge",
                        "--as-of",
                        "2026-07-01T12:00:00+00:00",
                    ]
                )
            started = json.loads(stream.getvalue())
            canary_id = started["canary_id"]

            for command in ("status", "report"):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    main(["money", "canary", command, canary_id, "--state-dir", tmp_name])
                payload = json.loads(stream.getvalue())
                self.assertEqual(payload["canary_id"], canary_id)
                self.assertTrue(payload["paper_only"])

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(
                    [
                        "money",
                        "canary",
                        "resume",
                        canary_id,
                        "--state-dir",
                        tmp_name,
                        "--as-of",
                        "2026-07-02T12:00:00+00:00",
                    ]
                )
            self.assertEqual(json.loads(stream.getvalue())["snapshot"]["snapshot_index"], 2)

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "canary", "invalidate", canary_id, "--state-dir", tmp_name, "--reason", "audit fixture invalidation"])
            self.assertEqual(json.loads(stream.getvalue())["status"], "invalidated")

    def test_fixture_canaries_start_three_labs_without_real_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            payload = start_fixture_canaries(tmp_name)

            self.assertEqual(len(payload["canaries"]), 3)
            self.assertEqual({item["lab"] for item in payload["canaries"]}, {"weather_edge", "etf_risk", "offerlab_seller_pilot"})
            self.assertTrue(payload["ledger_valid"])
            self.assertFalse(any(payload["production_state"].values()))
            store_text = (Path(tmp_name) / "canaries.jsonl").read_text(encoding="utf-8")
            self.assertIn("canary_started", store_text)
            self.assertIn("canary_snapshot", store_text)


if __name__ == "__main__":
    unittest.main()
