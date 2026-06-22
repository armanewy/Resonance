from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

from behavior_lab.money.canary import MoneyCanaryManager  # noqa: E402
from behavior_lab.money.operations import MoneyOperations  # noqa: E402


class FinanceOperationsReleaseAuditRegressions(unittest.TestCase):
    def test_forged_readiness_with_invalid_cost_basis_cannot_start_seller_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readiness_path = Path(tmp) / "forged_readiness.json"
            readiness_path.write_text(
                json.dumps(
                    {
                        "schema_version": "offerlab_seller_pilot_onboarding.v1",
                        "data_readiness": {
                            "readiness_gate": {"passed": True},
                            "canary_start_allowed": True,
                            "never_silently_impute_material_costs": True,
                            "material_value_summary": {
                                "cost_basis": {
                                    "unit_cost_amount": {
                                        "row_count": 12,
                                        "valid_count": 0,
                                        "blank_or_invalid_count": 12,
                                    }
                                },
                                "fees": {
                                    "fee_amount": {
                                        "row_count": 12,
                                        "valid_count": 12,
                                        "blank_or_invalid_count": 0,
                                    }
                                },
                                "shipping_costs": {
                                    "shipping_cost_amount": {
                                        "row_count": 12,
                                        "valid_count": 12,
                                        "blank_or_invalid_count": 0,
                                    }
                                },
                                "orders": {
                                    "sale_price_amount": {
                                        "row_count": 12,
                                        "valid_count": 12,
                                        "blank_or_invalid_count": 0,
                                    }
                                },
                            },
                        },
                        "mapping_approval": {
                            "human_approval_required": False,
                            "material_ambiguities": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            started = MoneyOperations(Path(tmp) / "ops").start(
                as_of="2026-07-01T12:00:00+00:00",
                seller_readiness_report=readiness_path,
            )

            self.assertFalse(started["manifest"]["seller_readiness"]["passed"])
            self.assertFalse(started["manifest"]["seller_readiness"]["canary_start_allowed"])
            self.assertEqual(
                started["manifest"]["canary_hashes"]["offerlab_seller_pilot"].get("status"),
                "blocked",
            )

    def test_invalidated_elapsed_canary_keeps_final_evidence_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            operations = MoneyOperations(Path(tmp) / "ops")
            started = operations.start(as_of="2026-07-01T00:00:00+00:00")
            canary_id = started["manifest"]["canary_hashes"]["weather_edge"]["canary_id"]
            manager = MoneyCanaryManager(Path(tmp) / "ops" / "canaries")
            start = datetime(2026, 7, 1, tzinfo=timezone.utc)

            for offset in range(1, 60):
                manager.resume(canary_id, as_of=(start + timedelta(days=offset)).isoformat())

            elapsed = manager.report(canary_id)
            self.assertTrue(elapsed["metrics"]["minimum_duration_elapsed"])
            self.assertTrue(elapsed["final_evidence_report"]["available"])

            manager.invalidate(canary_id, reason="audit invalidation", as_of="2026-08-30T00:00:00+00:00")
            invalidated = manager.report(canary_id)

            self.assertTrue(invalidated["metrics"]["minimum_duration_elapsed"])
            self.assertTrue(invalidated["invalidated"])
            self.assertFalse(invalidated["final_evidence_report"]["available"])
            self.assertFalse(invalidated["final_evidence_report"]["real_money_allowed"])


if __name__ == "__main__":
    unittest.main()
