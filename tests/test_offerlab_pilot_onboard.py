from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.cli import main
from behavior_lab.money.integration import _write_offerlab_fixture
from behavior_lab.offerlab_pilot import onboard_input


class OfferLabPilotOnboardingTests(unittest.TestCase):
    def test_onboard_proposes_mappings_and_blocks_canary_until_readiness_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)

            report = onboard_input(source)

            self.assertTrue(report["local_only"])
            self.assertFalse(report["uploads_seller_data"])
            self.assertFalse(report["executes_seller_actions"])
            self.assertFalse(report["llm_boundary"]["llm_used"])
            self.assertTrue(report["llm_boundary"]["deterministic_validation_controls_readiness"])
            self.assertIn("offer_amount", set(report["datasets"]["offers"]["proposed_column_mapping"].values()))
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])
            self.assertFalse(report["data_readiness"]["readiness_gate"]["passed"])
            self.assertTrue(report["data_readiness"]["never_silently_impute_material_costs"])

    def test_onboard_flags_material_mapping_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)
            _append_column(source / "offers.csv", "Offer Price", "91.00")

            report = onboard_input(source)

            self.assertTrue(report["mapping_approval"]["human_approval_required"])
            self.assertTrue(
                any(item["reason"] == "ambiguous_column_mapping" for item in report["mapping_approval"]["material_ambiguities"])
            )
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])

    def test_onboard_flags_missing_cost_basis_as_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)
            (source / "cost_basis.csv").unlink()

            report = onboard_input(source)

            self.assertTrue(report["mapping_approval"]["human_approval_required"])
            self.assertTrue(
                any(item["dataset"] == "cost_basis" and item["reason"] == "missing_material_dataset" for item in report["mapping_approval"]["material_ambiguities"])
            )
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])

    def test_onboard_rejects_blank_material_cost_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_many_complete_rows(source, rows=30, blank_cost_basis=True)

            report = onboard_input(source)

            self.assertFalse(report["data_readiness"]["canary_start_allowed"])
            self.assertFalse(report["data_readiness"]["readiness_gate"]["passed"])
            material = report["data_readiness"]["material_value_summary"]
            self.assertEqual(material["cost_basis"]["unit_cost_amount"]["valid_count"], 0)
            self.assertEqual(material["cost_basis"]["unit_cost_amount"]["blank_or_invalid_count"], 30)
            self.assertTrue(
                any(item["reason"] == "blank_or_invalid_material_values" for item in report["mapping_approval"]["material_ambiguities"])
            )

    def test_cli_onboard_writes_data_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            output = Path(tmp) / "readiness.json"
            source.mkdir()
            _write_offerlab_fixture(source)

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["offerlab-pilot", "onboard", str(source), "--output", str(output)])

            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["report_hash"], json.loads(output.read_text(encoding="utf-8"))["report_hash"])
            self.assertFalse(payload["data_readiness"]["canary_start_allowed"])


def _append_column(path: Path, column: str, value: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) + [column]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, column: value})


def _write_many_complete_rows(root: Path, *, rows: int, blank_cost_basis: bool) -> None:
    base = "2026-01-01T12:00:00+00:00"
    available = "2026-01-01T13:00:00+00:00"
    paid = "2026-01-01T15:00:00+00:00"
    completed = "2026-01-02T15:00:00+00:00"
    matured = "2026-02-15T00:00:00+00:00"
    datasets = {
        "listings": (
            ["listing_id", "event_time", "available_at", "asking_price_amount", "currency", "category", "listing_status"],
            [
                {
                    "listing_id": f"listing_{index:03d}",
                    "event_time": base,
                    "available_at": available,
                    "asking_price_amount": "100.00",
                    "currency": "USD",
                    "category": "electronics",
                    "listing_status": "sold",
                }
                for index in range(rows)
            ],
        ),
        "offers": (
            ["offer_id", "listing_id", "event_time", "available_at", "offer_amount", "currency", "offer_state", "seller_response", "seller_response_time"],
            [
                {
                    "offer_id": f"offer_{index:03d}",
                    "listing_id": f"listing_{index:03d}",
                    "event_time": base,
                    "available_at": available,
                    "offer_amount": "90.00",
                    "currency": "USD",
                    "offer_state": "accepted",
                    "seller_response": "accepted",
                    "seller_response_time": "2026-01-01T14:00:00+00:00",
                }
                for index in range(rows)
            ],
        ),
        "orders": (
            ["order_id", "listing_id", "offer_id", "event_time", "available_at", "sale_price_amount", "currency", "order_status", "paid_at", "completed_at", "return_window_matured_at", "quantity"],
            [
                {
                    "order_id": f"order_{index:03d}",
                    "listing_id": f"listing_{index:03d}",
                    "offer_id": f"offer_{index:03d}",
                    "event_time": paid,
                    "available_at": paid,
                    "sale_price_amount": "90.00",
                    "currency": "USD",
                    "order_status": "completed",
                    "paid_at": paid,
                    "completed_at": completed,
                    "return_window_matured_at": matured,
                    "quantity": "1",
                }
                for index in range(rows)
            ],
        ),
        "fees": (
            ["fee_id", "order_id", "event_time", "available_at", "fee_amount", "currency", "fee_type"],
            [
                {
                    "fee_id": f"fee_{index:03d}",
                    "order_id": f"order_{index:03d}",
                    "event_time": paid,
                    "available_at": paid,
                    "fee_amount": "12.00",
                    "currency": "USD",
                    "fee_type": "final_value",
                }
                for index in range(rows)
            ],
        ),
        "shipping_costs": (
            ["shipping_id", "order_id", "event_time", "available_at", "shipping_cost_amount", "currency"],
            [
                {
                    "shipping_id": f"ship_{index:03d}",
                    "order_id": f"order_{index:03d}",
                    "event_time": paid,
                    "available_at": paid,
                    "shipping_cost_amount": "8.00",
                    "currency": "USD",
                }
                for index in range(rows)
            ],
        ),
        "cost_basis": (
            ["cost_basis_id", "listing_id", "event_time", "available_at", "unit_cost_amount", "currency"],
            [
                {
                    "cost_basis_id": f"cost_{index:03d}",
                    "listing_id": f"listing_{index:03d}",
                    "event_time": base,
                    "available_at": available,
                    "unit_cost_amount": "" if blank_cost_basis else "40.00",
                    "currency": "USD",
                }
                for index in range(rows)
            ],
        ),
        "cancellations_unpaid": (["cancellation_id", "event_time", "available_at", "event_type", "currency"], []),
        "returns_refunds": (["return_id", "order_id", "event_time", "available_at", "refund_amount", "currency"], []),
        "inventory": (
            ["inventory_id", "listing_id", "event_time", "available_at", "quantity_available"],
            [
                {
                    "inventory_id": f"inventory_{index:03d}",
                    "listing_id": f"listing_{index:03d}",
                    "event_time": base,
                    "available_at": available,
                    "quantity_available": "1",
                }
                for index in range(rows)
            ],
        ),
        "traffic": (
            ["traffic_id", "listing_id", "event_time", "available_at", "impressions", "views"],
            [
                {
                    "traffic_id": f"traffic_{index:03d}",
                    "listing_id": f"listing_{index:03d}",
                    "event_time": base,
                    "available_at": available,
                    "impressions": "10",
                    "views": "2",
                }
                for index in range(rows)
            ],
        ),
    }
    for dataset, (fieldnames, payload_rows) in datasets.items():
        with (root / f"{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload_rows)


if __name__ == "__main__":
    unittest.main()
