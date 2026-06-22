from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from behavior_lab.offerlab_pilot import (
    OfferLabPilotError,
    audit_pilot,
    import_pilot,
    inspect_input,
    write_template,
)


BASE_TIME = "2026-01-01T12:00:00+00:00"
AVAILABLE_TIME = "2026-01-01T13:00:00+00:00"
PAID_TIME = "2026-01-01T14:00:00+00:00"
COMPLETED_TIME = "2026-01-02T14:00:00+00:00"
MATURED_TIME = "2026-02-15T00:00:00+00:00"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pilot_files(root: Path, *, count: int = 30, omit_last_cost: bool = False) -> None:
    listings = []
    offers = []
    orders = []
    fees = []
    shipping = []
    costs = []
    inventory = []
    for index in range(count):
        listing_id = f"listing_{index:03d}"
        offer_id = f"offer_{index:03d}"
        order_id = f"order_{index:03d}"
        listings.append(
            {
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "asking_price_amount": "100.00",
                "currency": "USD",
                "category": "collectibles" if index % 2 else "electronics",
                "listing_status": "sold",
            }
        )
        offers.append(
            {
                "offer_id": offer_id,
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "offer_amount": "90.00",
                "currency": "USD",
                "offer_state": "accepted",
                "seller_response": "accepted",
                "seller_response_time": "2026-01-01T15:00:00+00:00",
            }
        )
        orders.append(
            {
                "order_id": order_id,
                "listing_id": listing_id,
                "offer_id": offer_id,
                "event_time": PAID_TIME,
                "available_at": PAID_TIME,
                "sale_price_amount": "90.00",
                "currency": "USD",
                "order_status": "completed",
                "paid_at": PAID_TIME,
                "completed_at": COMPLETED_TIME,
                "return_window_matured_at": MATURED_TIME,
                "quantity": "1",
            }
        )
        fees.append(
            {
                "fee_id": f"fee_{index:03d}",
                "order_id": order_id,
                "event_time": PAID_TIME,
                "available_at": PAID_TIME,
                "fee_amount": "12.00",
                "currency": "USD",
                "fee_type": "final_value",
            }
        )
        shipping.append(
            {
                "shipping_id": f"ship_{index:03d}",
                "order_id": order_id,
                "event_time": PAID_TIME,
                "available_at": PAID_TIME,
                "shipping_cost_amount": "8.00",
                "currency": "USD",
            }
        )
        if not (omit_last_cost and index == count - 1):
            costs.append(
                {
                    "cost_basis_id": f"cost_{index:03d}",
                    "listing_id": listing_id,
                    "event_time": BASE_TIME,
                    "available_at": AVAILABLE_TIME,
                    "unit_cost_amount": "40.00",
                    "currency": "USD",
                }
            )
        inventory.append(
            {
                "inventory_id": f"inv_{index:03d}",
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "quantity_available": "0",
                "inventory_age_days": "45",
            }
        )
    _write_csv(root / "listings.csv", listings)
    _write_csv(root / "offers.csv", offers)
    _write_csv(root / "orders.csv", orders)
    _write_csv(root / "fees.csv", fees)
    _write_csv(root / "shipping_costs.csv", shipping)
    _write_csv(root / "cost_basis.csv", costs)
    _write_csv(root / "inventory.csv", inventory)
    (root / "traffic.json").write_text(
        json.dumps(
            [
                {
                    "traffic_id": "traffic_001",
                    "listing_id": "listing_000",
                    "event_time": BASE_TIME,
                    "available_at": AVAILABLE_TIME,
                    "impressions": 100,
                    "views": 10,
                }
            ]
        ),
        encoding="utf-8",
    )


class OfferLabPilotTests(unittest.TestCase):
    def test_template_writes_all_dataset_templates_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_template(tmp)
            self.assertTrue(Path(result["manifest"]).exists())
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertIn("listings", manifest["datasets"])
            self.assertIn("returns_refunds", manifest["datasets"])
            self.assertTrue((Path(tmp) / "offers.csv").exists())

    def test_inspect_requires_every_source_column_to_be_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "listings.csv",
                [
                    {
                        "listing_id": "listing_1",
                        "event_time": BASE_TIME,
                        "available_at": AVAILABLE_TIME,
                        "asking_price_amount": "100.00",
                        "currency": "USD",
                        "category": "electronics",
                        "listing_status": "active",
                        "unexpected_export_column": "x",
                    }
                ],
            )
            report = inspect_input(root)
            self.assertFalse(report["ready_to_import"])
            self.assertIn("unmapped source columns", report["errors"][0])

    def test_import_hashes_versions_and_audit_reports_mature_margin(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_files(source)
            result = import_pilot(source, data_root=data_tmp, pilot_id="pilot_test")
            self.assertEqual(result.imported_rows, 211)
            second = import_pilot(source, data_root=data_tmp, pilot_id="pilot_test")
            self.assertTrue(second.skipped_existing)
            audit = audit_pilot("pilot_test", data_root=data_tmp)
            self.assertFalse(audit["executes_seller_actions"])
            self.assertEqual(audit["offer_funnel"]["seller_accepted"], 30)
            self.assertEqual(audit["offer_funnel"]["buyer_paid"], 30)
            self.assertEqual(audit["offer_funnel"]["order_completed"], 30)
            self.assertEqual(audit["offer_funnel"]["return_window_matured"], 30)
            self.assertEqual(audit["mature_contribution_margin"]["total"], 900.0)
            self.assertEqual(audit["realized_price_vs_asking"]["average_ratio"], 0.9)
            self.assertTrue(audit["readiness_gate"]["passed"])
            self.assertTrue(audit["shadow_evaluation_possible"])

    def test_audit_reports_missing_cost_basis_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_files(source, omit_last_cost=True)
            import_pilot(source, data_root=data_tmp, pilot_id="pilot_missing_cost")
            audit = audit_pilot("pilot_missing_cost", data_root=data_tmp)
            self.assertIn("listing_029", audit["data_quality_gaps"]["missing_cost_basis_listing_ids"])
            self.assertEqual(audit["mature_contribution_margin"]["orders"], 29)
            self.assertFalse(audit["readiness_gate"]["passed"])
            self.assertTrue(audit["data_quality_gaps"]["never_imputed_costs"])

    def test_import_rejects_bad_currency(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_files(source, count=1)
            text = (source / "listings.csv").read_text(encoding="utf-8")
            (source / "listings.csv").write_text(text.replace("USD", "usd", 1), encoding="utf-8")
            with self.assertRaises(OfferLabPilotError):
                import_pilot(source, data_root=data_tmp, pilot_id="bad_currency")

    def test_cli_offerlab_pilot_import_and_audit_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_files(source)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            imported = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "behavior_lab",
                    "offerlab-pilot",
                    "import",
                    str(source),
                    "--data-root",
                    data_tmp,
                    "--pilot-id",
                    "cli_pilot",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            audited = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "behavior_lab",
                    "offerlab-pilot",
                    "audit",
                    "cli_pilot",
                    "--data-root",
                    data_tmp,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(json.loads(imported.stdout)["pilot_id"], "cli_pilot")
            self.assertEqual(json.loads(audited.stdout)["offer_funnel"]["mature_margin_count"], 30)


if __name__ == "__main__":
    unittest.main()
