from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: F401

from behavior_lab.labs.offerlab_money import evaluate, report
from behavior_lab.money.storage import MoneyStorage
from behavior_lab.offerlab_pilot import import_pilot


BASE_TIME = "2026-01-01T12:00:00+00:00"
AVAILABLE_TIME = "2026-01-01T13:00:00+00:00"
RESPONSE_TIME = "2026-01-01T14:00:00+00:00"
PAID_TIME = "2026-01-01T15:00:00+00:00"
COMPLETED_TIME = "2026-01-02T15:00:00+00:00"
MATURED_TIME = "2026-02-15T00:00:00+00:00"
EVALUATED_AT = "2026-03-01T00:00:00+00:00"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pilot_fixture(root: Path, *, omit_cost_listing: str | None = None) -> None:
    listings = []
    offers = []
    orders = []
    fees = []
    shipping = []
    costs = []
    cancellations = []
    returns = []
    inventory = []
    traffic = []
    scenarios = [
        ("001", "accepted", "completed"),
        ("002", "accepted", "returned"),
        ("003", "accepted", "paid"),
        ("004", "accepted", "accepted"),
        ("005", "accepted", "cancelled"),
        ("006", "declined", "unresolved"),
        ("007", "", "unresolved"),
    ]
    for suffix, response, status in scenarios:
        listing_id = f"listing_{suffix}"
        offer_id = f"offer_{suffix}"
        order_id = f"order_{suffix}"
        listings.append(
            {
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "asking_price_amount": "100.00",
                "currency": "USD",
                "category": "electronics",
                "listing_status": "sold" if status in {"completed", "returned", "paid"} else "active",
            }
        )
        offer = {
            "offer_id": offer_id,
            "listing_id": listing_id,
            "event_time": BASE_TIME,
            "available_at": AVAILABLE_TIME,
            "offer_amount": "90.00",
            "currency": "USD",
            "offer_state": response or "pending",
            "seller_response": response,
            "seller_response_time": RESPONSE_TIME if response else "",
            "seller_response_amount": "95.00" if response == "countered" else "",
            "decision_history_available_at": AVAILABLE_TIME,
            "expires_at": "2026-01-03T00:00:00+00:00",
        }
        offers.append(offer)
        if status in {"completed", "returned", "paid"}:
            orders.append(
                {
                    "order_id": order_id,
                    "listing_id": listing_id,
                    "offer_id": offer_id,
                    "event_time": PAID_TIME,
                    "available_at": PAID_TIME,
                    "sale_price_amount": "90.00",
                    "currency": "USD",
                    "order_status": "completed" if status in {"completed", "returned"} else "paid",
                    "paid_at": PAID_TIME,
                    "completed_at": COMPLETED_TIME if status in {"completed", "returned"} else "",
                    "return_window_matured_at": MATURED_TIME if status in {"completed", "returned"} else "",
                    "quantity": "1",
                }
            )
            fees.append(
                {
                    "fee_id": f"fee_{suffix}",
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
                    "shipping_id": f"ship_{suffix}",
                    "order_id": order_id,
                    "event_time": PAID_TIME,
                    "available_at": PAID_TIME,
                    "shipping_cost_amount": "8.00",
                    "currency": "USD",
                }
            )
        if status == "returned":
            returns.append(
                {
                    "return_id": f"return_{suffix}",
                    "order_id": order_id,
                    "event_time": "2026-01-10T00:00:00+00:00",
                    "available_at": "2026-01-10T00:00:00+00:00",
                    "refund_amount": "90.00",
                    "currency": "USD",
                    "listing_id": listing_id,
                    "return_opened_at": "2026-01-09T00:00:00+00:00",
                    "return_closed_at": "2026-01-10T00:00:00+00:00",
                    "return_window_matured_at": MATURED_TIME,
                    "return_status": "returned",
                }
            )
        if status == "cancelled":
            cancellations.append(
                {
                    "cancellation_id": f"cancel_{suffix}",
                    "event_time": "2026-01-01T16:00:00+00:00",
                    "available_at": "2026-01-01T16:00:00+00:00",
                    "event_type": "cancellation",
                    "currency": "USD",
                    "order_id": "",
                    "listing_id": listing_id,
                    "offer_id": offer_id,
                    "amount": "0.00",
                }
            )
        if omit_cost_listing != listing_id:
            costs.append(
                {
                    "cost_basis_id": f"cost_{suffix}",
                    "listing_id": listing_id,
                    "event_time": BASE_TIME,
                    "available_at": AVAILABLE_TIME,
                    "unit_cost_amount": "40.00",
                    "currency": "USD",
                    "sku": f"sku_{suffix}",
                    "cost_source": "seller_documented",
                }
            )
        inventory.append(
            {
                "inventory_id": f"inventory_{suffix}",
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "quantity_available": "1",
                "inventory_age_days": "45",
            }
        )
        traffic.append(
            {
                "traffic_id": f"traffic_{suffix}",
                "listing_id": listing_id,
                "event_time": BASE_TIME,
                "available_at": AVAILABLE_TIME,
                "impressions": "10",
                "views": "2",
            }
        )

    _write_csv(
        root / "listings.csv",
        ["listing_id", "event_time", "available_at", "asking_price_amount", "currency", "category", "listing_status"],
        listings,
    )
    _write_csv(
        root / "offers.csv",
        [
            "offer_id",
            "listing_id",
            "event_time",
            "available_at",
            "offer_amount",
            "currency",
            "offer_state",
            "seller_response",
            "seller_response_time",
            "seller_response_amount",
            "decision_history_available_at",
            "expires_at",
        ],
        offers,
    )
    _write_csv(
        root / "orders.csv",
        [
            "order_id",
            "listing_id",
            "offer_id",
            "event_time",
            "available_at",
            "sale_price_amount",
            "currency",
            "order_status",
            "paid_at",
            "completed_at",
            "return_window_matured_at",
            "quantity",
        ],
        orders,
    )
    _write_csv(root / "fees.csv", ["fee_id", "order_id", "event_time", "available_at", "fee_amount", "currency", "fee_type"], fees)
    _write_csv(
        root / "shipping_costs.csv",
        ["shipping_id", "order_id", "event_time", "available_at", "shipping_cost_amount", "currency"],
        shipping,
    )
    _write_csv(
        root / "cost_basis.csv",
        ["cost_basis_id", "listing_id", "event_time", "available_at", "unit_cost_amount", "currency", "sku", "cost_source"],
        costs,
    )
    _write_csv(
        root / "cancellations_unpaid.csv",
        ["cancellation_id", "event_time", "available_at", "event_type", "currency", "order_id", "listing_id", "offer_id", "amount"],
        cancellations,
    )
    _write_csv(
        root / "returns_refunds.csv",
        [
            "return_id",
            "order_id",
            "event_time",
            "available_at",
            "refund_amount",
            "currency",
            "listing_id",
            "return_opened_at",
            "return_closed_at",
            "return_window_matured_at",
            "return_status",
        ],
        returns,
    )
    _write_csv(
        root / "inventory.csv",
        ["inventory_id", "listing_id", "event_time", "available_at", "quantity_available", "inventory_age_days"],
        inventory,
    )
    _write_csv(
        root / "traffic.csv",
        ["traffic_id", "listing_id", "event_time", "available_at", "impressions", "views"],
        traffic,
    )


class OfferLabMoneyEvaluationTests(unittest.TestCase):
    def test_evaluate_maps_each_seller_decision_to_paper_contract_and_entry(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_fixture(source)
            import_pilot(source, data_root=data_tmp, pilot_id="wave2a")
            money_root = Path(data_tmp) / "money"

            result = evaluate("wave2a", data_root=data_tmp, money_root=money_root, evaluation_timestamp=EVALUATED_AT)

            self.assertEqual(result["decisions_seen"], 7)
            self.assertEqual(result["contracts_written"], 7)
            self.assertEqual(result["ledger_entries_appended"], 7)
            self.assertFalse(result["executes_seller_actions"])
            self.assertFalse(result["submits_seller_actions"])
            self.assertFalse(result["financial_action"])
            self.assertFalse(result["notifications_allowed"])
            self.assertFalse(result["causal_profit_lift_claimed"])
            self.assertEqual(result["benchmark_v2_evidence"]["role"], "research_only_evidence")
            self.assertEqual(result["status_counts"]["completed"], 1)
            self.assertEqual(result["status_counts"]["returned"], 1)
            self.assertEqual(result["status_counts"]["paid"], 1)
            self.assertEqual(result["status_counts"]["accepted"], 1)
            self.assertEqual(result["status_counts"]["cancelled"], 1)
            self.assertEqual(result["status_counts"]["unresolved"], 2)

            storage = MoneyStorage(money_root)
            entries = {entry.provenance["seller_row_keys"]["offer_id"]: entry for entry in storage.ledger.latest_entries()}
            self.assertEqual(set(entries), {f"offer_{suffix}" for suffix in ("001", "002", "003", "004", "005", "006", "007")})
            completed = entries["offer_001"]
            self.assertEqual(completed.selected_action, "accept")
            self.assertEqual(completed.provenance["financial_status"], "completed")
            self.assertEqual(completed.provenance["timestamps"]["offer_timestamp"], BASE_TIME)
            self.assertEqual(completed.provenance["timestamps"]["response_timestamp"], RESPONSE_TIME)
            self.assertEqual(completed.provenance["timestamps"]["payment_timestamp"], PAID_TIME)
            self.assertEqual(completed.provenance["timestamps"]["return_window_matured_at"], MATURED_TIME)
            self.assertEqual(completed.fees, 12.0)
            self.assertEqual(completed.shipping, 8.0)
            self.assertEqual(completed.holding_costs, 40.0)
            self.assertEqual(completed.conservative_expected_net_value, 30.0)
            self.assertEqual(completed.provenance["cost_basis"]["total_cost_basis"], 40.0)

            returned = entries["offer_002"]
            self.assertEqual(returned.provenance["financial_status"], "returned")
            self.assertEqual(returned.return_refund_allowance, 90.0)
            self.assertEqual(returned.conservative_expected_net_value, -60.0)

            self.assertEqual(entries["offer_003"].provenance["financial_status"], "paid")
            self.assertIn("return_window_not_matured", entries["offer_003"].ineligibility_reasons)
            self.assertEqual(entries["offer_004"].provenance["financial_status"], "accepted")
            self.assertEqual(entries["offer_005"].provenance["financial_status"], "cancelled")
            self.assertEqual(entries["offer_005"].provenance["cancellation"]["events"][0]["event_time"], "2026-01-01T16:00:00+00:00")
            self.assertEqual(entries["offer_006"].selected_action, "decline")
            self.assertEqual(entries["offer_006"].conservative_expected_net_value, 0.0)
            self.assertEqual(entries["offer_007"].selected_action, "abstain")
            self.assertTrue(entries["offer_007"].provenance["explicit_silence"])
            self.assertIn("insufficient_seller_data", entries["offer_007"].ineligibility_reasons)

            for contract in storage.list_contracts():
                actions = {action["action_id"]: action for action in contract["available_actions"]}
                self.assertEqual(set(actions), {"abstain", "decline", "accept", "counter"})
                self.assertTrue(actions["counter"]["parameters"]["bounded_values_only"])
                self.assertTrue(contract["paper_only"])
                self.assertFalse(contract["risk_policy"]["seller_action_submitted"])
                self.assertFalse(contract["notification_threshold"]["notifications_allowed"])

            second = evaluate("wave2a", data_root=data_tmp, money_root=money_root, evaluation_timestamp=EVALUATED_AT)
            self.assertEqual(second["ledger_entries_appended"], 0)
            self.assertEqual(second["ledger_entries_skipped_existing"], 7)

    def test_unknown_cost_basis_makes_net_profit_claim_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_fixture(source, omit_cost_listing="listing_001")
            import_pilot(source, data_root=data_tmp, pilot_id="missing_cost")
            money_root = Path(data_tmp) / "money"

            result = evaluate("missing_cost", data_root=data_tmp, money_root=money_root, evaluation_timestamp=EVALUATED_AT)

            self.assertEqual(result["unknown_cost_basis_count"], 1)
            entries = {entry.provenance["seller_row_keys"]["offer_id"]: entry for entry in MoneyStorage(money_root).ledger.latest_entries()}
            completed = entries["offer_001"]
            self.assertFalse(completed.material_costs_known)
            self.assertIsNone(completed.conservative_expected_net_value)
            self.assertIn("unknown_cost_basis", completed.ineligibility_reasons)
            self.assertFalse(completed.provenance["net_profit_claim_eligible"])

    def test_report_summarizes_shadow_value_without_causal_lift_claims(self) -> None:
        with tempfile.TemporaryDirectory() as input_tmp, tempfile.TemporaryDirectory() as data_tmp:
            source = Path(input_tmp)
            _pilot_fixture(source)
            import_pilot(source, data_root=data_tmp, pilot_id="reportable")
            money_root = Path(data_tmp) / "money"
            evaluate("reportable", data_root=data_tmp, money_root=money_root, evaluation_timestamp=EVALUATED_AT)

            payload = report("reportable", data_root=data_tmp, money_root=money_root)

            self.assertFalse(payload["causal_profit_lift_claimed"])
            self.assertEqual(payload["historical_policy_claim"], "descriptive_shadow_value_against_documented_historical_action_not_causal_lift")
            self.assertEqual(payload["benchmark_v2_evidence"]["role"], "research_only_evidence")
            self.assertEqual(payload["decision_count"], 7)
            self.assertEqual(payload["explicit_silence_count"], 1)
            self.assertEqual(payload["financial_status_counts"]["completed"], 1)
            self.assertEqual(payload["financial_status_counts"]["returned"], 1)
            self.assertEqual(payload["conservative_shadow_value"]["basis"], "seller_documented_historical_action")
            self.assertFalse(payload["conservative_shadow_value"]["causal_lift_claimed"])
            self.assertEqual(payload["conservative_shadow_value"]["total"], -30.0)

    def test_report_is_explicitly_silent_without_evaluation_entries(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            payload = report("not_evaluated", data_root=data_tmp, money_root=Path(data_tmp) / "money")

            self.assertTrue(payload["explicit_silence"]["applies"])
            self.assertEqual(payload["explicit_silence"]["reason"], "no_offerlab_money_evaluation_entries")
            self.assertFalse(payload["executes_seller_actions"])
            self.assertEqual(payload["decision_count"], 0)


if __name__ == "__main__":
    unittest.main()
