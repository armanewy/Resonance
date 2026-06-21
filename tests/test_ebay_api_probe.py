from __future__ import annotations

import _bootstrap  # noqa: F401

import unittest

from tools.ebay_api_probe import EbayApiProbe, ProbeError, StaticProbeClient
from tools.ebay_api_probe.probe import READ_ONLY_SCOPES


class EbayApiProbeTests(unittest.TestCase):
    def test_probe_builds_redacted_field_and_permission_matrix(self) -> None:
        client = StaticProbeClient(
            {
                "seller_owned_best_offers": {"status": 200, "bestOffers": [{"price": {"value": "72.00"}, "buyer": {"userId": "x"}}]},
                "buyer_participated_best_offers": {"status": 200, "bestOffers": [{"offerPrice": {"value": "70.00"}}]},
                "unrelated_best_offers_denied": {"status": 403, "error": "denied"},
                "inventory_read": {"status": 200, "inventoryItemGroupKey": "g1"},
                "orders_read": {"status": 200, "orderId": "o1", "lineItems": []},
                "finances_read": {"status": 200, "feeType": "FINAL_VALUE_FEE", "amount": {"value": "1.00"}},
                "traffic_read": {"status": 200, "impressions": 10, "views": 2},
            }
        )
        report = EbayApiProbe(client).run(
            scopes=sorted(READ_ONLY_SCOPES),
            seller_owned_listing_id="seller-item",
            buyer_participated_listing_id="buyer-item",
            unrelated_listing_id="other-item",
        )
        self.assertTrue(report["read_only"])
        self.assertFalse(report["mutation_endpoints_called"])
        self.assertFalse(report["message_content_collected"])
        self.assertFalse(report["message_content_detected"])
        self.assertFalse(report["field_matrix"]["seller_owned_best_offers"]["message_content"])
        self.assertTrue(report["field_matrix"]["seller_owned_best_offers"]["offer_amount"])
        self.assertTrue(report["permission_matrix"]["unrelated_best_offers_denied"]["denied_as_expected"])
        self.assertEqual(len(client.calls), 7)

    def test_probe_rejects_broad_scopes_and_mutation_paths(self) -> None:
        probe = EbayApiProbe(StaticProbeClient({}))
        with self.assertRaises(ProbeError):
            probe.run(
                scopes=["https://api.ebay.com/oauth/api_scope/sell.inventory"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
            )
        with self.assertRaises(ProbeError):
            probe._validate_path("/ws/api.dll?callname=RespondToBestOffer")
        with self.assertRaises(ProbeError):
            probe._validate_path("/ws/api.dll?callname=AddItem")
        with self.assertRaises(ProbeError):
            probe._validate_path("/sell/inventory/v1/inventory_item/sku")

    def test_probe_rejects_unrelated_success_and_detects_message_fields(self) -> None:
        responses = {
            "seller_owned_best_offers": {"status": 200, "bestOffers": [{"message": "do not collect"}]},
            "buyer_participated_best_offers": {"status": 200},
            "unrelated_best_offers_denied": {"status": 200},
        }
        for key in ["inventory_read", "orders_read", "finances_read", "traffic_read"]:
            responses[key] = {"status": 200}
        with self.assertRaises(ProbeError):
            EbayApiProbe(StaticProbeClient(responses)).run(
                scopes=["https://api.ebay.com/oauth/api_scope"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
            )
        responses["unrelated_best_offers_denied"] = {"status": 403}
        report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
        )
        self.assertTrue(report["message_content_detected"])
        self.assertTrue(report["message_content_violation"])
        self.assertFalse(report["field_matrix"]["seller_owned_best_offers"]["message_content"])

    def test_compare_modes_returns_field_matrix_diff_without_raw_payloads(self) -> None:
        sandbox = {"mode": "sandbox", "field_matrix": {"traffic_read": {"traffic": False}}}
        production = {"mode": "production", "field_matrix": {"traffic_read": {"traffic": True}}}
        comparison = EbayApiProbe.compare_modes(sandbox, production)
        self.assertFalse(comparison["raw_payloads_retained"])
        self.assertFalse(comparison["field_matrix_comparison"]["traffic_read"]["traffic"]["sandbox"])
        self.assertTrue(comparison["field_matrix_comparison"]["traffic_read"]["traffic"]["production"])


if __name__ == "__main__":
    unittest.main()
