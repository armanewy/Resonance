from __future__ import annotations

import _bootstrap  # noqa: F401

import unittest

from tools.ebay_api_probe import EbayApiProbe, ProbeError, StaticProbeClient
from tools.ebay_api_probe.cli import build_parser
from tools.ebay_api_probe.probe import READ_ONLY_SCOPES, ROLE_REQUESTS


class EbayApiProbeTests(unittest.TestCase):
    def test_probe_builds_redacted_field_and_permission_matrix(self) -> None:
        client = StaticProbeClient(
            {
                "seller_owned_best_offers": {
                    "status": 200,
                    "bestOffers": [
                        {
                            "price": {"value": "72.00", "currencyID": "USD"},
                            "buyer": {"userId": "raw-buyer-id"},
                            "bestOfferStatus": "Active",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "orderId": "raw-order-id",
                        }
                    ],
                },
                "buyer_participated_best_offers": {
                    "status": 200,
                    "bestOffers": [
                        {
                            "offerPrice": {"value": "70.00", "currency": "USD"},
                            "buyerUserId": "raw-buyer-id",
                            "offerStatus": "Accepted",
                            "timestamp": "2026-01-02T00:00:00Z",
                        }
                    ],
                },
                ROLE_REQUESTS["unrelated_public_ended"]: {"status": 403, "error": "denied"},
            }
        )
        report = EbayApiProbe(client).run(
            scopes=sorted(READ_ONLY_SCOPES),
            seller_owned_listing_id="seller-item",
            buyer_participated_listing_id="buyer-item",
            unrelated_listing_id="other-item",
            authorized_production_user_token=True,
        )
        rendered = str(report)
        self.assertTrue(report["read_only"])
        self.assertFalse(report["mutation_endpoints_called"])
        self.assertFalse(report["message_content_collected"])
        self.assertFalse(report["message_content_detected"])
        self.assertFalse(report["field_matrix"]["seller_owned_best_offers"]["message_content"])
        self.assertTrue(report["field_matrix"]["seller_owned_best_offers"]["offer_amount"])
        self.assertTrue(report["access_context_matrix"]["seller_owned"]["offer_history_available"])
        self.assertTrue(report["access_context_matrix"]["seller_owned"]["offer_actor_available_redacted"])
        self.assertEqual(report["feasibility"], "technically feasible")
        self.assertEqual(report["permission_matrix"][ROLE_REQUESTS["unrelated_public_ended"]]["observed_result"], "denied")
        self.assertEqual(report["unrelated_visibility_observation"], "denied")
        self.assertEqual(len(client.calls), 3)
        self.assertNotIn("seller-item", rendered)
        self.assertNotIn("buyer-item", rendered)
        self.assertNotIn("other-item", rendered)
        self.assertNotIn("raw-buyer-id", rendered)

    def test_probe_rejects_broad_scopes_and_mutation_paths(self) -> None:
        probe = EbayApiProbe(StaticProbeClient({}))
        with self.assertRaises(ProbeError):
            probe.run(
                scopes=["https://api.ebay.com/oauth/api_scope/sell.inventory"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
                authorized_production_user_token=True,
            )
        with self.assertRaises(ProbeError):
            probe._validate_path("/ws/api.dll?callname=RespondToBestOffer")
        with self.assertRaises(ProbeError):
            probe._validate_path("/ws/api.dll?callname=AddItem")
        with self.assertRaises(ProbeError):
            probe._validate_path("/sell/inventory/v1/inventory_item")

    def test_cli_requires_explicit_production_token_env_and_scopes(self) -> None:
        parser = build_parser()
        valid = parser.parse_args(
            [
                "--mode",
                "production",
                "--token-env",
                "EBAY_PRODUCTION_USER_TOKEN",
                "--scope",
                "https://api.ebay.com/oauth/api_scope",
                "--seller-owned-listing-id",
                "seller",
                "--buyer-participated-listing-id",
                "buyer",
                "--unrelated-listing-id",
                "unrelated",
            ]
        )
        self.assertEqual(valid.mode, "production")
        self.assertEqual(valid.token_env, "EBAY_PRODUCTION_USER_TOKEN")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--mode",
                    "sandbox",
                    "--token-env",
                    "EBAY_SANDBOX_ACCESS_TOKEN",
                    "--scope",
                    "https://api.ebay.com/oauth/api_scope",
                    "--seller-owned-listing-id",
                    "seller",
                    "--buyer-participated-listing-id",
                    "buyer",
                    "--unrelated-listing-id",
                    "unrelated",
                ]
            )

    def test_probe_requires_production_authorization_and_manual_distinct_ids(self) -> None:
        probe = EbayApiProbe(StaticProbeClient({}))
        with self.assertRaisesRegex(ProbeError, "authorized production user token"):
            probe.run(scopes=["https://api.ebay.com/oauth/api_scope"], seller_owned_listing_id="a", buyer_participated_listing_id="b", unrelated_listing_id="c")
        with self.assertRaisesRegex(ProbeError, "distinct"):
            probe.run(
                scopes=["https://api.ebay.com/oauth/api_scope"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="a",
                unrelated_listing_id="c",
                authorized_production_user_token=True,
            )
        with self.assertRaisesRegex(ProbeError, "exactly one"):
            probe.run(
                scopes=["https://api.ebay.com/oauth/api_scope"],
                seller_owned_listing_id="a,b",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
                authorized_production_user_token=True,
            )
        with self.assertRaisesRegex(ProbeError, "production mode"):
            EbayApiProbe(StaticProbeClient({}), sandbox=True).run(
                scopes=["https://api.ebay.com/oauth/api_scope"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
                authorized_production_user_token=True,
            )

    def test_probe_rejects_unrelated_success_and_detects_message_fields(self) -> None:
        responses = {
            "seller_owned_best_offers": {"status": 200, "bestOffers": [{"message": "do not collect"}]},
            "buyer_participated_best_offers": {"status": 200},
            ROLE_REQUESTS["unrelated_public_ended"]: {"status": 200, "bestOffers": [{"offerPrice": {"value": "42.00"}}]},
        }
        accessible_report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )
        self.assertEqual(accessible_report["unrelated_visibility_observation"], "accessible")
        responses[ROLE_REQUESTS["unrelated_public_ended"]] = {"status": 200}
        empty_report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )
        self.assertEqual(empty_report["unrelated_visibility_observation"], "empty")
        responses[ROLE_REQUESTS["unrelated_public_ended"]] = {"status": 403}
        report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )
        self.assertTrue(report["message_content_detected"])
        self.assertTrue(report["message_content_violation"])
        self.assertFalse(report["field_matrix"]["seller_owned_best_offers"]["message_content"])

    def test_summary_wrappers_do_not_create_false_accessibility(self) -> None:
        responses = {
            "seller_owned_best_offers": {
                "status": 200,
                "ack": "Success",
                "field_keys": ["Ack", "BestOffer", "Price", "currencyID"],
                "offer_count": 1,
                "amount_field_visible": True,
                "currency_field_visible": True,
                "message_content_detected": True,
                "message_content_discarded": True,
                "raw_payload_retained": False,
            },
            "buyer_participated_best_offers": {
                "status": 200,
                "ack": "Success",
                "field_keys": ["Ack"],
                "offer_count": 0,
                "raw_payload_retained": False,
            },
            ROLE_REQUESTS["unrelated_public_ended"]: {
                "status": 200,
                "ack": "Failure",
                "error_codes": ["219"],
                "field_keys": ["Ack", "Errors", "ErrorCode"],
                "offer_count": 0,
                "raw_payload_retained": False,
            },
        }

        report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )

        self.assertTrue(report["field_matrix"]["seller_owned_best_offers"]["offer_amount"])
        self.assertTrue(report["field_matrix"]["seller_owned_best_offers"]["offer_currency"])
        self.assertTrue(report["message_content_detected"])
        self.assertFalse(report["message_content_violation"])
        self.assertEqual(report["permission_matrix"]["buyer_participated_best_offers"]["observed_result"], "empty")
        self.assertEqual(report["permission_matrix"][ROLE_REQUESTS["unrelated_public_ended"]]["observed_result"], "indeterminate")
        self.assertFalse(report["permission_matrix"][ROLE_REQUESTS["unrelated_public_ended"]]["accessible"])

    def test_compare_modes_returns_field_matrix_diff_without_raw_payloads(self) -> None:
        sandbox = {"mode": "sandbox", "field_matrix": {"traffic_read": {"traffic": False}}}
        production = {"mode": "production", "field_matrix": {"traffic_read": {"traffic": True}}}
        comparison = EbayApiProbe.compare_modes(sandbox, production)
        self.assertFalse(comparison["raw_payloads_retained"])
        self.assertFalse(comparison["field_matrix_comparison"]["traffic_read"]["traffic"]["sandbox"])
        self.assertTrue(comparison["field_matrix_comparison"]["traffic_read"]["traffic"]["production"])

    def test_non_unrelated_denials_are_not_marked_expected(self) -> None:
        request_names = [
            "seller_owned_best_offers",
            "buyer_participated_best_offers",
        ]
        for denied_name in request_names:
            responses = {
                "seller_owned_best_offers": {"status": 200},
                "buyer_participated_best_offers": {"status": 200},
                ROLE_REQUESTS["unrelated_public_ended"]: {"status": 403},
            }
            responses[denied_name] = {"status": 403}

            report = EbayApiProbe(StaticProbeClient(responses)).run(
                scopes=["https://api.ebay.com/oauth/api_scope"],
                seller_owned_listing_id="a",
                buyer_participated_listing_id="b",
                unrelated_listing_id="c",
                authorized_production_user_token=True,
            )

            denied = report["permission_matrix"][denied_name]
            self.assertEqual(denied["observed_result"], "denied")
            self.assertFalse(denied["expected_denial"])
            self.assertFalse(denied["denied_as_expected"])
            self.assertIsNone(report["permission_matrix"][ROLE_REQUESTS["unrelated_public_ended"]]["denied_as_expected"])

    def test_one_failed_request_does_not_drive_platform_wide_indeterminate(self) -> None:
        report = EbayApiProbe(
            StaticProbeClient(
                {
                    "seller_owned_best_offers": {
                        "status": 200,
                        "amount_field_visible": True,
                        "currency_field_visible": True,
                        "status_field_visible": True,
                    },
                    "buyer_participated_best_offers": {"status": 0, "transport_error": "timeout"},
                    ROLE_REQUESTS["unrelated_public_ended"]: {"status": 403},
                }
            )
        ).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )

        self.assertEqual(report["failed_request_count"], 1)
        self.assertFalse(report["failed_requests_are_platform_wide_conclusion"])
        self.assertEqual(report["feasibility"], "partially feasible")


if __name__ == "__main__":
    unittest.main()
