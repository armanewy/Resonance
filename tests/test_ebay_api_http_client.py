from __future__ import annotations

import _bootstrap  # noqa: F401

from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tools.ebay_api_probe.http_client import EbayHttpClientError, EbayHttpProbeClient

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ebay_api_probe"


class _Headers(dict):
    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(key, default)


class _Response:
    def __init__(self, status: int, payload: bytes, content_type: str) -> None:
        self.status = status
        self._payload = payload
        self.headers = _Headers({"Content-Type": content_type})

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class EbayApiHttpClientTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["EBAY_SANDBOX_ACCESS_TOKEN"] = "secret-token"

    def tearDown(self) -> None:
        os.environ.pop("EBAY_SANDBOX_ACCESS_TOKEN", None)
        os.environ.pop("EBAY_PRODUCTION_USER_TOKEN", None)

    def test_trading_xml_response_is_redacted_and_token_not_returned(self) -> None:
        xml = (FIXTURES / "get_best_offers_response.xml").read_bytes()
        with patch("tools.ebay_api_probe.http_client.urlopen", return_value=_Response(200, xml, "text/xml")) as mocked:
            client = EbayHttpProbeClient(sandbox=True)
            summary = client.get("seller_owned_best_offers", "/ws/api.dll?callname=GetBestOffers", {"item_id": "123"})

        sent_request = mocked.call_args.args[0]
        self.assertEqual(sent_request.headers["X-ebay-api-call-name"], "GetBestOffers")
        self.assertIn(b"<ItemID>123</ItemID>", sent_request.data)
        rendered = json.dumps(summary, sort_keys=True)
        self.assertEqual(summary["ack"], "Success")
        self.assertEqual(summary["offer_count"], 1)
        self.assertTrue(summary["amount_field_visible"])
        self.assertTrue(summary["currency_field_visible"])
        self.assertTrue(summary["status_field_visible"])
        self.assertTrue(summary["type_field_visible"])
        self.assertTrue(summary["timestamp_field_visible"])
        self.assertTrue(summary["identifier_field_visible"])
        self.assertTrue(summary["message_content_detected"])
        self.assertTrue(summary["message_content_discarded"])
        self.assertTrue(summary["pii_content_detected"])
        self.assertIn("currencyID", summary["field_keys"])
        self.assertEqual(len(summary["listing_id_hashes"]), 1)
        self.assertTrue(summary["listing_id_hashes"][0].startswith("sha256:"))
        self.assertNotIn("redacted_payload", summary)
        self.assertNotIn("synthetic-buyer-id", rendered)
        self.assertNotIn("buyer@example.invalid", rendered)
        self.assertNotIn("synthetic private message", rendered)
        self.assertNotIn('"123"', rendered)
        self.assertNotIn("secret-token", rendered)
        self.assertFalse(summary["raw_payload_retained"])

    def test_rest_json_response_redacts_pii_and_retries_rate_limit(self) -> None:
        payload = (FIXTURES / "order_response.json").read_bytes()
        rate_limit = HTTPError(
            "https://api.sandbox.ebay.com/sell/fulfillment/v1/order",
            429,
            "Too Many Requests",
            _Headers({"Content-Type": "application/json", "Retry-After": "0"}),
            BytesIO(b'{"errorId": "rate"}'),
        )
        with patch("tools.ebay_api_probe.http_client.urlopen", side_effect=[rate_limit, _Response(200, payload, "application/json")]):
            summary = EbayHttpProbeClient(sandbox=True).get("orders_read", "/sell/fulfillment/v1/order", {"limit": 1})

        rendered = json.dumps(summary, sort_keys=True)
        self.assertEqual(summary["status"], 200)
        self.assertTrue(summary["identifier_field_visible"])
        self.assertTrue(summary["message_content_detected"])
        self.assertTrue(summary["message_content_discarded"])
        self.assertTrue(summary["pii_content_detected"])
        self.assertTrue(summary["pii_content_discarded"])
        self.assertNotIn("redacted_payload", summary)
        self.assertNotIn("synthetic-buyer-id", rendered)
        self.assertNotIn("buyer@example.invalid", rendered)
        self.assertNotIn("Synthetic Buyer", rendered)
        self.assertNotIn("Synthetic Buyer Full", rendered)
        self.assertNotIn("1 Example Street", rendered)
        self.assertNotIn("00000", rendered)
        self.assertNotIn("555-0100", rendered)
        self.assertNotIn("synthetic private message", rendered)
        self.assertNotIn("synthetic buyer private message", rendered)

    def test_missing_token_is_explicit_without_printing_credentials(self) -> None:
        os.environ.pop("EBAY_SANDBOX_ACCESS_TOKEN", None)
        with self.assertRaisesRegex(EbayHttpClientError, "EBAY_SANDBOX_ACCESS_TOKEN"):
            EbayHttpProbeClient(sandbox=True).get("orders_read", "/sell/fulfillment/v1/order", {})

    def test_production_token_env_must_be_explicit(self) -> None:
        os.environ["EBAY_PRODUCTION_USER_TOKEN"] = "secret-production-token"
        with self.assertRaisesRegex(EbayHttpClientError, "explicit token environment variable"):
            EbayHttpProbeClient(sandbox=False).get("seller_owned_best_offers", "/ws/api.dll?callname=GetBestOffers", {"item_id": "123"})

        xml = (FIXTURES / "get_best_offers_response.xml").read_bytes()
        with patch("tools.ebay_api_probe.http_client.urlopen", return_value=_Response(200, xml, "text/xml")) as mocked:
            summary = EbayHttpProbeClient(sandbox=False, token_env="EBAY_PRODUCTION_USER_TOKEN").get(
                "seller_owned_best_offers",
                "/ws/api.dll?callname=GetBestOffers",
                {"item_id": "123"},
            )

        sent_request = mocked.call_args.args[0]
        rendered = json.dumps(summary, sort_keys=True)
        self.assertEqual(sent_request.full_url, "https://api.ebay.com/ws/api.dll")
        self.assertNotIn("secret-production-token", rendered)

    def test_dynamic_identifier_object_keys_are_not_retained(self) -> None:
        payload = {
            "orders": {
                "123456789012": {
                    "orderId": "9876543210",
                    "buyerUserId": "synthetic-buyer-id",
                    "total": {"value": "10.00", "currency": "USD"},
                },
                "buyer@example.invalid": {
                    "status": "COMPLETE",
                },
            }
        }
        with patch(
            "tools.ebay_api_probe.http_client.urlopen",
            return_value=_Response(200, json.dumps(payload).encode("utf-8"), "application/json"),
        ):
            summary = EbayHttpProbeClient(sandbox=True).get("orders_read", "/sell/fulfillment/v1/order", {"limit": 1})

        rendered = json.dumps(summary, sort_keys=True)
        self.assertTrue(summary["identifier_field_visible"])
        self.assertTrue(summary["pii_content_detected"])
        self.assertNotIn("123456789012", rendered)
        self.assertNotIn("9876543210", rendered)
        self.assertNotIn("orderId", rendered)
        self.assertNotIn("buyer@example.invalid", rendered)
        self.assertNotIn("synthetic-buyer-id", rendered)
        self.assertIn("field_key_sha256:", rendered)
        self.assertIn("__pii_field__", summary["field_keys"])
        self.assertIn("__identifier_field__", summary["field_keys"])


if __name__ == "__main__":
    unittest.main()
