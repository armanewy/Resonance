from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


READ_ONLY_SCOPES = {
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory.readonly",
    "https://api.ebay.com/oauth/api_scope/sell.account.readonly",
    "https://api.ebay.com/oauth/api_scope/sell.fulfillment.readonly",
    "https://api.ebay.com/oauth/api_scope/sell.finances",
    "https://api.ebay.com/oauth/api_scope/sell.analytics.readonly",
}

FORBIDDEN_SCOPE_FRAGMENTS = {
    "sell.inventory",  # non-readonly inventory can create/update offers/listings
    "sell.marketing",
    "sell.negotiation",
}

FORBIDDEN_METHOD_FRAGMENTS = {
    "respond",
    "sendoffer",
    "createoffer",
    "updateoffer",
    "publishoffer",
    "createpromotion",
    "pausepromotion",
    "discount",
    "message",
}

ALLOWED_READ_PATHS = {
    "/ws/api.dll?callname=getbestoffers",
    "/sell/inventory/v1/inventory_item",
    "/sell/fulfillment/v1/order",
    "/sell/finances/v1/transaction",
    "/sell/analytics/v1/traffic_report",
}


class ProbeError(ValueError):
    pass


class ProbeClient(Protocol):
    def get(self, request_name: str, path: str, params: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ProbeRequest:
    name: str
    path: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StaticProbeClient:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[ProbeRequest] = []

    def get(self, request_name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(ProbeRequest(request_name, path, dict(params)))
        return dict(self.responses.get(request_name, {}))


class EbayApiProbe:
    """Read-only eBay API feasibility probe.

    The probe does not mint tokens and does not perform HTTP itself. A caller
    supplies a client already authorized for one test account. This class only
    validates the requested scopes/endpoints and summarizes field availability.
    """

    def __init__(self, client: ProbeClient, *, marketplace_id: str = "EBAY_US", sandbox: bool = True) -> None:
        self.client = client
        self.marketplace_id = marketplace_id
        self.sandbox = sandbox

    def run(self, *, scopes: list[str], seller_owned_listing_id: str, buyer_participated_listing_id: str, unrelated_listing_id: str) -> dict[str, Any]:
        self._validate_scopes(scopes)
        if not seller_owned_listing_id or not buyer_participated_listing_id or not unrelated_listing_id:
            raise ProbeError("listing identifiers are required")
        requests = [
            ProbeRequest("seller_owned_best_offers", "/ws/api.dll?callname=GetBestOffers", {"item_id": seller_owned_listing_id}),
            ProbeRequest("buyer_participated_best_offers", "/ws/api.dll?callname=GetBestOffers", {"item_id": buyer_participated_listing_id}),
            ProbeRequest("unrelated_best_offers_probe", "/ws/api.dll?callname=GetBestOffers", {"item_id": unrelated_listing_id}),
            ProbeRequest("inventory_read", "/sell/inventory/v1/inventory_item", {"limit": 10}),
            ProbeRequest("orders_read", "/sell/fulfillment/v1/order", {"limit": 10}),
            ProbeRequest("finances_read", "/sell/finances/v1/transaction", {"limit": 10}),
            ProbeRequest("traffic_read", "/sell/analytics/v1/traffic_report", {"marketplace_id": self.marketplace_id}),
        ]
        responses = {}
        for request in requests:
            self._validate_path(request.path)
            responses[request.name] = self.client.get(request.name, request.path, request.params)
        field_matrix = self._field_matrix(responses)
        permission_matrix = self._permission_matrix(responses)
        unrelated = permission_matrix["unrelated_best_offers_probe"]
        message_content_detected = any(row["message_content_detected"] for row in field_matrix.values())
        return {
            "mode": "sandbox" if self.sandbox else "production",
            "read_only": True,
            "marketplace_id": self.marketplace_id,
            "authorized_scopes": sorted(scopes),
            "mutation_endpoints_called": False,
            "message_content_collected": False,
            "message_content_detected": message_content_detected,
            "message_content_violation": message_content_detected,
            "field_matrix": field_matrix,
            "permission_matrix": permission_matrix,
            "unrelated_visibility_observation": unrelated["observed_result"],
            "unrelated_visibility_conclusion": "empirical probe result only; no access assumption is hard-coded",
            "redacted": True,
            "raw_payloads_retained": False,
        }

    @staticmethod
    def compare_modes(sandbox_report: dict[str, Any], production_report: dict[str, Any]) -> dict[str, Any]:
        sandbox_fields = sandbox_report.get("field_matrix", {})
        production_fields = production_report.get("field_matrix", {})
        request_names = sorted(set(sandbox_fields) | set(production_fields))
        comparison = {}
        for name in request_names:
            field_names = sorted(set(sandbox_fields.get(name, {})) | set(production_fields.get(name, {})))
            comparison[name] = {
                field: {
                    "sandbox": bool(sandbox_fields.get(name, {}).get(field, False)),
                    "production": bool(production_fields.get(name, {}).get(field, False)),
                }
                for field in field_names
            }
        return {
            "sandbox_mode": sandbox_report.get("mode"),
            "production_mode": production_report.get("mode"),
            "field_matrix_comparison": comparison,
            "raw_payloads_retained": False,
        }

    def _validate_scopes(self, scopes: list[str]) -> None:
        unknown = sorted(set(scopes) - READ_ONLY_SCOPES)
        if unknown:
            raise ProbeError(f"scope is not on the read-only probe allowlist: {unknown}")
        for scope in scopes:
            for fragment in FORBIDDEN_SCOPE_FRAGMENTS:
                if scope.endswith(fragment):
                    raise ProbeError(f"scope is too broad for read-only probe: {scope}")

    def _validate_path(self, path: str) -> None:
        lowered = path.lower()
        if lowered not in ALLOWED_READ_PATHS:
            raise ProbeError(f"path is not on the read-only probe allowlist: {path}")
        for fragment in FORBIDDEN_METHOD_FRAGMENTS:
            if fragment in lowered:
                raise ProbeError(f"mutation or message endpoint is forbidden: {path}")

    def _field_matrix(self, responses: dict[str, dict[str, Any]]) -> dict[str, dict[str, bool]]:
        fields = {
            "offer_amount": ["bestOffers", "price", "offerPrice", "amount"],
            "offer_currency": ["currency", "currencyID", "currency_id"],
            "buyer_obfuscated_id": ["buyer", "buyerUserId", "userId"],
            "listing_id": ["itemId", "listingId", "inventoryItemGroupKey"],
            "traffic": ["impressions", "views", "clickThroughRate"],
            "completed_sale": ["orderId", "total", "lineItems"],
            "fees": ["feeType", "bookingEntry", "amount"],
            "message_content": ["message", "text", "body"],
        }
        matrix: dict[str, dict[str, bool]] = {}
        for name, response in responses.items():
            flattened = _field_keys(response)
            matrix[name] = {field: any(candidate.lower() in flattened for candidate in candidates) for field, candidates in fields.items()}
            matrix[name]["offer_amount"] = matrix[name]["offer_amount"] or bool(response.get("amount_field_visible"))
            matrix[name]["offer_currency"] = matrix[name]["offer_currency"] or bool(response.get("currency_field_visible"))
            matrix[name]["buyer_obfuscated_id"] = matrix[name]["buyer_obfuscated_id"] or bool(response.get("identifier_field_visible"))
            matrix[name]["message_content_detected"] = matrix[name]["message_content"] or bool(response.get("message_content_detected"))
            matrix[name]["message_content"] = False
        return matrix

    def _permission_matrix(self, responses: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        matrix = {}
        for name, response in responses.items():
            status = int(response.get("status", 200))
            observed_result = _observed_permission_result(status, response)
            matrix[name] = {
                "status": status,
                "accessible": observed_result == "accessible",
                "expected_denial": None if name == "unrelated_best_offers_probe" else False,
                "denied_as_expected": None if name == "unrelated_best_offers_probe" else False,
                "observed_result": observed_result,
            }
        return matrix


def _flatten_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_flatten_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_flatten_keys(item))
    return keys


def _field_keys(response: dict[str, Any]) -> set[str]:
    keys = _flatten_keys(response)
    for key in response.get("field_keys", []) or []:
        keys.add(str(key).lower())
    return keys


def _observed_permission_result(status: int, response: dict[str, Any]) -> str:
    if status == 0 or response.get("transport_error") or response.get("parse_error"):
        return "indeterminate"
    if status in {401, 403}:
        return "denied"
    ack = str(response.get("ack") or "").strip().lower()
    if ack in {"failure", "partialfailure"} or response.get("error_codes"):
        return "indeterminate"
    if not (200 <= status < 300):
        return "indeterminate"
    keys = _field_keys(response)
    offer_markers = {"bestoffers", "bestoffer", "offerprice", "price", "amount", "currency", "currencyid", "@currencyid"}
    summary_markers = {
        "amount_field_visible",
        "currency_field_visible",
        "status_field_visible",
        "type_field_visible",
        "timestamp_field_visible",
        "identifier_field_visible",
    }
    has_summary = bool({"field_keys", "offer_count", "raw_payload_retained"} & set(response))
    if any(bool(response.get(marker)) for marker in summary_markers):
        return "accessible"
    try:
        if int(response.get("offer_count", 0)) > 0:
            return "accessible"
    except (TypeError, ValueError):
        pass
    if keys & offer_markers:
        return "accessible"
    wrapper_keys = {
        "status",
        "ack",
        "request_name",
        "transport",
        "error_codes",
        "warning_codes",
        "field_keys",
        "offer_count",
        "amount_field_visible",
        "currency_field_visible",
        "status_field_visible",
        "type_field_visible",
        "timestamp_field_visible",
        "identifier_field_visible",
        "message_content_detected",
        "message_content_discarded",
        "pii_content_detected",
        "pii_content_discarded",
        "listing_id_hashes",
        "raw_payload_retained",
    }
    non_status_keys = keys - wrapper_keys
    if has_summary or not non_status_keys:
        return "empty"
    return "accessible"
