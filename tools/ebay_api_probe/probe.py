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
}

ROLE_REQUESTS = {
    "seller_owned": "seller_owned_best_offers",
    "buyer_participated": "buyer_participated_best_offers",
    "unrelated_public_ended": "unrelated_public_ended_best_offers",
}

FEASIBILITY_VALUES = {
    "technically feasible",
    "partially feasible",
    "not feasible",
    "indeterminate",
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
    access_context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StaticProbeClient:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[ProbeRequest] = []

    def get(self, request_name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        access_context = next((context for context, name in ROLE_REQUESTS.items() if name == request_name), "unknown")
        self.calls.append(ProbeRequest(request_name, path, dict(params), access_context))
        return dict(self.responses.get(request_name, {}))


class EbayApiProbe:
    """Read-only eBay API feasibility probe.

    The probe does not mint tokens and does not perform HTTP itself. A caller
    supplies a client already authorized for one test account. This class only
    validates the requested scopes/endpoints and summarizes field availability.
    """

    def __init__(self, client: ProbeClient, *, marketplace_id: str = "EBAY_US", sandbox: bool = False) -> None:
        self.client = client
        self.marketplace_id = marketplace_id
        self.sandbox = sandbox

    def run(
        self,
        *,
        scopes: list[str],
        seller_owned_listing_id: str,
        buyer_participated_listing_id: str,
        unrelated_listing_id: str,
        authorized_production_user_token: bool = False,
    ) -> dict[str, Any]:
        if self.sandbox:
            raise ProbeError("authorized eBay feasibility probe must run in production mode")
        if not authorized_production_user_token:
            raise ProbeError("authorized production user token must be explicitly supplied")
        self._validate_scopes(scopes)
        listing_ids = {
            "seller_owned": self._validate_listing_id(seller_owned_listing_id, "seller_owned_listing_id"),
            "buyer_participated": self._validate_listing_id(buyer_participated_listing_id, "buyer_participated_listing_id"),
            "unrelated_public_ended": self._validate_listing_id(unrelated_listing_id, "unrelated_listing_id"),
        }
        if len(set(listing_ids.values())) != 3:
            raise ProbeError("seller-owned, buyer-participated, and unrelated listing identifiers must be distinct")
        requests = [
            ProbeRequest(
                ROLE_REQUESTS["seller_owned"],
                "/ws/api.dll?callname=GetBestOffers",
                {"item_id": listing_ids["seller_owned"]},
                "seller_owned",
            ),
            ProbeRequest(
                ROLE_REQUESTS["buyer_participated"],
                "/ws/api.dll?callname=GetBestOffers",
                {"item_id": listing_ids["buyer_participated"]},
                "buyer_participated",
            ),
            ProbeRequest(
                ROLE_REQUESTS["unrelated_public_ended"],
                "/ws/api.dll?callname=GetBestOffers",
                {"item_id": listing_ids["unrelated_public_ended"]},
                "unrelated_public_ended",
            ),
        ]
        responses = {}
        for request in requests:
            self._validate_path(request.path)
            responses[request.name] = self.client.get(request.name, request.path, request.params)
        field_matrix = self._field_matrix(responses)
        permission_matrix = self._permission_matrix(responses)
        message_content_detected = any(row["message_content_detected"] for row in field_matrix.values())
        raw_payloads_retained = any(bool(response.get("raw_payload_retained")) for response in responses.values())
        undiscarded_message_content = any(
            bool(field_matrix[name].get("message_content_detected")) and not bool(responses[name].get("message_content_discarded"))
            for name in responses
        )
        unsafe_artifact_retention = raw_payloads_retained or undiscarded_message_content
        access_context_matrix = self._access_context_matrix(field_matrix, permission_matrix)
        feasibility = self._feasibility(access_context_matrix, unsafe_artifact_retention=unsafe_artifact_retention)
        unrelated = permission_matrix[ROLE_REQUESTS["unrelated_public_ended"]]
        return {
            "schema_version": "ebay_authorized_read_only_probe.v2",
            "mode": "sandbox" if self.sandbox else "production",
            "read_only": True,
            "marketplace_id": self.marketplace_id,
            "authorized_scopes": sorted(scopes),
            "authorized_production_user_token_required": True,
            "authorized_production_user_token_supplied": True,
            "manual_listing_ids_required": True,
            "listing_id_source": "operator_supplied_only",
            "crawling_or_listing_discovery_allowed": False,
            "request_count": len(requests),
            "documented_read_only_requests": [request.name for request in requests],
            "mutation_endpoints_called": False,
            "message_content_collected": False,
            "message_content_detected": message_content_detected,
            "message_content_violation": undiscarded_message_content,
            "field_matrix": field_matrix,
            "permission_matrix": permission_matrix,
            "access_context_matrix": access_context_matrix,
            "capability_summary": feasibility["capability_summary"],
            "failed_request_count": feasibility["failed_request_count"],
            "failed_requests_are_platform_wide_conclusion": False,
            "feasibility": feasibility["verdict"],
            "feasibility_reasons": feasibility["reasons"],
            "unrelated_visibility_observation": unrelated["observed_result"],
            "unrelated_visibility_conclusion": "empirical probe result only; no access assumption is hard-coded",
            "redacted": True,
            "raw_payloads_retained": raw_payloads_retained,
            "raw_private_payloads_retained_by_default": False,
            "seller_export_pilot_blocked": False,
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

    @staticmethod
    def _validate_listing_id(value: str, field_name: str) -> str:
        listing_id = str(value or "").strip()
        if not listing_id:
            raise ProbeError(f"{field_name} is required")
        if any(separator in listing_id for separator in [",", ";", "\n", "\r", "\t", " "]):
            raise ProbeError(f"{field_name} must contain exactly one manually supplied listing identifier")
        lowered = listing_id.lower()
        if lowered.startswith(("http://", "https://")) or "*" in listing_id:
            raise ProbeError(f"{field_name} must be a listing identifier, not a discovery pattern or URL")
        return listing_id

    def _field_matrix(self, responses: dict[str, dict[str, Any]]) -> dict[str, dict[str, bool]]:
        fields = {
            "offer_amount": ["bestOffers", "price", "offerPrice", "amount"],
            "offer_currency": ["currency", "currencyID", "currency_id"],
            "buyer_obfuscated_id": ["buyer", "buyerUserId", "userId"],
            "listing_id": ["itemId", "listingId", "inventoryItemGroupKey"],
            "traffic": ["impressions", "views", "clickThroughRate"],
            "completed_sale": ["orderId", "total", "lineItems", "transactionId", "paidTime", "checkoutStatus"],
            "fees": ["feeType", "bookingEntry", "amount"],
            "offer_status": ["bestOfferStatus", "offerStatus", "status"],
            "offer_type": ["bestOfferType", "offerType", "type"],
            "offer_timestamp": ["time", "date", "timestamp", "expirationTime"],
            "message_content": ["message", "text", "body"],
        }
        matrix: dict[str, dict[str, bool]] = {}
        for name, response in responses.items():
            flattened = _field_keys(response)
            matrix[name] = {field: any(candidate.lower() in flattened for candidate in candidates) for field, candidates in fields.items()}
            matrix[name]["offer_amount"] = matrix[name]["offer_amount"] or bool(response.get("amount_field_visible"))
            matrix[name]["offer_currency"] = matrix[name]["offer_currency"] or bool(response.get("currency_field_visible"))
            matrix[name]["buyer_obfuscated_id"] = matrix[name]["buyer_obfuscated_id"] or bool(response.get("identifier_field_visible"))
            matrix[name]["offer_status"] = matrix[name]["offer_status"] or bool(response.get("status_field_visible"))
            matrix[name]["offer_type"] = matrix[name]["offer_type"] or bool(response.get("type_field_visible"))
            matrix[name]["offer_timestamp"] = matrix[name]["offer_timestamp"] or bool(response.get("timestamp_field_visible"))
            matrix[name]["offer_history"] = matrix[name]["offer_amount"] and matrix[name]["offer_currency"]
            matrix[name]["offer_actors_redacted"] = matrix[name]["buyer_obfuscated_id"]
            matrix[name]["response_state"] = matrix[name]["offer_status"] or matrix[name]["offer_type"]
            matrix[name]["completed_outcome_linkage"] = matrix[name]["completed_sale"]
            matrix[name]["message_content_detected"] = matrix[name]["message_content"] or bool(response.get("message_content_detected"))
            matrix[name]["message_content"] = False
        return matrix

    def _permission_matrix(self, responses: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        matrix = {}
        for name, response in responses.items():
            status = int(response.get("status", 200))
            observed_result = _observed_permission_result(status, response)
            is_unrelated = name == ROLE_REQUESTS["unrelated_public_ended"]
            matrix[name] = {
                "status": status,
                "accessible": observed_result == "accessible",
                "expected_denial": None if is_unrelated else False,
                "denied_as_expected": None if is_unrelated else False,
                "observed_result": observed_result,
            }
        return matrix

    def _access_context_matrix(
        self,
        field_matrix: dict[str, dict[str, bool]],
        permission_matrix: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        matrix: dict[str, dict[str, Any]] = {}
        for context, request_name in ROLE_REQUESTS.items():
            fields = field_matrix.get(request_name, {})
            permission = permission_matrix.get(request_name, {})
            matrix[context] = {
                "request": request_name,
                "observed_result": permission.get("observed_result", "indeterminate"),
                "accessible": bool(permission.get("accessible")),
                "offer_history_available": bool(fields.get("offer_history")),
                "offer_actor_available_redacted": bool(fields.get("offer_actors_redacted")),
                "timestamp_available": bool(fields.get("offer_timestamp")),
                "response_state_available": bool(fields.get("response_state")),
                "completed_outcome_linkage_available": bool(fields.get("completed_outcome_linkage")),
            }
        return matrix

    def _feasibility(
        self,
        access_context_matrix: dict[str, dict[str, Any]],
        *,
        unsafe_artifact_retention: bool,
    ) -> dict[str, Any]:
        role_rows = list(access_context_matrix.values())
        failed_requests = [row["request"] for row in role_rows if row["observed_result"] == "indeterminate"]
        accessible_rows = [row for row in role_rows if row["accessible"]]
        seller = access_context_matrix["seller_owned"]
        buyer = access_context_matrix["buyer_participated"]
        capability_names = [
            "offer_history_available",
            "offer_actor_available_redacted",
            "timestamp_available",
            "response_state_available",
            "completed_outcome_linkage_available",
        ]
        capability_summary = {
            capability: any(bool(row.get(capability)) for row in role_rows)
            for capability in capability_names
        }
        reasons: list[str] = []
        if failed_requests:
            reasons.append("one_or_more_role_requests_indeterminate")
        if not seller["accessible"]:
            reasons.append("seller_owned_access_not_observed")
        if not buyer["accessible"]:
            reasons.append("buyer_participated_access_not_observed")
        missing_capabilities = [name for name, available in capability_summary.items() if not available]
        if missing_capabilities:
            reasons.append("missing_capabilities:" + ",".join(missing_capabilities))
        if unsafe_artifact_retention:
            reasons.append("unsafe_artifact_retention")

        if unsafe_artifact_retention:
            verdict = "not feasible"
        elif seller["accessible"] and buyer["accessible"] and not missing_capabilities:
            verdict = "technically feasible"
        elif accessible_rows and any(capability_summary.values()):
            verdict = "partially feasible"
        elif failed_requests and len(failed_requests) == len(role_rows):
            verdict = "indeterminate"
        elif failed_requests and not accessible_rows:
            verdict = "indeterminate"
        else:
            verdict = "not feasible"
        if verdict not in FEASIBILITY_VALUES:
            raise ProbeError(f"invalid feasibility verdict: {verdict}")
        return {
            "verdict": verdict,
            "reasons": reasons or ["all_required_observations_available"],
            "capability_summary": capability_summary,
            "failed_request_count": len(failed_requests),
        }


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
