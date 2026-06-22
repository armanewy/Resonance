from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


TRADING_PRODUCTION_URL = "https://api.ebay.com/ws/api.dll"
TRADING_SANDBOX_URL = "https://api.sandbox.ebay.com/ws/api.dll"
REST_PRODUCTION_URL = "https://api.ebay.com"
REST_SANDBOX_URL = "https://api.sandbox.ebay.com"
TRADING_COMPATIBILITY_LEVEL = "1455"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

MESSAGE_KEY_FRAGMENTS = ("message", "text", "body", "description")
PII_KEY_FRAGMENTS = ("email", "address", "street", "name", "phone", "postal", "zip")
IDENTIFIER_KEY_FRAGMENTS = ("userid", "user_id", "buyeruserid", "selleruserid", "orderid")
LISTING_ID_KEY_FRAGMENTS = ("itemid", "listingid", "inventoryitemgroupkey")
AMOUNT_FIELD_NAMES = {"price", "offerprice", "amount", "value", "convertedcurrentprice"}
CURRENCY_FIELD_NAMES = {"currency", "currencyid", "currency_id", "@currencyid"}
STATUS_FIELD_NAMES = {"bestofferstatus", "offerstatus", "status"}
TYPE_FIELD_NAMES = {"bestoffertype", "offertype", "type"}
TIME_FIELD_FRAGMENTS = ("time", "date", "timestamp")
SAFE_FIELD_KEY_NAMES = {
    "ack",
    "amount",
    "bestoffer",
    "bestoffers",
    "bestofferstatus",
    "bestoffertype",
    "bookingentry",
    "convertedcurrentprice",
    "currency",
    "currencyid",
    "errorcode",
    "errorid",
    "fee",
    "feetype",
    "impressions",
    "lineitems",
    "order",
    "orders",
    "offer",
    "offercount",
    "offerprice",
    "offerstatus",
    "price",
    "status",
    "timestamp",
    "total",
    "transaction",
    "transactions",
    "type",
    "value",
    "views",
    "warningcode",
}


class EbayHttpClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class EbayHttpProbeClient:
    """Read-only eBay HTTP client for probe summaries.

    The client returns redacted response summaries compatible with
    ``EbayApiProbe``. It never returns raw payload text or tokens.
    """

    sandbox: bool = False
    token_env: str | None = None
    marketplace_id: str = "EBAY_US"
    timeout_seconds: float = 30.0
    max_attempts: int = 3

    def get(self, request_name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        token = self._token()
        if path.lower().startswith("/ws/api.dll?callname=getbestoffers"):
            return self._trading_get_best_offers(request_name, token=token, params=params)
        return self._rest_get(request_name, token=token, path=path, params=params)

    def _token(self) -> str:
        if not self.sandbox and not self.token_env:
            raise EbayHttpClientError("Production probing requires an explicit token environment variable name")
        env_name = self.token_env or ("EBAY_SANDBOX_ACCESS_TOKEN" if self.sandbox else "EBAY_ACCESS_TOKEN")
        token = os.environ.get(env_name)
        if not token:
            raise EbayHttpClientError(f"Missing eBay access token in environment variable {env_name}")
        return token

    def _trading_get_best_offers(self, request_name: str, *, token: str, params: dict[str, Any]) -> dict[str, Any]:
        item_id = str(params.get("item_id") or params.get("ItemID") or "").strip()
        best_offer_status = str(params.get("best_offer_status") or params.get("BestOfferStatus") or "All")
        body = _trading_xml("GetBestOffers", {"ItemID": item_id, "BestOfferStatus": best_offer_status})
        headers = {
            "Content-Type": "text/xml",
            "X-EBAY-API-CALL-NAME": "GetBestOffers",
            "X-EBAY-API-SITEID": "0",
            "X-EBAY-API-COMPATIBILITY-LEVEL": TRADING_COMPATIBILITY_LEVEL,
            "X-EBAY-API-IAF-TOKEN": token,
        }
        status, payload, content_type = self._request(
            Request(
                TRADING_SANDBOX_URL if self.sandbox else TRADING_PRODUCTION_URL,
                data=body.encode("utf-8"),
                headers=headers,
                method="POST",
            )
        )
        parsed = _parse_payload(payload, content_type=content_type)
        return _response_summary(request_name, status=status, parsed=parsed, transport="trading_xml")

    def _rest_get(self, request_name: str, *, token: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{REST_SANDBOX_URL if self.sandbox else REST_PRODUCTION_URL}{path}"
        if query:
            url = f"{url}?{query}"
        status, payload, content_type = self._request(
            Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
                },
                method="GET",
            )
        )
        parsed = _parse_payload(payload, content_type=content_type)
        return _response_summary(request_name, status=status, parsed=parsed, transport="rest_json")

    def _request(self, request: Request) -> tuple[int, bytes, str]:
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return int(response.status), response.read(), str(response.headers.get("Content-Type", ""))
            except HTTPError as exc:
                payload = exc.read()
                if exc.code not in TRANSIENT_STATUS_CODES or attempt >= self.max_attempts:
                    return int(exc.code), payload, str(exc.headers.get("Content-Type", ""))
                _sleep_for_retry(exc.headers.get("Retry-After"), attempt)
            except URLError as exc:
                last_error = exc.reason.__class__.__name__
                if attempt >= self.max_attempts:
                    return 0, json.dumps({"transport_error": last_error}).encode("utf-8"), "application/json"
                _sleep_for_retry(None, attempt)
        return 0, json.dumps({"transport_error": last_error or "unknown"}).encode("utf-8"), "application/json"


def _trading_xml(call_name: str, fields: dict[str, str]) -> str:
    lines = [f'<?xml version="1.0" encoding="utf-8"?>', f'<{call_name}Request xmlns="urn:ebay:apis:eBLBaseComponents">']
    for key, value in fields.items():
        if value:
            lines.append(f"  <{key}>{_escape_xml(value)}</{key}>")
    lines.append(f"  <Version>{TRADING_COMPATIBILITY_LEVEL}</Version>")
    lines.append(f"</{call_name}Request>")
    return "\n".join(lines)


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _sleep_for_retry(retry_after: str | None, attempt: int) -> None:
    try:
        delay = float(retry_after) if retry_after else 0.2 * attempt
    except ValueError:
        delay = 0.2 * attempt
    time.sleep(min(delay, 2.0))


def _parse_payload(payload: bytes, *, content_type: str) -> Any:
    text = payload.decode("utf-8", errors="replace")
    if "xml" in content_type.lower() or text.lstrip().startswith("<"):
        try:
            return _xml_to_dict(ET.fromstring(text))
        except ET.ParseError:
            return {"parse_error": "xml_parse_error"}
    try:
        return json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {"parse_error": "json_parse_error"}


def _xml_to_dict(element: ET.Element) -> dict[str, Any]:
    children = list(element)
    name = _strip_namespace(element.tag)
    node: dict[str, Any] = {}
    for key, value in element.attrib.items():
        attr_name = _strip_namespace(key)
        node[attr_name] = value
        node[f"@{attr_name}"] = value
    text = (element.text or "").strip()
    if not children:
        if node:
            if text:
                node["value"] = text
            return {name: node}
        return {name: text}
    grouped: dict[str, list[Any]] = {}
    for child in children:
        child_dict = _xml_to_dict(child)
        for key, value in child_dict.items():
            grouped.setdefault(key, []).append(value)
    if text:
        node["value"] = text
    node.update({key: values[0] if len(values) == 1 else values for key, values in grouped.items()})
    return {name: node}


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _response_summary(request_name: str, *, status: int, parsed: Any, transport: str) -> dict[str, Any]:
    flattened = _flatten(parsed)
    ack = _first_value(flattened, "ack")
    error_codes = sorted({str(value) for key, value in flattened if key.lower().endswith("errorcode") or key.lower() == "errorid"})
    warning_codes = sorted({str(value) for key, value in flattened if key.lower().endswith("warningcode")})
    raw_field_keys = sorted({key for key, _value in flattened})
    field_keys = _redacted_field_keys(raw_field_keys)
    normalized_keys = {_normalize_key(key) for key in field_keys}
    raw_normalized_keys = {_normalize_key(key) for key in raw_field_keys}
    message_detected = any(_is_message_key(key) for key in raw_field_keys)
    pii_detected = any(_is_pii_key(key) for key in raw_field_keys)
    identifier_visible = any(_is_identifier_key(key) or _is_listing_id_key(key) for key in raw_field_keys)
    listing_id_hashes = sorted(
        {
            _hash_identifier(value)
            for key, value in flattened
            if _is_listing_id_key(key) and not isinstance(value, (dict, list))
        }
    )
    return {
        "request_name": request_name,
        "status": status,
        "transport": transport,
        "ack": ack,
        "error_codes": error_codes,
        "warning_codes": warning_codes,
        "field_keys": field_keys,
        "offer_count": _count_offer_nodes(parsed),
        "amount_field_visible": bool(raw_normalized_keys & AMOUNT_FIELD_NAMES),
        "currency_field_visible": bool(raw_normalized_keys & CURRENCY_FIELD_NAMES),
        "status_field_visible": bool(raw_normalized_keys & STATUS_FIELD_NAMES),
        "type_field_visible": bool(raw_normalized_keys & TYPE_FIELD_NAMES),
        "timestamp_field_visible": any(fragment in key for key in raw_normalized_keys for fragment in TIME_FIELD_FRAGMENTS),
        "identifier_field_visible": identifier_visible,
        "listing_id_hashes": listing_id_hashes,
        "message_content_detected": message_detected,
        "message_content_discarded": message_detected,
        "pii_content_detected": pii_detected,
        "pii_content_discarded": pii_detected,
        "parse_error": _first_value(flattened, "parse_error"),
        "transport_error": _first_value(flattened, "transport_error"),
        "raw_payload_retained": False,
    }


def _hash_identifier(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hash_field_key(value: str) -> str:
    return "field_key_sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_key(key: str) -> str:
    return str(key).lower().replace("-", "").replace("_", "")


def _redacted_field_keys(keys: list[str]) -> list[str]:
    return sorted({_redacted_field_key(key) for key in keys})


def _redacted_field_key(key: str) -> str:
    normalized = _normalize_key(key)
    if _is_message_key(key):
        return "__message_field__"
    if _is_pii_key(key):
        return "__pii_field__"
    if _is_identifier_key(key) or _is_listing_id_key(key):
        return "__identifier_field__"
    if normalized in SAFE_FIELD_KEY_NAMES:
        return key
    if _looks_like_dynamic_key(key):
        return _hash_field_key(str(key))
    return key


def _looks_like_dynamic_key(key: str) -> bool:
    text = str(key)
    normalized = _normalize_key(text)
    if any(character.isdigit() for character in text):
        return True
    if len(text) >= 16 and any(character.isalpha() for character in text) and any(character in "-_:" for character in text):
        return True
    if len(normalized) >= 20:
        return True
    return False


def _is_message_key(key: str) -> bool:
    lowered = _normalize_key(key)
    return any(fragment in lowered for fragment in MESSAGE_KEY_FRAGMENTS)


def _is_pii_key(key: str) -> bool:
    lowered = _normalize_key(key)
    text = str(key).lower()
    return "@" in text or any(fragment in lowered for fragment in PII_KEY_FRAGMENTS)


def _is_identifier_key(key: str) -> bool:
    lowered = _normalize_key(key)
    return any(fragment in lowered for fragment in IDENTIFIER_KEY_FRAGMENTS)


def _is_listing_id_key(key: str) -> bool:
    lowered = _normalize_key(key)
    return any(fragment in lowered for fragment in LISTING_ID_KEY_FRAGMENTS)


def _count_offer_nodes(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_offer_nodes(item) for item in value)
    if not isinstance(value, dict):
        return 0
    total = 0
    for key, item in value.items():
        lowered = _normalize_key(key)
        if lowered == "bestoffer":
            total += len(item) if isinstance(item, list) else 1
        elif lowered == "bestoffers" and isinstance(item, list):
            total += len(item)
        else:
            total += _count_offer_nodes(item)
    return total


def _flatten(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            rows.append((str(key), item))
            rows.extend(_flatten(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten(item))
        return rows
    return []


def _first_value(flattened: list[tuple[str, Any]], key_name: str) -> Any:
    for key, value in flattened:
        if key.lower() == key_name.lower() and not isinstance(value, (dict, list)):
            return value
    return None
