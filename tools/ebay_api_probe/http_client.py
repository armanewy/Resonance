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

MESSAGE_KEYS = {"message", "text", "body", "description"}
PII_KEYS = {"email", "emailaddress", "street", "street1", "street2", "address", "name", "firstname", "lastname", "phone"}
IDENTIFIER_KEYS = {"userid", "buyeruserid", "selleruserid"}
LISTING_ID_KEYS = {"itemid", "listingid", "inventoryitemgroupkey"}


class EbayHttpClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class EbayHttpProbeClient:
    """Read-only eBay HTTP client for probe summaries.

    The client returns redacted response summaries compatible with
    ``EbayApiProbe``. It never returns raw payload text or tokens.
    """

    sandbox: bool = True
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
    if not children:
        return {name: element.text or ""}
    grouped: dict[str, list[Any]] = {}
    for child in children:
        child_dict = _xml_to_dict(child)
        for key, value in child_dict.items():
            grouped.setdefault(key, []).append(value)
    return {name: {key: values[0] if len(values) == 1 else values for key, values in grouped.items()}}


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _response_summary(request_name: str, *, status: int, parsed: Any, transport: str) -> dict[str, Any]:
    redacted = _redact(parsed)
    flattened = _flatten(redacted)
    ack = _first_value(flattened, "ack")
    error_codes = sorted({str(value) for key, value in flattened if key.lower().endswith("errorcode") or key.lower() == "errorid"})
    warning_codes = sorted({str(value) for key, value in flattened if key.lower().endswith("warningcode")})
    return {
        "request_name": request_name,
        "status": status,
        "transport": transport,
        "ack": ack,
        "error_codes": error_codes,
        "warning_codes": warning_codes,
        "field_keys": sorted({key for key, _value in flattened}),
        "redacted_payload": redacted,
        "raw_payload_retained": False,
    }


def _redact(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if lowered in MESSAGE_KEYS:
            output[key] = {"detected": True, "discarded": True}
        elif lowered in PII_KEYS:
            output[key] = "[redacted]"
        elif lowered in IDENTIFIER_KEYS:
            output[key] = _hash_identifier(item)
        elif lowered in LISTING_ID_KEYS:
            output[key] = _hash_identifier(item)
            output[f"{key}_hashed"] = True
        else:
            output[key] = _redact(item)
    return output


def _hash_identifier(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
