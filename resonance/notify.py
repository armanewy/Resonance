from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from resonance.config import NotificationConfig
from resonance.storage import CorrelationFinding
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


NEW_VERIFIED_RELATIONSHIP = "new_verified_relationship"
STABLE_RELATIONSHIP_BROKEN = "stable_relationship_broken"
MAJOR_STRENGTHENING = "major_strengthening"

NOTIFIABLE_EVENT_TYPES = {
    NEW_VERIFIED_RELATIONSHIP,
    STABLE_RELATIONSHIP_BROKEN,
    MAJOR_STRENGTHENING,
}


@dataclass(frozen=True)
class FindingLifecycleEvent:
    event_type: str
    finding: CorrelationFinding
    classified_at_utc: datetime
    strengthening_delta: float | None = None


@dataclass(frozen=True)
class NotificationResult:
    sent: bool
    skipped: bool
    reason: str
    destination: str | None = None
    error: str | None = None


def notify_lifecycle_event(
    event: FindingLifecycleEvent,
    config: NotificationConfig,
    *,
    now: datetime | None = None,
) -> NotificationResult:
    if event.event_type not in NOTIFIABLE_EVENT_TYPES:
        raise ValueError(f"Unsupported notification event type: {event.event_type}")

    sent_at_utc = ensure_utc(now or utc_now()).replace(microsecond=0)
    if not config.enabled:
        return NotificationResult(False, True, "disabled")

    if event.event_type == MAJOR_STRENGTHENING:
        if event.strengthening_delta is None:
            return NotificationResult(False, True, "missing_strengthening_delta")
        if event.strengthening_delta < config.major_strengthening_threshold:
            return NotificationResult(False, True, "below_strengthening_threshold")

    history = _load_history(config.history_path)
    fingerprint = _event_fingerprint(event)
    if fingerprint in _sent_fingerprints(history):
        return NotificationResult(False, True, "duplicate")

    finding_key = _finding_key(event.finding)
    if _in_finding_cooldown(history, finding_key, sent_at_utc, config.finding_cooldown_hours):
        return NotificationResult(False, True, "finding_cooldown")

    if event.event_type == NEW_VERIFIED_RELATIONSHIP and _in_discovery_cooldown(
        history,
        sent_at_utc,
        config.discovery_cooldown_hours,
    ):
        return NotificationResult(False, True, "discovery_cooldown")

    message = format_notification_message(event, config.dashboard_url)
    destination = "stdout" if config.dry_run_stdout else "ntfy"
    if config.dry_run_stdout:
        print(message, file=sys.stdout)
    else:
        if not config.ntfy_endpoint:
            return NotificationResult(False, True, "missing_ntfy_endpoint")
        error = _post_ntfy(config.ntfy_endpoint, message, config.request_timeout_seconds)
        if error is not None:
            return NotificationResult(False, False, "send_failed", destination=destination, error=error)

    _record_sent(history, event, fingerprint, sent_at_utc)
    history_error = _save_history(config.history_path, history)
    if history_error is not None:
        return NotificationResult(True, False, "sent_history_not_saved", destination=destination, error=history_error)
    return NotificationResult(True, False, "sent", destination=destination)


def format_notification_message(event: FindingLifecycleEvent, dashboard_url: str) -> str:
    finding = event.finding
    return "\n".join(
        (
            "Resonance finding update",
            f"Event: {_event_label(event.event_type)}",
            f"Metrics: {finding.x_metric} <> {finding.y_metric}",
            f"Lag: {finding.lag_seconds} seconds",
            f"Holdout rho: {_format_float(finding.holdout_rho)}",
            f"Stability: {_format_float(finding.stability)}",
            f"Dashboard: {dashboard_url}",
        )
    )


def _post_ntfy(endpoint: str, message: str, timeout_seconds: int) -> str | None:
    request = urllib.request.Request(
        endpoint,
        data=message.encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Title": "Resonance finding",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if 200 <= int(status) < 300:
                return None
            return f"HTTP {status}"
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return str(exc)


def _event_label(event_type: str) -> str:
    labels = {
        NEW_VERIFIED_RELATIONSHIP: "new verified relationship",
        STABLE_RELATIONSHIP_BROKEN: "previously stable relationship broken",
        MAJOR_STRENGTHENING: "major strengthening",
    }
    return labels[event_type]


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _load_history(path: str) -> dict[str, Any]:
    history_path = Path(path)
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"sent": []}
    if not isinstance(raw, dict) or not isinstance(raw.get("sent"), list):
        return {"sent": []}
    return raw


def _save_history(path: str, history: dict[str, Any]) -> str | None:
    history_path = Path(path)
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        return str(exc)
    return None


def _record_sent(
    history: dict[str, Any],
    event: FindingLifecycleEvent,
    fingerprint: str,
    sent_at_utc: datetime,
) -> None:
    history.setdefault("sent", []).append(
        {
            "fingerprint": fingerprint,
            "finding_key": _finding_key(event.finding),
            "event_type": event.event_type,
            "sent_at_utc": to_utc_iso(sent_at_utc),
        }
    )


def _sent_fingerprints(history: dict[str, Any]) -> set[str]:
    return {
        str(item.get("fingerprint"))
        for item in history.get("sent", [])
        if isinstance(item, dict) and item.get("fingerprint")
    }


def _in_finding_cooldown(
    history: dict[str, Any],
    finding_key: str,
    now: datetime,
    cooldown_hours: int,
) -> bool:
    threshold = now - timedelta(hours=cooldown_hours)
    for item in history.get("sent", []):
        if not isinstance(item, dict) or item.get("finding_key") != finding_key:
            continue
        sent_at = _parse_history_time(item.get("sent_at_utc"))
        if sent_at is not None and sent_at > threshold:
            return True
    return False


def _in_discovery_cooldown(
    history: dict[str, Any],
    now: datetime,
    cooldown_hours: int,
) -> bool:
    threshold = now - timedelta(hours=cooldown_hours)
    for item in history.get("sent", []):
        if not isinstance(item, dict) or item.get("event_type") != NEW_VERIFIED_RELATIONSHIP:
            continue
        sent_at = _parse_history_time(item.get("sent_at_utc"))
        if sent_at is not None and sent_at > threshold:
            return True
    return False


def _parse_history_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_utc(value)
    except ValueError:
        return None


def _event_fingerprint(event: FindingLifecycleEvent) -> str:
    finding = event.finding
    payload = {
        "event_type": event.event_type,
        "finding_key": _finding_key(finding),
        "lag_seconds": finding.lag_seconds,
        "holdout_rho": round(float(finding.holdout_rho), 6),
        "stability": round(float(finding.stability), 6),
        "strengthening_delta": (
            None
            if event.strengthening_delta is None
            else round(float(event.strengthening_delta), 6)
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finding_key(finding: CorrelationFinding) -> str:
    return "|".join((finding.x_metric, finding.y_metric, finding.transform))


__all__ = [
    "FindingLifecycleEvent",
    "MAJOR_STRENGTHENING",
    "NEW_VERIFIED_RELATIONSHIP",
    "NotificationResult",
    "STABLE_RELATIONSHIP_BROKEN",
    "format_notification_message",
    "notify_lifecycle_event",
]
