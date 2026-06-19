from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from resonance.config import NotificationConfig
from resonance.notify import (
    MAJOR_STRENGTHENING,
    NEW_VERIFIED_RELATIONSHIP,
    STABLE_RELATIONSHIP_BROKEN,
    FindingLifecycleEvent,
    format_notification_message,
    notify_lifecycle_event,
)
from resonance.storage import CorrelationFinding


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_disabled_notifications_skip_without_history(tmp_path, capsys) -> None:
    history_path = tmp_path / "history.json"

    result = notify_lifecycle_event(
        FindingLifecycleEvent(NEW_VERIFIED_RELATIONSHIP, _finding(), NOW),
        _config(history_path, enabled=False),
        now=NOW,
    )

    assert result.skipped is True
    assert result.reason == "disabled"
    assert not history_path.exists()
    assert capsys.readouterr().out == ""


def test_stdout_dry_run_writes_message_and_history(tmp_path, capsys) -> None:
    history_path = tmp_path / "history.json"

    result = notify_lifecycle_event(
        FindingLifecycleEvent(NEW_VERIFIED_RELATIONSHIP, _finding(), NOW),
        _config(history_path),
        now=NOW,
    )

    output = capsys.readouterr().out
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert result.sent is True
    assert result.destination == "stdout"
    assert "Metrics: cpu_percent <> tcp_latency_ms" in output
    assert "Lag: 900 seconds" in output
    assert "Holdout rho: 0.720" in output
    assert "Stability: 0.910" in output
    assert "Dashboard: http://127.0.0.1:8501" in output
    assert len(history["sent"]) == 1


def test_message_avoids_causal_wording() -> None:
    message = format_notification_message(
        FindingLifecycleEvent(STABLE_RELATIONSHIP_BROKEN, _finding(), NOW),
        "http://127.0.0.1:8501",
    ).lower()

    for word in ("cause", "caused", "because", "due to", "drives", "leads to"):
        assert word not in message


def test_duplicate_notification_is_skipped(tmp_path, capsys) -> None:
    history_path = tmp_path / "history.json"
    event = FindingLifecycleEvent(NEW_VERIFIED_RELATIONSHIP, _finding(), NOW)
    config = _config(history_path)

    first = notify_lifecycle_event(event, config, now=NOW)
    second = notify_lifecycle_event(event, config, now=NOW + timedelta(hours=25))

    assert first.sent is True
    assert second.skipped is True
    assert second.reason == "duplicate"
    assert capsys.readouterr().out.count("Resonance finding update") == 1


def test_discovery_notification_is_limited_to_one_per_day(tmp_path, capsys) -> None:
    history_path = tmp_path / "history.json"
    config = _config(history_path)

    first = notify_lifecycle_event(
        FindingLifecycleEvent(NEW_VERIFIED_RELATIONSHIP, _finding("cpu_percent"), NOW),
        config,
        now=NOW,
    )
    second = notify_lifecycle_event(
        FindingLifecycleEvent(
            NEW_VERIFIED_RELATIONSHIP,
            _finding("memory_percent"),
            NOW + timedelta(hours=1),
        ),
        config,
        now=NOW + timedelta(hours=1),
    )

    assert first.sent is True
    assert second.skipped is True
    assert second.reason == "discovery_cooldown"
    assert capsys.readouterr().out.count("Resonance finding update") == 1


def test_per_finding_cooldown_blocks_different_event(tmp_path, capsys) -> None:
    history_path = tmp_path / "history.json"
    config = _config(history_path)
    finding = _finding()

    first = notify_lifecycle_event(
        FindingLifecycleEvent(STABLE_RELATIONSHIP_BROKEN, finding, NOW),
        config,
        now=NOW,
    )
    second = notify_lifecycle_event(
        FindingLifecycleEvent(MAJOR_STRENGTHENING, finding, NOW + timedelta(hours=1), 0.35),
        config,
        now=NOW + timedelta(hours=1),
    )

    assert first.sent is True
    assert second.skipped is True
    assert second.reason == "finding_cooldown"
    assert capsys.readouterr().out.count("Resonance finding update") == 1


def test_major_strengthening_requires_configured_threshold(tmp_path, capsys) -> None:
    result = notify_lifecycle_event(
        FindingLifecycleEvent(MAJOR_STRENGTHENING, _finding(), NOW, strengthening_delta=0.10),
        _config(tmp_path / "history.json"),
        now=NOW,
    )

    assert result.skipped is True
    assert result.reason == "below_strengthening_threshold"
    assert capsys.readouterr().out == ""


def test_ntfy_post_uses_mocked_http_and_records_history(tmp_path, monkeypatch) -> None:
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    history_path = tmp_path / "history.json"

    result = notify_lifecycle_event(
        FindingLifecycleEvent(STABLE_RELATIONSHIP_BROKEN, _finding(), NOW),
        _config(history_path, dry_run_stdout=False, ntfy_endpoint="https://ntfy.example/topic"),
        now=NOW,
    )

    assert result.sent is True
    assert result.destination == "ntfy"
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "https://ntfy.example/topic"
    assert request.get_method() == "POST"
    assert timeout == 5
    assert b"Holdout rho: 0.720" in request.data
    assert len(json.loads(history_path.read_text(encoding="utf-8"))["sent"]) == 1


def test_network_failure_returns_error_without_history(tmp_path, monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    history_path = tmp_path / "history.json"

    result = notify_lifecycle_event(
        FindingLifecycleEvent(STABLE_RELATIONSHIP_BROKEN, _finding(), NOW),
        _config(history_path, dry_run_stdout=False, ntfy_endpoint="https://ntfy.example/topic"),
        now=NOW,
    )

    assert result.sent is False
    assert result.skipped is False
    assert result.reason == "send_failed"
    assert "network unavailable" in str(result.error)
    assert not history_path.exists()


def _config(
    history_path,
    *,
    enabled: bool = True,
    dry_run_stdout: bool = True,
    ntfy_endpoint: str = "",
) -> NotificationConfig:
    return NotificationConfig(
        enabled=enabled,
        dry_run_stdout=dry_run_stdout,
        ntfy_endpoint=ntfy_endpoint,
        history_path=str(history_path),
        dashboard_url="http://127.0.0.1:8501",
        discovery_cooldown_hours=24,
        finding_cooldown_hours=24,
        major_strengthening_threshold=0.20,
        request_timeout_seconds=5,
    )


def _finding(x_metric: str = "cpu_percent") -> CorrelationFinding:
    return CorrelationFinding(
        x_metric=x_metric,
        y_metric="tcp_latency_ms",
        transform="first_difference",
        lag_seconds=900,
        discovery_rho=0.81,
        holdout_rho=0.72,
        corrected_q=0.001,
        stability=0.91,
        overlap_count=42,
        first_seen_utc=NOW - timedelta(days=1),
        last_verified_utc=NOW,
        status="active",
        evidence={"association_only": True},
    )
