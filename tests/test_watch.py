from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.config import NotificationConfig
from resonance.notify import MAJOR_STRENGTHENING, NEW_VERIFIED_RELATIONSHIP, NotificationResult
from resonance.storage import CorrelationFinding
from resonance.watch import DEFAULT_WATCH_INTERVAL_HOURS, run_scan_cycle, watch_loop


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_run_scan_cycle_applies_lifecycle_and_notification_adapter(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    notification_events = []
    findings = [
        _finding(holdout_rho=0.72, aligned_end_utc=NOW),
        _finding(holdout_rho=0.72, aligned_end_utc=NOW + timedelta(hours=6)),
        _finding(holdout_rho=0.72, aligned_end_utc=NOW + timedelta(hours=12)),
        _finding(holdout_rho=0.72, aligned_end_utc=NOW + timedelta(hours=18)),
    ]

    def scanner(*_args, **_kwargs):
        return (findings.pop(0),)

    def notifier(event, _config, *, now):
        notification_events.append((event, now))
        return NotificationResult(sent=True, skipped=False, reason="sent", destination="test")

    first = run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW,
    )
    second = run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=6),
    )
    third = run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=12),
    )
    promoted = run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=18),
    )

    assert [event.status for event in first.lifecycle_events] == ["incubating"]
    assert [event.status for event in second.lifecycle_events] == ["incubating"]
    assert [event.status for event in third.lifecycle_events] == ["incubating"]
    assert [event.status for event in promoted.lifecycle_events] == ["new"]
    assert len(promoted.notification_results) == 1
    assert notification_events[0][0].event_type == NEW_VERIFIED_RELATIONSHIP
    assert notification_events[0][1] == NOW + timedelta(hours=18)


def test_run_scan_cycle_notifies_major_strengthening(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    notification_events = []
    findings = [
        _finding(holdout_rho=0.60, aligned_end_utc=NOW),
        _finding(holdout_rho=0.60, aligned_end_utc=NOW + timedelta(hours=6)),
        _finding(holdout_rho=0.60, aligned_end_utc=NOW + timedelta(hours=12)),
        _finding(holdout_rho=0.60, aligned_end_utc=NOW + timedelta(hours=18)),
        _finding(holdout_rho=0.86, aligned_end_utc=NOW + timedelta(hours=24)),
    ]

    def scanner(*_args, **_kwargs):
        return (findings.pop(0),)

    def notifier(event, _config, *, now):
        notification_events.append(event)
        return NotificationResult(sent=True, skipped=False, reason="sent", destination="test")

    run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW,
    )
    run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=6),
    )
    run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=12),
    )
    run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=18),
    )
    result = run_scan_cycle(
        db_path,
        hours=168,
        notification_config=_notification_config(tmp_path),
        scanner=scanner,
        notifier=notifier,
        scan_utc=NOW + timedelta(hours=24),
    )

    assert [event.status for event in result.lifecycle_events] == ["strengthened"]
    assert notification_events[-1].event_type == MAJOR_STRENGTHENING
    assert notification_events[-1].strengthening_delta == 0.26


def test_watch_loop_uses_six_hour_default_interval_between_scans(tmp_path) -> None:
    stop_requested = _StopAfterWait()

    code = watch_loop(
        tmp_path / "resonance.db",
        hours=168,
        interval_seconds=DEFAULT_WATCH_INTERVAL_HOURS * 60 * 60,
        notification_config=_notification_config(tmp_path),
        stop_requested=stop_requested,
        scanner=lambda *_args, **_kwargs: (),
        notifier=lambda *_args, **_kwargs: NotificationResult(False, True, "disabled"),
        now=lambda: NOW,
    )

    assert code == 0
    assert stop_requested.wait_calls == [21600.0]


class _StopAfterWait:
    def __init__(self) -> None:
        self.stopped = False
        self.wait_calls: list[float] = []

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, seconds: float) -> None:
        self.wait_calls.append(seconds)
        self.stopped = True


def _notification_config(tmp_path) -> NotificationConfig:
    return NotificationConfig(
        enabled=True,
        dry_run_stdout=True,
        ntfy_endpoint="",
        history_path=str(tmp_path / "history.json"),
        dashboard_url="http://127.0.0.1:8501",
        discovery_cooldown_hours=24,
        finding_cooldown_hours=24,
        major_strengthening_threshold=0.20,
        request_timeout_seconds=5,
    )


def _finding(*, holdout_rho: float, aligned_end_utc: datetime = NOW) -> CorrelationFinding:
    return CorrelationFinding(
        x_metric="cpu_percent",
        y_metric="tcp_latency_ms",
        transform="first_difference",
        lag_seconds=900,
        discovery_rho=0.81,
        holdout_rho=holdout_rho,
        corrected_q=0.001,
        stability=0.91,
        overlap_count=42,
        first_seen_utc=NOW - timedelta(days=1),
        last_verified_utc=NOW,
        status="active",
        evidence={"association_only": True, "aligned_end_utc": aligned_end_utc.isoformat().replace("+00:00", "Z")},
    )
