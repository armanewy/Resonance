from __future__ import annotations

import argparse
import math
import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from resonance.analysis.lifecycle import LifecycleEvent, update_finding_lifecycle
from resonance.analysis.scanner import ScannerOptions, scan_correlations
from resonance.config import DEFAULT_CONFIG_PATH, ConfigError, NotificationConfig, load_config
from resonance.notify import (
    MAJOR_STRENGTHENING,
    NEW_VERIFIED_RELATIONSHIP,
    STABLE_RELATIONSHIP_BROKEN,
    FindingLifecycleEvent,
    NotificationResult,
    notify_lifecycle_event,
)
from resonance.storage import (
    DEFAULT_DB_PATH,
    CorrelationFinding,
    correlation_finding_from_row,
    ensure_database,
    fetch_correlation_findings,
)
from resonance.time_utils import ensure_utc, utc_now


DEFAULT_WATCH_INTERVAL_HOURS = 6.0
DEFAULT_SCAN_HOURS = 168.0

Scanner = Callable[..., tuple[CorrelationFinding, ...]]
Notifier = Callable[..., NotificationResult]


@dataclass(frozen=True)
class WatchCycleResult:
    findings: tuple[CorrelationFinding, ...]
    lifecycle_events: tuple[LifecycleEvent, ...]
    notification_results: tuple[NotificationResult, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run continuous conservative Resonance correlation scans.")
    parser.add_argument(
        "--hours",
        type=_positive_float,
        default=DEFAULT_SCAN_HOURS,
        help=f"Lookback window in hours. Defaults to {DEFAULT_SCAN_HOURS:g}.",
    )
    parser.add_argument(
        "--interval-hours",
        type=_positive_float,
        default=DEFAULT_WATCH_INTERVAL_HOURS,
        help=f"Hours to sleep between scans. Defaults to {DEFAULT_WATCH_INTERVAL_HOURS:g}.",
    )
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Config path for notification settings. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", flush=True)
        return 2

    stop_requested = threading.Event()
    _install_signal_handlers(stop_requested)
    return watch_loop(
        Path(args.database),
        hours=args.hours,
        interval_seconds=args.interval_hours * 60 * 60,
        notification_config=config.notifications,
        scanner_options=ScannerOptions(calendar_timezone=config.location.timezone),
        stop_requested=stop_requested,
        once=args.once,
    )


def watch_loop(
    database_path: str | Path,
    *,
    hours: float,
    interval_seconds: float,
    notification_config: NotificationConfig,
    stop_requested: threading.Event | None = None,
    once: bool = False,
    scanner: Scanner = scan_correlations,
    notifier: Notifier = notify_lifecycle_event,
    scanner_options: ScannerOptions | None = None,
    now: Callable[[], datetime] = utc_now,
) -> int:
    if hours <= 0:
        raise ValueError("hours must be greater than 0")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    stop_event = stop_requested or threading.Event()
    database = Path(database_path)

    while not stop_event.is_set():
        scan_time = ensure_utc(now()).replace(microsecond=0)
        result = run_scan_cycle(
            database,
            hours=hours,
            notification_config=notification_config,
            scanner=scanner,
            notifier=notifier,
            scanner_options=scanner_options,
            scan_utc=scan_time,
        )
        print(
            "Resonance watcher scan complete: "
            f"{len(result.findings)} findings, "
            f"{len(result.lifecycle_events)} lifecycle events, "
            f"{len(result.notification_results)} notification attempts.",
            flush=True,
        )
        if once:
            break
        stop_event.wait(interval_seconds)
    return 0


def run_scan_cycle(
    database_path: str | Path,
    *,
    hours: float,
    notification_config: NotificationConfig,
    scanner: Scanner = scan_correlations,
    notifier: Notifier = notify_lifecycle_event,
    scanner_options: ScannerOptions | None = None,
    scan_utc: datetime | None = None,
) -> WatchCycleResult:
    scan_time = ensure_utc(scan_utc or utc_now()).replace(microsecond=0)
    database = Path(database_path)

    conn = ensure_database(database)
    try:
        previous_findings = tuple(correlation_finding_from_row(row) for row in fetch_correlation_findings(conn))
    finally:
        conn.close()

    findings = scanner(
        database,
        hours=hours,
        dry_run=False,
        now=scan_time,
        options=scanner_options,
    )

    conn = ensure_database(database)
    try:
        lifecycle_events = update_finding_lifecycle(
            conn,
            findings,
            previous_findings=previous_findings,
            scan_utc=scan_time,
        )
    finally:
        conn.close()

    notification_results = tuple(
        result
        for result in (
            _notify_for_lifecycle_event(
                event,
                notification_config,
                notifier=notifier,
                now=scan_time,
            )
            for event in lifecycle_events
        )
        if result is not None
    )
    return WatchCycleResult(findings, lifecycle_events, notification_results)


def _notify_for_lifecycle_event(
    event: LifecycleEvent,
    config: NotificationConfig,
    *,
    notifier: Notifier,
    now: datetime,
) -> NotificationResult | None:
    notification_event = _notification_event(event)
    if notification_event is None:
        return None
    return notifier(notification_event, config, now=now)


def _notification_event(event: LifecycleEvent) -> FindingLifecycleEvent | None:
    finding = _finding_from_lifecycle_event(event)
    if finding is None:
        return None
    if event.status == "new":
        return FindingLifecycleEvent(NEW_VERIFIED_RELATIONSHIP, finding, event.event_utc)
    if event.status == "broken":
        return FindingLifecycleEvent(STABLE_RELATIONSHIP_BROKEN, finding, event.event_utc)
    if event.status == "strengthened":
        return FindingLifecycleEvent(
            MAJOR_STRENGTHENING,
            finding,
            event.event_utc,
            strengthening_delta=_strengthening_delta(event),
        )
    return None


def _finding_from_lifecycle_event(event: LifecycleEvent) -> CorrelationFinding | None:
    if event.discovery_rho is None or event.holdout_rho is None:
        return None
    if event.corrected_q is None or event.stability is None or event.overlap_count is None:
        return None
    return CorrelationFinding(
        x_metric=event.x_metric,
        y_metric=event.y_metric,
        transform=event.transform,
        lag_seconds=event.lag_seconds,
        discovery_rho=event.discovery_rho,
        holdout_rho=event.holdout_rho,
        corrected_q=event.corrected_q,
        stability=event.stability,
        overlap_count=event.overlap_count,
        first_seen_utc=event.event_utc,
        last_verified_utc=event.event_utc,
        status=event.status,
        evidence={},
    )


def _strengthening_delta(event: LifecycleEvent) -> float | None:
    value = event.details.get("coefficient_delta")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _install_signal_handlers(stop_requested: threading.Event) -> None:
    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
