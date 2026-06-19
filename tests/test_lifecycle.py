from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.analysis.lifecycle import (
    LifecycleOptions,
    fetch_lifecycle_events,
    update_finding_lifecycle,
)
from resonance.storage import CorrelationFinding


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_repeated_scans_verify_same_relationship_with_small_lag_drift(sqlite_conn) -> None:
    options = LifecycleOptions(lag_tolerance_seconds=120, coefficient_epsilon=0.05)
    first = _finding(lag_seconds=900, holdout_rho=0.70)
    drifted = _finding(lag_seconds=960, holdout_rho=0.72)

    first_events = update_finding_lifecycle(
        sqlite_conn,
        [first],
        scan_utc=NOW,
        options=options,
    )
    second_events = update_finding_lifecycle(
        sqlite_conn,
        [drifted],
        previous_findings=[first],
        scan_utc=NOW + timedelta(minutes=5),
        options=options,
    )

    rows = fetch_lifecycle_events(sqlite_conn)
    assert [event.status for event in first_events] == ["new"]
    assert [event.status for event in second_events] == ["verified"]
    assert len(rows) == 2
    assert rows[0].relationship_key == rows[1].relationship_key
    assert rows[1].lag_seconds == 960


def test_strengthening_and_weakening_ignore_tiny_coefficient_changes(sqlite_conn) -> None:
    options = LifecycleOptions(coefficient_epsilon=0.05)

    update_finding_lifecycle(sqlite_conn, [_finding(holdout_rho=0.60)], scan_utc=NOW, options=options)
    tiny_change = update_finding_lifecycle(
        sqlite_conn,
        [_finding(holdout_rho=0.63)],
        scan_utc=NOW + timedelta(minutes=5),
        options=options,
    )
    strengthened = update_finding_lifecycle(
        sqlite_conn,
        [_finding(holdout_rho=0.72)],
        scan_utc=NOW + timedelta(minutes=10),
        options=options,
    )
    weakened = update_finding_lifecycle(
        sqlite_conn,
        [_finding(holdout_rho=0.61)],
        scan_utc=NOW + timedelta(minutes=15),
        options=options,
    )

    assert [event.status for event in tiny_change] == ["verified"]
    assert [event.status for event in strengthened] == ["strengthened"]
    assert [event.status for event in weakened] == ["weakened"]
    assert [event.status for event in fetch_lifecycle_events(sqlite_conn)] == [
        "new",
        "verified",
        "strengthened",
        "weakened",
    ]


def test_validation_failures_break_only_after_repeated_failures(sqlite_conn) -> None:
    options = LifecycleOptions(broken_after_failures=2)
    finding = _finding()

    update_finding_lifecycle(sqlite_conn, [finding], scan_utc=NOW, options=options)
    first_failure = update_finding_lifecycle(
        sqlite_conn,
        [],
        previous_findings=[finding],
        scan_utc=NOW + timedelta(minutes=5),
        options=options,
    )
    second_failure = update_finding_lifecycle(
        sqlite_conn,
        [],
        previous_findings=[finding],
        scan_utc=NOW + timedelta(minutes=10),
        options=options,
    )

    assert [event.status for event in first_failure] == ["weakened"]
    assert first_failure[0].failure_count == 1
    assert [event.status for event in second_failure] == ["broken"]
    assert second_failure[0].failure_count == 2


def test_current_scan_recovers_broken_relationship(sqlite_conn) -> None:
    options = LifecycleOptions(broken_after_failures=1)
    finding = _finding()

    update_finding_lifecycle(sqlite_conn, [finding], scan_utc=NOW, options=options)
    broken = update_finding_lifecycle(
        sqlite_conn,
        [],
        previous_findings=[finding],
        scan_utc=NOW + timedelta(minutes=5),
        options=options,
    )
    recovered = update_finding_lifecycle(
        sqlite_conn,
        [_finding(holdout_rho=0.74)],
        previous_findings=[finding],
        scan_utc=NOW + timedelta(minutes=10),
        options=options,
    )

    assert [event.status for event in broken] == ["broken"]
    assert [event.status for event in recovered] == ["verified"]
    assert recovered[0].previous_status == "broken"
    assert recovered[0].failure_count == 0
    assert recovered[0].details["recovered_from"] == "broken"


def _finding(
    *,
    lag_seconds: int = 900,
    discovery_rho: float = 0.84,
    holdout_rho: float = 0.70,
    corrected_q: float = 0.004,
    stability: float = 1.0,
    overlap_count: int = 80,
) -> CorrelationFinding:
    return CorrelationFinding(
        x_metric="cpu_percent",
        y_metric="tcp_latency_ms",
        transform="first_difference",
        lag_seconds=lag_seconds,
        discovery_rho=discovery_rho,
        holdout_rho=holdout_rho,
        corrected_q=corrected_q,
        stability=stability,
        overlap_count=overlap_count,
        first_seen_utc=NOW,
        last_verified_utc=NOW,
        status="active",
        evidence={"selected_on": "first_70_percent"},
    )
