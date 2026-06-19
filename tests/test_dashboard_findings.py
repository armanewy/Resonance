from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.analysis.lifecycle import ensure_lifecycle_schema
from resonance.dashboard import analysis_from_finding, dashboard_findings
from resonance.storage import CorrelationFinding, upsert_correlation_findings
from resonance.time_utils import to_utc_iso


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_dashboard_findings_filter_statuses_and_archived(sqlite_conn) -> None:
    verified = _finding("verified_x", holdout_rho=0.50, corrected_q=0.001, stability=0.80)
    strengthened = _finding("strengthened_x", holdout_rho=0.90, corrected_q=0.010, stability=1.0)
    weakened = _finding("weakened_x", holdout_rho=0.70, corrected_q=0.003, stability=0.90)
    broken = _finding("broken_x", holdout_rho=0.95, corrected_q=0.0001, stability=1.0)
    new = _finding("new_x", holdout_rho=0.99, corrected_q=0.00001, stability=1.0)

    upsert_correlation_findings(sqlite_conn, [verified, strengthened, weakened, broken, new])
    _insert_lifecycle(sqlite_conn, verified, "verified", event_id=1)
    _insert_lifecycle(sqlite_conn, strengthened, "strengthened", event_id=2)
    _insert_lifecycle(sqlite_conn, weakened, "weakened", event_id=3)
    _insert_lifecycle(sqlite_conn, broken, "broken", event_id=4)
    _insert_lifecycle(sqlite_conn, new, "new", event_id=5)

    current = dashboard_findings(sqlite_conn, show_archived=False)
    archived = dashboard_findings(sqlite_conn, show_archived=True)

    assert [(finding.x_metric, finding.status) for finding in current] == [
        ("verified_x", "verified"),
        ("weakened_x", "weakened"),
        ("strengthened_x", "strengthened"),
    ]
    assert [(finding.x_metric, finding.status) for finding in archived] == [
        ("broken_x", "broken"),
        ("verified_x", "verified"),
        ("weakened_x", "weakened"),
        ("strengthened_x", "strengthened"),
    ]
    assert "new_x" not in {finding.x_metric for finding in archived}


def test_analysis_from_finding_uses_stored_evidence_contract() -> None:
    finding = _finding(
        "cpu_percent",
        lag_seconds=600,
        evidence={
            "cadence_seconds": 300,
            "aligned_observation_count": 288,
            "aligned_start_utc": "2026-06-18T12:00:00Z",
            "aligned_end_utc": "2026-06-19T12:00:00Z",
            "x_coverage": 0.92,
            "y_coverage": 0.86,
            "discovery_overlap": 144,
            "permutation_p_value": 0.005,
            "window_scores": [{"window_index": 0, "rho": 0.7, "overlap_count": 72}],
            "warnings": ["small holdout"],
        },
    )

    analysis = analysis_from_finding(finding)

    assert analysis.aligned_pair.cadence_seconds == 300
    assert analysis.lag_result.best_lag_steps == 2
    assert analysis.validation_result.permutation_p_value == 0.005
    assert analysis.validation_result.window_scores == ({"window_index": 0, "rho": 0.7, "overlap_count": 72},)
    assert analysis.validation_result.warnings == ("small holdout",)


def _finding(
    x_metric: str,
    *,
    y_metric: str = "tcp_latency_ms",
    lag_seconds: int = 900,
    discovery_rho: float = 0.81,
    holdout_rho: float = 0.72,
    corrected_q: float = 0.001,
    stability: float = 0.91,
    overlap_count: int = 80,
    evidence: dict | None = None,
) -> CorrelationFinding:
    return CorrelationFinding(
        x_metric=x_metric,
        y_metric=y_metric,
        transform="first_difference",
        lag_seconds=lag_seconds,
        discovery_rho=discovery_rho,
        holdout_rho=holdout_rho,
        corrected_q=corrected_q,
        stability=stability,
        overlap_count=overlap_count,
        first_seen_utc=NOW - timedelta(days=1),
        last_verified_utc=NOW,
        status="active",
        evidence=evidence or {"selected_on": "first_70_percent"},
    )


def _insert_lifecycle(sqlite_conn, finding: CorrelationFinding, status: str, *, event_id: int) -> None:
    ensure_lifecycle_schema(sqlite_conn)
    sqlite_conn.execute(
        """
        INSERT INTO finding_lifecycle_events (
            event_id,
            relationship_key,
            event_utc,
            scan_utc,
            x_metric,
            y_metric,
            transform,
            lag_seconds,
            previous_status,
            status,
            failure_count,
            missing_since_utc,
            discovery_rho,
            holdout_rho,
            corrected_q,
            stability,
            overlap_count,
            details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            "|".join((finding.x_metric, finding.y_metric, finding.transform, str(finding.lag_seconds))),
            to_utc_iso(NOW + timedelta(minutes=event_id)),
            to_utc_iso(NOW + timedelta(minutes=event_id)),
            finding.x_metric,
            finding.y_metric,
            finding.transform,
            finding.lag_seconds,
            "verified",
            status,
            0,
            None,
            finding.discovery_rho,
            finding.holdout_rho,
            finding.corrected_q,
            finding.stability,
            finding.overlap_count,
            "{}",
        ),
    )
    sqlite_conn.commit()
