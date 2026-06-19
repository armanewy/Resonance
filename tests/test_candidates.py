from __future__ import annotations

from datetime import datetime, timedelta, timezone

from resonance.analysis.candidates import CandidateOptions, select_candidate_pairs
from resonance.storage import Measurement, ensure_database, insert_measurements


START = datetime(2026, 6, 18, 0, 0, tzinfo=timezone.utc)


def test_select_candidate_pairs_returns_valid_canonical_pairs(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_regular_metrics(db_path, ("cpu_percent", "tcp_latency_ms", "weather_temperature_c"), count=6)

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(minutes=25),
        options=CandidateOptions(min_observations=4, min_coverage=0.9, min_aligned_bins=4),
    )

    assert [(pair.x_metric, pair.y_metric) for pair in result.pairs] == [
        ("cpu_percent", "tcp_latency_ms"),
        ("cpu_percent", "weather_temperature_c"),
        ("tcp_latency_ms", "weather_temperature_c"),
    ]
    assert all(pair.aligned_bins == 6 for pair in result.pairs)
    assert result.rejections == ()


def test_select_candidate_pairs_rejects_duplicate_requested_metrics(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_regular_metrics(db_path, ("cpu_percent", "tcp_latency_ms"), count=5)

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(minutes=20),
        metrics=("cpu_percent", "cpu_percent", "tcp_latency_ms"),
        options=CandidateOptions(min_observations=4, min_coverage=0.9, min_aligned_bins=4),
    )

    assert [(pair.x_metric, pair.y_metric) for pair in result.pairs] == [
        ("cpu_percent", "tcp_latency_ms")
    ]
    assert _rejection_reasons(result) == {"identical_metrics"}


def test_select_candidate_pairs_rejects_low_coverage_metrics(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_regular_metrics(db_path, ("cpu_percent",), count=6)
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                Measurement(START, "gappy_pressure_hpa", 1011.0, "hPa", "synthetic"),
                Measurement(START + timedelta(minutes=10), "gappy_pressure_hpa", 1012.0, "hPa", "synthetic"),
            ],
        )
    finally:
        conn.close()

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(minutes=25),
        options=CandidateOptions(min_observations=2, min_coverage=0.8, min_aligned_bins=2),
    )

    assert result.pairs == ()
    assert ("gappy_pressure_hpa",) in [rejection.metrics for rejection in result.rejections]
    assert _rejection_reasons(result) == {"low_coverage"}


def test_select_candidate_pairs_rejects_status_and_boolean_metrics(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_regular_metrics(db_path, ("cpu_percent",), count=5)
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                Measurement(START + timedelta(minutes=5 * index), "tcp_success", 1.0, "boolean", "synthetic")
                for index in range(5)
            ]
            + [
                Measurement(START + timedelta(minutes=5 * index), "weather_code", 2.0, "code", "synthetic")
                for index in range(5)
            ],
        )
    finally:
        conn.close()

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(minutes=20),
        options=CandidateOptions(min_observations=4, min_coverage=0.9, min_aligned_bins=4),
    )

    assert result.pairs == ()
    assert _rejections_by_metric(result)["tcp_success"] == "status_or_flag_metric"
    assert _rejections_by_metric(result)["weather_code"] == "status_or_flag_metric"


def test_select_candidate_pairs_rejects_known_direct_derivations(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                measurement
                for index in range(5)
                for measurement in (
                    Measurement(
                        START + timedelta(minutes=5 * index),
                        "network_recv_bytes",
                        1000.0 + index,
                        "bytes",
                        "synthetic",
                    ),
                    Measurement(
                        START + timedelta(minutes=5 * index),
                        "network_recv_bytes_per_second",
                        float(index),
                        "bytes/second",
                        "synthetic",
                        {"derived_from": "network_recv_bytes"},
                    ),
                )
            ],
        )
    finally:
        conn.close()

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(minutes=20),
        options=CandidateOptions(min_observations=4, min_coverage=0.9, min_aligned_bins=4),
    )

    assert result.pairs == ()
    assert _rejection_reasons(result) == {"direct_derivation"}


def test_select_candidate_pairs_rejects_coarsest_cadence_with_too_few_aligned_bins(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                Measurement(START + timedelta(minutes=5 * index), "cpu_percent", float(index), "percent", "synthetic")
                for index in range(61)
            ]
            + [
                Measurement(START + timedelta(hours=index), "weather_temperature_c", 20.0 + index, "C", "synthetic")
                for index in range(6)
            ],
        )
    finally:
        conn.close()

    result = select_candidate_pairs(
        db_path,
        START,
        START + timedelta(hours=5),
        options=CandidateOptions(min_observations=5, min_coverage=0.9, min_aligned_bins=8),
    )

    assert result.pairs == ()
    assert _rejection_reasons(result) == {"too_few_aligned_bins"}


def _write_regular_metrics(db_path, metric_names: tuple[str, ...], *, count: int) -> None:
    conn = ensure_database(db_path)
    try:
        insert_measurements(
            conn,
            [
                Measurement(
                    START + timedelta(minutes=5 * index),
                    metric,
                    float(index + metric_index),
                    "percent" if metric.endswith("percent") else "ms",
                    "synthetic",
                )
                for index in range(count)
                for metric_index, metric in enumerate(metric_names)
            ],
        )
    finally:
        conn.close()


def _rejection_reasons(result) -> set[str]:
    return {rejection.reason for rejection in result.rejections}


def _rejections_by_metric(result) -> dict[str, str]:
    return {
        rejection.metrics[0]: rejection.reason
        for rejection in result.rejections
        if len(rejection.metrics) == 1
    }
