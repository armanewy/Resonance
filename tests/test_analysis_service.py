from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from resonance.analysis.contracts import PairAnalysis
from resonance.analysis.service import (
    MetricPairAnalysis,
    ValidationOptions,
    analyze_metric_pair,
    list_analyzable_metrics,
)
from resonance.storage import Measurement, ensure_database, insert_measurements


START = datetime(2026, 6, 18, 0, 0, tzinfo=timezone.utc)
END = START + timedelta(minutes=5 * 119)


def test_list_analyzable_metrics_returns_typed_metric_metadata(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_pair_db(db_path)

    metrics = list_analyzable_metrics(db_path, START, END)

    by_name = {metric.metric: metric for metric in metrics}
    assert set(by_name) == {"ambient_temp_c", "cpu_percent", "tcp_latency_ms"}
    assert by_name["tcp_latency_ms"].units == ("ms",)
    assert by_name["tcp_latency_ms"].sources == ("synthetic",)
    assert by_name["tcp_latency_ms"].sample_count == 120
    assert by_name["tcp_latency_ms"].cadence_seconds == 300
    assert by_name["tcp_latency_ms"].coverage == 1.0
    assert by_name["tcp_latency_ms"].warnings == ()


def test_list_analyzable_metrics_uses_read_only_database_access(tmp_path) -> None:
    missing_db_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        list_analyzable_metrics(missing_db_path, START, END)

    assert not missing_db_path.exists()


def test_analyze_metric_pair_returns_typed_pair_analysis_with_metadata(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    _write_pair_db(db_path)

    analysis = analyze_metric_pair(
        db_path,
        "tcp_latency_ms",
        "cpu_percent",
        START,
        END,
        "raw",
        max_lag_steps=5,
        validation_options=ValidationOptions(
            min_aligned_points=40,
            min_overlap=20,
            permutations=19,
            permutation_seed=7,
        ),
    )

    assert isinstance(analysis, PairAnalysis)
    assert isinstance(analysis, MetricPairAnalysis)
    assert analysis.x_metric_summary.units == ("ms",)
    assert analysis.y_metric_summary.units == ("percent",)
    assert analysis.x_metric_summary.sample_count == 120
    assert analysis.y_metric_summary.sample_count == 120
    assert analysis.aligned_pair.cadence_seconds == 300
    assert analysis.aligned_pair.x_coverage == 1.0
    assert analysis.aligned_pair.y_coverage == 1.0
    assert analysis.lag_result.best_lag_steps == 2
    assert analysis.lag_result.best_lag_seconds == 600
    assert analysis.lag_result.best_rho is not None
    assert analysis.lag_result.best_rho > 0.98
    assert analysis.validation_result.holdout_overlap >= 20


def _write_pair_db(db_path) -> None:
    rng = random.Random(123)
    x_values = [rng.gauss(0, 1) for _ in range(120)]
    measurements = []
    for index, x_value in enumerate(x_values):
        timestamp = START + timedelta(minutes=5 * index)
        lagged_source = x_values[index - 2] if index >= 2 else rng.gauss(0, 1)
        y_value = (lagged_source * 10) + rng.gauss(0, 0.01)
        measurements.extend(
            (
                Measurement(timestamp, "tcp_latency_ms", x_value, "ms", "synthetic"),
                Measurement(timestamp, "cpu_percent", y_value, "percent", "synthetic"),
                Measurement(timestamp, "ambient_temp_c", 20.0 + (index % 5), "celsius", "synthetic"),
            )
        )

    conn = ensure_database(db_path)
    try:
        insert_measurements(conn, measurements)
    finally:
        conn.close()
