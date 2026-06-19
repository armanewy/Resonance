from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from resonance.science.selection import SelectionGates, select_candidate
from resonance.science.snapshots import create_snapshot
from resonance.storage import Measurement, init_db, insert_measurements
from resonance.synthetic import generate_synthetic_series


def test_strong_lag_produces_plausible_winner_and_records_artifact(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    fit = _fit_from_exploration(dataset, lag_seconds=900)
    candidate = _candidate("lagged-x", lag_seconds=900, parameters=fit)

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
    )
    evaluation = result.evaluations[0]

    assert result.selected_candidate_id == "lagged-x"
    assert evaluation.default_winner is True
    assert evaluation.tuning_spearman_rho is not None
    assert evaluation.tuning_spearman_rho > 0.95
    assert evaluation.baseline_improvement is not None
    assert evaluation.baseline_improvement > 0.45
    assert result.artifact is not None
    assert (tmp_path / "artifacts" / result.artifact["path"]).exists()


def test_selection_reuses_fitted_parameters_without_refitting(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    candidate = _candidate("frozen-bad-fit", lag_seconds=900, parameters={"scale": 0.0, "offset": 0.0})

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
    )

    assert result.selected_candidate_id is None
    assert result.evaluations[0].passed_gates["baseline_improvement"] is False


def test_shared_seasonality_only_does_not_beat_persistence_baseline(tmp_path: Path) -> None:
    dataset = generate_synthetic_series(
        "shared_seasonality_only",
        duration_hours=168,
        noise=0.6,
        seed=42,
    )
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    fit = _fit_from_exploration(dataset, lag_seconds=900)
    candidate = _candidate("seasonal-lag", lag_seconds=900, parameters=fit)

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
    )

    assert result.selected_candidate_id is None
    assert result.evaluations[0].passed_gates["baseline_improvement"] is False


def test_single_shared_outlier_fails_stability(tmp_path: Path) -> None:
    dataset = generate_synthetic_series(
        "single_shared_outlier",
        duration_hours=96,
        noise=0.35,
        seed=21,
    )
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=0)
    candidate = _candidate("outlier-fit", lag_seconds=0, parameters={"scale": 1.1, "offset": 0.0})

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
        gates=SelectionGates(min_abs_spearman=0.05, min_baseline_improvement=-1.0),
    )

    assert result.selected_candidate_id is None
    assert result.evaluations[0].passed_gates["window_stability"] is False


def test_relationship_break_fails_tuning_or_stability(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("relationship_break", duration_hours=96, noise=0.25, seed=34)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=0)
    fit = _fit_from_exploration(dataset, lag_seconds=0)
    candidate = _candidate("broken-relationship", lag_seconds=0, parameters=fit)

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
    )
    gates = result.evaluations[0].passed_gates

    assert result.selected_candidate_id is None
    assert gates["spearman"] is False or gates["window_stability"] is False


def test_independent_autocorrelated_is_not_convincing(tmp_path: Path) -> None:
    dataset = generate_synthetic_series(
        "independent_autocorrelated",
        duration_hours=96,
        noise=0.7,
        seed=55,
    )
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=0)
    fit = _fit_from_exploration(dataset, lag_seconds=0)
    candidate = _candidate("independent-fit", lag_seconds=0, parameters=fit)

    result = select_candidate(
        manifest["snapshot_id"],
        [candidate],
        artifact_root=tmp_path / "artifacts",
    )

    assert result.selected_candidate_id is None
    assert not all(result.evaluations[0].passed_gates.values())


def test_ties_are_deterministic_and_only_one_default_winner(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    fit = _fit_from_exploration(dataset, lag_seconds=900)
    candidates = [
        _candidate("b-candidate", lag_seconds=900, parameters=fit),
        _candidate("a-candidate", lag_seconds=900, parameters=fit),
    ]

    result = select_candidate(
        manifest["snapshot_id"],
        candidates,
        artifact_root=tmp_path / "artifacts",
    )

    assert result.ranking[:2] == ("a-candidate", "b-candidate")
    assert result.selected_candidate_id == "a-candidate"
    assert sum(evaluation.default_winner for evaluation in result.evaluations) == 1


def test_candidate_selection_never_loads_blind_artifact(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    blind_path = tmp_path / "artifacts" / manifest["artifacts"]["blind"]["path"]
    blind_path.write_bytes(b"this is not a valid blind artifact")
    fit = _fit_from_exploration(dataset, lag_seconds=900)

    result = select_candidate(
        manifest["snapshot_id"],
        [_candidate("lagged-x", lag_seconds=900, parameters=fit)],
        artifact_root=tmp_path / "artifacts",
    )

    assert result.selected_candidate_id == "lagged-x"


def test_pareto_front_exposes_performance_versus_complexity(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    fit = _fit_from_exploration(dataset, lag_seconds=900)
    candidates = [
        _candidate("simple", lag_seconds=900, parameters=fit),
        _candidate("bad-simple", lag_seconds=900, parameters={"scale": 0.0, "offset": 0.0}),
        _candidate("extra-complex", lag_seconds=900, parameters={**fit, "unused": 1.0}, extra_complexity=True),
    ]

    result = select_candidate(
        manifest["snapshot_id"],
        candidates,
        artifact_root=tmp_path / "artifacts",
    )

    assert "simple" in result.pareto_front
    assert "bad-simple" not in result.pareto_front


def _snapshot_from_dataset(tmp_path: Path, dataset, *, max_lag_seconds: int) -> dict:
    db_path = tmp_path / f"{dataset.metadata['scenario']}-{max_lag_seconds}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    measurements = []
    for sample in dataset.samples:
        if sample.x is not None:
            measurements.append(Measurement(sample.timestamp_utc, "x", sample.x, "unit", "synthetic"))
        if sample.y is not None:
            measurements.append(Measurement(sample.timestamp_utc, "y", sample.y, "unit", "synthetic"))
    insert_measurements(conn, measurements)
    conn.close()
    return create_snapshot(
        db_path=db_path,
        hours=int(math.ceil(dataset.metadata["duration_hours"] + 1)),
        metrics=["x", "y"],
        max_lag_seconds=max_lag_seconds,
        artifact_root=tmp_path / "artifacts",
    )


def _candidate(
    candidate_id: str,
    *,
    lag_seconds: int,
    parameters: dict[str, float],
    extra_complexity: bool = False,
) -> dict:
    expression: dict = {
        "node": "add",
        "left": {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": {
                "node": "lag",
                "input": {"node": "metric", "metric": "x"},
                "lag_seconds": lag_seconds,
            },
        },
        "right": {"node": "fitted_parameter", "parameter": "offset"},
    }
    if extra_complexity:
        expression = {
            "node": "add",
            "left": expression,
            "right": {"node": "numeric_constant", "value": 0.0},
        }
    return {
        "candidate_id": candidate_id,
        "hypothesis": {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": f"{candidate_id} predicts y",
            "concise_claim": "Synthetic x is associated with synthetic y in this dataset.",
            "rationale": "Test fixture candidate.",
            "target_metric": "y",
            "input_metrics": ["x"],
            "target_transform": "identity",
            "expression": expression,
            "parameter_bounds": {
                "scale": {"lower": -5.0, "upper": 5.0},
                "offset": {"lower": -20.0, "upper": 20.0},
            },
            "expected_direction": "positive",
            "maximum_lag_seconds": lag_seconds,
            "fitting_metric": "rmse",
            "tuning_metric": "mae",
            "blind_metrics": ["rmse", "spearman_r"],
            "minimum_blind_effect": 0.1,
            "minimum_baseline_improvement": 0.02,
            "negative_controls": [],
            "falsification_conditions": [
                {"description": "Tuning gates do not support preregistration."}
            ],
            "complexity_budget": {"max_ast_nodes": 8, "max_source_metrics": 1},
            "origin": "manual",
            "parent_hypothesis_ids": [],
            "random_seed": 20260619,
        },
        "fit_result": {
            "fit_result_id": f"fit-{candidate_id}",
            "fitted_parameters": parameters,
        },
    }


def _fit_from_exploration(dataset, *, lag_seconds: int) -> dict[str, float]:
    sample_interval = int(dataset.metadata["sample_interval_seconds"])
    lag_steps = lag_seconds // sample_interval if lag_seconds else 0
    exploration_end = len(dataset.samples) // 2
    pairs = []
    for index in range(lag_steps, exploration_end):
        x_value = dataset.samples[index - lag_steps].x
        y_value = dataset.samples[index].y
        if x_value is not None and y_value is not None:
            pairs.append((float(x_value), float(y_value)))
    if len(pairs) < 2:
        return {"scale": 0.0, "offset": 0.0}
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    variance = sum((x - x_mean) ** 2 for x, _ in pairs)
    if variance == 0:
        return {"scale": 0.0, "offset": y_mean}
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    scale = covariance / variance
    offset = y_mean - scale * x_mean
    return {"scale": scale, "offset": offset}
