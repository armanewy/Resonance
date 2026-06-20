from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from resonance.science.contracts import HypothesisSpec, expression_node_count
from resonance.science.ledger import read_entries, verify_ledger
from resonance.science.program_search import run_program_search
from resonance.science.search_cli import main as search_main
from resonance.science.snapshots import create_snapshot
from resonance.storage import Measurement, init_db, insert_measurements
from resonance.synthetic import generate_synthetic_series


def test_program_search_improves_known_lagged_relation_and_records_lineage(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)

    result = run_program_search(
        [_seed_hypothesis(lag_seconds=0)],
        snapshot_id=manifest["snapshot_id"],
        budget=16,
        beam_width=6,
        complexity_penalty=0.001,
        random_seed=7,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
    )

    assert 1 <= result.evaluated_count <= 16
    assert result.selected_candidate_id is not None
    selected = _candidate(result, result.selected_candidate_id)
    seed = _candidate(result, result.ranking[-1])
    assert selected.selection_evaluation.tuning_mae is not None
    assert selected.selection_evaluation.passed is True
    assert any(candidate.parent_candidate_id for candidate in result.candidates)
    assert result.pareto_front
    assert selected.selection_evaluation.tuning_mae <= seed.selection_evaluation.tuning_mae

    for candidate in result.candidates:
        artifact_path = tmp_path / "artifacts" / candidate.artifact["path"]
        assert artifact_path.exists()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["raw_blind_values_exposed"] is False

    entries = read_entries(tmp_path / "ledger.jsonl")
    search_entries = [entry for entry in entries if entry["event_type"] == "program_search_completed"]
    assert len(search_entries) == 1
    assert search_entries[0]["payload"]["lineage"]
    assert verify_ledger(tmp_path / "ledger.jsonl").valid is True


def test_program_search_avoids_promoting_null_scenario(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("independent_autocorrelated", duration_hours=96, noise=0.7, seed=55)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=0)

    result = run_program_search(
        [_seed_hypothesis(lag_seconds=0)],
        snapshot_id=manifest["snapshot_id"],
        budget=8,
        beam_width=4,
        complexity_penalty=0.001,
        random_seed=4,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
    )

    assert result.selected_candidate_id is None
    assert not any(candidate.selection_evaluation.default_winner for candidate in result.candidates)


def test_program_search_respects_budget_and_never_reads_blind_values(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    blind_path = tmp_path / "artifacts" / manifest["artifacts"]["blind"]["path"]
    blind_path.write_bytes(b"not valid blind data")

    result = run_program_search(
        [_seed_hypothesis(lag_seconds=900)],
        snapshot_id=manifest["snapshot_id"],
        budget=3,
        beam_width=10,
        complexity_penalty=0.001,
        random_seed=13,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
        record_ledger=False,
    )

    assert result.evaluated_count <= 3
    assert result.config.budget == 3
    assert result.config.beam_width == 10
    assert result.to_dict()["raw_blind_values_exposed"] is False


def test_program_search_caps_defaults_to_prompt_limits(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)

    result = run_program_search(
        [_seed_hypothesis(lag_seconds=900)],
        snapshot_id=manifest["snapshot_id"],
        budget=1000,
        beam_width=100,
        complexity_penalty=0.001,
        random_seed=17,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
        record_ledger=False,
    )

    assert result.config.budget == 100
    assert result.config.beam_width == 10
    assert result.evaluated_count <= 100
    assert result.config.max_depth == _seed_hypothesis(lag_seconds=900).complexity_budget.max_ast_nodes


def test_program_search_prefers_simpler_expression_when_performance_ties(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    simple = _seed_hypothesis(lag_seconds=900)
    complex_equivalent = _seed_hypothesis(
        lag_seconds=900,
        expression={
            "node": "add",
            "left": simple.expression.model_dump(mode="json"),
            "right": {"node": "numeric_constant", "value": 0.0},
        },
        max_ast_nodes=12,
    )

    result = run_program_search(
        [complex_equivalent, simple],
        snapshot_id=manifest["snapshot_id"],
        budget=2,
        beam_width=2,
        complexity_penalty=0.001,
        random_seed=2,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
        record_ledger=False,
    )

    selected = _candidate(result, result.selected_candidate_id)
    assert expression_node_count(selected.hypothesis.expression) == expression_node_count(simple.expression)


def test_program_search_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)

    first = run_program_search(
        [_seed_hypothesis(lag_seconds=0)],
        snapshot_id=manifest["snapshot_id"],
        budget=10,
        beam_width=5,
        complexity_penalty=0.001,
        random_seed=42,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "first.jsonl",
    )
    second = run_program_search(
        [_seed_hypothesis(lag_seconds=0)],
        snapshot_id=manifest["snapshot_id"],
        budget=10,
        beam_width=5,
        complexity_penalty=0.001,
        random_seed=42,
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "second.jsonl",
    )

    assert first.ranking == second.ranking
    assert first.selected_candidate_id == second.selected_candidate_id
    assert [candidate.hypothesis.hypothesis_hash() for candidate in first.candidates] == [
        candidate.hypothesis.hypothesis_hash() for candidate in second.candidates
    ]


def test_search_cli_runs_explicit_local_search(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = generate_synthetic_series("strong_lag", duration_hours=96, noise=0.25, seed=11)
    manifest = _snapshot_from_dataset(tmp_path, dataset, max_lag_seconds=900)
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(_seed_hypothesis(lag_seconds=900).model_dump(mode="json")),
        encoding="utf-8",
    )

    assert search_main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "run",
            "--snapshot",
            manifest["snapshot_id"],
            "--seed-hypothesis",
            str(seed_path),
            "--budget",
            "4",
            "--beam-width",
            "2",
            "--random-seed",
            "9",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["snapshot_id"] == manifest["snapshot_id"]
    assert output["evaluated_count"] <= 4
    assert output["raw_blind_values_exposed"] is False


def _candidate(result, candidate_id: str | None):
    assert candidate_id is not None
    return next(candidate for candidate in result.candidates if candidate.candidate_id == candidate_id)


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
        measurements.append(Measurement(sample.timestamp_utc, "control", 0.0, "unit", "synthetic"))
    insert_measurements(conn, measurements)
    conn.close()
    return create_snapshot(
        db_path=db_path,
        hours=int(math.ceil(dataset.metadata["duration_hours"] + 1)),
        metrics=["control", "x", "y"],
        max_lag_seconds=max_lag_seconds,
        artifact_root=tmp_path / "artifacts",
    )


def _seed_hypothesis(
    *,
    lag_seconds: int,
    expression: dict | None = None,
    max_ast_nodes: int = 8,
) -> HypothesisSpec:
    expression = expression or {
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
    return HypothesisSpec.model_validate(
        {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": "x predicts y",
            "concise_claim": "Synthetic x predicts y in this dataset.",
            "rationale": "Deterministic program-search fixture.",
            "target_metric": "y",
            "input_metrics": ["x"],
            "target_transform": "identity",
            "expression": expression,
            "parameter_bounds": {
                "scale": {"lower": -5.0, "upper": 5.0},
                "offset": {"lower": -20.0, "upper": 20.0},
            },
            "expected_direction": "positive",
            "maximum_lag_seconds": max(lag_seconds, 900),
            "fitting_metric": "rmse",
            "tuning_metric": "rmse",
            "blind_metrics": ["rmse", "spearman_r"],
            "minimum_blind_effect": 0.1,
            "minimum_baseline_improvement": 0.02,
            "negative_controls": [{"metric": "control", "rationale": "Synthetic null control."}],
            "falsification_conditions": [{"description": "Tuning gates do not support preregistration."}],
            "complexity_budget": {"max_ast_nodes": max_ast_nodes, "max_source_metrics": 1},
            "origin": "manual",
            "parent_hypothesis_ids": [],
            "random_seed": 20260619,
        }
    )
