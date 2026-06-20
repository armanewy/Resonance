from __future__ import annotations

import json
from pathlib import Path

from resonance.science.ablation import run_ablation
from resonance.science.cli import main
from resonance.science.ledger import read_entries, verify_ledger


def test_ablation_compares_llm_and_required_baselines_without_blind(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"

    result = run_ablation(
        scenarios=["strong_lag", "shared_seasonality_only"],
        provider_name="mock",
        seed=123,
        candidate_budget=4,
        duration_hours=48,
        snapshot_hours=120,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    assert result["raw_blind_values_exposed"] is False
    assert result["blind_evaluation"]["run"] is False
    assert {scenario["scenario"] for scenario in result["scenarios"]} == {
        "strong_lag",
        "shared_seasonality_only",
    }

    strong_lag = next(item for item in result["scenarios"] if item["scenario"] == "strong_lag")
    generator_names = {item["generator_name"] for item in strong_lag["generators"]}
    assert generator_names == {
        "llm_mock",
        "pairwise_lag",
        "random_dsl",
        "linear_combo",
        "persistence_zero_residual",
    }
    assert any(
        generator["recovery"]["any_candidate_recovered_expected_relation"]
        for generator in strong_lag["generators"]
    )

    shared = next(item for item in result["scenarios"] if item["scenario"] == "shared_seasonality_only")
    assert all(generator["recovery"]["known_positive"] is False for generator in shared["generators"])
    assert all(generator["blind_performance"] is None for generator in shared["generators"])
    assert all(generator["cost_metadata"]["total_cost_usd"] == 0.0 for generator in shared["generators"])

    report = _artifact(artifact_root, result["run_id"])
    assert report["record_type"] == "science_llm_ablation_report"
    assert report["conclusion"] == result["conclusion"]
    assert verify_ledger(ledger_path).valid is True
    entries = read_entries(ledger_path)
    assert any(
        entry["event_type"] == "experiment_completed"
        and entry["payload"]["experiment_type"] == "science_llm_ablation"
        for entry in entries
    )
    assert not any(entry["event_type"] == "blind_evaluation_completed" for entry in entries)


def test_ablation_cli_records_report_artifact(tmp_path: Path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"

    status = main(
        [
            "--artifact-root",
            str(artifact_root),
            "--ledger",
            str(ledger_path),
            "ablate",
            "--scenarios",
            "strong_lag,shared_seasonality_only",
            "--provider",
            "mock",
            "--seed",
            "123",
            "--candidate-budget",
            "4",
            "--duration-hours",
            "48",
            "--snapshot-hours",
            "120",
        ]
    )

    assert status == 0
    result = json.loads(capsys.readouterr().out)
    assert result["provider"] == "mock"
    assert result["candidate_budget"] == 4
    assert result["blind_evaluation"]["run"] is False
    assert (artifact_root / result["artifact"]["path"]).exists()


def _artifact(root: Path, digest: str) -> dict:
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    return json.loads(path.read_text(encoding="utf-8"))
