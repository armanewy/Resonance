from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resonance.science import imagination
from resonance.science.cli import main
from resonance.science.discovery_brief import DiscoveryBrief
from resonance.science.ledger import read_entries, verify_ledger


def test_imagination_brief_sent_to_provider_excludes_tuning_and_blind_sentinels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    seen: dict[str, Any] = {}

    class CapturingProvider:
        name = "mock"
        model = "capturing-mock"
        prompt_version = "capturing-v1"
        request_config: dict[str, Any] = {}
        last_raw_proposals: tuple[Any, ...] = ()

        def propose(
            self,
            brief: DiscoveryBrief,
            max_hypotheses: int,
            seed: int,
        ):
            seen["brief_json"] = brief.canonical_json()
            seen["max_hypotheses"] = max_hypotheses
            self.last_raw_proposals = (_valid_hypothesis(_catalog_id()),)
            return []

    monkeypatch.setattr(
        imagination,
        "load_snapshot_manifest",
        lambda snapshot_id, *, artifact_root: _manifest(),
    )
    monkeypatch.setattr(
        imagination,
        "load_exploration_view",
        lambda snapshot_id, *, artifact_root: {
            "snapshot_id": snapshot_id,
            "partition": "exploration",
            "rows": [
                {
                    "timestamp_utc": "2026-06-01T00:00:00Z",
                    "metrics": {
                        "x": [{"value": 1.0, "unit": "synthetic", "source": "test"}],
                        "y": [{"value": 2.0, "unit": "synthetic", "source": "test"}],
                        "control": [{"value": 3.0, "unit": "synthetic", "source": "test"}],
                    },
                }
            ],
            "metadata": {
                "tuning": {"sentinel": "TUNING_SENTINEL_SHOULD_NOT_LEAK"},
                "blind": {"sentinel": "BLIND_SENTINEL_SHOULD_NOT_LEAK"},
            },
        },
    )

    result = imagination.imagine_hypotheses(
        snapshot_id="snapshot-123",
        provider_name="mock",
        max_hypotheses=8,
        provider=CapturingProvider(),
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )

    assert result["rejected_provider_count"] == 0
    assert seen["max_hypotheses"] == 8
    assert "TUNING_SENTINEL_SHOULD_NOT_LEAK" not in seen["brief_json"]
    assert "BLIND_SENTINEL_SHOULD_NOT_LEAK" not in seen["brief_json"]
    run = _artifact(artifact_root, result["run_id"])
    assert run["loaded_partitions_before_provider"] == ["exploration"]
    assert run["raw_blind_values_exposed"] is False


def test_imagination_cli_requires_approval_before_fit_and_never_auto_blinds(
    tmp_path: Path,
    capsys,
) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    base = ["--artifact-root", str(artifact_root), "--ledger", str(ledger_path)]

    snapshot = _run_ok(
        capsys,
        [
            *base,
            "snapshot",
            "synthetic",
            "--scenario",
            "strong_lag",
            "--duration-hours",
            "48",
            "--hours",
            "120",
        ],
    )
    imagined = _run_ok(
        capsys,
        [
            *base,
            "imagine",
            "--snapshot",
            snapshot["snapshot_id"],
            "--provider",
            "mock",
            "--max-hypotheses",
            "8",
        ],
    )
    run_id = imagined["run_id"]
    shown = _run_ok(capsys, [*base, "review", run_id])
    assert shown["approval_count"] == 0
    assert shown["proposals"][0]["approved"] is False

    assert main([*base, "fit-approved", run_id]) == 1
    failed = capsys.readouterr().out
    assert "no approved review-accepted proposals" in failed

    approved = _run_ok(capsys, [*base, "review", run_id, "--approve", "0"])
    assert approved["approval_count"] == 1
    fit = _run_ok(capsys, [*base, "fit-approved", run_id])
    assert fit["selected_candidate_id"] == fit["tuning"]["selected_candidate_id"]
    assert fit["raw_blind_values_exposed"] is False

    entries = read_entries(ledger_path)
    assert not any(entry["event_type"] == "blind_evaluation_completed" for entry in entries)
    assert any(
        entry["event_type"] == "result_interpreted"
        and entry["payload"]["interpretation_type"] == "imagination_human_approval"
        for entry in entries
    )
    assert any(
        entry["event_type"] == "result_interpreted"
        and entry["payload"]["interpretation_type"] == "imagination_tuning_selection"
        for entry in entries
    )

    candidate_id = fit["selected_candidate_id"]
    preregistration = _run_ok(capsys, [*base, "preregister", "--candidate", candidate_id])
    evaluation = _run_ok(capsys, [*base, "blind-evaluate", preregistration["preregistration_id"]])
    report = _run_ok(capsys, [*base, "report", preregistration["preregistration_id"]])
    assert evaluation["status"] == "pass"
    assert evaluation["raw_blind_values_exposed"] is False
    assert report["snapshot_id"] == snapshot["snapshot_id"]
    assert verify_ledger(ledger_path).valid is True


def test_imagine_cli_rejects_more_than_eight_hypotheses(tmp_path: Path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    base = ["--artifact-root", str(artifact_root), "--ledger", str(ledger_path)]
    snapshot = _run_ok(
        capsys,
        [
            *base,
            "snapshot",
            "synthetic",
            "--scenario",
            "strong_lag",
            "--duration-hours",
            "24",
            "--hours",
            "120",
        ],
    )

    status = main(
        [
            *base,
            "imagine",
            "--snapshot",
            snapshot["snapshot_id"],
            "--provider",
            "mock",
            "--max-hypotheses",
            "9",
        ]
    )

    assert status == 1
    assert "at most 8" in capsys.readouterr().out


def _run_ok(capsys, args: list[str]) -> dict[str, Any]:
    assert main(args) == 0
    return json.loads(capsys.readouterr().out)


def _artifact(root: Path, digest: str) -> dict[str, Any]:
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_id() -> str:
    return "a" * 64


def _manifest() -> dict[str, Any]:
    return {
        "snapshot_id": "snapshot-123",
        "max_lag_seconds": 900,
        "metric_catalog": {
            "catalog_id": _catalog_id(),
            "metric_names": ["control", "x", "y"],
            "metrics": [
                {"name": "control", "units": ["synthetic"], "sources": ["test"]},
                {"name": "x", "units": ["synthetic"], "sources": ["test"]},
                {"name": "y", "units": ["synthetic"], "sources": ["test"]},
            ],
        },
    }


def _valid_hypothesis(catalog_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "hypothesis_type": "observational_prediction",
        "title": "Mock lagged x predicts y",
        "concise_claim": "Lagged x is associated with y.",
        "rationale": "Synthetic mock proposal for provider sentinel tests.",
        "target_metric": "y",
        "input_metrics": ["x"],
        "target_transform": "identity",
        "expression": {
            "node": "add",
            "left": {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": "scale"},
                "right": {
                    "node": "lag",
                    "input": {"node": "metric", "metric": "x"},
                    "lag_seconds": 900,
                },
            },
            "right": {"node": "fitted_parameter", "parameter": "offset"},
        },
        "parameter_bounds": {
            "scale": {"lower": 0.0, "upper": 3.0},
            "offset": {"lower": -5.0, "upper": 5.0},
        },
        "expected_direction": "positive",
        "maximum_lag_seconds": 900,
        "fitting_metric": "rmse",
        "tuning_metric": "rmse",
        "blind_metrics": ["mae", "rmse", "spearman_r"],
        "minimum_blind_effect": 0.5,
        "minimum_baseline_improvement": 0.05,
        "negative_controls": [
            {"metric": "control", "rationale": "Control should not track prediction."}
        ],
        "falsification_conditions": [
            {"description": "Tuning performance does not improve over baseline."}
        ],
        "complexity_budget": {"max_ast_nodes": 8, "max_source_metrics": 1},
        "origin": "llm",
        "parent_hypothesis_ids": [],
        "snapshot_metric_catalog_id": catalog_id,
        "random_seed": 20260619,
    }
