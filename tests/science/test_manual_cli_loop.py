from __future__ import annotations

import json
from pathlib import Path

import pytest

from resonance.science.cli import main
from resonance.science.ledger import read_entries, verify_ledger


HYPOTHESIS_PATH = Path("examples/science/strong_lag_hypothesis.json")


def test_manual_cli_runs_complete_strong_lag_loop_and_refuses_repeat(tmp_path: Path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"

    result = _run_loop(
        capsys,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        scenario="strong_lag",
    )

    assert result["evaluation"]["status"] == "pass"
    assert result["evaluation"]["raw_blind_values_exposed"] is False
    assert result["report"]["snapshot_id"] == result["snapshot_id"]
    assert result["report"]["snapshot_git_commit"]
    assert result["report"]["evaluation_code_commit"]
    assert result["report"]["report_code_commit"]
    assert result["report"]["llm_used"] is False
    assert result["report"]["arbitrary_generated_code_executable"] is False

    repeat_status = main(
        [
            "--artifact-root",
            str(artifact_root),
            "--ledger",
            str(ledger_path),
            "blind-evaluate",
            result["preregistration_id"],
        ]
    )
    repeat_output = capsys.readouterr().out
    assert repeat_status == 1
    assert "blind evaluation already completed" in repeat_output

    entries = read_entries(ledger_path)
    event_types = [entry["event_type"] for entry in entries]
    assert "hypothesis_proposed" in event_types
    assert "fit_completed" in event_types
    assert "hypothesis_preregistered" in event_types
    assert "blind_evaluation_completed" in event_types
    assert any(
        entry["event_type"] == "result_interpreted"
        and entry["payload"]["interpretation_type"] == "blind_verdict_report"
        for entry in entries
    )
    assert len([entry for entry in entries if entry["event_type"] == "blind_evaluation_completed"]) == 1
    assert verify_ledger(ledger_path).valid is True


@pytest.mark.parametrize(
    "scenario",
    [
        "shared_seasonality_only",
        "single_shared_outlier",
        "relationship_break",
        "independent_autocorrelated",
    ],
)
def test_manual_cli_null_and_adversarial_scenarios_do_not_pass(
    tmp_path: Path,
    capsys,
    scenario: str,
) -> None:
    result = _run_loop(
        capsys,
        artifact_root=tmp_path / scenario / "artifacts",
        ledger_path=tmp_path / scenario / "ledger.jsonl",
        scenario=scenario,
    )

    assert result["evaluation"]["status"] in {"fail", "inconclusive"}
    assert result["evaluation"]["status"] != "pass"
    assert result["report"]["status"] == result["evaluation"]["status"]


def _run_loop(
    capsys,
    *,
    artifact_root: Path,
    ledger_path: Path,
    scenario: str,
) -> dict:
    base = ["--artifact-root", str(artifact_root), "--ledger", str(ledger_path)]

    snapshot = _run_ok(
        capsys,
        [
            *base,
            "snapshot",
            "synthetic",
            "--scenario",
            scenario,
            "--duration-hours",
            "48",
            "--hours",
            "120",
        ],
    )
    snapshot_id = snapshot["snapshot_id"]

    proposal = _run_ok(
        capsys,
        [
            *base,
            "hypothesis",
            "validate",
            str(HYPOTHESIS_PATH),
            "--snapshot",
            snapshot_id,
        ],
    )
    assert proposal["llm_used"] is False

    fit = _run_ok(capsys, [*base, "fit", str(HYPOTHESIS_PATH), "--snapshot", snapshot_id])
    tuning = _run_ok(capsys, [*base, "tune", "--run", fit["run_id"]])
    preregistration = _run_ok(
        capsys,
        [*base, "preregister", "--candidate", proposal["candidate_id"]],
    )
    evaluation = _run_ok(capsys, [*base, "blind-evaluate", preregistration["preregistration_id"]])
    report = _run_ok(capsys, [*base, "report", preregistration["preregistration_id"]])

    return {
        "snapshot_id": snapshot_id,
        "proposal": proposal,
        "fit": fit,
        "tuning": tuning,
        "preregistration_id": preregistration["preregistration_id"],
        "evaluation": evaluation,
        "report": report,
    }


def _run_ok(capsys, args: list[str]) -> dict:
    assert main(args) == 0
    output = capsys.readouterr().out
    return json.loads(output)
