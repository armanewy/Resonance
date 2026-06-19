from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from resonance.science.ledger import (
    GENESIS_PREVIOUS_HASH,
    LedgerError,
    append_event,
    read_entries,
    verify_ledger,
)
from resonance.science.ledger_cli import main


CODE_COMMIT = "a" * 40
NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def test_valid_append_continuation_records_reproducibility_fields(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"

    first = append_event(
        "hypothesis_preregistered",
        {
            "dataset_snapshot_id": "snapshot-001",
            "hypothesis_hash": "hypothesis-a",
            "evaluator_version": "evaluator-1",
            "random_seed": 1234,
            "parameters": {"lag_minutes": 15},
            "metrics": {"primary": "r"},
        },
        artifact_hashes={"frozen_program": "sha256:abc", "report": "sha256:def"},
        code_commit=CODE_COMMIT,
        ledger_path=ledger_path,
        timestamp_utc=NOW,
    )
    second = append_event(
        "blind_evaluation_completed",
        {
            "dataset_snapshot_id": "snapshot-001",
            "hypothesis_hash": "hypothesis-a",
            "evaluator_version": "evaluator-1",
            "random_seed": 1234,
            "parameters": {"lag_minutes": 15},
            "metrics": {"primary": 0.42, "status": "inconclusive"},
        },
        artifact_hashes={"graph": "sha256:123", "report": "sha256:456"},
        code_commit=CODE_COMMIT,
        ledger_path=ledger_path,
        timestamp_utc=NOW,
    )

    verification = verify_ledger(ledger_path)
    entries = read_entries(ledger_path)
    assert verification.valid is True
    assert verification.entry_count == 2
    assert first["sequence_number"] == 1
    assert first["previous_entry_hash"] == GENESIS_PREVIOUS_HASH
    assert second["sequence_number"] == 2
    assert second["previous_entry_hash"] == first["entry_hash"]
    assert entries[1]["payload"]["dataset_snapshot_id"] == "snapshot-001"
    assert entries[1]["artifact_hashes"] == {"graph": "sha256:123", "report": "sha256:456"}


def test_verify_detects_payload_edit(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 2)
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["metrics"]["score"] = 99
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _write_lines(ledger_path, lines)

    verification = verify_ledger(ledger_path)

    assert verification.valid is False
    assert any("payload_hash does not match payload" in error for error in verification.errors)


def test_verify_detects_line_deletion(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 3)
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    _write_lines(ledger_path, [lines[0], lines[2]])

    verification = verify_ledger(ledger_path)

    assert verification.valid is False
    assert any("sequence_number" in error for error in verification.errors)


def test_verify_detects_reordering(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 3)
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    _write_lines(ledger_path, [lines[0], lines[2], lines[1]])

    verification = verify_ledger(ledger_path)

    assert verification.valid is False
    assert any("sequence_number" in error for error in verification.errors)


def test_verify_detects_broken_previous_hash(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 2)
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["previous_entry_hash"] = "b" * 64
    second["entry_hash"] = "c" * 64
    lines[1] = json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _write_lines(ledger_path, lines)

    verification = verify_ledger(ledger_path)

    assert verification.valid is False
    assert any("previous_entry_hash does not match prior entry" in error for error in verification.errors)


def test_verify_detects_truncated_final_line(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 2)
    ledger_path.write_bytes(ledger_path.read_bytes().rstrip(b"\n"))

    verification = verify_ledger(ledger_path)

    assert verification.valid is False
    assert any("truncated" in error for error in verification.errors)


def test_append_refuses_invalid_ledger(tmp_path: Path) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 1)
    ledger_path.write_bytes(ledger_path.read_bytes().rstrip(b"\n"))

    with pytest.raises(LedgerError):
        append_event(
            "result_interpreted",
            {"dataset_snapshot_id": "snapshot-001"},
            artifact_hashes={},
            code_commit=CODE_COMMIT,
            ledger_path=ledger_path,
            timestamp_utc=NOW,
        )

    assert verify_ledger(ledger_path).valid is False


def test_cli_verify_and_show(tmp_path: Path, capsys) -> None:
    ledger_path = _ledger_with_entries(tmp_path, 2)

    assert main(["--ledger", str(ledger_path), "verify"]) == 0
    verify_output = capsys.readouterr().out
    assert "Ledger verified: 2 entries" in verify_output

    assert main(["--ledger", str(ledger_path), "show", "--limit", "1"]) == 0
    show_output = capsys.readouterr().out
    assert '"sequence_number": 2' in show_output
    assert '"sequence_number": 1' not in show_output


def _ledger_with_entries(tmp_path: Path, count: int) -> Path:
    ledger_path = tmp_path / "ledger.jsonl"
    for index in range(count):
        append_event(
            "fit_completed",
            {
                "dataset_snapshot_id": "snapshot-001",
                "hypothesis_hash": "hypothesis-a",
                "evaluator_version": "evaluator-1",
                "random_seed": 1234 + index,
                "parameters": {"index": index},
                "metrics": {"score": index},
            },
            artifact_hashes={"report": f"sha256:{index}"},
            code_commit=CODE_COMMIT,
            ledger_path=ledger_path,
            timestamp_utc=NOW,
        )
    return ledger_path


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
