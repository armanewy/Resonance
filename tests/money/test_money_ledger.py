from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.money.ledger import MoneyLedger, MoneyLedgerEntry, MoneyLedgerError


def _entry(decision_id: str = "decision_1", *, designation: str = "paper") -> MoneyLedgerEntry:
    return MoneyLedgerEntry(
        decision_id=decision_id,
        contract_hash="contract_hash",
        decision_timestamp="2026-01-01T12:00:00+00:00",
        data_cutoff="2026-01-01T11:00:00+00:00",
        target={"name": "seller_margin"},
        action_alternatives=["abstain", "accept"],
        selected_action="abstain",
        no_action_alternative="abstain",
        capital_required=0.0,
        maximum_possible_loss=0.0,
        expected_gross_value=0.0,
        uncertainty_adjustment=0.0,
        fees=0.0,
        slippage=0.0,
        shipping=0.0,
        taxes_or_tax_assumption_reference="not_applicable",
        holding_costs=0.0,
        return_refund_allowance=0.0,
        research_api_cost=0.0,
        conservative_expected_net_value=0.0,
        decision_deadline="2026-01-02T00:00:00+00:00",
        feature_program_hash="feature_hash",
        evidence_state="paper_decision" if designation == "paper" else "proposed",
        designation=designation,
        provenance={"strategy_id": "test", "source_id": "fixture"},
        artifact_hashes={"fixture": "abc"},
        assumption_versions={"costs": "v1"},
    )


class MoneyLedgerTests(unittest.TestCase):
    def test_ledger_appends_and_resolves_without_rewriting_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MoneyLedger(str(Path(tmp) / "money.jsonl"))
            first = ledger.append_entry(_entry())
            resolved = ledger.append_resolution(
                "decision_1",
                resolution={"outcome": "no_action", "realized_costs": {"fees": 0.0}},
                realized_gross_value=0.0,
                realized_net_value=0.0,
                mechanically_defined_no_action_outcome={"value": 0.0},
            )
            self.assertTrue(ledger.verify())
            records = ledger.records()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["record_hash"], first["record_hash"])
            self.assertEqual(records[1]["payload"]["supersedes_entry_hash"], first["record_hash"])
            self.assertEqual(resolved["payload"]["evidence_state"], "resolved_paper")
            self.assertIsNone(records[0]["payload"]["resolution"])

    def test_duplicate_decision_must_supersede_latest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MoneyLedger(str(Path(tmp) / "money.jsonl"))
            ledger.append_entry(_entry())
            with self.assertRaises(MoneyLedgerError):
                ledger.append_entry(_entry())

    def test_updates_cannot_rewrite_forecasts_or_mix_designations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MoneyLedger(str(Path(tmp) / "money.jsonl"))
            record = ledger.append_entry(_entry())
            with self.assertRaises(MoneyLedgerError):
                ledger.append_entry(
                    replace(
                        _entry(),
                        selected_action="accept",
                        supersedes_entry_hash=record["record_hash"],
                    )
                )
            with self.assertRaises(MoneyLedgerError):
                ledger.append_entry(
                    replace(
                        _entry(designation="real"),
                        supersedes_entry_hash=record["record_hash"],
                    )
                )

    def test_unknown_costs_and_manual_real_approval_are_rejected(self) -> None:
        with self.assertRaises(MoneyLedgerError):
            replace(_entry(), material_costs_known=False, conservative_expected_net_value=0.0)
        with self.assertRaises(MoneyLedgerError):
            replace(_entry(), fees=None)
        with self.assertRaises(MoneyLedgerError):
            replace(_entry(), fees=None, conservative_expected_net_value=None)
        with self.assertRaises(MoneyLedgerError):
            replace(_entry(), evidence_state="manually_approved_real", designation="real")
        with self.assertRaises(MoneyLedgerError):
            _entry(designation="real")

    def test_resolved_entries_require_no_action_outcome_and_reconciled_costs(self) -> None:
        base = _entry()
        with self.assertRaises(MoneyLedgerError):
            replace(
                base,
                evidence_state="resolved_paper",
                resolution={"outcome": "sold", "realized_costs": {"fees": 2.0}},
                realized_gross_value=10.0,
                realized_net_value=8.0,
            )
        with self.assertRaises(MoneyLedgerError):
            replace(
                base,
                evidence_state="resolved_paper",
                resolution={"outcome": "sold", "realized_costs": {"fees": -2.0}},
                realized_gross_value=10.0,
                realized_net_value=12.0,
                mechanically_defined_no_action_outcome={"value": 0.0},
            )
        with self.assertRaises(MoneyLedgerError):
            replace(
                base,
                evidence_state="resolved_paper",
                resolution={"outcome": "sold", "realized_costs": {"fees": None}},
                realized_gross_value=10.0,
                realized_net_value=10.0,
                mechanically_defined_no_action_outcome={"value": 0.0},
            )
        with self.assertRaises(MoneyLedgerError):
            replace(
                base,
                evidence_state="resolved_paper",
                resolution={"outcome": "sold", "realized_costs": {"fees": 2.0}},
                realized_gross_value=10.0,
                realized_net_value=9.0,
                mechanically_defined_no_action_outcome={"value": 0.0},
            )

    def test_corrections_append_superseding_record_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MoneyLedger(str(Path(tmp) / "money.jsonl"))
            first = ledger.append_entry(_entry())
            corrected = replace(
                _entry(),
                supersedes_entry_hash=first["record_hash"],
                artifact_hashes={"fixture": "def"},
            )
            with self.assertRaises(MoneyLedgerError):
                ledger.append_entry(corrected)
            ledger.append_correction(corrected, reason="correct artifact hash")
            self.assertEqual(len(ledger.records()), 2)
            self.assertEqual(ledger.records()[-1]["payload"]["correction_reason"], "correct artifact hash")


if __name__ == "__main__":
    unittest.main()
