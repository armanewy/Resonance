from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.money.contracts import Action, FinancialDecisionContract
from behavior_lab.money.storage import MoneyStorage


class MoneyStorageTests(unittest.TestCase):
    def test_contract_storage_round_trips_hash_and_contract(self) -> None:
        contract = FinancialDecisionContract(
            contract_id="storage_contract",
            domain="seller",
            target={"metric": "margin"},
            decision_horizon="1d",
            decision_deadline="2026-01-02T00:00:00+00:00",
            available_actions=[Action(action_id="abstain", action_type="no_action")],
            no_action_id="abstain",
            payoff_specification={"metric": "net"},
            cost_policy={"unknown": "ineligible"},
            risk_policy={"paper_only": True},
            liquidity_policy={"not_applicable": True},
            resolution_source={"source": "fixture"},
            data_cutoff_policy={"as_of": "decision"},
            prospective_requirement={"required": True},
            notification_threshold={"enabled": False},
            paper_only=True,
            contract_version="v1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            storage = MoneyStorage(tmp)
            path = storage.write_contract(contract)
            self.assertTrue(path.exists())
            stored = storage.list_contracts()[0]
            self.assertEqual(stored["contract_hash"], contract.contract_hash())
            loaded = storage.read_contract("storage_contract")
            self.assertEqual(loaded.contract_hash(), contract.contract_hash())
            self.assertTrue(storage.ledger.verify())

    def test_contract_storage_rejects_path_traversal_ids(self) -> None:
        contract = FinancialDecisionContract(
            contract_id="../escaped_contract",
            domain="seller",
            target={"metric": "margin"},
            decision_horizon="1d",
            decision_deadline="2026-01-02T00:00:00+00:00",
            available_actions=[Action(action_id="abstain", action_type="no_action")],
            no_action_id="abstain",
            payoff_specification={"metric": "net"},
            cost_policy={"unknown": "ineligible"},
            risk_policy={"paper_only": True},
            liquidity_policy={"not_applicable": True},
            resolution_source={"source": "fixture"},
            data_cutoff_policy={"as_of": "decision"},
            prospective_requirement={"required": True},
            notification_threshold={"enabled": False},
            paper_only=True,
            contract_version="v1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            storage = MoneyStorage(tmp)
            with self.assertRaises(ValueError):
                storage.write_contract(contract)
            with self.assertRaises(ValueError):
                storage.read_contract("../escaped_contract")


if __name__ == "__main__":
    unittest.main()
