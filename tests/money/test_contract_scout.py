from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
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

from behavior_lab.cli import main
from behavior_lab.money.contract_scout import ContractScout, OpportunityContractProposal
from behavior_lab.money_agents.roles import CONTRACT_SCOUT, MoneyAgentContext
from behavior_lab.money_agents.runtime import FinancialResearchAgentRuntime, ProviderResponse, StaticMoneyAgentProvider


class ContractScoutTests(unittest.TestCase):
    def test_viable_public_only_contract_becomes_experimentally_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scout = ContractScout(tmp)
            result = scout.run(proposals=[_proposal("public_cost_avoidance")], include_seed_families=False)

            self.assertEqual(result["accepted"], 1)
            item = result["items"]["eligible"][0]
            self.assertEqual(item["validation"]["status"], "eligible_experimental")
            self.assertTrue(item["validation"]["eligible_for_experimental_portfolio"])
            self.assertTrue(item["paper_only"])
            self.assertFalse(item["production_source_activation"])
            self.assertFalse(item["money_allocation"])

    def test_duplicate_prior_contract_is_preserved_but_not_reaccepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scout = ContractScout(tmp)
            proposal = _proposal("duplicate_contract")
            scout.run(proposals=[proposal], include_seed_families=False)
            second = scout.run(proposals=[proposal], include_seed_families=False)

            self.assertEqual(second["accepted"], 0)
            self.assertEqual(second["duplicates"], 1)
            duplicate = second["items"]["duplicates"][0]
            self.assertEqual(duplicate["validation"]["status"], "duplicate")
            self.assertIn("duplicate_prior_proposal", duplicate["validation"]["reasons"])

    def test_ambiguous_resolution_is_rejected(self) -> None:
        result = _run_single(_mutate({"resolution_source": {"source_id": "ambiguous_source", "ambiguous": True}}))

        self.assertEqual(result["items"]["rejected"][0]["validation"]["status"], "rejected")
        self.assertIn("ambiguous_resolution", result["items"]["rejected"][0]["validation"]["reasons"])

    def test_unknown_material_costs_are_rejected(self) -> None:
        proposal = _proposal("unknown_costs")
        proposal["material_costs"][0]["unknown"] = True
        result = _run_single(proposal)

        self.assertIn("unknown_material_costs", result["items"]["rejected"][0]["validation"]["reasons"])

    def test_unbounded_loss_is_rejected(self) -> None:
        result = _run_single(_mutate({"maximum_possible_loss": {"amount": None, "bounded": False}}))

        self.assertIn("unbounded_loss", result["items"]["rejected"][0]["validation"]["reasons"])

    def test_private_data_dependency_without_acquisition_path_is_rejected(self) -> None:
        proposal = _proposal("private_blocked")
        proposal["paper_mode_feasibility"] = {
            "paper_only": True,
            "requires_private_data": True,
            "can_collect_prospectively": False,
            "real_account_mutation_required": False,
        }
        proposal["currently_available_sources"] = []
        proposal["historical_depth"] = {}
        result = _run_single(proposal)

        reasons = result["items"]["rejected"][0]["validation"]["reasons"]
        self.assertIn("private_data_dependency_without_acquisition_path", reasons)

    def test_unclear_licensing_enters_approval_inbox(self) -> None:
        result = _run_single(_mutate({"licensing_concerns": ["unclear"]}))

        self.assertEqual(result["approval_required"], 1)
        approval = result["items"]["approval_required"][0]
        self.assertEqual(approval["validation"]["status"], "approval_required")
        self.assertEqual(approval["validation"]["approval_required"], ["unclear_license"])

    def test_missing_no_action_is_rejected(self) -> None:
        result = _run_single(_mutate({"no_action_alternative": "missing_no_action"}))

        self.assertIn("missing_no_action", result["items"]["rejected"][0]["validation"]["reasons"])

    def test_high_maintenance_low_value_is_rejected(self) -> None:
        result = _run_single(_mutate({"estimated_maintenance_burden": "high", "expected_information_value": "low"}))

        self.assertIn("high_maintenance_low_value", result["items"]["rejected"][0]["validation"]["reasons"])

    def test_no_viable_proposals_records_empty_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scout = ContractScout(tmp)
            result = scout.run(proposals=[], include_seed_families=False)

            self.assertEqual(result["accepted"], 0)
            self.assertEqual(result["rejected"], 0)
            self.assertEqual(scout.report()["proposal_counts"], {"eligible_experimental": 0, "approval_required": 0, "rejected": 0, "duplicate": 0})
            self.assertTrue(scout.verify())

    def test_approve_only_allows_eligible_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scout = ContractScout(tmp)
            scout.run(proposals=[_proposal("approvable")], include_seed_families=False)
            approved = scout.approve("approvable")

            self.assertEqual(approved["status"], "approved_for_experimental_portfolio")
            self.assertFalse(approved["production_source_activation"])
            self.assertFalse(approved["money_allocation"])

    def test_cli_run_proposals_report_and_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal_path = Path(tmp) / "proposals.json"
            proposal_path.write_text(json.dumps([_proposal("cli_contract")]), encoding="utf-8")
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "contract-scout", "run", "--state-dir", tmp, "--no-seed-families", "--proposals-json", str(proposal_path)])
            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["accepted"], 1)

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "contract-scout", "proposals", "--state-dir", tmp])
            proposals = json.loads(stream.getvalue())
            self.assertEqual(proposals["counts"]["eligible_experimental"], 1)

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "contract-scout", "reject", "cli_contract", "--state-dir", tmp, "--reason", "not_a_priority"])
            rejected = json.loads(stream.getvalue())
            self.assertEqual(rejected["status"], "manually_rejected")


def _run_single(proposal: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return ContractScout(tmp).run(proposals=[proposal], include_seed_families=False)


def _mutate(changes: dict) -> dict:
    proposal = _proposal("mutated_contract")
    for key, value in changes.items():
        proposal[key] = value
    return proposal


def _proposal(proposal_id: str) -> dict:
    return {
        "proposal_id": proposal_id,
        "title": "Cloud versus local batch compute cost-avoidance contract",
        "contract_family": "compute_cost_avoidance",
        "outcome": "Total completed batch-compute cost for a recurring workload.",
        "resolution_source": {"source_id": "provider_billing_export", "publisher": "official billing export", "ambiguous": False},
        "resolution_cadence": "daily",
        "decision_deadline": "before_batch_submission",
        "available_actions": [
            {"action_id": "run_local", "action_type": "no_action"},
            {"action_id": "run_cloud_spot", "action_type": "paper_cost_decision"},
        ],
        "no_action_alternative": "run_local",
        "payoff_formula": {"formula": "local_cost - cloud_cost - transfer_cost - research_cost", "executable": True},
        "material_costs": [
            {"name": "cloud_cost", "represented": True, "unknown": False},
            {"name": "local_power_cost", "represented": True, "unknown": False},
            {"name": "data_transfer_cost", "represented": True, "unknown": False},
        ],
        "capital_requirement": {"amount": 20.0, "currency": "USD", "bounded": True},
        "maximum_possible_loss": {"amount": 20.0, "currency": "USD", "bounded": True},
        "required_source_families": ["billing_export", "local_power_meter", "workload_runtime_log"],
        "currently_available_sources": ["billing_fixture", "runtime_log_fixture"],
        "missing_sources": [],
        "historical_depth": {"days": 90},
        "prospective_duration_required": "30 days",
        "expected_decision_frequency": "daily",
        "paper_mode_feasibility": {"paper_only": True, "can_collect_prospectively": True, "real_account_mutation_required": False},
        "platform_regulatory_dependencies": [],
        "credential_requirements": [],
        "licensing_concerns": [],
        "estimated_research_cost": {"usd": 3.0},
        "estimated_maintenance_burden": "low",
        "expected_information_value": "medium",
        "reason_it_may_fail": ["workload volume too low"],
        "citations": [{"title": "Local billing export", "url": "file:///local-only"}],
    }


class ProposalModelTests(unittest.TestCase):
    def test_proposal_hash_and_equivalence_are_stable(self) -> None:
        payload = _proposal("stable_contract")
        a = OpportunityContractProposal.from_dict(payload)
        b_payload = copy.deepcopy(payload)
        b_payload["status"] = "eligible_experimental"
        b = OpportunityContractProposal.from_dict(b_payload)

        self.assertEqual(a.proposal_hash(), b.proposal_hash())
        self.assertEqual(a.equivalence_key(), b.equivalence_key())


class ContractScoutAgentRoleTests(unittest.TestCase):
    def test_existing_money_agent_runtime_accepts_structured_contract_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = ProviderResponse(
                provider="mock",
                model="static",
                prompt_version="contract-scout-v1",
                content={"contract_proposals": [_proposal("agent_contract")], "rejections": []},
                tool_calls=[{"tool_name": "official_docs.search", "mode": "read_only"}],
                citations=[{"title": "Official billing docs", "url": "https://example.invalid/docs"}],
            )
            context = MoneyAgentContext(
                campaign_id="contract-scout-test",
                prompt_version="contract-scout-v1",
                explicit_budgets={"max_tool_calls": 1, "max_response_cost_usd": 0.0},
            )

            payload = FinancialResearchAgentRuntime(StaticMoneyAgentProvider(response), state_path=Path(tmp) / "agents.jsonl").run(CONTRACT_SCOUT, context)

            self.assertEqual(payload["role_id"], "financial_contract_scout")
            self.assertEqual(payload["content"]["contract_proposals"][0]["proposal_id"], "agent_contract")

    def test_contract_scout_agent_role_rejects_real_action_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = _proposal("real_action_attempt")
            proposal["paper_mode_feasibility"] = {"paper_only": False, "real_account_mutation_required": True}
            response = ProviderResponse(
                provider="mock",
                model="static",
                prompt_version="contract-scout-v1",
                content={"contract_proposals": [proposal], "rejections": []},
            )
            context = MoneyAgentContext(campaign_id="contract-scout-test", prompt_version="contract-scout-v1")

            with self.assertRaises(PermissionError):
                FinancialResearchAgentRuntime(StaticMoneyAgentProvider(response), state_path=Path(tmp) / "agents.jsonl").run(CONTRACT_SCOUT, context)


if __name__ == "__main__":
    unittest.main()
