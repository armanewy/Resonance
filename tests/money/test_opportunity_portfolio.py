from __future__ import annotations

from contextlib import redirect_stdout
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
from behavior_lab.money.portfolio import (
    AttentionBudget,
    AutonomousFinancialOpportunityPortfolio,
    OpportunityPortfolioError,
    PortfolioContract,
    REAL_ACTION_FLAGS,
)


class OpportunityPortfolioTests(unittest.TestCase):
    def test_weekly_cycle_runs_seeded_public_contracts_and_blocks_seller_without_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            portfolio = AutonomousFinancialOpportunityPortfolio(tmp, budget=AttentionBudget(approvals_per_week=5))

            result = portfolio.run_cycle(schedule="weekly", as_of="2026-06-22T12:00:00+00:00")

            self.assertTrue(result["paper_only"])
            self.assertFalse(any(result["production_state"].values()))
            self.assertIn("seed-seller-shadow", {item["contract_id"] for item in result["contracts"] if item["status"] == "blocked"})
            completed_contracts = {item["contract_id"] for item in result["paper_autopilot"]["completed_tasks"] if item.get("contract_id")}
            self.assertIn("seed-weather-edge", completed_contracts)
            self.assertIn("seed-etf-risk", completed_contracts)
            self.assertTrue(any(item["blocked_does_not_stop_portfolio"] for item in result["allocation"]["allocations"]))

    def test_budget_allocator_prioritizes_value_and_deprioritizes_dead_ends(self) -> None:
        good = PortfolioContract(
            contract_id="good",
            contract_family="weather_event_market",
            title="Good",
            lab="weather_edge",
            expected_economic_value=5.0,
            expected_information_gain=1.0,
            feedback_cadence_days=1,
            source_config={"provider": "fixture"},
        )
        dead = PortfolioContract(
            contract_id="dead",
            contract_family="compute_cost_avoidance",
            title="Dead",
            lab=None,
            expected_information_gain=0.1,
            source_acquisition_cost=5.0,
            source_maintenance_cost=5.0,
            prior_failure_rate=1.0,
            source_config={"provider": "fixture"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            allocation = AutonomousFinancialOpportunityPortfolio(tmp, contracts=[good, dead]).allocate_budget()

            shares = {item["contract_id"]: item["research_share"] for item in allocation["allocations"]}
            self.assertGreater(shares["good"], shares["dead"])
            self.assertEqual(shares["dead"], 0.0)

    def test_notifications_are_limited_to_allowed_classes_and_human_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            portfolio = AutonomousFinancialOpportunityPortfolio(tmp, budget=AttentionBudget(approvals_per_week=1, alerts_per_day=0))

            result = portfolio.run_cycle(schedule="weekly", as_of="2026-06-22T12:00:00+00:00")

            allowed = {"approval_required", "prospectively_verified_paper_opportunity", "operational_failure_requires_authority"}
            notifications = result["notifications"]["notifications"]
            self.assertTrue(all(item["kind"] in allowed for item in notifications))
            self.assertLessEqual(sum(1 for item in notifications if item["kind"] == "approval_required"), 1)
            self.assertTrue(result["notifications"]["forbidden_notifications_suppressed"])
            self.assertTrue(result["notifications"]["suppressed"])

    def test_data_mesh_source_acquisition_is_integrated_without_production_activation(self) -> None:
        contract = PortfolioContract(
            contract_id="compute",
            contract_family="compute_cost_avoidance",
            title="Compute cost paper contract",
            lab=None,
            status="experimental",
            expected_information_gain=0.5,
            source_config={"provider": "fixture", "source_id": "billing_export"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = AutonomousFinancialOpportunityPortfolio(tmp, contracts=[contract]).run_cycle(
                schedule="weekly",
                mesh_manifests=[_manifest()],
                fixtures_by_source={"official_json_cost_source": _fixture()},
                as_of="2026-06-22T12:00:00+00:00",
            )

            self.assertEqual(len(result["data_acquisition"]["activated_experimental_sources"]), 1)
            self.assertFalse(result["data_acquisition"]["production_state_mutated"])
            self.assertEqual(result["weekly_report"]["sources_gained"], 1)

    def test_weekly_report_leads_with_value_cost_and_attention_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = AutonomousFinancialOpportunityPortfolio(tmp).run_cycle(schedule="continuous", as_of="2026-06-22T12:00:00+00:00")
            report = result["weekly_report"]

            for field in (
                "resolved_paper_value",
                "conservative_prospective_value",
                "hypothetical_capital_at_risk",
                "drawdown",
                "no_action_rate",
                "research_cost",
                "maintenance_cost",
                "time_since_human_attention_required",
                "contract_allocation",
                "sources_gained",
                "sources_repaired",
                "sources_retired",
                "failures_not_repeated",
            ):
                self.assertIn(field, report)
            self.assertFalse(any(report["production_state"].values()))

    def test_monthly_reallocation_can_pause_low_value_contract_without_real_actions(self) -> None:
        low = PortfolioContract(
            contract_id="low",
            contract_family="compute_cost_avoidance",
            title="Low value",
            lab=None,
            status="active",
            source_acquisition_cost=10.0,
            source_maintenance_cost=10.0,
            prior_failure_rate=1.0,
            source_config={"provider": "fixture"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            portfolio = AutonomousFinancialOpportunityPortfolio(tmp, contracts=[low])
            portfolio.run_cycle(schedule="monthly", as_of="2026-06-22T12:00:00+00:00")
            status = portfolio.status()

            self.assertEqual(status["contracts"][0]["status"], "paused")
            self.assertFalse(any(status["production_state"].values()))

    def test_real_action_configuration_is_rejected(self) -> None:
        with self.assertRaises(OpportunityPortfolioError):
            PortfolioContract(
                contract_id="bad",
                contract_family="weather_event_market",
                title="Bad",
                lab="weather_edge",
                source_config={"connector": "broker.place_order"},
            )

    def test_seek_value_cli_runs_paper_portfolio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "seek-value", "--state-dir", tmp, "--mode", "paper", "--monthly-budget", "40", "--schedule", "continuous"])

            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["schedule"], "continuous")
            self.assertTrue(payload["paper_only"])
            self.assertEqual(payload["production_state"], REAL_ACTION_FLAGS)


def _fixture() -> dict:
    return {
        "records": [
            {"published_at": "2026-06-22T12:00:00+00:00", "available_at": "2026-06-22T12:01:00+00:00", "cost": 12.5},
        ]
    }


def _manifest() -> dict:
    return {
        "source_id": "official_json_cost_source",
        "version": "v1",
        "source_family": "billing_export",
        "display_name": "Official JSON Cost Source",
        "official_publisher": "Official Example Publisher",
        "adapter_type": "json_api",
        "endpoint": "https://official.example.invalid/costs",
        "request_parameters": {"records_path": "records"},
        "pagination": {"mode": "none"},
        "event_timestamp": {"field": "published_at", "semantics": "provider_event_time"},
        "availability_timestamp": {"field": "available_at", "semantics": "provider_publication_time"},
        "timezone": "UTC",
        "units": {"cost": "USD"},
        "geography": {"type": "global", "id": "001"},
        "cadence": {"seconds": 3600},
        "revision_behavior": {"mode": "all_available", "revision_id_field": "revision_id"},
        "missing_value_behavior": {"mode": "drop"},
        "license": {"status": "documented", "summary": "Official public terms permit research use", "url": "https://official.example.invalid/terms"},
        "rate_limits": {"bounded": True, "requests_per_minute": 30},
        "normalized_series": [
            {
                "series_id": "official_json_cost_source.hourly_cost",
                "display_name": "Hourly Cost",
                "observation_kind": "economic_release",
                "value_field": "cost",
                "event_time_field": "published_at",
                "availability_time_field": "available_at",
                "unit": "USD",
                "geography": {"type": "global", "id": "001"},
                "contract_usage": ["compute_cost_avoidance"],
            }
        ],
        "quality_checks": [{"name": "nonempty"}, {"name": "timestamp_parseable"}],
        "documentation_urls": ["https://official.example.invalid/docs"],
        "credential_requirements": [],
    }


if __name__ == "__main__":
    unittest.main()
