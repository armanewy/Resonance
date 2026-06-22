from __future__ import annotations

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

from behavior_lab.money_agents import (
    CONNECTOR_DIAGNOSTICIAN,
    HYPOTHESIS_SCIENTIST,
    ROLE_CONNECTOR_DIAGNOSTICIAN,
    ROLE_HYPOTHESIS_SCIENTIST,
    ROLE_SKEPTIC,
    ROLE_SOURCE_SCOUT,
    ROLE_WEEKLY_ALLOCATOR,
    SKEPTIC,
    SOURCE_SCOUT,
    WEEKLY_ALLOCATOR,
    FinancialResearchAgentRuntime,
    MoneyAgentBudgetError,
    MoneyAgentContext,
    MoneyAgentPermissionError,
    ProviderResponse,
    StaticMoneyAgentProvider,
    UsageRecord,
)


def _context() -> MoneyAgentContext:
    return MoneyAgentContext(
        campaign_id="finance-wave2e-test",
        prompt_version="finance_wave2e_v1",
        permitted_sources=("sec_companyfacts", "fred_releases"),
        permitted_connectors=("sec_companyfacts_connector", "fred_connector"),
        explicit_budgets={
            "max_response_cost_usd": 0.25,
            "max_response_tokens": 2000,
            "max_tool_calls": 4,
            "weekly_hours": 6.0,
            "cost_usd": 0.50,
            "tool_calls": 4.0,
        },
        prior_proposal_ids=("prior-hypothesis",),
    )


def _runtime(response: ProviderResponse, state_path: Path) -> FinancialResearchAgentRuntime:
    return FinancialResearchAgentRuntime(StaticMoneyAgentProvider(response), state_path=state_path)


def _response(content: dict[str, object], *, tool_name: str = "official.sec.metadata") -> ProviderResponse:
    return ProviderResponse(
        provider="mock-provider",
        model="mock-finance-model",
        prompt_version="finance_wave2e_v1",
        content=content,
        tool_calls=[{"tool_name": tool_name, "mode": "read_only", "purpose": "fixture"}],
        citations=[{"source_id": "sec_companyfacts", "url": "https://www.sec.gov/edgar/sec-api-documentation"}],
        usage=UsageRecord(input_tokens=100, output_tokens=80, total_tokens=180, cost_usd=0.01),
    )


class FinancialSourceScoutTests(unittest.TestCase):
    def test_source_scout_records_provider_usage_citations_and_lineage(self) -> None:
        content = {
            "source_candidates": [
                {
                    "source_id": "sec_companyfacts",
                    "official_provider": True,
                    "license_status": "documented",
                    "license_citation": "SEC fair access and API documentation.",
                    "rate_limit_summary": "Documented fair-access limits apply.",
                    "timestamp_policy": "provider filing acceptance timestamp is documented",
                    "proposed_metrics": ["filing_lag_days", "revision_count"],
                    "proposed_connectors": ["sec_companyfacts_connector"],
                    "activation_status": "proposed",
                    "availability_as_predictive_evidence": False,
                }
            ],
            "rejections": [{"source_id": "unknown_blog", "reason": "not an official provider"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "money-agents.jsonl"
            payload = _runtime(_response(content), state).run(SOURCE_SCOUT, _context())
            self.assertEqual(payload["provider"], "mock-provider")
            self.assertEqual(payload["model"], "mock-finance-model")
            self.assertEqual(payload["prompt_version"], "finance_wave2e_v1")
            self.assertEqual(payload["usage"]["total_tokens"], 180)
            self.assertEqual(payload["lineage"]["proposal_ids"], ["sec_companyfacts"])
            self.assertEqual(payload["lineage"]["rejection_ids"], ["unknown_blog"])
            events = [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event_type"], "money_agent_completed")

    def test_source_scout_rejects_unclear_license_activation_and_predictive_availability(self) -> None:
        bad = {
            "source_candidates": [
                {
                    "source_id": "fred_releases",
                    "official_provider": True,
                    "license_status": "unclear",
                    "license_citation": "",
                    "rate_limit_summary": "unknown",
                    "timestamp_policy": "inferred",
                    "proposed_metrics": ["release_delay"],
                    "proposed_connectors": ["fred_connector"],
                    "activation_status": "active",
                    "availability_as_predictive_evidence": True,
                }
            ],
            "rejections": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "money-agents.jsonl"
            with self.assertRaises(MoneyAgentPermissionError):
                _runtime(_response(bad), state).run(SOURCE_SCOUT, _context())
            event = json.loads(state.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["event_type"], "money_agent_rejected")
            self.assertEqual(event["payload"]["role_id"], ROLE_SOURCE_SCOUT)


class FinancialHypothesisScientistTests(unittest.TestCase):
    def test_hypothesis_scientist_accepts_structured_executable_hypotheses_only(self) -> None:
        content = {
            "hypotheses": [
                {
                    "hypothesis_id": "lagged-filings-001",
                    "hypothesis_type": "lagged_features",
                    "target": "next_week_index_volatility_bucket",
                    "features": ["filing_lag_days_t_minus_1", "revision_count_t_minus_4"],
                    "executable_spec": {
                        "family": "feature_transform_plan",
                        "operators": ["lag", "rolling_count"],
                    },
                    "falsification_tests": ["fails if chronological development loss does not beat no-change baseline"],
                    "lineage": {"parent_ids": ["prior-hypothesis"]},
                }
            ],
            "rejections": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = _runtime(_response(content), Path(tmp) / "state.jsonl").run(
                HYPOTHESIS_SCIENTIST,
                _context(),
                parent_ids=["prior-hypothesis"],
            )
            self.assertEqual(payload["role_id"], ROLE_HYPOTHESIS_SCIENTIST)
            self.assertEqual(payload["lineage"]["parent_ids"], ["prior-hypothesis"])
            self.assertEqual(payload["lineage"]["proposal_ids"], ["lagged-filings-001"])

    def test_hypothesis_scientist_rejects_strategy_source_code(self) -> None:
        content = {
            "hypotheses": [
                {
                    "hypothesis_id": "bad-code",
                    "hypothesis_type": "liquidity_effect",
                    "executable_spec": {"family": "feature_plan"},
                    "falsification_tests": ["fails on development"],
                    "strategy_code": "def trade(row): return 'buy'",
                }
            ],
            "rejections": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAgentPermissionError):
                _runtime(_response(content), Path(tmp) / "state.jsonl").run(HYPOTHESIS_SCIENTIST, _context())


class FinancialSkepticTests(unittest.TestCase):
    def test_skeptic_must_cover_required_financial_validity_risks(self) -> None:
        checks = sorted(
            {
                "timing_leakage",
                "survivorship_bias",
                "corporate_action_leakage",
                "selection_bias",
                "target_leakage",
                "stale_pricing",
                "non_executable_prices",
                "omitted_costs",
                "correlated_outcomes",
                "regime_concentration",
                "prior_failed_equivalent_hypotheses",
            }
        )
        content = {
            "checked_risk_types": checks,
            "audit_findings": [
                {
                    "finding_id": "risk-001",
                    "risk_type": "timing_leakage",
                    "severity": "blocker",
                    "affected_proposal_ids": ["lagged-filings-001"],
                    "evidence": "candidate feature used filing acceptance times after the decision cutoff",
                    "recommendation": "reject until data cutoff is rebuilt",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = _runtime(_response(content, tool_name="offline.audit_replay"), Path(tmp) / "state.jsonl").run(SKEPTIC, _context())
            self.assertEqual(payload["role_id"], ROLE_SKEPTIC)
            self.assertEqual(payload["lineage"]["proposal_ids"], ["risk-001"])

    def test_skeptic_cannot_declare_strategy_valid(self) -> None:
        content = {
            "checked_risk_types": list(SKEPTIC.required_checks),
            "audit_findings": [{"finding_id": "validity", "verdict": "valid", "risk_type": "none"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAgentPermissionError):
                _runtime(_response(content), Path(tmp) / "state.jsonl").run(SKEPTIC, _context())


class ConnectorAndAllocatorTests(unittest.TestCase):
    def test_connector_diagnostician_is_read_only_and_connector_scoped(self) -> None:
        content = {
            "diagnostics": [
                {
                    "diagnostic_id": "diag-001",
                    "connector_id": "sec_companyfacts_connector",
                    "check_mode": "offline_replay",
                    "symptoms": ["schema field missing in cached response"],
                    "read_only_checks": ["replay fixture and compare normalized columns"],
                    "repair_plan": ["add parser guard in connector-specific code"],
                    "activation_change": False,
                    "mutation_required": False,
                }
            ],
            "maintenance_tickets": [{"ticket_id": "ticket-001", "diagnostic_id": "diag-001"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = _runtime(_response(content, tool_name="offline.fixture_replay"), Path(tmp) / "state.jsonl").run(
                CONNECTOR_DIAGNOSTICIAN,
                _context(),
            )
            self.assertEqual(payload["role_id"], ROLE_CONNECTOR_DIAGNOSTICIAN)
            self.assertEqual(payload["lineage"]["proposal_ids"], ["diag-001"])

    def test_weekly_allocator_cannot_exceed_explicit_budget(self) -> None:
        content = {
            "allocations": [
                {
                    "work_item_id": "week-001",
                    "role_id": ROLE_SOURCE_SCOUT,
                    "hours": 4.0,
                    "cost_usd": 0.20,
                    "tool_calls": 2,
                    "depends_on": [],
                },
                {
                    "work_item_id": "week-002",
                    "role_id": ROLE_SKEPTIC,
                    "hours": 3.0,
                    "cost_usd": 0.20,
                    "tool_calls": 1,
                    "depends_on": ["week-001"],
                },
            ],
            "deferred": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAgentBudgetError):
                _runtime(_response(content, tool_name="metadata.calendar"), Path(tmp) / "state.jsonl").run(WEEKLY_ALLOCATOR, _context())

    def test_weekly_allocator_records_bounded_allocations(self) -> None:
        content = {
            "allocations": [
                {
                    "work_item_id": "week-001",
                    "role_id": ROLE_SOURCE_SCOUT,
                    "hours": 2.0,
                    "cost_usd": 0.10,
                    "tool_calls": 1,
                    "depends_on": [],
                },
                {
                    "work_item_id": "week-002",
                    "role_id": ROLE_SKEPTIC,
                    "hours": 2.0,
                    "cost_usd": 0.10,
                    "tool_calls": 1,
                    "depends_on": ["week-001"],
                },
            ],
            "deferred": [{"work_item_id": "week-003", "reason": "insufficient explicit hours"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = _runtime(_response(content, tool_name="metadata.calendar"), Path(tmp) / "state.jsonl").run(WEEKLY_ALLOCATOR, _context())
            self.assertEqual(payload["role_id"], ROLE_WEEKLY_ALLOCATOR)
            self.assertEqual(payload["lineage"]["proposal_ids"], ["week-001", "week-002"])
            self.assertEqual(payload["lineage"]["rejection_ids"], ["week-003"])


class RuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_rejects_mutating_tool_calls_and_excess_costs(self) -> None:
        content = {
            "allocations": [
                {
                    "work_item_id": "ok",
                    "role_id": ROLE_HYPOTHESIS_SCIENTIST,
                    "hours": 1.0,
                    "cost_usd": 0.10,
                    "tool_calls": 1,
                }
            ],
            "deferred": [],
        }
        bad_tool = _response(content, tool_name="broker.place_order")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAgentPermissionError):
                _runtime(bad_tool, Path(tmp) / "state.jsonl").run(WEEKLY_ALLOCATOR, _context())
        expensive = ProviderResponse(
            provider="mock-provider",
            model="mock-finance-model",
            prompt_version="finance_wave2e_v1",
            content=content,
            tool_calls=[{"tool_name": "metadata.calendar", "mode": "read_only"}],
            citations=[],
            usage=UsageRecord(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=1.00),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAgentBudgetError):
                _runtime(expensive, Path(tmp) / "state.jsonl").run(WEEKLY_ALLOCATOR, _context())


if __name__ == "__main__":
    unittest.main()
