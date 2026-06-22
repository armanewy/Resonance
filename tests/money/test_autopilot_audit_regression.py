from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.money.autopilot import MoneyAutopilot, load_portfolio


def _write_portfolio(
    tmp: Path,
    *,
    budgets: dict[str, float] | None = None,
    contracts: list[dict[str, object]] | None = None,
) -> Path:
    path = tmp / "money-lab.yaml"
    path.write_text(
        json.dumps(
            {
                "portfolio_id": "audit-money-lab",
                "state_dir": str(tmp / "state"),
                "budgets": {
                    "llm_monthly_cost_usd": 10.0,
                    "web_searches": 10.0,
                    "connector_attempts": 10.0,
                    "candidate_evaluations": 20.0,
                    "max_concurrency": 1.0,
                    "alerts_per_day": 10.0,
                    "approvals_per_week": 10.0,
                    **(budgets or {}),
                },
                "contracts": contracts
                or [
                    {
                        "contract_id": "weather",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {"provider": "fixture"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _decision(**overrides: object) -> dict[str, object]:
    decision: dict[str, object] = {
        "contract_id": "weather",
        "lab": "weather_edge",
        "selected_action": "buy_yes",
        "capital_required": 1.0,
        "maximum_possible_loss": 1.0,
        "conservative_expected_net_value": 2.0,
        "decision_id": "audit-decision",
        "strategy_id": "audit-strategy",
        "source_id": "audit-source",
        "seller_shadow_value": 0.0,
        "paper_only": True,
        "unknown_cost_basis_count": 0,
        "material_costs_known": True,
        "forecast_current": True,
        "liquidity_capacity_ok": True,
        "deadline_open": True,
        "action_mode": "reactive",
    }
    decision.update(overrides)
    return decision


class MoneyAutopilotAuditRegressionTests(unittest.TestCase):
    def test_real_yaml_nested_contract_configuration_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            path = tmp / "money-lab.yaml"
            path.write_text(
                f"""
portfolio_id: audit-yaml
state_dir: {tmp / "state"}
budgets:
  llm_monthly_cost_usd: 1.0
  web_searches: 2
  connector_attempts: 2
  candidate_evaluations: 3
  max_concurrency: 1
  alerts_per_day: 1
  approvals_per_week: 1
contracts:
  - contract_id: weather
    lab: weather_edge
    provider: fixture
    paper_capital_limit: 20.0
    alert_threshold: 0.0
    target_inputs:
      city: New York
    source_config:
      provider: fixture
      source_id: fixture-weather
    research_budget:
      web_searches: 1
    schedule:
      paper_decision: daily
    evidence_thresholds:
      min_history_days: 4
    prospective_requirements:
      min_unseen_episodes: 1
""".strip(),
                encoding="utf-8",
            )

            portfolio = load_portfolio(path)

            self.assertEqual(portfolio.budgets["llm_monthly_cost_usd"], 1.0)
            self.assertEqual(portfolio.contracts[0].source_config["source_id"], "fixture-weather")
            self.assertEqual(portfolio.contracts[0].target_inputs["city"], "New York")
            self.assertEqual(portfolio.contracts[0].prospective_requirements["min_unseen_episodes"], 1)

    def test_no_paper_opportunity_without_blind_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = _write_portfolio(
                Path(tmp_name),
                budgets={
                    "candidate_evaluations": 0.0,
                    "llm_monthly_cost_usd": 0.0,
                    "web_searches": 0.0,
                    "connector_attempts": 0.0,
                },
            )
            autopilot = MoneyAutopilot.from_path(path)
            result = autopilot.run_once()

            self.assertTrue(any(item.get("task_type") == "blind_evaluation" for item in result["skipped_tasks"]))
            self.assertEqual([], autopilot._events("autopilot_blind_evaluation_consumed"))
            self.assertEqual([], autopilot._events("autopilot_paper_opportunity"))

    def test_missing_credential_blocks_paper_opportunity_until_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = _write_portfolio(
                Path(tmp_name),
                contracts=[
                    {
                        "contract_id": "weather",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {
                            "provider": "fixture",
                            "requires_credential": True,
                            "credential_available": False,
                        },
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            self.assertEqual({"missing_credential"}, {item["reason"] for item in autopilot.approvals()["approvals"]})
            self.assertEqual([], autopilot._events("autopilot_paper_opportunity"))

    def test_approvals_per_week_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = _write_portfolio(
                Path(tmp_name),
                budgets={"approvals_per_week": 1.0},
                contracts=[
                    {
                        "contract_id": "needs-approvals",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {
                            "provider": "fixture",
                            "requires_credential": True,
                            "credential_available": False,
                            "license_status": "unclear",
                            "paid_source": True,
                            "private_data_ambiguity": True,
                            "production_source_promotion": True,
                        },
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            self.assertLessEqual(len(autopilot.approvals()["approvals"]), 1)

    def test_connector_names_for_live_exchange_or_order_actions_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = _write_portfolio(
                Path(tmp_name),
                contracts=[
                    {
                        "contract_id": "dangerous",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {"provider": "fixture", "connector": "exchange.place_live_order"},
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            failures = autopilot._events("autopilot_connector_attempt_failed")
            self.assertTrue(failures)
            self.assertEqual("malicious_connector_blocked", failures[0]["payload"]["reason"])

    def test_credentials_embedded_in_source_configuration_do_not_leak_to_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            path = _write_portfolio(
                tmp,
                contracts=[
                    {
                        "contract_id": "secret",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {
                            "provider": "fixture",
                            "connector": "fixture?token=AUDIT_SECRET_TOKEN",
                            "api_key": "AUDIT_SECRET_TOKEN",
                        },
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            store_text = (tmp / "state" / "audit-money-lab" / "autopilot.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("AUDIT_SECRET_TOKEN", store_text)

    def test_auth_query_credentials_are_not_persisted_to_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            secret = "SUPER_SECRET_CREDENTIAL_123"
            path = _write_portfolio(
                tmp,
                contracts=[
                    {
                        "contract_id": "secret-auth",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {
                            "provider": "fixture",
                            "connector": f"fixture_feed?auth={secret}",
                            "source_id": f"weather_feed?auth={secret}",
                        },
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            store_text = (tmp / "state" / "audit-money-lab" / "autopilot.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(secret, store_text)

    def test_secret_bearing_decision_source_id_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            secret = "SOURCE_AUTH_SECRET_456"
            autopilot = MoneyAutopilot.from_path(_write_portfolio(tmp))
            with patch.object(
                MoneyAutopilot,
                "_run_weather_decision",
                return_value=_decision(source_id=f"vendor_feed?auth={secret}"),
            ):
                autopilot.run_once()

            store_text = (tmp / "state" / "audit-money-lab" / "autopilot.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(secret, store_text)

    def test_plain_secret_bearing_decision_source_id_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            secret = "PLAIN_SOURCE_SECRET_456"
            autopilot = MoneyAutopilot.from_path(_write_portfolio(tmp))
            with patch.object(
                MoneyAutopilot,
                "_run_weather_decision",
                return_value=_decision(source_id=secret),
            ):
                autopilot.run_once()

            store_text = (tmp / "state" / "audit-money-lab" / "autopilot.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(secret, store_text)

    def test_bearer_credentials_in_failure_messages_are_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            secret = "BEARER_SECRET_789"
            autopilot = MoneyAutopilot.from_path(_write_portfolio(tmp))
            with patch.object(
                MoneyAutopilot,
                "_run_weather_decision",
                side_effect=RuntimeError(f"upstream rejected Authorization: Bearer {secret}"),
            ):
                autopilot.run_once()

            store_text = (tmp / "state" / "audit-money-lab" / "autopilot.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(secret, store_text)

    def test_paper_opportunity_requires_explicit_safety_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            autopilot = MoneyAutopilot.from_path(_write_portfolio(Path(tmp_name)))
            with patch.object(
                MoneyAutopilot,
                "_run_weather_decision",
                return_value={
                    "contract_id": "weather",
                    "lab": "weather_edge",
                    "selected_action": "buy_yes",
                    "capital_required": 1.0,
                    "maximum_possible_loss": 1.0,
                    "conservative_expected_net_value": 2.0,
                    "decision_id": "missing-gate-attestations",
                    "strategy_id": "audit-strategy",
                    "source_id": "audit-source",
                    "seller_shadow_value": 0.0,
                    "paper_only": True,
                },
            ):
                autopilot.run_once()

            self.assertEqual([], autopilot._events("autopilot_paper_opportunity"))

    def test_weekly_report_realized_value_excludes_unresolved_expected_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            autopilot = MoneyAutopilot.from_path(_write_portfolio(Path(tmp_name)))
            result = autopilot.run_once()

            self.assertEqual(0, result["weekly_report"]["realized_paper_value"])
            self.assertGreater(result["weekly_report"]["prospective_paper_pnl"], 0)


if __name__ == "__main__":
    unittest.main()
