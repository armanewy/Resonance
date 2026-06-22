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
from behavior_lab.money.autopilot import MoneyAutopilot, MoneyAutopilotError, PAPER_NOTICE, load_portfolio


def _portfolio(tmp: Path, *, budgets: dict[str, float] | None = None, extra_contracts: list[dict[str, object]] | None = None) -> Path:
    contracts: list[dict[str, object]] = [
        {
            "contract_id": "seller",
            "lab": "offerlab_seller_pilot",
            "paper_capital_limit": 500.0,
            "alert_threshold": 1.0,
            "source_config": {"provider": "fixture"},
        },
        {
            "contract_id": "weather",
            "lab": "weather_edge",
            "paper_capital_limit": 20.0,
            "alert_threshold": 0.01,
            "source_config": {"provider": "fixture"},
        },
        {
            "contract_id": "etf",
            "lab": "etf_risk",
            "paper_capital_limit": 100000.0,
            "alert_threshold": 0.0,
            "source_config": {"provider": "fixture", "session_count": 90, "min_history_days": 35},
        },
    ]
    contracts.extend(extra_contracts or [])
    path = tmp / "money-lab.yaml"
    path.write_text(
        json.dumps(
            {
                "portfolio_id": "test-money-lab",
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
                "contracts": contracts,
            }
        ),
        encoding="utf-8",
    )
    return path


class MoneyAutopilotTests(unittest.TestCase):
    def test_three_contracts_run_concurrently_as_paper_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            autopilot = MoneyAutopilot(load_portfolio(_portfolio(Path(tmp))))
            result = autopilot.run_once()
            status = autopilot.status()

            self.assertTrue(result["ledger_valid"])
            self.assertTrue({"offerlab_seller_pilot", "weather_edge", "authorized_fixture"} <= set(result["weekly_report"]["prospective_value_by_source"]))
            self.assertEqual({contract["contract_id"] for contract in status["contracts"]}, {"seller", "weather", "etf"})
            self.assertGreaterEqual(result["weekly_report"]["decision_count"], 3)
            self.assertFalse(any(result["production_state"].values()))
            self.assertTrue(all("PAPER" in item["payload"]["notice"] for item in autopilot._events("autopilot_paper_opportunity")))

    def test_one_lab_failure_isolated_from_other_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(
                Path(tmp),
                extra_contracts=[
                    {
                        "contract_id": "broken",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {"simulate_failure": True},
                    }
                ],
            )
            result = MoneyAutopilot.from_path(path).run_once()

            self.assertEqual(len(result["failed_tasks"]), 1)
            completed_contracts = {item["contract_id"] for item in result["completed_tasks"] if item["contract_id"]}
            self.assertTrue({"seller", "weather", "etf"} <= completed_contracts)

    def test_restart_skips_completed_tasks_blind_and_source_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(Path(tmp))
            first = MoneyAutopilot.from_path(path).run_once()
            second = MoneyAutopilot.from_path(path).run_once()
            autopilot = MoneyAutopilot.from_path(path)

            self.assertGreater(len(first["completed_tasks"]), 0)
            self.assertEqual(second["completed_tasks"], [])
            self.assertTrue(any(item["reason"] == "already_completed" for item in second["skipped_tasks"]))
            self.assertEqual(len(autopilot._events("autopilot_blind_evaluation_consumed")), 3)
            self.assertTrue(all(event["payload"]["repeat_allowed"] is False for event in autopilot._events("autopilot_blind_evaluation_consumed")))
            self.assertTrue(all(event["payload"]["refit_performed"] is False for event in autopilot._events("autopilot_prospective_incubation")))

    def test_exhausted_research_budget_still_allows_collection_and_paper_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = MoneyAutopilot.from_path(
                _portfolio(
                    Path(tmp),
                    budgets={
                        "llm_monthly_cost_usd": 0.0,
                        "web_searches": 0.0,
                        "connector_attempts": 0.0,
                        "candidate_evaluations": 0.0,
                    },
                )
            ).run_once()

            skipped_research = [item for item in result["skipped_tasks"] if item.get("reason") == "research_budget_exhausted"]
            completed_types = {item["task_type"] for item in result["completed_tasks"]}
            self.assertTrue(skipped_research)
            self.assertIn("source_update", completed_types)
            self.assertIn("paper_decision", completed_types)

    def test_no_signal_and_no_action_winner_are_not_notified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(
                Path(tmp),
                extra_contracts=[
                    {
                        "contract_id": "quiet",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {"force_no_signal": True},
                    },
                    {
                        "contract_id": "noaction",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "alert_threshold": 0.0,
                        "source_config": {"force_no_action": True},
                    },
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()

            opportunities = [event["payload"] for event in autopilot._events("autopilot_paper_opportunity")]
            self.assertNotIn("quiet", {item["contract_id"] for item in opportunities})
            self.assertNotIn("noaction", {item["contract_id"] for item in opportunities})
            self.assertTrue(all(item["notice"] == PAPER_NOTICE for item in opportunities))

    def test_duplicate_alerts_are_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(Path(tmp))
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()
            first_count = len(autopilot._events("autopilot_paper_opportunity"))
            autopilot.run_once()

            self.assertEqual(len(autopilot._events("autopilot_paper_opportunity")), first_count)

    def test_malicious_connector_and_real_action_config_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MoneyAutopilotError):
                load_portfolio(
                    _portfolio(
                        Path(tmp),
                        extra_contracts=[
                            {
                                "contract_id": "bad-real",
                                "lab": "weather_edge",
                                "paper_capital_limit": 1.0,
                                "source_config": {"real_action": True},
                            }
                        ],
                    )
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(
                Path(tmp),
                extra_contracts=[
                    {
                        "contract_id": "bad-connector",
                        "lab": "weather_edge",
                        "paper_capital_limit": 1.0,
                        "source_config": {"connector": "broker.place_order"},
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.run_once()
            failures = autopilot._events("autopilot_connector_attempt_failed")
            self.assertTrue(any(item["payload"]["reason"] == "malicious_connector_blocked" for item in failures))
            retry_result = autopilot.run_once()
            self.assertTrue(any(item.get("reason") == "failed_connector_without_changed_evidence" for item in retry_result["skipped_tasks"]))

    def test_approval_inbox_only_contains_allowed_reasons_and_pause_resume_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(
                Path(tmp),
                extra_contracts=[
                    {
                        "contract_id": "needs-approval",
                        "lab": "weather_edge",
                        "paper_capital_limit": 20.0,
                        "source_config": {
                            "requires_credential": True,
                            "license_status": "unclear",
                            "paid_source": True,
                            "private_data_ambiguity": True,
                        },
                    }
                ],
            )
            autopilot = MoneyAutopilot.from_path(path)
            autopilot.pause("seller")
            self.assertTrue(autopilot.status()["contracts"][0]["paused"])
            autopilot.resume("seller")
            autopilot.run_once()
            reasons = {item["reason"] for item in autopilot.approvals()["approvals"]}
            self.assertEqual(reasons, {"missing_credential", "unclear_license", "paid_source", "private_data_ambiguity"})

    def test_cli_default_run_status_and_weekly_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _portfolio(Path(tmp))
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "autopilot", "--portfolio", str(path)])
            self.assertTrue(json.loads(stream.getvalue())["ledger_valid"])

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "autopilot", "status", "--portfolio", str(path)])
            self.assertEqual(json.loads(stream.getvalue())["portfolio_id"], "test-money-lab")

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["money", "autopilot", "weekly-report", "--portfolio", str(path)])
            self.assertIn("PAPER", json.loads(stream.getvalue())["notice"])


if __name__ == "__main__":
    unittest.main()
