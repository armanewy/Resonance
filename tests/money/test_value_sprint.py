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
from behavior_lab.money.value_sprint import PROHIBITIONS, SPRINT_DECISIONS, ValueSprintConfig, run_autonomous_value_sprint


class AutonomousValueSprintTests(unittest.TestCase):
    def test_run_writes_required_artifacts_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = run_autonomous_value_sprint(ValueSprintConfig(state_dir=root / "state", output_dir=root / "reports", days=30))

            self.assertTrue((root / "reports" / "AUTONOMOUS_VALUE_SPRINT.json").exists())
            self.assertTrue((root / "reports" / "AUTONOMOUS_VALUE_SPRINT.html").exists())
            self.assertTrue((root / "reports" / "VALUE_SYSTEM_DECISION.md").exists())
            self.assertIn(result["top_level_decision"], SPRINT_DECISIONS)
            self.assertEqual(len(result["daily_runs"]), 30)
            self.assertEqual(result["prohibitions"], PROHIBITIONS)
            self.assertFalse(any(result["production_state"].values()))

    def test_success_criteria_and_required_evidence_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_autonomous_value_sprint(ValueSprintConfig(state_dir=Path(tmp) / "state", output_dir=Path(tmp) / "reports", days=30))

            evidence = result["required_evidence"]
            criteria = result["success_criteria"]
            for field in (
                "user_attention_minutes",
                "approvals_requested",
                "active_contracts",
                "usable_sources",
                "automatically_added_sources",
                "automatically_repaired_sources",
                "repeated_failures_avoided_through_memory",
                "candidate_counts",
                "blind_survivors",
                "prospective_survivors",
                "paper_opportunities",
                "no_action_decisions",
                "resolved_paper_value",
                "research_api_cost",
                "source_maintenance_cost",
            ):
                self.assertIn(field, evidence)
            self.assertTrue(criteria["runs_for_30_days_without_manual_data_wrangling"])
            self.assertTrue(criteria["at_least_three_active_public_only_contracts"])
            self.assertTrue(criteria["at_least_one_source_autonomously_added"])
            self.assertTrue(criteria["no_repeated_blind_evaluation"])
            self.assertTrue(criteria["no_production_mutation"])
            self.assertGreaterEqual(evidence["automatically_repaired_sources"], 1)
            self.assertEqual(criteria["at_least_one_source_failure_repaired_or_substituted"], evidence["automatically_repaired_sources"] >= 1)
            self.assertEqual(criteria["candidate_reaches_prospective_or_defensible_no_edge"], evidence["prospective_survivors"] > 0 or evidence["no_action_decisions"] > 0)

    def test_short_run_does_not_claim_30_day_success_or_source_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_autonomous_value_sprint(ValueSprintConfig(state_dir=Path(tmp) / "state", output_dir=Path(tmp) / "reports", days=1))

            evidence = result["required_evidence"]
            criteria = result["success_criteria"]
            self.assertFalse(criteria["runs_for_30_days_without_manual_data_wrangling"])
            self.assertEqual(evidence["automatically_repaired_sources"], 0)
            self.assertFalse(criteria["at_least_one_source_failure_repaired_or_substituted"])
            self.assertEqual(result["top_level_decision"], "RESEARCH_ENGINE_IS_UNTRUSTWORTHY")

    def test_serializable_payload_does_not_leak_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = run_autonomous_value_sprint(ValueSprintConfig(state_dir=root / "state", output_dir=root / "reports", days=30))
            serialized = json.dumps(result, sort_keys=True)

            self.assertNotIn(str(root), serialized)
            self.assertEqual(result["artifacts"]["json"], "AUTONOMOUS_VALUE_SPRINT.json")
            self.assertEqual(result["artifacts"]["html"], "AUTONOMOUS_VALUE_SPRINT.html")
            self.assertEqual(result["artifacts"]["decision"], "VALUE_SYSTEM_DECISION.md")

    def test_cli_value_sprint_run_outputs_json_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = io.StringIO()

            with redirect_stdout(stream):
                main(
                    [
                        "money",
                        "value-sprint",
                        "run",
                        "--state-dir",
                        str(root / "state"),
                        "--output-dir",
                        str(root / "reports"),
                        "--days",
                        "30",
                    ]
                )

            payload = json.loads(stream.getvalue())
            self.assertIn(payload["top_level_decision"], SPRINT_DECISIONS)
            self.assertTrue((root / "reports" / "AUTONOMOUS_VALUE_SPRINT.json").exists())
            self.assertFalse(any(payload["production_state"].values()))


if __name__ == "__main__":
    unittest.main()
