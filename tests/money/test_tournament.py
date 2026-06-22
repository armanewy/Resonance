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
from behavior_lab.money.accounting import summarize_money_entries
from behavior_lab.money.tournament import ALLOWED_CLASSIFICATIONS, ALLOWED_TOP_LEVEL_RESULTS, DIMENSION_KEYS, run_financial_tournament


class FinancialTournamentTests(unittest.TestCase):
    def test_tournament_writes_reports_and_refuses_fixture_only_winner(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp, tempfile.TemporaryDirectory() as docs_tmp:
            workspace = Path(workspace_tmp)
            output = Path(output_tmp)
            docs = Path(docs_tmp)

            payload = run_financial_tournament(output_dir=output, docs_dir=docs, workspace=workspace)

            self.assertIn(payload["top_level_result"], ALLOWED_TOP_LEVEL_RESULTS)
            self.assertEqual(payload["top_level_result"], "CONTINUE_MULTIPLE_CANARIES")
            self.assertIsNone(payload["selected_wedge"])
            self.assertTrue(payload["fixture_only"])
            self.assertFalse(payload["real_actions_executed"])
            self.assertFalse(any(payload["production_state"].values()))
            self.assertTrue((output / "FINANCIAL_TOURNAMENT.json").exists())
            self.assertTrue((output / "FINANCIAL_TOURNAMENT.html").exists())
            self.assertTrue((docs / "FINANCIAL_WEDGE_DECISION.md").exists())

            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn(str(workspace), serialized)
            self.assertNotIn(str(output), serialized)
            self.assertNotIn(str(docs), serialized)

    def test_each_contract_has_required_dimensions_and_reproducible_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp, tempfile.TemporaryDirectory() as docs_tmp:
            payload = run_financial_tournament(output_dir=output_tmp, docs_dir=docs_tmp, workspace=workspace_tmp)

            self.assertEqual(set(payload["contracts"]), {"offerlab_seller_pilot", "weather_edge", "etf_risk"})
            for contract_id, assessment in payload["contracts"].items():
                self.assertIn(assessment["classification"], ALLOWED_CLASSIFICATIONS)
                self.assertTrue(set(DIMENSION_KEYS) <= set(assessment["dimensions"]))
                records = assessment["raw_economic_records"]
                reproduced = summarize_money_entries(records)
                self.assertEqual(reproduced, assessment["economic_reproduction"]["summary_from_raw_ledger"], contract_id)
                self.assertFalse(assessment["dimensions"]["paper_or_shadow_value"]["presented_as_realized_pnl"])
                self.assertFalse(assessment["canary"]["real_money_allowed"])

            self.assertEqual(payload["contracts"]["weather_edge"]["classification"], "DATA_STARVED")
            self.assertGreater(payload["contracts"]["offerlab_seller_pilot"]["dimensions"]["target_data_quality"]["unknown_cost_basis_count"], 0)

    def test_cli_runs_financial_tournament(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp, tempfile.TemporaryDirectory() as docs_tmp:
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(
                    [
                        "money",
                        "tournament",
                        "run",
                        "--workspace",
                        workspace_tmp,
                        "--output-dir",
                        output_tmp,
                        "--docs-dir",
                        docs_tmp,
                    ]
                )

            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["top_level_result"], "CONTINUE_MULTIPLE_CANARIES")
            self.assertTrue((Path(output_tmp) / "FINANCIAL_TOURNAMENT.json").exists())
            self.assertTrue((Path(output_tmp) / "FINANCIAL_TOURNAMENT.html").exists())
            self.assertTrue((Path(docs_tmp) / "FINANCIAL_WEDGE_DECISION.md").exists())


if __name__ == "__main__":
    unittest.main()
