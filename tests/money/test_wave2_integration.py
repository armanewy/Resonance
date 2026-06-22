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
from behavior_lab.money.integration import run_wave2_integration_proof


class FinanceWave2IntegrationTests(unittest.TestCase):
    def test_fixture_proof_connects_labs_without_real_actions_or_path_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp:
            workspace = Path(workspace_tmp)
            output = Path(output_tmp)

            proof = run_wave2_integration_proof(output_dir=output, workspace=workspace)

            self.assertTrue(proof["all_required_checks_passed"])
            self.assertTrue(all(proof["required_integration_proof"].values()))
            self.assertTrue(proof["components"]["offerlab_money"]["unknown_material_costs_prevent_eligibility"])
            self.assertIn(proof["components"]["weather_edge"]["selected_action"], {"buy_yes", "no_trade"})
            self.assertIn(proof["components"]["etf_risk"]["selected_action"], {"cash", "low_exposure", "normal_exposure"})
            self.assertFalse(proof["components"]["money_agents"]["candidate_queue"]["determines_verdict"])
            self.assertTrue(proof["components"]["finance_data"]["cutoff_audit_passed"])
            self.assertFalse(any(proof["production_state"].values()))

            json_path = output / "wave_2_integration.json"
            html_path = output / "wave_2_integration.html"
            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())
            serialized = json_path.read_text(encoding="utf-8")
            self.assertNotIn(str(workspace), serialized)
            self.assertEqual(json.loads(serialized)["proof_hash"], proof["proof_hash"])

    def test_cli_writes_wave2_integration_reports(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp, tempfile.TemporaryDirectory() as output_tmp:
            stream = io.StringIO()
            with redirect_stdout(stream):
                main(
                    [
                        "money",
                        "wave2-integration",
                        "report",
                        "--workspace",
                        workspace_tmp,
                        "--output-dir",
                        output_tmp,
                    ]
                )

            payload = json.loads(stream.getvalue())
            self.assertTrue(payload["all_required_checks_passed"])
            self.assertTrue((Path(output_tmp) / "wave_2_integration.json").exists())
            self.assertTrue((Path(output_tmp) / "wave_2_integration.html").exists())


if __name__ == "__main__":
    unittest.main()
