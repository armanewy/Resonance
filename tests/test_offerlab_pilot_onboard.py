from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.cli import main
from behavior_lab.money.integration import _write_offerlab_fixture
from behavior_lab.offerlab_pilot import onboard_input


class OfferLabPilotOnboardingTests(unittest.TestCase):
    def test_onboard_proposes_mappings_and_blocks_canary_until_readiness_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)

            report = onboard_input(source)

            self.assertTrue(report["local_only"])
            self.assertFalse(report["uploads_seller_data"])
            self.assertFalse(report["executes_seller_actions"])
            self.assertFalse(report["llm_boundary"]["llm_used"])
            self.assertTrue(report["llm_boundary"]["deterministic_validation_controls_readiness"])
            self.assertIn("offer_amount", set(report["datasets"]["offers"]["proposed_column_mapping"].values()))
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])
            self.assertFalse(report["data_readiness"]["readiness_gate"]["passed"])
            self.assertTrue(report["data_readiness"]["never_silently_impute_material_costs"])

    def test_onboard_flags_material_mapping_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)
            _append_column(source / "offers.csv", "Offer Price", "91.00")

            report = onboard_input(source)

            self.assertTrue(report["mapping_approval"]["human_approval_required"])
            self.assertTrue(
                any(item["reason"] == "ambiguous_column_mapping" for item in report["mapping_approval"]["material_ambiguities"])
            )
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])

    def test_onboard_flags_missing_cost_basis_as_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            source.mkdir()
            _write_offerlab_fixture(source)
            (source / "cost_basis.csv").unlink()

            report = onboard_input(source)

            self.assertTrue(report["mapping_approval"]["human_approval_required"])
            self.assertTrue(
                any(item["dataset"] == "cost_basis" and item["reason"] == "missing_material_dataset" for item in report["mapping_approval"]["material_ambiguities"])
            )
            self.assertFalse(report["data_readiness"]["canary_start_allowed"])

    def test_cli_onboard_writes_data_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seller_exports"
            output = Path(tmp) / "readiness.json"
            source.mkdir()
            _write_offerlab_fixture(source)

            stream = io.StringIO()
            with redirect_stdout(stream):
                main(["offerlab-pilot", "onboard", str(source), "--output", str(output)])

            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["report_hash"], json.loads(output.read_text(encoding="utf-8"))["report_hash"])
            self.assertFalse(payload["data_readiness"]["canary_start_allowed"])


def _append_column(path: Path, column: str, value: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) + [column]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, column: value})


if __name__ == "__main__":
    unittest.main()
