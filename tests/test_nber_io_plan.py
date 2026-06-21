from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_nber_io.py"
PLAN = ROOT / "docs" / "research" / "NBER_SCALE_PLAN.md"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_nber_io", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark_nber_io.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NberIoPlanTest(unittest.TestCase):
    def test_scale_plan_records_negotiation_first_constraints(self) -> None:
        text = PLAN.read_text(encoding="utf-8")
        required = [
            "negotiation-first",
            "anon_bo_threads.csv.gz",
            "anon_bo_lists.csv.gz",
            "non-negotiated listings omitted",
            "atomic final manifest",
            "deterministic partitions",
            "Windows-compatible paths",
            "restart after interruption",
            "complete 98M-listing normalization",
        ]
        for phrase in required:
            self.assertIn(phrase, text)

    def test_partitioning_is_deterministic(self) -> None:
        module = load_benchmark_module()
        first = module.stable_partition("L000000000042", 64)
        second = module.stable_partition("L000000000042", 64)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 64)

    def test_python_fallback_benchmark_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--rows",
                    "2000",
                    "--listing-rows",
                    "5000",
                    "--engines",
                    "python",
                    "--work-dir",
                    tmp,
                    "--partitions",
                    "8",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["data"]["thread_rows"], 2000)
        self.assertEqual(summary["data"]["listing_rows"], 5000)
        self.assertEqual(len(summary["data"]["threads_sha256"]), 64)
        self.assertEqual(len(summary["data"]["listings_sha256"]), 64)
        self.assertEqual(summary["engines"][0]["status"], "ok")
        self.assertEqual(summary["engines"][0]["thread_rows"], 2000)
        self.assertEqual(summary["engines"][0]["listing_rows"], 5000)
        self.assertEqual(summary["engines"][0]["matched_listing_rows"], summary["engines"][0]["distinct_thread_listing_ids"])


if __name__ == "__main__":
    unittest.main()
