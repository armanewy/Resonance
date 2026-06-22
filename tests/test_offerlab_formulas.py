from __future__ import annotations

import _bootstrap  # noqa: F401

from pathlib import Path
import tempfile
import unittest

from behavior_lab.benchmarks.splits import chronological_split
from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.datasets.nber_best_offer.tasks import build_tasks
from behavior_lab.offerlab_models.formulas import FormulaHiddenLockbox, build_formula_candidates, evaluate_formula_candidates, fit_formula


def _seller_rows() -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_sample_dataset(root / "raw")
        normalize_dataset(root / "raw", root / "normalized")
        return build_tasks(root / "normalized")["seller_next_action"]


class OfferLabFormulaTests(unittest.TestCase):
    def test_formula_candidates_are_small_and_falsifiable(self) -> None:
        candidates = build_formula_candidates()
        self.assertGreaterEqual(len(candidates), 5)
        terms = {term for candidate in candidates for term in candidate.terms}
        self.assertIn("relative_offer", terms)
        self.assertIn("gap_to_listing", terms)
        self.assertIn("prior_counter_count", terms)
        self.assertNotIn("concession_size", terms)
        self.assertIn("timing_hour", terms)
        self.assertIn("relative_offer_x_round", terms)
        for candidate in candidates:
            self.assertLessEqual(candidate.complexity, 3)
            self.assertIn("Fails if", candidate.falsification_condition)

    def test_formula_evaluation_uses_hidden_once(self) -> None:
        split = chronological_split(_seller_rows(), time_key="timestamp")
        report = evaluate_formula_candidates(split.train, split.development, split.hidden, black_box_model_id="regularized_glm", black_box_hidden_loss=1.0)
        self.assertFalse(report["production_export_allowed"])
        self.assertTrue(report["research_only"])
        self.assertEqual(report["scope"]["evidence_scope"], "bounded_smoke_or_semantics")
        self.assertTrue(report["falsification_enforced"])
        self.assertFalse(report["hidden_lockbox"]["submitted"])
        self.assertFalse(report["black_box_comparison"]["compared"])
        self.assertIn("chronological development falsification", report["black_box_comparison"]["claim"])

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "formula-hidden.jsonl"
            submitted = evaluate_formula_candidates(
                split.train,
                split.development,
                split.hidden,
                black_box_model_id="regularized_glm",
                black_box_hidden_loss=1.0,
                hidden_lockbox_id="formula-test",
                hidden_lockbox_store_path=store,
            )
            self.assertEqual(submitted["hidden_lockbox"]["hidden_submission_count"], 1)
            self.assertTrue(submitted["black_box_comparison"]["compared"])
            with self.assertRaises(RuntimeError):
                evaluate_formula_candidates(
                    split.train,
                    split.development,
                    split.hidden,
                    hidden_lockbox_id="formula-test",
                    hidden_lockbox_store_path=store,
                )

            model = fit_formula(build_formula_candidates()[0], split.train)
            with self.assertRaises(TypeError):
                FormulaHiddenLockbox(split.hidden)  # type: ignore[call-arg]
            lockbox = FormulaHiddenLockbox(split.hidden, lockbox_id="formula-direct", store_path=Path(tmp) / "formula-direct.jsonl")
            lockbox.submit_once(model)
            with self.assertRaises(RuntimeError):
                lockbox.submit_once(model)


if __name__ == "__main__":
    unittest.main()
