from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from behavior_lab.causal import TreatmentComparator
from behavior_lab.experiments import ExperimentScheduler
from behavior_lab.gym import WorldGym
from behavior_lab.runner import BatchConfig, SyntheticBatchRunner
from behavior_lab.stress import LabStressTester
from behavior_lab.worlds import HabitPlusOverrideWorld


class CausalAndStressTests(unittest.TestCase):
    def test_assignment_probability_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = ExperimentScheduler(WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=1)).ledger)
            with self.assertRaises(ValueError):
                scheduler.assign_intervention({}, treatment="a", comparator="b", probability=1.0)
            with self.assertRaises(ValueError):
                scheduler.assign_intervention({}, treatment="a", comparator="b", probability=0.0)

    def test_treatment_comparator_stratifies_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=2))
            scheduler = ExperimentScheduler(gym.ledger, seed=3)
            pre = scheduler.preregister(
                question="test",
                treatment="explicit_first_step",
                comparator="generic_task_description",
                target="started_within_10_minutes",
                population="tasks",
                planned_trials=6,
                stopping_rule="fixed",
                analysis_plan="difference in means",
                approval_required=False,
            )
            for i in range(6):
                assignment = scheduler.assign_intervention(
                    {"fatigue": 0.8 if i % 2 else 0.1, "task_size_large": 0.0},
                    treatment="explicit_first_step",
                    comparator="generic_task_description",
                    preregistration_id=pre,
                )
                assigned = assignment["assignment"]["assigned_treatment"]
                scheduler.record_trial_outcome(
                    assignment,
                    {"started_within_10_minutes": assigned == "explicit_first_step"},
                )
            result = TreatmentComparator(gym.ledger).compare(
                treatment="explicit_first_step",
                comparator="generic_task_description",
                outcome_name="started_within_10_minutes",
                preregistration_id=pre,
            )
            self.assertEqual(result.treatment_n + result.comparator_n, 6)
            self.assertTrue(result.by_block)
            self.assertGreaterEqual(result.standard_error, 0.0)

    def test_lab_stress_tester_runs_and_redacts_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = LabStressTester().run(Path(tmp), episodes=80, seed=4)
            self.assertTrue(report["temporal_firewall_ok"])
            self.assertTrue(report["hidden_payload_redacted"])
            self.assertIn("best_formula_mechanism_recall", report)
            self.assertIn("formula_language_driver_recall_probe", report)
            self.assertGreaterEqual(report["formula_language_driver_recall_probe"], 0.5)

    def test_batch_runner_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = BatchConfig(worlds=["habit"], seeds=[9], episode_counts=[40])
            runner = SyntheticBatchRunner(Path(tmp))
            first = runner.run(config)
            second = runner.run(config)
            self.assertEqual(first[0]["status"], "complete")
            self.assertEqual(second[0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
