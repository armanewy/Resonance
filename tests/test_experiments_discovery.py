from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from behavior_lab.discovery import DiscoveryLoop, LLMHypothesisGenerator
from behavior_lab.core import HypothesisSpec
from behavior_lab.experiments import ExperimentScheduler
from behavior_lab.gym import WorldGym
from behavior_lab.personal_lab import PersonalLab
from behavior_lab.registry import ModelRegistry
from behavior_lab.research_api import EvaluationBudgetExceeded, ResearchAPI
from behavior_lab.worlds import HabitPlusOverrideWorld


class ExperimentDiscoveryTests(unittest.TestCase):
    def test_randomized_assignment_and_effect_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = PersonalLab(Path(tmp))
            prereg = lab.preregister_task_start_experiment(planned_trials=4)
            context = {"fatigue": 0.4, "ambiguity": 0.8, "task_size_large": 1.0}
            assignment = lab.assign_for_task(
                context,
                treatment="explicit_first_step",
                comparator="generic_task_description",
                preregistration_id=prereg,
            )
            self.assertIn(assignment["assignment"]["assigned_treatment"], {"explicit_first_step", "generic_task_description"})
            lab.capture_trial_outcome(
                assignment,
                started_within_10_minutes=True,
                time_to_start_seconds=120,
                completed_within_day=False,
            )
            effect = lab.estimate_effect("explicit_first_step", "generic_task_description")
            self.assertEqual(effect["treatment_n"] + effect["comparator_n"], 1)

    def test_real_intervention_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = ExperimentScheduler(WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=4)).ledger)
            with self.assertRaises(PermissionError):
                scheduler.launch_real_intervention(None, approved_by_human=False)  # type: ignore[arg-type]

    def test_discovery_loop_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=5))
            gym.seed(120)
            report = DiscoveryLoop(gym).run(iterations=2, offline_trials_per_iteration=3)
            self.assertEqual(len(report["iterations"]), 2)
            self.assertTrue(gym.ledger.verify_hash_chain())
            self.assertGreater(len(gym.ledger.payloads("intervention_trial")), 0)

    def test_research_api_facade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=6))
            gym.seed(80)
            api = ResearchAPI(gym)
            self.assertIn("ambiguity", api.list_variables())
            models = api.fit_model_zoo()
            result = api.gym.blind_server().evaluate(models[0], split="hidden")
            self.assertEqual(result["details"]["redacted"], "hidden labels and failure rows are not exposed")
            proposal = api.propose_experiment([model.model_id for model in models[:3]])
            self.assertGreaterEqual(proposal.expected_hypothesis_separation, 0.0)

    def test_split_manifest_does_not_migrate_existing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=7))
            gym.seed(60)
            first = gym.split_assignments()
            self.assertFalse(first)
            original = {row["case_id"]: split for split, rows in gym.splits().items() for row in rows}
            gym.seed(5)
            updated = {row["case_id"]: split for split, rows in gym.splits().items() for row in rows}
            for case_id, split in original.items():
                self.assertEqual(updated[case_id], split)
            self.assertGreater(len(gym.ledger.payloads("split_assignment")), len(original))
            ModelRegistry(gym.ledger).freeze_candidate("m_test", "prospective", "test freeze")
            gym.seed(2)
            after_freeze = {row["case_id"]: split for split, rows in gym.splits().items() for row in rows}
            new_case_ids = set(after_freeze) - set(updated)
            self.assertTrue(new_case_ids)
            self.assertTrue(all(after_freeze[case_id] == "prospective" for case_id in new_case_ids))

    def test_research_api_budgets_reload_and_offline_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=8))
            gym.seed(90)
            api = ResearchAPI(gym, campaign_id="budget-test")
            spec = HypothesisSpec.formula(
                "h_budget_reload",
                gym.target_name,
                ["deadline_near", "fatigue", "explicit_first_step * indicator(ambiguity > 0.6)"],
            )
            api.submit_hypothesis(spec)
            fit = api.fit_hypothesis(spec.hypothesis_id)
            model_id = fit["model_id"]
            api.evaluate_hypothesis(model_id, split="hidden")
            with self.assertRaises(EvaluationBudgetExceeded):
                api.evaluate_hypothesis(model_id, split="hidden")

            reloaded = ResearchAPI(gym, campaign_id="reload-test")
            self.assertIn(model_id, reloaded.models)
            proposal = reloaded.propose_experiment([model_id])
            before = len(gym.ledger.payloads("intervention_trial"))
            summary = reloaded.run_offline_experiment(proposal, trials=4)
            self.assertTrue(summary["ledger_valid"])
            self.assertEqual(summary["trials_appended"], 4)
            self.assertEqual(len(gym.ledger.payloads("intervention_trial")), before + 4)

    def test_llm_hypothesis_generator_validates_dsl_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=9))
            gym.seed(50)
            api = ResearchAPI(gym, campaign_id="llm-adapter-test")
            generator = LLMHypothesisGenerator(
                lambda _: [
                    {
                        "hypothesis_id": "h_llm_valid",
                        "terms": ["deadline_near", "fatigue"],
                        "assumptions": ["synthetic safe adapter test"],
                        "falsification_conditions": ["fails on development"],
                    }
                ]
            )
            specs = generator.propose(api)
            self.assertEqual(specs[0].hypothesis_id, "h_llm_valid")

            invalid = LLMHypothesisGenerator(lambda _: [{"terms": ["hidden_label_from_future"]}])
            with self.assertRaises(ValueError):
                invalid.propose(api)


if __name__ == "__main__":
    unittest.main()
