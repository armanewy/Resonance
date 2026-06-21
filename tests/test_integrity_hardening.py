from __future__ import annotations

import _bootstrap  # noqa: F401

import copy
from datetime import timedelta
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from behavior_lab.causal import TreatmentComparator
from behavior_lab.core import DecisionEpisode, HypothesisSpec, InterventionTrial, parse_time, stable_hash
from behavior_lab.discovery import HypothesisGenerator, LLMHypothesisGenerator
from behavior_lab.dsl import Formula, FormulaSyntaxError
from behavior_lab.experiments import ExperimentIntegrityError, ExperimentScheduler
from behavior_lab.gym import WorldGym
from behavior_lab.models import BaseRateModel, model_from_artifact, model_to_artifact
from behavior_lab.personal_lab import PersonalLab
from behavior_lab.research_api import ResearchAPI
from behavior_lab.temporal import split_rows
from behavior_lab.runner import RunLock
from behavior_lab.worlds import HabitPlusOverrideWorld


class IntegrityHardeningTests(unittest.TestCase):
    def test_hidden_result_omits_direct_prevalence_and_baseline_lift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=31))
            gym.seed(60)
            api = ResearchAPI(gym, campaign_id="redaction")
            model = api.fit_model_zoo()[0]
            api.freeze_candidate(model.model_id)
            result = api.evaluate_hypothesis(model.model_id, "hidden")
            self.assertNotIn("base_rate", result)
            self.assertNotIn("lift_over_base_log_loss", result)
            self.assertIn("aggregate score", result["details"]["redacted"])

    def test_manual_hypothesis_must_match_target_and_known_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=32))
            gym.seed(60)
            api = ResearchAPI(gym, campaign_id="validation")
            unknown = HypothesisSpec.formula("h_unknown", gym.target_name, ["typo_feature"])
            api.submit_hypothesis(unknown)
            with self.assertRaises(ValueError):
                api.fit_hypothesis(unknown.hypothesis_id)

            wrong_target = HypothesisSpec.formula("h_wrong_target", "different_target", ["fatigue"])
            api.submit_hypothesis(wrong_target)
            with self.assertRaises(ValueError):
                api.fit_hypothesis(wrong_target.hypothesis_id)

    def test_llm_adapter_rejects_string_instead_of_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=33))
            gym.seed(50)
            api = ResearchAPI(gym, campaign_id="llm-shape")
            with self.assertRaises(ValueError):
                LLMHypothesisGenerator(lambda _: [{"terms": "fatigue"}]).propose(api)
            with self.assertRaises(ValueError):
                LLMHypothesisGenerator(
                    lambda _: [{"terms": ["fatigue"], "assumptions": "not a list"}]
                ).propose(api)

    def test_randomization_is_reproducible_across_clean_runs_and_restarts(self) -> None:
        def allocation(path: Path, *, restart_after: int | None) -> list[str]:
            gym = WorldGym(path, world=HabitPlusOverrideWorld(seed=101))
            scheduler = ExperimentScheduler(gym.ledger, seed=77)
            prereg = scheduler.preregister(
                question="reproducible allocation",
                treatment="explicit_first_step",
                comparator="generic_task_description",
                target="started_within_10_minutes",
                population="tasks",
                planned_trials=8,
                stopping_rule="eight assignments",
                analysis_plan="intention to treat",
                approval_required=False,
            )
            arms: list[str] = []
            for index in range(8):
                if restart_after is not None and index == restart_after:
                    scheduler = ExperimentScheduler(gym.ledger, seed=77)
                assignment = scheduler.assign_intervention(
                    {"fatigue": index / 10},
                    treatment="explicit_first_step",
                    comparator="generic_task_description",
                    preregistration_id=prereg,
                )
                self.assertEqual(assignment["assignment_index"], index)
                arms.append(assignment["assignment"]["assigned_treatment"])
            return arms

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            uninterrupted = allocation(Path(first), restart_after=None)
            restarted = allocation(Path(second), restart_after=3)
            self.assertEqual(uninterrupted, restarted)

    def test_outcome_cannot_rewrite_randomized_assignment_or_canonical_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=34))
            scheduler = ExperimentScheduler(gym.ledger, seed=4)
            prereg = scheduler.preregister(
                question="assignment integrity",
                treatment="explicit_first_step",
                comparator="generic_task_description",
                target="started_within_10_minutes",
                population="tasks",
                planned_trials=2,
                stopping_rule="two assignments",
                analysis_plan="intention to treat",
                approval_required=False,
            )
            assignment = scheduler.assign_intervention(
                {"fatigue": 0.4, "explicit_first_step": 0.0},
                treatment="explicit_first_step",
                comparator="generic_task_description",
                preregistration_id=prereg,
            )
            tampered = copy.deepcopy(assignment)
            tampered["context_snapshot"]["fatigue"] = 0.99
            with self.assertRaises(ExperimentIntegrityError):
                scheduler.record_trial_outcome(
                    tampered,
                    {"started_within_10_minutes": True},
                )

            scheduler.record_trial_outcome(
                assignment,
                {"started_within_10_minutes": True},
                data_provenance={
                    "context_snapshot": {"fatigue": 1.0},
                    "intervened_context": {"fatigue": 1.0},
                    "adapter": "test",
                },
            )
            stored = gym.ledger.payloads("intervention_trial")[-1]["data_provenance"]
            self.assertEqual(stored["context_snapshot"], assignment["context_snapshot"])
            self.assertEqual(stored["adapter"], "test")

    def test_intervention_trial_rejects_inconsistent_assigned_probability(self) -> None:
        with self.assertRaises(ValueError):
            InterventionTrial.create(
                subject_id="s",
                context_snapshot_id="c",
                comparison={"treatment": "a", "comparator": "b"},
                assignment={
                    "assigned_treatment": "a",
                    "treatment_probability": 0.7,
                    "assigned_probability": 0.3,
                },
                adherence={"treatment_delivered": True},
                outcomes={"y": True},
                measurement_horizons=["now"],
            )

    def test_variable_propensity_uses_weighted_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=35))
            probabilities = [0.2, 0.8, 0.3, 0.7]
            arms = ["a", "a", "b", "b"]
            outcomes = [1, 0, 1, 0]
            for index, (probability, arm, outcome) in enumerate(zip(probabilities, arms, outcomes, strict=True)):
                trial = InterventionTrial.create(
                    subject_id="s",
                    context_snapshot_id=f"c{index}",
                    comparison={"treatment": "a", "comparator": "b"},
                    assignment={
                        "assigned_treatment": arm,
                        "treatment_probability": probability,
                        "assigned_probability": probability if arm == "a" else 1.0 - probability,
                        "block": {},
                    },
                    adherence={"treatment_delivered": True, "treatment_seen": True},
                    outcomes={"y": outcome},
                    measurement_horizons=["now"],
                )
                gym.ledger.append("intervention_trial", trial, record_id=trial.trial_id, unique_record_id=True)
            result = TreatmentComparator(gym.ledger).compare(treatment="a", comparator="b", outcome_name="y")
            self.assertEqual(result.estimator, "hajek_inverse_probability_weighted")
            self.assertEqual(result.assignment_probability_range, [0.2, 0.8])
            self.assertIn("varied", result.warning or "")

    def test_model_artifact_semantic_validation_rejects_impossible_probability(self) -> None:
        model = BaseRateModel("m", 0.5)
        artifact = model_to_artifact(model, [])
        artifact["rate"] = 2.0
        artifact["artifact_hash"] = stable_hash(
            {key: value for key, value in artifact.items() if key != "artifact_hash"}
        )
        with self.assertRaises(ValueError):
            model_from_artifact(artifact)

    def test_formula_function_arity_is_checked_at_parse_time(self) -> None:
        with self.assertRaises(FormulaSyntaxError):
            Formula.parse(["threshold(fatigue)"])
        with self.assertRaises(FormulaSyntaxError):
            Formula.parse(["interaction(fatigue)"])
        with self.assertRaises(FormulaSyntaxError):
            Formula.parse(["sqrt(fatigue, ambiguity)"])

    def test_invalid_old_run_lock_can_be_recovered_after_stale_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".run.lock"
            path.write_text("not-json", encoding="utf-8")
            old = time.time() - 10
            os.utime(path, (old, old))
            with RunLock(path, stale_after_seconds=0.01):
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())

    def test_delayed_historical_backfill_is_not_prospective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=37))
            gym.seed(60)
            api = ResearchAPI(gym, campaign_id="backfill")
            model = api.fit_model_zoo()[0]
            freeze = api.freeze_candidate(model.model_id)
            cutoff = parse_time(freeze["payload"]["data_cutoff_time"])

            historical_time = cutoff - timedelta(days=1)
            backfill = DecisionEpisode.create(
                subject_id=gym.world.subject_id,
                decision_time=historical_time.isoformat(),
                observation_cutoff=(historical_time - timedelta(seconds=1)).isoformat(),
                situation={"type": "late_import", "description": "historical backfill"},
                available_actions=["start_now", "defer"],
                pre_decision_context={"fatigue": 0.4, "ambiguity": 0.5},
                observed_action={"action": "defer"},
                later_outcomes={gym.target_name: False},
                data_provenance={"source": "delayed_import"},
            )
            gym.ledger.append(
                "decision_episode",
                backfill,
                record_id=backfill.episode_id,
                unique_record_id=True,
            )
            gym.seed(1)
            assignments = gym.ensure_split_manifest(campaign_id="backfill")

            self.assertEqual(assignments[backfill.episode_id], "staging")
            prospective = gym.prospective_rows_for_freeze(
                freeze["payload"]["freeze_id"],
                "backfill",
            )
            self.assertEqual(len(prospective), 1)
            self.assertNotEqual(prospective[0]["case_id"], backfill.episode_id)
            self.assertEqual(api.submit_frozen_candidate(model.model_id)["n"], 1)

    def test_personal_lab_refuses_false_prospective_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = PersonalLab(Path(tmp))
            with self.assertRaises(RuntimeError):
                lab.freeze_model_for_prospective_block("unregistered", "no artifact")


    def test_hidden_budget_is_explicitly_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=37))
            gym.seed(20)
            with self.assertRaises(ValueError):
                ResearchAPI(gym, campaign_id="bad-budget", hidden_budget=2)

    def test_offline_experiment_is_blocked_after_candidate_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=38))
            gym.seed(60)
            api = ResearchAPI(gym, campaign_id="frozen-experiment")
            models = api.fit_model_zoo()
            proposal = api.propose_experiment([model.model_id for model in models[:3]])
            api.freeze_candidate(models[0].model_id)
            with self.assertRaises(PermissionError):
                api.run_offline_experiment(proposal, trials=2)

    def test_tiny_chronological_splits_prioritize_development_before_hidden(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"case_id": str(index), "decision_time": (start + timedelta(minutes=index)).isoformat()}
            for index in range(3)
        ]
        two = split_rows(rows[:2])
        self.assertEqual({name: len(values) for name, values in two.items()}, {
            "training": 1,
            "development": 1,
            "hidden": 0,
            "prospective": 0,
        })
        three = split_rows(rows)
        self.assertEqual({name: len(values) for name, values in three.items()}, {
            "training": 1,
            "development": 1,
            "hidden": 1,
            "prospective": 0,
        })

    def test_mutation_ids_include_parent_lineage_and_do_not_conflict(self) -> None:
        generator = HypothesisGenerator()
        parent_a = HypothesisSpec.formula("parent_a", "started_within_10_minutes", ["fatigue"])
        parent_b = HypothesisSpec.formula("parent_b", "started_within_10_minutes", ["fatigue"])
        mutation_a = generator.mutate_from_residuals(parent_a, [], "started_within_10_minutes")
        mutation_b = generator.mutate_from_residuals(parent_b, [], "started_within_10_minutes")
        self.assertNotEqual(mutation_a.hypothesis_id, mutation_b.hypothesis_id)
        self.assertEqual(mutation_a.parent_ids, ["parent_a"])
        self.assertEqual(mutation_b.parent_ids, ["parent_b"])

    def test_identical_preregistrations_have_independent_randomization_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gym = WorldGym(Path(tmp), world=HabitPlusOverrideWorld(seed=39))
            scheduler = ExperimentScheduler(gym.ledger, seed=7)

            def register() -> str:
                return scheduler.preregister(
                    question="Does a first step help?",
                    treatment="explicit_first_step",
                    comparator="generic_task_description",
                    target="started_within_10_minutes",
                    population="tasks",
                    planned_trials=2,
                    stopping_rule="two assignments",
                    analysis_plan="intention to treat",
                    approval_required=False,
                )

            first_prereg = register()
            second_prereg = register()
            context = {"fatigue": 0.4, "ambiguity": 0.8}
            first = scheduler.assign_intervention(
                context,
                treatment="explicit_first_step",
                comparator="generic_task_description",
                preregistration_id=first_prereg,
            )
            second = scheduler.assign_intervention(
                context,
                treatment="explicit_first_step",
                comparator="generic_task_description",
                preregistration_id=second_prereg,
            )

            self.assertEqual(
                first["randomization"]["experiment_signature"],
                second["randomization"]["experiment_signature"],
            )
            self.assertNotEqual(first["assignment_id"], second["assignment_id"])
            self.assertNotEqual(
                first["randomization"]["randomization_namespace"],
                second["randomization"]["randomization_namespace"],
            )


    def test_public_version_identifiers_stay_in_sync(self) -> None:
        import tomllib
        import behavior_lab
        from behavior_lab.models import SOFTWARE_VERSION

        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        declared = project["project"]["version"]
        self.assertEqual(declared, "0.4.0")
        self.assertEqual(behavior_lab.__version__, declared)
        self.assertEqual(SOFTWARE_VERSION, declared)



if __name__ == "__main__":
    unittest.main()
