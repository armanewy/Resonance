from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any, Callable

from behavior_lab.core import HypothesisSpec, new_id
from behavior_lab.dsl import Formula
from behavior_lab.evaluation import evaluate_model, residuals
from behavior_lab.experiments import DisagreementFinder, ExperimentScheduler
from behavior_lab.gym import TARGET, WorldGym
from behavior_lab.models import LogisticFormulaHypothesis, ModelFoundry
from behavior_lab.registry import ModelRegistry
from behavior_lab.temporal import feature_catalog


class HypothesisGenerator:
    def seed_hypotheses(self, rows: list[dict[str, Any]], target_name: str = TARGET) -> list[HypothesisSpec]:
        variables = set(feature_catalog(rows))
        specs = []
        simple = [name for name in ["ambiguity", "fatigue", "deadline_near", "public_commitment", "explicit_first_step"] if name in variables]
        if simple:
            specs.append(
                HypothesisSpec.formula(
                    "h_simple_pressures_v1",
                    target_name,
                    simple,
                    falsification_conditions=["does not beat the base rate on development or prospective cases"],
                )
            )
        if "explicit_first_step" in variables and "ambiguity" in variables:
            specs.append(
                HypothesisSpec.formula(
                    "h_ambiguity_interaction_v1",
                    target_name,
                    [
                        "explicit_first_step",
                        "indicator(ambiguity > 0.6)",
                        "explicit_first_step * indicator(ambiguity > 0.6)",
                        "fatigue",
                    ],
                    falsification_conditions=["interaction term fails to improve high-ambiguity cases"],
                )
            )
        if "social_cost" in variables:
            specs.append(
                HypothesisSpec.formula(
                    "h_social_threshold_v1",
                    target_name,
                    ["indicator(social_cost > 0.7)", "public_commitment", "deadline_near"],
                    falsification_conditions=["threshold term is unstable across new contexts"],
                )
            )
        return specs

    def mutate_from_residuals(
        self,
        parent: HypothesisSpec,
        residual_rows: list[dict[str, Any]],
        target_name: str = TARGET,
    ) -> HypothesisSpec:
        existing_terms = list(parent.structure.get("terms", []))
        candidates = self._residual_terms(residual_rows)
        for term in candidates:
            if term not in existing_terms:
                return HypothesisSpec.formula(
                    new_id("h_mut"),
                    target_name,
                    existing_terms + [term],
                    parent_ids=[parent.hypothesis_id],
                    falsification_conditions=[f"new residual term {term!r} fails on development and prospective splits"],
                )
        return HypothesisSpec.formula(
            new_id("h_mut"),
            target_name,
            existing_terms,
            parent_ids=[parent.hypothesis_id],
            falsification_conditions=["no residual mutation improved generalization"],
        )

    def _residual_terms(self, residual_rows: list[dict[str, Any]]) -> list[str]:
        if not residual_rows:
            return []
        numeric_names = sorted(
            {
                name
                for row in residual_rows
                for name, value in row.get("features", {}).items()
                if name != "bias" and isinstance(value, (int, float, bool))
            }
        )
        scores = []
        for name in numeric_names:
            high_values = [float(row["features"].get(name, 0.0)) for row in residual_rows[: max(1, len(residual_rows) // 2)]]
            score = abs(mean(high_values)) if high_values else 0.0
            scores.append((score, name))
        scores.sort(reverse=True)
        terms = []
        for _, name in scores:
            if name in {"ambiguity", "fatigue", "social_cost"}:
                terms.append(f"indicator({name} > 0.6)")
            else:
                terms.append(name)
        if {"explicit_first_step", "ambiguity"}.issubset(set(numeric_names)):
            terms.insert(0, "explicit_first_step * indicator(ambiguity > 0.6)")
        return terms


class LLMHypothesisGenerator:
    """Validated adapter seam for an external hypothesis-proposal model."""

    def __init__(self, proposer: Callable[[dict[str, Any]], list[dict[str, Any]]]):
        self.proposer = proposer

    def propose(self, api: Any, *, max_hypotheses: int = 5) -> list[HypothesisSpec]:
        variables = set(api.list_variables())
        request = {
            "schema": api.inspect_schema(),
            "target": api.describe_target(),
            "variables": sorted(variables),
            "rules": [
                "Return small executable formulas only.",
                "Use only listed variables.",
                "Include assumptions and falsification conditions.",
                "Do not claim causality from observational association.",
            ],
            "max_hypotheses": max_hypotheses,
        }
        specs: list[HypothesisSpec] = []
        for index, candidate in enumerate(self.proposer(request), start=1):
            terms = [str(term) for term in candidate.get("terms", [])]
            if not terms:
                continue
            formula = Formula.parse(terms)
            used_variables = set(formula.variables)
            unknown = used_variables - variables
            if unknown:
                raise ValueError(f"LLM hypothesis used unknown variables: {sorted(unknown)}")
            specs.append(
                HypothesisSpec.formula(
                    hypothesis_id=str(candidate.get("hypothesis_id") or new_id("h_llm")),
                    target_name=api.gym.target_name,
                    terms=terms,
                    assumptions=[str(item) for item in candidate.get("assumptions", [])]
                    or ["LLM-proposed formula passed DSL and variable validation"],
                    falsification_conditions=[str(item) for item in candidate.get("falsification_conditions", [])]
                    or ["Does not beat base-rate and recent-rate baselines on development feedback"],
                )
            )
            if len(specs) >= max_hypotheses:
                break
        return specs


class DiscoveryLoop:
    def __init__(self, gym: WorldGym):
        self.gym = gym
        self.registry = ModelRegistry(gym.ledger)
        self.scheduler = ExperimentScheduler(gym.ledger)
        self.generator = HypothesisGenerator()
        self.finder = DisagreementFinder()

    def run(self, iterations: int = 3, offline_trials_per_iteration: int = 8) -> dict[str, Any]:
        report: dict[str, Any] = {"iterations": []}
        active_specs: list[HypothesisSpec] = []
        for iteration in range(iterations):
            splits = self.gym.splits()
            training = splits["training"]
            development = splits["development"]
            if not active_specs:
                active_specs = self.generator.seed_hypotheses(training, self.gym.target_name)
            fitted = []
            for spec in active_specs:
                self.registry.submit_hypothesis(spec)
                model = LogisticFormulaHypothesis(spec).fit(training)
                fitted.append((spec, model))
                self.registry.record_fit(model, spec.hypothesis_id, "training", len(training))
            zoo = ModelFoundry().fit_zoo(training, development, self.gym.target_name)
            candidates = [model for _, model in fitted] + zoo
            scored = [(model, evaluate_model(model, development, split="development", include_details=True)) for model in candidates]
            scored.sort(key=lambda item: item[1].log_loss)
            best_model, best_metrics = scored[0]
            self.registry.record_evaluation(best_metrics)
            if hasattr(best_model, "hypothesis_id"):
                best_spec = next((spec for spec, model in fitted if model.model_id == best_model.model_id), None)
            else:
                best_spec = None
            if best_metrics.lift_over_base_log_loss > 0:
                self.registry.promote_hypothesis(getattr(best_model, "hypothesis_id", best_model.model_id), best_model.model_id, "best development log loss this iteration")
            for model, metrics in scored[-2:]:
                self.registry.retire_hypothesis(
                    getattr(model, "hypothesis_id", model.model_id),
                    "dominated on development split",
                    {"log_loss": metrics.log_loss, "best_log_loss": best_metrics.log_loss},
                )
            residual_summary = residuals(best_model, development, limit=10)
            if best_spec:
                mutation = self.generator.mutate_from_residuals(best_spec, residual_summary, self.gym.target_name)
                active_specs = [best_spec, mutation]
            else:
                active_specs = self.generator.seed_hypotheses(training, self.gym.target_name)

            contexts = [self.gym.world.sample_context() for _ in range(25)]
            proposal = self.finder.propose(candidates[: min(6, len(candidates))], contexts)
            prereg_id = self.scheduler.preregister(
                question="Which context best separates the surviving task-start theories?",
                treatment=proposal.treatment,
                comparator=proposal.comparator,
                target=self.gym.target_name,
                population="synthetic world gym contexts",
                planned_trials=offline_trials_per_iteration,
                stopping_rule="Fixed number of synthetic trials per iteration.",
                analysis_plan="Compare randomized treatment and comparator outcomes; feed observations into next offline fit.",
                approval_required=False,
            )
            for _ in range(offline_trials_per_iteration):
                assigned = proposal.treatment if self.gym.world.random.random() < 0.5 else proposal.comparator
                trial = self.gym.world.run_intervention_trial(
                    proposal.context,
                    proposal.treatment,
                    proposal.comparator,
                    assigned,
                    0.5,
                    preregistration_id=prereg_id,
                )
                self.gym.ledger.append("intervention_trial", trial, record_id=trial.trial_id)

            report["iterations"].append(
                {
                    "iteration": iteration + 1,
                    "training_cases": len(training),
                    "development_cases": len(development),
                    "best_model_id": best_model.model_id,
                    "best_log_loss": best_metrics.log_loss,
                    "best_lift_over_base": best_metrics.lift_over_base_log_loss,
                    "proposal": asdict(proposal),
                    "new_trials": offline_trials_per_iteration,
                }
            )
        final_models = self.gym.fit_model_zoo()
        server = self.gym.blind_server()
        hidden_results = [server.evaluate(model, split="hidden") for model in final_models]
        hidden_results.sort(key=lambda item: item["log_loss"])
        development_scores = [(model, evaluate_model(model, self.gym.splits()["development"], split="development")) for model in final_models]
        development_scores.sort(key=lambda item: item[1].log_loss)
        frozen_model = development_scores[0][0]
        prospective_result = server.submit_frozen_candidate(frozen_model)
        self.registry.freeze_candidate(frozen_model.model_id, "prospective", "final loop candidate")
        report["hidden_leaderboard"] = hidden_results[:5]
        report["prospective_result"] = prospective_result
        return report
