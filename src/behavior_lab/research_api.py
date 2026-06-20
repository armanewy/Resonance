from __future__ import annotations

from dataclasses import asdict
from typing import Any

from behavior_lab.core import HypothesisSpec
from behavior_lab.evaluation import counterexamples, paired_compare, residuals
from behavior_lab.experiments import DisagreementFinder, ExperimentProposal, ExperimentScheduler
from behavior_lab.gym import TARGET, WorldGym
from behavior_lab.models import LogisticFormulaHypothesis, ModelFoundry, model_from_artifact, model_to_artifact
from behavior_lab.registry import EvaluationBudgetError, ModelRegistry
from behavior_lab.temporal import feature_catalog


EvaluationBudgetExceeded = EvaluationBudgetError


class ResearchAPI:
    """Typed LLM-facing facade.

    This deliberately avoids raw ledger access. Training labels are visible, development
    failures are limited summaries, and hidden/prospective labels are never exposed.
    """

    def __init__(self, gym: WorldGym, *, campaign_id: str = "default", hidden_budget: int = 1, prospective_budget: int = 1):
        self.gym = gym
        self.registry = ModelRegistry(gym.ledger)
        self.models: dict[str, Any] = {}
        self.hypotheses: dict[str, HypothesisSpec] = {}
        self.campaign_id = campaign_id
        self.hidden_budget = hidden_budget
        self.prospective_budget = prospective_budget
        self._load_registry_state()

    def inspect_schema(self) -> dict[str, Any]:
        return {
            "record_types": [
                "decision_episode",
                "intervention_trial",
                "hypothesis",
                "model_fit",
                "evaluation",
                "evaluation_budget_use",
                "experiment_preregistration",
                "intervention_assignment",
                "split_assignment",
                "research_run_start",
                "research_run_end",
                "frozen_candidate",
                "model_obituary",
            ],
            "target": {"name": self.gym.target_name, "type": "binary"},
            "splits": {name: len(rows) for name, rows in self.gym.splits().items()},
        }

    def list_variables(self) -> list[str]:
        return feature_catalog(self.gym.splits()["training"])

    def describe_target(self, target_id: str = TARGET) -> dict[str, Any]:
        rows = self.gym.splits()["training"]
        positives = sum(row["target"] for row in rows)
        return {
            "target_id": target_id,
            "definition": "Whether the intended task began within ten minutes.",
            "training_cases": len(rows),
            "training_base_rate": positives / len(rows) if rows else 0.0,
        }

    def query_training_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.gym.blind_server().query_training_data(limit=limit)

    def inspect_model_registry(self) -> dict[str, Any]:
        return self.registry.inspect_model_registry()

    def inspect_model_lineage(self, model_id: str | None = None) -> dict[str, Any]:
        graph = self.registry.lineage_graph()
        if model_id is None:
            return graph
        return {
            "nodes": {key: value for key, value in graph["nodes"].items() if key == model_id or model_id in str(value)},
            "edges": [edge for edge in graph["edges"] if model_id in {edge["from"], edge["to"]}],
        }

    def submit_hypothesis(self, hypothesis_spec: HypothesisSpec) -> dict[str, Any]:
        self.hypotheses[hypothesis_spec.hypothesis_id] = hypothesis_spec
        return self.registry.submit_hypothesis(hypothesis_spec)

    def fit_hypothesis(self, hypothesis_id: str) -> dict[str, Any]:
        spec = self.hypotheses.get(hypothesis_id)
        if spec is None:
            payload = self.gym.ledger.latest_by_payload_key("hypothesis", "hypothesis_id", hypothesis_id)
            if payload is None:
                raise KeyError(f"Unknown hypothesis: {hypothesis_id}")
            spec = HypothesisSpec(**payload)
            self.hypotheses[hypothesis_id] = spec
        training_rows = self.gym.splits()["training"]
        model = LogisticFormulaHypothesis(spec).fit(training_rows)
        self.models[model.model_id] = model
        artifact = model_to_artifact(model, training_rows)
        self.registry.record_fit(model, spec.hypothesis_id, "training", len(training_rows), artifact=artifact)
        return {"model_id": model.model_id, "hypothesis_id": hypothesis_id, "parameters": model.parameters}

    def evaluate_hypothesis(self, model_id: str, split: str = "development") -> dict[str, Any]:
        model = self._model(model_id)
        if split == "prospective":
            raise PermissionError("Use submit_frozen_candidate for prospective evaluation.")
        if split == "hidden":
            self.registry.assert_evaluation_budget_available(campaign_id=self.campaign_id, split=split, limit=self.hidden_budget)
        result = self.gym.blind_server().evaluate(model, split=split)
        if split in {"development", "hidden", "prospective"}:
            self.registry.record_evaluation_from_payload(result)
        if split == "hidden":
            self.registry.record_evaluation_budget_use(
                campaign_id=self.campaign_id,
                model_id=model_id,
                split=split,
                limit=self.hidden_budget,
            )
        return result

    def compare_models(self, model_a: str, model_b: str, split: str = "development") -> dict[str, Any]:
        rows = self.gym.splits()[split]
        return paired_compare(self._model(model_a), self._model(model_b), rows)

    def inspect_residuals(self, model_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return residuals(self._model(model_id), self.gym.splits()["development"], limit=limit)

    def inspect_counterexamples(self, model_a: str, model_b: str, limit: int = 10) -> list[dict[str, Any]]:
        return counterexamples(self._model(model_a), self._model(model_b), self.gym.splits()["development"], limit=limit)

    def propose_experiment(self, model_ids: list[str] | None = None) -> ExperimentProposal:
        if model_ids is None:
            if not self.models:
                for model in self.fit_model_zoo():
                    self.models[model.model_id] = model
            model_ids = list(self.models)[:6]
        models = [self._model(model_id) for model_id in model_ids]
        contexts = [self.gym.world.sample_context() for _ in range(20)]
        return DisagreementFinder().propose(models, contexts)

    def simulate_experiment(self, proposal: ExperimentProposal, trials: int = 12) -> list[dict[str, Any]]:
        simulated = []
        for index in range(trials):
            assigned = proposal.treatment if index % 2 == 0 else proposal.comparator
            trial = self.gym.world.run_intervention_trial(
                proposal.context,
                proposal.treatment,
                proposal.comparator,
                assigned,
                0.5,
            )
            simulated.append(asdict(trial))
        return simulated

    def run_offline_experiment(self, proposal: ExperimentProposal, trials: int = 12) -> dict[str, Any]:
        scheduler = ExperimentScheduler(self.gym.ledger)
        preregistration_id = scheduler.preregister(
            question="Synthetic experiment proposed through ResearchAPI.",
            treatment=proposal.treatment,
            comparator=proposal.comparator,
            target=self.gym.target_name,
            population="synthetic world gym contexts",
            planned_trials=trials,
            stopping_rule=f"Stop after exactly {trials} synthetic assignments.",
            analysis_plan="Estimate randomized difference in means and append all trials to the immutable ledger.",
            approval_required=False,
        )
        for _ in range(trials):
            assignment = scheduler.assign_intervention(
                proposal.context,
                treatment=proposal.treatment,
                comparator=proposal.comparator,
                probability=0.5,
                preregistration_id=preregistration_id,
            )
            assigned = assignment["assignment"]["assigned_treatment"]
            synthetic_trial = self.gym.world.run_intervention_trial(
                proposal.context,
                proposal.treatment,
                proposal.comparator,
                assigned,
                0.5,
                preregistration_id=preregistration_id,
            )
            scheduler.record_trial_outcome(
                assignment,
                synthetic_trial.outcomes,
                adherence=synthetic_trial.adherence,
                measurement_horizons=synthetic_trial.measurement_horizons,
                subject_id=synthetic_trial.subject_id,
            )
        self.gym.ledger.verify_hash_chain()
        effect = scheduler.estimate_treatment_effect(
            treatment=proposal.treatment,
            comparator=proposal.comparator,
            outcome_name=self.gym.target_name,
        )
        return {
            "preregistration_id": preregistration_id,
            "trials_appended": trials,
            "ledger_valid": True,
            "effect_estimate": effect,
        }

    def submit_frozen_candidate(self, model_id: str) -> dict[str, Any]:
        model = self._model(model_id)
        self.registry.assert_evaluation_budget_available(
            campaign_id=self.campaign_id,
            split="prospective",
            limit=self.prospective_budget,
        )
        self.registry.freeze_candidate(model_id, "prospective", "submitted through ResearchAPI", campaign_id=self.campaign_id)
        result = self.gym.blind_server().submit_frozen_candidate(model)
        self.registry.record_evaluation_from_payload(result)
        self.registry.record_evaluation_budget_use(
            campaign_id=self.campaign_id,
            model_id=model_id,
            split="prospective",
            limit=self.prospective_budget,
        )
        return result

    def fit_model_zoo(self) -> list[Any]:
        splits = self.gym.splits()
        models = ModelFoundry().fit_zoo(splits["training"], splits["development"], self.gym.target_name)
        for model in models:
            self.models[model.model_id] = model
            self.registry.record_fit(
                model,
                getattr(model, "hypothesis_id", model.model_id),
                "training",
                len(splits["training"]),
                artifact=model_to_artifact(model, splits["training"]),
            )
        return models

    def mechanism_score(self, hypothesis_id: str) -> dict[str, Any]:
        spec = self.hypotheses[hypothesis_id]
        terms = list(spec.structure.get("terms", []))
        return {
            "hypothesis_id": hypothesis_id,
            "synthetic_world_only": True,
            "driver_recall": self.gym.world.mechanism_equivalence_score(terms),
        }

    def _model(self, model_id: str) -> Any:
        if model_id not in self.models:
            self._load_registry_state()
        if model_id not in self.models:
            raise KeyError(f"Model {model_id!r} is not fitted in this ResearchAPI session or persisted registry")
        return self.models[model_id]

    def _load_registry_state(self) -> None:
        for payload in self.gym.ledger.payloads("hypothesis"):
            try:
                self.hypotheses[payload["hypothesis_id"]] = HypothesisSpec(**payload)
            except TypeError:
                continue
        for payload in self.gym.ledger.payloads("model_fit"):
            artifact = payload.get("artifact", {})
            if not artifact or artifact.get("family") == "unknown":
                continue
            try:
                model = model_from_artifact(artifact)
            except (KeyError, TypeError, ValueError):
                continue
            self.models[model.model_id] = model
