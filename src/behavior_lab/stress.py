from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from behavior_lab.core import HypothesisSpec
from behavior_lab.evaluation import evaluate_model
from behavior_lab.gym import WorldGym
from behavior_lab.models import LogisticFormulaHypothesis, ModelFoundry
from behavior_lab.temporal import assert_snapshot_is_pre_decision, pre_decision_snapshot
from behavior_lab.worlds import make_world


class LabStressTester:
    """Small self-audit suite for the discovery infrastructure.

    This is intentionally code, not just documentation: the lab should be able
    to challenge its own assumptions after each wave. The checks target the
    exact failure modes we care about: temporal leakage, baselines beating fancy
    hypotheses, hidden labels being redacted, and formula discovery recovering
    at least part of a known synthetic mechanism.
    """

    def run(self, data_dir: str | Path, *, episodes: int = 160, seed: int = 17, world: str = "habit") -> dict[str, Any]:
        gym = WorldGym(data_dir, world=make_world(world, seed=seed))
        if not gym.decision_episodes():
            gym.seed(episodes)
        splits = gym.splits()
        models = ModelFoundry().fit_zoo(splits["training"], splits["development"], gym.target_name)
        dev = [evaluate_model(model, splits["development"], split="development") for model in models]
        dev.sort(key=lambda metric: metric.log_loss)
        base_rate = next((metric for metric in dev if metric.complexity == 1), None)
        hidden_payload = gym.blind_server().evaluate(models[0], split="hidden")
        leakage_ok = self._check_temporal_firewall(gym)
        language_probe_score = self._formula_language_driver_recall_probe(gym)
        best_formula = self._best_formula_model(models, dev)
        best_formula_terms = self._formula_terms(best_formula)
        best_formula_score = gym.world.mechanism_equivalence_score(best_formula_terms)
        intervention_direction_accuracy = self._intervention_direction_accuracy(gym, best_formula)
        best = dev[0]
        return {
            "world": gym.world.name,
            "episodes": len(gym.decision_episodes()),
            "splits": {name: len(rows) for name, rows in splits.items()},
            "temporal_firewall_ok": leakage_ok,
            "hidden_payload_redacted": hidden_payload.get("details", {}).get("redacted") is not None,
            "best_development_model": asdict(best),
            "base_rate_development_model": asdict(base_rate) if base_rate else None,
            "best_beats_base_log_loss": bool(base_rate and best.log_loss <= base_rate.log_loss),
            "best_formula_mechanism_recall": best_formula_score,
            "best_formula_terms": best_formula_terms,
            "intervention_direction_accuracy": intervention_direction_accuracy,
            "formula_language_driver_recall_probe": language_probe_score,
            "warnings": self._warnings(splits, base_rate is not None and best.log_loss <= base_rate.log_loss),
        }

    def run_world_matrix(self, data_dir: str | Path, *, episodes: int = 140, seed: int = 23) -> list[dict[str, Any]]:
        reports = []
        for world_name in ["habit", "two_mode", "threshold", "nonstationary", "confounded"]:
            gym = WorldGym(Path(data_dir) / world_name, world=make_world(world_name, seed=seed))
            if not gym.decision_episodes():
                gym.seed(episodes)
            splits = gym.splits()
            models = ModelFoundry().fit_zoo(splits["training"], splits["development"], gym.target_name)
            metrics = [evaluate_model(model, splits["development"], split="development") for model in models]
            metrics.sort(key=lambda item: item.log_loss)
            best_formula = self._best_formula_model(models, metrics)
            best_formula_terms = self._formula_terms(best_formula)
            best_formula_score = gym.world.mechanism_equivalence_score(best_formula_terms)
            language_probe_score = self._formula_language_driver_recall_probe(gym)
            reports.append(
                {
                    "world": gym.world.name,
                    "splits": {name: len(rows) for name, rows in splits.items()},
                    "best_model_id": metrics[0].model_id,
                    "best_log_loss": metrics[0].log_loss,
                    "best_complexity": metrics[0].complexity,
                    "best_formula_mechanism_recall": best_formula_score,
                    "best_formula_terms": best_formula_terms,
                    "intervention_direction_accuracy": self._intervention_direction_accuracy(gym, best_formula),
                    "formula_language_driver_recall_probe": language_probe_score,
                }
            )
        return reports

    def _check_temporal_firewall(self, gym: WorldGym) -> bool:
        for episode in gym.decision_episodes()[:20]:
            snapshot = pre_decision_snapshot(episode)
            assert_snapshot_is_pre_decision(snapshot)
            if "observed_action" in snapshot or "later_outcomes" in snapshot:
                return False
        return True

    def _best_formula_model(
        self,
        models: list[Any],
        metrics: list[Any],
    ) -> Any | None:
        ranked_model_ids = [metric.model_id for metric in metrics]
        ranked_models = sorted(
            [model for model in models if hasattr(model, "formula")],
            key=lambda model: ranked_model_ids.index(model.model_id) if model.model_id in ranked_model_ids else len(ranked_model_ids),
        )
        if not ranked_models:
            return None
        return ranked_models[0]

    def _formula_terms(self, model: Any | None) -> list[str]:
        if model is None or not hasattr(model, "formula"):
            return []
        return [term.expression for term in model.formula.terms]

    def _intervention_direction_accuracy(self, gym: WorldGym, model: Any | None) -> float | None:
        if model is None:
            return None
        comparisons = [
            ("explicit_first_step", "generic_task_description"),
            ("visible_commitment", "no_intervention"),
            ("two_minute_countdown", "no_intervention"),
        ]
        correct = 0
        total = 0
        for treatment, comparator in comparisons:
            for _ in range(8):
                context = gym.world.sample_context()
                treatment_context = self._apply_intervention(context, treatment)
                comparator_context = self._apply_intervention(context, comparator)
                true_effect = gym.world.probability_start(treatment_context) - gym.world.probability_start(comparator_context)
                predicted_effect = model.predict_proba(treatment_context) - model.predict_proba(comparator_context)
                if abs(true_effect) < 0.02:
                    total += 1
                    correct += 1 if abs(predicted_effect) < 0.05 else 0
                else:
                    total += 1
                    correct += 1 if true_effect * predicted_effect > 0 else 0
        return correct / total if total else None

    def _apply_intervention(self, context: dict[str, Any], intervention: str) -> dict[str, Any]:
        updated = dict(context)
        if intervention == "explicit_first_step":
            updated["explicit_first_step"] = 1.0
        elif intervention in {"generic_task_description", "no_intervention"}:
            if intervention == "generic_task_description":
                updated["explicit_first_step"] = 0.0
        elif intervention == "visible_commitment":
            updated["public_commitment"] = 1.0
        elif intervention == "two_minute_countdown":
            updated["deadline_near"] = 1.0
        return updated

    def _formula_language_driver_recall_probe(self, gym: WorldGym) -> float:
        rows = gym.splits()["training"]
        spec = HypothesisSpec.formula(
            "stress_known_driver_probe",
            gym.target_name,
            [
                "explicit_first_step",
                "indicator(ambiguity > 0.6)",
                "explicit_first_step * indicator(ambiguity > 0.6)",
                "fatigue",
                "deadline_near",
                "public_commitment",
                "recent_context_switches",
            ],
        )
        model = LogisticFormulaHypothesis(spec).fit(rows)
        terms = [term.expression for term in model.formula.terms]
        return gym.world.mechanism_equivalence_score(terms)

    def _warnings(self, splits: dict[str, list[dict[str, Any]]], beats_base: bool) -> list[str]:
        warnings: list[str] = []
        if len(splits.get("prospective", [])) == 0:
            warnings.append("prospective split is empty; freeze-and-forward claims are not meaningful yet")
        if not beats_base:
            warnings.append("no discovered model beat the base-rate baseline on development")
        return warnings
