from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Any

from behavior_lab.core import InterventionTrial, new_id, utc_now
from behavior_lab.causal import TreatmentComparator
from behavior_lab.evaluation import BinaryPredictor
from behavior_lab.ledger import ImmutableLedger


@dataclass(frozen=True)
class ExperimentProposal:
    experiment_id: str
    mode: str
    context: dict[str, Any]
    treatment: str
    comparator: str
    expected_hypothesis_separation: float
    cost: float
    risk: float
    participant_burden: float
    utility: float
    model_predictions: dict[str, dict[str, float]]


class ExperimentScheduler:
    def __init__(self, ledger: ImmutableLedger, seed: int = 11):
        self.ledger = ledger
        self.random = random.Random(seed)

    def preregister(
        self,
        *,
        question: str,
        treatment: str,
        comparator: str,
        target: str,
        population: str,
        planned_trials: int,
        stopping_rule: str,
        analysis_plan: str,
        approval_required: bool = True,
    ) -> str:
        preregistration_id = new_id("pre")
        self.ledger.append(
            "experiment_preregistration",
            {
                "preregistration_id": preregistration_id,
                "question": question,
                "comparison": {"treatment": treatment, "comparator": comparator},
                "target": target,
                "population": population,
                "planned_trials": planned_trials,
                "stopping_rule": stopping_rule,
                "analysis_plan": analysis_plan,
                "approval_required": approval_required,
                "created_at": utc_now(),
            },
            record_id=preregistration_id,
        )
        return preregistration_id

    def assign_intervention(
        self,
        context: dict[str, Any],
        *,
        treatment: str,
        comparator: str,
        probability: float = 0.5,
        preregistration_id: str | None = None,
    ) -> dict[str, Any]:
        if not 0.0 < probability < 1.0:
            raise ValueError("Assignment probability must be strictly between 0 and 1")
        assigned = treatment if self.random.random() < probability else comparator
        assignment = {
            "assignment_id": new_id("a"),
            "context_snapshot": dict(context),
            "comparison": {"treatment": treatment, "comparator": comparator},
            "assignment": {
                "method": "randomized_block",
                "assigned_treatment": assigned,
                "probability": probability,
                "block": self._block(context),
            },
            "preregistration_id": preregistration_id,
            "assigned_at": utc_now(),
        }
        self.ledger.append("intervention_assignment", assignment, record_id=assignment["assignment_id"])
        return assignment

    def record_trial_outcome(
        self,
        assignment: dict[str, Any],
        outcomes: dict[str, Any],
        *,
        adherence: dict[str, Any] | None = None,
        measurement_horizons: list[str] | None = None,
        subject_id: str = "arman",
    ) -> InterventionTrial:
        intervened_context = _apply_intervention(
            assignment["context_snapshot"],
            assignment["assignment"]["assigned_treatment"],
        )
        trial = InterventionTrial.create(
            subject_id=subject_id,
            context_snapshot_id=assignment["assignment_id"],
            comparison=assignment["comparison"],
            assignment=assignment["assignment"],
            adherence=adherence or {"treatment_delivered": True, "treatment_seen": True},
            outcomes=outcomes,
            measurement_horizons=measurement_horizons or ["10_minutes", "2_hours", "1_day"],
            preregistration_id=assignment.get("preregistration_id"),
            data_provenance={
                "context_snapshot": assignment["context_snapshot"],
                "intervened_context": intervened_context,
                "manual_or_adapter_capture": True,
            },
        )
        self.ledger.append("intervention_trial", trial, record_id=trial.trial_id)
        return trial

    def estimate_treatment_effect(
        self,
        *,
        treatment: str,
        comparator: str,
        outcome_name: str,
    ) -> dict[str, Any]:
        return TreatmentComparator(self.ledger).compare(
            treatment=treatment,
            comparator=comparator,
            outcome_name=outcome_name,
        ).to_dict()

    def launch_real_intervention(self, proposal: ExperimentProposal, *, approved_by_human: bool = False) -> dict[str, Any]:
        if not approved_by_human:
            raise PermissionError("Real interventions require explicit human approval.")
        assignment = self.assign_intervention(
            proposal.context,
            treatment=proposal.treatment,
            comparator=proposal.comparator,
            probability=0.5,
        )
        self.ledger.append(
            "real_intervention_launch",
            {"proposal": proposal.__dict__, "assignment_id": assignment["assignment_id"], "launched_at": utc_now()},
        )
        return assignment

    def _block(self, context: dict[str, Any]) -> dict[str, str]:
        fatigue = float(context.get("fatigue", 0.0))
        return {
            "time_of_day": "morning" if context.get("time_of_day_morning", 0.0) else "not_morning",
            "fatigue_band": "high" if fatigue > 0.66 else "medium" if fatigue > 0.33 else "low",
            "task_size": "large" if context.get("task_size_large", 0.0) else "small_or_medium",
        }


def _binomial_se(rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(rate * (1.0 - rate), 1e-9) / n)


class DisagreementFinder:
    def propose(
        self,
        models: list[BinaryPredictor],
        candidate_contexts: list[dict[str, Any]],
        *,
        treatment: str = "explicit_first_step",
        comparator: str = "generic_task_description",
        mode: str = "science",
        lambda_cost: float = 0.1,
        lambda_risk: float = 0.1,
        lambda_burden: float = 0.1,
    ) -> ExperimentProposal:
        best: ExperimentProposal | None = None
        for context in candidate_contexts:
            treatment_context = _apply_intervention(context, treatment)
            comparator_context = _apply_intervention(context, comparator)
            predictions: dict[str, dict[str, float]] = {}
            treatment_values = []
            comparator_values = []
            for model in models:
                pt = model.predict_proba(treatment_context)
                pc = model.predict_proba(comparator_context)
                predictions[model.model_id] = {"treatment": pt, "comparator": pc, "effect": pt - pc}
                treatment_values.append(pt)
                comparator_values.append(pc)
            if mode == "science":
                separation = max(
                    max(treatment_values) - min(treatment_values),
                    max(comparator_values) - min(comparator_values),
                    max(value["effect"] for value in predictions.values()) - min(value["effect"] for value in predictions.values()),
                )
            else:
                separation = mean(value["effect"] for value in predictions.values())
            cost = 0.2 if context.get("task_size_large", 0.0) else 0.1
            risk = 0.05
            burden = 0.15 + 0.1 * float(context.get("fatigue", 0.0))
            utility = separation - lambda_cost * cost - lambda_risk * risk - lambda_burden * burden
            proposal = ExperimentProposal(
                experiment_id=new_id("x"),
                mode=mode,
                context=dict(context),
                treatment=treatment,
                comparator=comparator,
                expected_hypothesis_separation=separation,
                cost=cost,
                risk=risk,
                participant_burden=burden,
                utility=utility,
                model_predictions=predictions,
            )
            if best is None or proposal.utility > best.utility:
                best = proposal
        if best is None:
            raise ValueError("No candidate contexts available for experiment proposal")
        return best


def _apply_intervention(context: dict[str, Any], intervention: str) -> dict[str, Any]:
    updated = dict(context)
    if intervention == "explicit_first_step":
        updated["explicit_first_step"] = 1.0
    elif intervention == "generic_task_description":
        updated["explicit_first_step"] = 0.0
    elif intervention == "visible_commitment":
        updated["public_commitment"] = 1.0
    elif intervention == "two_minute_countdown":
        updated["deadline_near"] = 1.0
    return updated
