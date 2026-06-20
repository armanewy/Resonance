from __future__ import annotations

from dataclasses import asdict
from typing import Any

from behavior_lab.core import EvaluationMetrics, FittedHypothesisRecord, HypothesisSpec, new_id, to_jsonable, utc_now
from behavior_lab.ledger import ImmutableLedger


class EvaluationBudgetError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, ledger: ImmutableLedger):
        self.ledger = ledger

    def submit_hypothesis(self, spec: HypothesisSpec) -> dict[str, Any]:
        return self.ledger.append("hypothesis", asdict(spec), record_id=spec.hypothesis_id)

    def record_fit(
        self,
        model: Any,
        hypothesis_id: str,
        training_split: str,
        training_cases: int,
        artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = getattr(model, "parameters", {})
        record = FittedHypothesisRecord(
            model_id=model.model_id,
            hypothesis_id=hypothesis_id,
            fitted_at=utc_now(),
            training_split=training_split,
            training_cases=training_cases,
            parameters=parameters,
            artifact=artifact or {"class": type(model).__name__},
        )
        return self.ledger.append("model_fit", asdict(record), record_id=model.model_id)

    def record_evaluation(self, metrics: EvaluationMetrics) -> dict[str, Any]:
        return self.ledger.append("evaluation", asdict(metrics), record_id=f"{metrics.model_id}_{metrics.split}")

    def record_evaluation_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.ledger.append("evaluation", payload, record_id=f"{payload['model_id']}_{payload['split']}")

    def assert_evaluation_budget_available(self, *, campaign_id: str, split: str, limit: int = 1) -> None:
        uses = [
            payload
            for payload in self.ledger.payloads("evaluation_budget_use")
            if payload.get("campaign_id") == campaign_id and payload.get("split") == split
        ]
        if len(uses) >= limit:
            raise EvaluationBudgetError(
                f"Evaluation budget exhausted for campaign {campaign_id!r} on split {split!r}; "
                f"limit is {limit}"
            )

    def record_evaluation_budget_use(self, *, campaign_id: str, model_id: str, split: str, limit: int = 1) -> dict[str, Any]:
        self.assert_evaluation_budget_available(campaign_id=campaign_id, split=split, limit=limit)
        return self.ledger.append(
            "evaluation_budget_use",
            {
                "budget_use_id": new_id("budget"),
                "campaign_id": campaign_id,
                "model_id": model_id,
                "split": split,
                "limit": limit,
                "used_at": utc_now(),
            },
        )

    def freeze_candidate(self, model_id: str, split: str, reason: str, campaign_id: str | None = None) -> dict[str, Any]:
        return self.ledger.append(
            "frozen_candidate",
            {"model_id": model_id, "split": split, "frozen_at": utc_now(), "reason": reason, "campaign_id": campaign_id},
        )

    def promote_hypothesis(self, hypothesis_id: str, model_id: str, reason: str) -> dict[str, Any]:
        return self.ledger.append(
            "hypothesis_status",
            {
                "hypothesis_id": hypothesis_id,
                "model_id": model_id,
                "status": "promoted",
                "reason": reason,
                "written_at": utc_now(),
            },
        )

    def retire_hypothesis(self, hypothesis_id: str, reason: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.ledger.append(
            "hypothesis_status",
            {
                "hypothesis_id": hypothesis_id,
                "status": "retired",
                "reason": reason,
                "evidence": evidence or {},
                "written_at": utc_now(),
            },
        )

    def model_obituary(self, hypothesis_id: str, body: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return self.ledger.append(
            "model_obituary",
            {
                "hypothesis_id": hypothesis_id,
                "body": body,
                "evidence": evidence,
                "written_at": utc_now(),
            },
        )

    def inspect_model_registry(self) -> dict[str, Any]:
        return {
            "hypotheses": self.ledger.payloads("hypothesis"),
            "fits": self.ledger.payloads("model_fit"),
            "evaluations": self.ledger.payloads("evaluation"),
            "evaluation_budget_uses": self.ledger.payloads("evaluation_budget_use"),
            "status_events": self.ledger.payloads("hypothesis_status"),
            "frozen_candidates": self.ledger.payloads("frozen_candidate"),
        }

    def lineage_graph(self) -> dict[str, Any]:
        nodes = {}
        edges = []
        for hypothesis in self.ledger.payloads("hypothesis"):
            hypothesis_id = hypothesis["hypothesis_id"]
            nodes[hypothesis_id] = hypothesis
            for parent in hypothesis.get("parent_ids", []):
                edges.append({"from": parent, "to": hypothesis_id, "kind": "parent"})
        for status in self.ledger.payloads("hypothesis_status"):
            nodes.setdefault(status["hypothesis_id"], {})["latest_status"] = status["status"]
        return {"nodes": to_jsonable(nodes), "edges": edges}
