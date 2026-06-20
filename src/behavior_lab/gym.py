from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from behavior_lab.core import HypothesisSpec, utc_now
from behavior_lab.evaluation import evaluate_model, pareto_frontier
from behavior_lab.ledger import ImmutableLedger
from behavior_lab.models import LogisticFormulaHypothesis, ModelFoundry
from behavior_lab.temporal import split_rows, supervised_rows
from behavior_lab.worlds import HiddenWorld, make_world


TARGET = "started_within_10_minutes"


class BlindEvaluationServer:
    def __init__(
        self,
        splits: dict[str, list[dict[str, Any]]],
        target_name: str = TARGET,
        frozen_candidates: set[str] | None = None,
    ):
        self.splits = splits
        self.target_name = target_name
        self._frozen_candidates: set[str] = set(frozen_candidates or set())

    def query_training_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.splits.get("training", [])
        visible = rows[:limit] if limit else rows
        return [{"case_id": row["case_id"], "features": row["features"], "target": row["target"]} for row in visible]

    def evaluate(self, model: Any, split: str = "development") -> dict[str, Any]:
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
        if split == "prospective" and model.model_id not in self._frozen_candidates:
            raise PermissionError("Prospective evaluation requires a frozen candidate.")
        include_details = split == "development"
        metrics = evaluate_model(model, self.splits[split], split=split, include_details=include_details)
        payload = asdict(metrics)
        if split in {"hidden", "prospective"}:
            payload["details"] = {"redacted": "hidden labels and failure rows are not exposed"}
        return payload

    def submit_frozen_candidate(self, model: Any) -> dict[str, Any]:
        self._frozen_candidates.add(model.model_id)
        return self.evaluate(model, split="prospective")


class WorldGym:
    def __init__(self, data_dir: str | Path, world: HiddenWorld | None = None, target_name: str = TARGET):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ImmutableLedger(self.data_dir / "ledger.jsonl")
        self.world = world or make_world("habit")
        self.target_name = target_name

    def seed(self, episodes: int) -> int:
        generated = self.world.generate_dataset(episodes)
        for episode in generated:
            self.ledger.append("decision_episode", episode, record_id=episode.episode_id)
        return len(generated)

    def decision_episodes(self) -> list[dict[str, Any]]:
        return self.ledger.payloads("decision_episode")

    def intervention_trial_rows(self) -> list[dict[str, Any]]:
        rows = []
        for trial in self.ledger.payloads("intervention_trial"):
            outcomes = trial.get("outcomes", {})
            if self.target_name not in outcomes and "started_within_10_minutes" not in outcomes:
                continue
            provenance = trial.get("data_provenance", {})
            features = provenance.get("intervened_context") or provenance.get("context_snapshot")
            if not features:
                continue
            target_value = outcomes.get(self.target_name, outcomes.get("started_within_10_minutes"))
            rows.append(
                {
                    "case_id": trial["trial_id"],
                    "decision_time": trial.get("recorded_at", trial["trial_id"]),
                    "features": dict(features, bias=1.0),
                    "target": 1 if target_value else 0,
                    "snapshot": {"trial_id": trial["trial_id"], "pre_decision_context": features},
                }
            )
        return rows

    def rows(self) -> list[dict[str, Any]]:
        rows = supervised_rows(self.decision_episodes(), self.target_name)
        rows.extend(self.intervention_trial_rows())
        rows.sort(key=lambda item: item["decision_time"])
        return rows

    def splits(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.rows()
        assignments = self.ensure_split_manifest(rows)
        grouped: dict[str, list[dict[str, Any]]] = {"training": [], "development": [], "hidden": [], "prospective": []}
        for row in rows:
            split = assignments.get(row["case_id"])
            if split:
                grouped.setdefault(split, []).append(row)
        return grouped

    def split_assignments(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for payload in self.ledger.payloads("split_assignment"):
            case_id = payload.get("case_id") or payload.get("episode_id")
            if case_id and payload.get("split"):
                assignments[str(case_id)] = str(payload["split"])
        return assignments

    def ensure_split_manifest(self, rows: list[dict[str, Any]] | None = None) -> dict[str, str]:
        """Create append-only split assignments without migrating existing cases.

        Initial datasets use the old temporal fractions once. New rows are assigned
        to training until a model has been frozen, after which new rows become
        prospective by definition.
        """

        rows = rows if rows is not None else self.rows()
        assignments = self.split_assignments()
        missing = [row for row in rows if row["case_id"] not in assignments]
        if not missing:
            return assignments

        if not assignments:
            initial = split_rows(rows)
            for split, split_rows_ in initial.items():
                for row in split_rows_:
                    self._record_split_assignment(row["case_id"], split, "temporal_fraction_manifest_v1")
                    assignments[row["case_id"]] = split
            return assignments

        has_frozen_candidate = bool(self.ledger.payloads("frozen_candidate"))
        split = "prospective" if has_frozen_candidate else "training"
        policy = "post_freeze_prospective_v1" if has_frozen_candidate else "pre_freeze_append_training_v1"
        for row in missing:
            self._record_split_assignment(row["case_id"], split, policy)
            assignments[row["case_id"]] = split
        return assignments

    def _record_split_assignment(self, case_id: str, split: str, policy_version: str) -> None:
        self.ledger.append(
            "split_assignment",
            {
                "episode_id": case_id,
                "case_id": case_id,
                "split": split,
                "assigned_at": utc_now(),
                "split_policy_version": policy_version,
            },
            record_id=f"split_{case_id}",
        )

    def blind_server(self) -> BlindEvaluationServer:
        return BlindEvaluationServer(self.splits(), self.target_name, self.frozen_model_ids())

    def frozen_model_ids(self) -> set[str]:
        return {payload["model_id"] for payload in self.ledger.payloads("frozen_candidate") if "model_id" in payload}

    def fit_hypothesis(self, spec: HypothesisSpec) -> Any:
        return LogisticFormulaHypothesis(spec).fit(self.splits()["training"])

    def fit_model_zoo(self) -> list[Any]:
        splits = self.splits()
        return ModelFoundry().fit_zoo(splits["training"], splits["development"], self.target_name)

    def leaderboard(self, split: str = "development") -> list[dict[str, Any]]:
        server = self.blind_server()
        results = [server.evaluate(model, split=split) for model in self.fit_model_zoo()]
        results.sort(key=lambda item: item["log_loss"])
        return results

    def complexity_frontier(self) -> list[dict[str, Any]]:
        splits = self.splits()
        models = self.fit_model_zoo()
        metrics = [evaluate_model(model, splits["development"], split="development") for model in models]
        return pareto_frontier(metrics)
