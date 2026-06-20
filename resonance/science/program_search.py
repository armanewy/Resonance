from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from resonance.science.contracts import (
    HypothesisSpec,
    expression_metrics,
    expression_node_count,
    stable_hash,
)
from resonance.science.fitting import FitResult, FittingError, fit_hypothesis
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.science.mutation import (
    MutationConfig,
    MutationError,
    MutationOperator,
    mutate_hypothesis,
)
from resonance.science.selection import CandidateEvaluation, SelectionError, SelectionResult, select_candidate
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT, load_exploration_view, load_snapshot_manifest
from resonance.time_utils import parse_utc


EVALUATOR_VERSION = "bounded-program-search-v1"
DEFAULT_BUDGET = 100
DEFAULT_BEAM_WIDTH = 10
# Program search is exploratory and deliberately budgeted. Stop after the
# first generation that fails to improve rather than burning the full default
# budget on cosmetically different expressions.
DEFAULT_STALL_ROUNDS = 1
DEFAULT_IMPROVEMENT_EPSILON = 1.0e-6


class ProgramSearchError(ValueError):
    """Raised when bounded scientific program search cannot complete."""


@dataclass(frozen=True)
class SearchConfig:
    budget: int = DEFAULT_BUDGET
    beam_width: int = DEFAULT_BEAM_WIDTH
    complexity_penalty: float = 0.001
    random_seed: int = 0
    max_depth: int | None = None
    stall_rounds: int = DEFAULT_STALL_ROUNDS
    improvement_epsilon: float = DEFAULT_IMPROVEMENT_EPSILON


@dataclass(frozen=True)
class EvaluatedCandidate:
    candidate_id: str
    hypothesis: HypothesisSpec
    depth: int
    parent_candidate_id: str | None
    parent_hypothesis_hash: str | None
    fit_result: FitResult
    selection_evaluation: CandidateEvaluation
    artifact: dict[str, str]

    @property
    def adjusted_score(self) -> float:
        score = self.selection_evaluation.score
        if not math.isfinite(score):
            return -math.inf
        return float(score - self.selection_evaluation.complexity["ast_nodes"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis_hash": self.hypothesis.hypothesis_hash(),
            "depth": self.depth,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_hypothesis_hash": self.parent_hypothesis_hash,
            "fit_result_id": self.fit_result.artifact_hash(),
            "selection": self.selection_evaluation.to_dict(),
            "artifact": dict(self.artifact),
        }


@dataclass(frozen=True)
class ProgramSearchResult:
    search_id: str
    snapshot_id: str
    evaluator_version: str
    config: SearchConfig
    evaluated_count: int
    stopped_reason: str
    selected_candidate_id: str | None
    selected_hypothesis_hash: str | None
    ranking: tuple[str, ...]
    pareto_front: tuple[str, ...]
    candidates: tuple[EvaluatedCandidate, ...]
    artifacts: dict[str, dict[str, str]]
    ledger_entry_hash: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "snapshot_id": self.snapshot_id,
            "evaluator_version": self.evaluator_version,
            "config": {
                "budget": self.config.budget,
                "beam_width": self.config.beam_width,
                "complexity_penalty": self.config.complexity_penalty,
                "random_seed": self.config.random_seed,
                "max_depth": self.config.max_depth,
                "stall_rounds": self.config.stall_rounds,
                "improvement_epsilon": self.config.improvement_epsilon,
            },
            "evaluated_count": self.evaluated_count,
            "stopped_reason": self.stopped_reason,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_hypothesis_hash": self.selected_hypothesis_hash,
            "ranking": list(self.ranking),
            "pareto_front": list(self.pareto_front),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "artifacts": {name: dict(artifact) for name, artifact in self.artifacts.items()},
            "ledger_entry_hash": self.ledger_entry_hash,
            "warnings": list(self.warnings),
            "raw_blind_values_exposed": False,
        }


def run_program_search(
    seed_hypotheses: Sequence[HypothesisSpec | Mapping[str, Any]],
    *,
    snapshot_id: str,
    budget: int = DEFAULT_BUDGET,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    complexity_penalty: float = 0.001,
    random_seed: int = 0,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    record_ledger: bool = True,
) -> ProgramSearchResult:
    if not seed_hypotheses:
        raise ProgramSearchError("at least one seed hypothesis is required")
    config = _bounded_config(
        SearchConfig(
            budget=budget,
            beam_width=beam_width,
            complexity_penalty=complexity_penalty,
            random_seed=random_seed,
        )
    )
    root = Path(artifact_root)
    manifest = load_snapshot_manifest(snapshot_id, artifact_root=root)
    exploration_view = load_exploration_view(snapshot_id, artifact_root=root)
    exploration_frame = _frame_from_rows(exploration_view["rows"])
    metric_catalog = manifest.get("metric_catalog")
    seeds = tuple(_parse_hypothesis(seed, metric_catalog) for seed in seed_hypotheses)
    max_depth = min(seed.complexity_budget.max_ast_nodes for seed in seeds)
    config = SearchConfig(**{**config.__dict__, "max_depth": max_depth})
    rng = random.Random(config.random_seed)
    search_id = stable_hash(
        {
            "snapshot_id": snapshot_id,
            "seed_hashes": [seed.hypothesis_hash() for seed in seeds],
            "config": config.__dict__,
        }
    )

    evaluated: list[EvaluatedCandidate] = []
    warnings: list[str] = []
    seen_hashes: set[str] = set()
    frontier = _seed_frontier(
        seeds,
        metric_catalog=metric_catalog,
        snapshot_max_lag_seconds=int(manifest.get("max_lag_seconds", 0)),
        budget=config.budget,
        seed=config.random_seed,
    )
    best_score = -math.inf
    stalled_rounds = 0
    stopped_reason = "budget_exhausted"

    while frontier and len(evaluated) < config.budget:
        current_round: list[EvaluatedCandidate] = []
        for hypothesis, depth, parent_candidate_id, parent_hash in frontier:
            if len(evaluated) >= config.budget:
                break
            if hypothesis.hypothesis_hash() in seen_hashes:
                continue
            seen_hashes.add(hypothesis.hypothesis_hash())
            try:
                candidate = _evaluate_candidate(
                    hypothesis,
                    snapshot_id=snapshot_id,
                    root=root,
                    exploration_frame=exploration_frame,
                    depth=depth,
                    parent_candidate_id=parent_candidate_id,
                    parent_hypothesis_hash=parent_hash,
                    sequence=len(evaluated) + 1,
                    complexity_penalty=config.complexity_penalty,
                )
            except ProgramSearchError as exc:
                warnings.append(str(exc))
                continue
            evaluated.append(candidate)
            current_round.append(candidate)

        if not current_round:
            stopped_reason = "search_space_exhausted" if evaluated else "no_fit_candidates"
            break

        round_best = max((_search_score(candidate, config.complexity_penalty) for candidate in current_round), default=-math.inf)
        if round_best > best_score + config.improvement_epsilon:
            best_score = round_best
            stalled_rounds = 0
        else:
            stalled_rounds += 1
            if stalled_rounds >= config.stall_rounds:
                stopped_reason = "stalled"
                break

        if len(evaluated) >= config.budget:
            stopped_reason = "budget_exhausted"
            break
        if max(candidate.depth for candidate in evaluated) >= max_depth:
            stopped_reason = "max_depth_reached"
            break

        beam = _diverse_beam(evaluated, config.beam_width, config.complexity_penalty)
        children: list[tuple[HypothesisSpec, int, str | None, str | None]] = []
        for parent in beam:
            remaining = config.budget - len(evaluated) - len(children)
            if remaining <= 0:
                break
            try:
                mutated = mutate_hypothesis(
                    parent.hypothesis,
                    seed=rng.randrange(0, 2**31),
                    config=MutationConfig(max_children=min(config.beam_width, remaining)),
                    metric_catalog=metric_catalog,
                    snapshot_max_lag_seconds=int(manifest.get("max_lag_seconds", 0)),
                )
            except MutationError as exc:
                warnings.append(f"mutation skipped for {parent.candidate_id}: {exc}")
                continue
            for child in mutated:
                if child.hypothesis_hash() not in seen_hashes:
                    children.append(
                        (
                            child,
                            parent.depth + 1,
                            parent.candidate_id,
                            parent.hypothesis.hypothesis_hash(),
                        )
                    )
        frontier = _dedupe_frontier(children)

    if not evaluated:
        raise ProgramSearchError("; ".join(warnings) or "no candidates could be evaluated")

    final_selection = select_candidate(
        snapshot_id,
        [_selection_input(candidate) for candidate in evaluated],
        artifact_root=root,
        record_artifact=True,
    )
    ranking = tuple(_tie_aware_ranking(final_selection, config.complexity_penalty))
    selected_id = _selected_from_ranking(final_selection, ranking)
    selected_hash = next(
        (candidate.hypothesis.hypothesis_hash() for candidate in evaluated if candidate.candidate_id == selected_id),
        None,
    )
    pareto_front = _pareto_front(evaluated)
    candidate_artifacts = {candidate.candidate_id: candidate.artifact for candidate in evaluated}
    summary_payload = {
        "search_id": search_id,
        "snapshot_id": snapshot_id,
        "evaluator_version": EVALUATOR_VERSION,
        "config": config.__dict__,
        "evaluated_count": len(evaluated),
        "stopped_reason": stopped_reason,
        "selected_candidate_id": selected_id,
        "selected_hypothesis_hash": selected_hash,
        "ranking": list(ranking),
        "pareto_front": list(pareto_front),
        "candidate_artifacts": candidate_artifacts,
        "selection_artifact": final_selection.artifact,
        "raw_blind_values_exposed": False,
        "code_commit": current_code_commit(),
    }
    search_artifact = _store_json_artifact(root, summary_payload)
    artifacts = {
        "program_search": search_artifact,
        **{f"candidate:{candidate_id}": artifact for candidate_id, artifact in candidate_artifacts.items()},
    }
    if final_selection.artifact is not None:
        artifacts["selection"] = final_selection.artifact

    ledger_entry_hash: str | None = None
    if record_ledger:
        entry = append_event(
            "program_search_completed",
            {
                "search_id": search_id,
                "snapshot_id": snapshot_id,
                "artifact_root": str(root.resolve()),
                "evaluator_version": EVALUATOR_VERSION,
                "config": config.__dict__,
                "evaluated_count": len(evaluated),
                "stopped_reason": stopped_reason,
                "selected_candidate_id": selected_id,
                "selected_hypothesis_hash": selected_hash,
                "ranking": list(ranking),
                "pareto_front": list(pareto_front),
                "artifacts": artifacts,
                "lineage": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "hypothesis_hash": candidate.hypothesis.hypothesis_hash(),
                        "parent_candidate_id": candidate.parent_candidate_id,
                        "parent_hypothesis_hash": candidate.parent_hypothesis_hash,
                        "depth": candidate.depth,
                    }
                    for candidate in evaluated
                ],
                "raw_blind_values_exposed": False,
            },
            artifact_hashes={name: artifact["sha256"] for name, artifact in artifacts.items()},
            ledger_path=ledger_path,
        )
        ledger_entry_hash = entry["entry_hash"]

    return ProgramSearchResult(
        search_id=search_id,
        snapshot_id=snapshot_id,
        evaluator_version=EVALUATOR_VERSION,
        config=config,
        evaluated_count=len(evaluated),
        stopped_reason=stopped_reason,
        selected_candidate_id=selected_id,
        selected_hypothesis_hash=selected_hash,
        ranking=ranking,
        pareto_front=pareto_front,
        candidates=tuple(sorted(evaluated, key=lambda candidate: candidate.candidate_id)),
        artifacts=artifacts,
        ledger_entry_hash=ledger_entry_hash,
        warnings=tuple(warnings),
    )


def _bounded_config(config: SearchConfig) -> SearchConfig:
    if config.budget <= 0:
        raise ProgramSearchError("budget must be positive")
    if config.beam_width <= 0:
        raise ProgramSearchError("beam_width must be positive")
    if config.complexity_penalty < 0:
        raise ProgramSearchError("complexity_penalty must be non-negative")
    return SearchConfig(
        budget=min(config.budget, DEFAULT_BUDGET),
        beam_width=min(config.beam_width, DEFAULT_BEAM_WIDTH),
        complexity_penalty=config.complexity_penalty,
        random_seed=config.random_seed,
        max_depth=config.max_depth,
        stall_rounds=max(1, config.stall_rounds),
        improvement_epsilon=config.improvement_epsilon,
    )


def _parse_hypothesis(value: HypothesisSpec | Mapping[str, Any], metric_catalog: Any) -> HypothesisSpec:
    if isinstance(value, HypothesisSpec):
        value.validate_metric_catalog(metric_catalog)
        return value
    return HypothesisSpec.model_validate(value, context={"metric_catalog": metric_catalog})


def _seed_frontier(
    seeds: Sequence[HypothesisSpec],
    *,
    metric_catalog: Any,
    snapshot_max_lag_seconds: int,
    budget: int,
    seed: int,
) -> list[tuple[HypothesisSpec, int, str | None, str | None]]:
    """Include cheap lag variants before open-ended structural mutation.

    Lag is a core numerical search dimension for time-series hypotheses. A
    random subset of all mutation operators must not be able to omit every
    plausible lag and make the search miss an otherwise obvious relation.
    """

    frontier: list[tuple[HypothesisSpec, int, str | None, str | None]] = [
        (hypothesis, 0, None, None) for hypothesis in seeds
    ]
    remaining = max(0, budget - len(frontier))
    for index, hypothesis in enumerate(seeds):
        if remaining <= 0 or snapshot_max_lag_seconds <= 0:
            break
        try:
            variants = mutate_hypothesis(
                hypothesis,
                seed=seed + index,
                config=MutationConfig(
                    max_children=remaining,
                    operators=(MutationOperator.CHANGE_LAG, MutationOperator.ADD_LAG),
                ),
                metric_catalog=metric_catalog,
                snapshot_max_lag_seconds=snapshot_max_lag_seconds,
            )
        except MutationError:
            continue
        parent_hash = hypothesis.hypothesis_hash()
        for child in variants:
            frontier.append((child, 1, f"seed-{parent_hash[:12]}", parent_hash))
            remaining -= 1
            if remaining <= 0:
                break
    return _dedupe_frontier(frontier)


def _evaluate_candidate(
    hypothesis: HypothesisSpec,
    *,
    snapshot_id: str,
    root: Path,
    exploration_frame: pd.DataFrame,
    depth: int,
    parent_candidate_id: str | None,
    parent_hypothesis_hash: str | None,
    sequence: int,
    complexity_penalty: float,
) -> EvaluatedCandidate:
    candidate_id = f"search-{sequence:04d}-{hypothesis.hypothesis_hash()[:12]}"
    try:
        fit_result = fit_hypothesis(hypothesis, exploration_frame, complexity_weight=complexity_penalty)
        selection = select_candidate(
            snapshot_id,
            [
                {
                    "candidate_id": candidate_id,
                    "hypothesis": hypothesis,
                    "fit_result": {
                        "fit_result_id": fit_result.artifact_hash(),
                        "fitted_parameters": fit_result.fitted_parameters,
                    },
                }
            ],
            artifact_root=root,
            record_artifact=False,
        )
        evaluation = selection.evaluations[0]
    except (FittingError, SelectionError, ValueError) as exc:
        raise ProgramSearchError(f"candidate {candidate_id} could not be evaluated: {exc}") from exc

    payload = {
        "candidate_id": candidate_id,
        "snapshot_id": snapshot_id,
        "evaluator_version": EVALUATOR_VERSION,
        "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
        "hypothesis_hash": hypothesis.hypothesis_hash(),
        "depth": depth,
        "parent_candidate_id": parent_candidate_id,
        "parent_hypothesis_hash": parent_hypothesis_hash,
        "fit_result": fit_result.deterministic_fit_artifact,
        "selection": evaluation.to_dict(),
        "raw_blind_values_exposed": False,
    }
    artifact = _store_json_artifact(root, payload)
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        hypothesis=hypothesis,
        depth=depth,
        parent_candidate_id=parent_candidate_id,
        parent_hypothesis_hash=parent_hypothesis_hash,
        fit_result=fit_result,
        selection_evaluation=evaluation,
        artifact=artifact,
    )


def _selection_input(candidate: EvaluatedCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "hypothesis": candidate.hypothesis,
        "fit_result": {
            "fit_result_id": candidate.fit_result.artifact_hash(),
            "fitted_parameters": candidate.fit_result.fitted_parameters,
        },
    }


def _diverse_beam(
    candidates: Sequence[EvaluatedCandidate],
    beam_width: int,
    complexity_penalty: float,
) -> tuple[EvaluatedCandidate, ...]:
    ranked = sorted(candidates, key=lambda candidate: _beam_key(candidate, complexity_penalty))
    selected: list[EvaluatedCandidate] = []
    signatures: set[tuple[Any, ...]] = set()
    for candidate in ranked:
        signature = _structural_signature(candidate.hypothesis)
        if signature in signatures:
            continue
        selected.append(candidate)
        signatures.add(signature)
        if len(selected) >= beam_width:
            return tuple(selected)
    for candidate in ranked:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) >= beam_width:
                break
    return tuple(selected)


def _beam_key(candidate: EvaluatedCandidate, complexity_penalty: float) -> tuple[Any, ...]:
    evaluation = candidate.selection_evaluation
    score = _search_score(candidate, complexity_penalty)
    mae = evaluation.tuning_mae if evaluation.tuning_mae is not None else math.inf
    return (
        not evaluation.passed,
        -score,
        mae,
        expression_node_count(candidate.hypothesis.expression),
        candidate.hypothesis.hypothesis_hash(),
        candidate.candidate_id,
    )


def _search_score(candidate: EvaluatedCandidate, complexity_penalty: float) -> float:
    score = candidate.selection_evaluation.score
    if not math.isfinite(score):
        return -math.inf
    complexity = expression_node_count(candidate.hypothesis.expression)
    return float(score - (complexity_penalty * complexity))


def _structural_signature(hypothesis: HypothesisSpec) -> tuple[Any, ...]:
    expression = hypothesis.expression.model_dump(mode="json")
    return (
        expression.get("node"),
        tuple(sorted(expression_metrics(hypothesis.expression))),
        expression_node_count(hypothesis.expression),
    )


def _dedupe_frontier(
    frontier: Sequence[tuple[HypothesisSpec, int, str | None, str | None]]
) -> list[tuple[HypothesisSpec, int, str | None, str | None]]:
    seen: set[str] = set()
    deduped = []
    for item in frontier:
        digest = item[0].hypothesis_hash()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(item)
    return deduped


def _tie_aware_ranking(selection: SelectionResult, complexity_penalty: float) -> tuple[str, ...]:
    by_id = {evaluation.candidate_id: evaluation for evaluation in selection.evaluations}

    def key(candidate_id: str) -> tuple[Any, ...]:
        evaluation = by_id[candidate_id]
        score = evaluation.score if math.isfinite(evaluation.score) else -math.inf
        adjusted = score - (complexity_penalty * evaluation.complexity["ast_nodes"])
        mae = evaluation.tuning_mae if evaluation.tuning_mae is not None else math.inf
        rmse = evaluation.tuning_rmse if evaluation.tuning_rmse is not None else math.inf
        return (
            not evaluation.passed,
            -round(adjusted, 6),
            round(mae, 6) if math.isfinite(mae) else mae,
            round(rmse, 6) if math.isfinite(rmse) else rmse,
            evaluation.complexity["ast_nodes"],
            evaluation.hypothesis_hash,
            candidate_id,
        )

    return tuple(sorted(selection.ranking, key=key))


def _selected_from_ranking(selection: SelectionResult, ranking: Sequence[str]) -> str | None:
    by_id = {evaluation.candidate_id: evaluation for evaluation in selection.evaluations}
    for candidate_id in ranking:
        if by_id[candidate_id].passed:
            return candidate_id
    return None


def _pareto_front(candidates: Sequence[EvaluatedCandidate]) -> tuple[str, ...]:
    front: list[EvaluatedCandidate] = []
    for candidate in candidates:
        mae = candidate.selection_evaluation.tuning_mae
        if mae is None:
            continue
        complexity = expression_node_count(candidate.hypothesis.expression)
        dominated = False
        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            other_mae = other.selection_evaluation.tuning_mae
            if other_mae is None:
                continue
            other_complexity = expression_node_count(other.hypothesis.expression)
            if (
                other_mae <= mae
                and other_complexity <= complexity
                and (other_mae < mae or other_complexity < complexity)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(
        key=lambda candidate: (
            candidate.selection_evaluation.tuning_mae,
            expression_node_count(candidate.hypothesis.expression),
            candidate.hypothesis.hypothesis_hash(),
            candidate.candidate_id,
        )
    )
    return tuple(candidate.candidate_id for candidate in front)


def _frame_from_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"timestamp_utc": parse_utc(str(row["timestamp_utc"]))}
        for metric, observations in row.get("metrics", {}).items():
            values = [
                float(observation["value"])
                for observation in observations
                if _is_finite_number(observation.get("value"))
            ]
            if values:
                record[str(metric)] = sum(values) / len(values)
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records).set_index("timestamp_utc").sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path = root / "sha256" / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProgramSearchError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": f"sha256/{digest[:2]}/{digest}.json", "format": "json"}


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def load_seed_hypotheses(paths: Sequence[str | Path]) -> tuple[HypothesisSpec, ...]:
    hypotheses = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            hypotheses.extend(HypothesisSpec.model_validate(item) for item in payload)
        else:
            hypotheses.append(HypothesisSpec.model_validate(payload))
    return tuple(hypotheses)


def read_json_gz_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))


__all__ = [
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_BUDGET",
    "EVALUATOR_VERSION",
    "EvaluatedCandidate",
    "ProgramSearchError",
    "ProgramSearchResult",
    "SearchConfig",
    "load_seed_hypotheses",
    "run_program_search",
]
