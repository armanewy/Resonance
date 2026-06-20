from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from resonance.science.contracts import (
    HypothesisSpec,
    Origin,
    expression_lag_seconds,
    expression_metrics,
    stable_hash,
)
from resonance.science.discovery_brief import discovery_brief_from_exploration_view
from resonance.science.fitting import EVALUATOR_VERSION as FITTING_VERSION
from resonance.science.fitting import FittingError, fit_hypothesis
from resonance.science.interpreter import frame_from_snapshot_rows
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.science.providers import MockProvider, run_provider
from resonance.science.selection import EVALUATOR_VERSION as SELECTION_VERSION
from resonance.science.selection import select_candidate
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT, create_snapshot, load_exploration_view
from resonance.storage import Measurement, init_db, insert_measurements
from resonance.synthetic import DEFAULT_SEED, LAGGED_SCENARIOS, generate_synthetic_series


ABLATION_VERSION = "science-llm-ablation-v1"
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_SCENARIOS = ("strong_lag", "shared_seasonality_only")
DEFAULT_CANDIDATE_BUDGET = 6
DEFAULT_DURATION_HOURS = 96.0
DEFAULT_SNAPSHOT_HOURS = 240
DEFAULT_SAMPLE_INTERVAL_SECONDS = 300
DEFAULT_MAX_LAG_SECONDS = 900
DEFAULT_NOISE = 0.6


class AblationError(RuntimeError):
    """Raised when the LLM ablation experiment cannot proceed."""


def run_ablation(
    *,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    provider_name: str = "mock",
    seed: int = DEFAULT_SEED,
    candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    duration_hours: float = DEFAULT_DURATION_HOURS,
    snapshot_hours: int = DEFAULT_SNAPSHOT_HOURS,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    max_lag_seconds: int = DEFAULT_MAX_LAG_SECONDS,
    noise: float = DEFAULT_NOISE,
) -> dict[str, Any]:
    """Compare mock LLM hypothesis generation with baseline generators.

    The default path intentionally evaluates on exploration and tuning only.
    Blind data is neither loaded nor exposed; each scenario receives its own
    fresh sealed synthetic snapshot so future blind extensions can preregister
    from a clean snapshot.
    """

    if provider_name != "mock":
        raise AblationError("ablation currently supports --provider mock only")
    if candidate_budget < 1:
        raise AblationError("candidate_budget must be positive")
    if candidate_budget > 8:
        raise AblationError("candidate_budget must be at most 8")

    root = Path(artifact_root)
    ledger = Path(ledger_path)
    scenario_results = [
        _run_scenario(
            scenario=scenario,
            provider_name=provider_name,
            seed=seed + index * 10_000,
            candidate_budget=candidate_budget,
            artifact_root=root,
            ledger_path=ledger,
            duration_hours=duration_hours,
            snapshot_hours=snapshot_hours,
            sample_interval_seconds=sample_interval_seconds,
            max_lag_seconds=max_lag_seconds,
            noise=noise,
        )
        for index, scenario in enumerate(scenarios)
    ]
    conclusion = _value_conclusion(scenario_results)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "science_llm_ablation_report",
        "ablation_version": ABLATION_VERSION,
        "provider": provider_name,
        "seed": int(seed),
        "candidate_budget": int(candidate_budget),
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "llm_adds_value": conclusion["llm_adds_value"],
        "conclusion": conclusion["conclusion"],
        "blind_evaluation": {
            "run": False,
            "reason": "Default ablation path is tuning-only; blind data was not loaded or exposed.",
        },
        "raw_blind_values_exposed": False,
        "code_commit": current_code_commit(),
    }
    artifact = _store_json_artifact(root, payload)
    append_event(
        "experiment_completed",
        {
            "experiment_type": "science_llm_ablation",
            "ablation_version": ABLATION_VERSION,
            "provider": provider_name,
            "seed": int(seed),
            "candidate_budget": int(candidate_budget),
            "scenario_count": len(scenario_results),
            "snapshot_ids": [result["snapshot_id"] for result in scenario_results],
            "llm_adds_value": conclusion["llm_adds_value"],
            "conclusion": conclusion["conclusion"],
            "raw_blind_values_exposed": False,
            "artifact_root": str(root.resolve()),
            "artifacts": {"ablation_report": artifact},
        },
        artifact_hashes={"ablation_report": artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger,
    )
    return {
        "run_id": artifact["sha256"],
        "ablation_version": ABLATION_VERSION,
        "provider": provider_name,
        "seed": int(seed),
        "candidate_budget": int(candidate_budget),
        "scenarios": scenario_results,
        "llm_adds_value": conclusion["llm_adds_value"],
        "conclusion": conclusion["conclusion"],
        "blind_evaluation": payload["blind_evaluation"],
        "raw_blind_values_exposed": False,
        "artifact": artifact,
    }


def _run_scenario(
    *,
    scenario: str,
    provider_name: str,
    seed: int,
    candidate_budget: int,
    artifact_root: Path,
    ledger_path: Path,
    duration_hours: float,
    snapshot_hours: int,
    sample_interval_seconds: int,
    max_lag_seconds: int,
    noise: float,
) -> dict[str, Any]:
    db_path = artifact_root.parent / "ablation" / f"{scenario}-{seed}.db"
    synthetic_metadata = _write_synthetic_database(
        scenario=scenario,
        db_path=db_path,
        duration_hours=duration_hours,
        sample_interval_seconds=sample_interval_seconds,
        noise=noise,
        seed=seed,
    )
    manifest = create_snapshot(
        db_path=db_path,
        hours=snapshot_hours,
        metrics=["control", "x", "y"],
        max_lag_seconds=max_lag_seconds,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
    )
    exploration = load_exploration_view(manifest["snapshot_id"], artifact_root=artifact_root)
    frame = frame_from_snapshot_rows(exploration["rows"])
    catalog_id = str(manifest["metric_catalog"]["catalog_id"])
    true_lag = synthetic_metadata.get("true_lag_seconds")
    generator_specs = _generator_specs(
        provider_name=provider_name,
        manifest=manifest,
        exploration=exploration,
        seed=seed,
        candidate_budget=candidate_budget,
    )
    generator_results = [
        _evaluate_generator(
            scenario=scenario,
            snapshot_id=manifest["snapshot_id"],
            generator=generator,
            true_lag_seconds=true_lag,
            exploration_frame=frame,
            artifact_root=artifact_root,
            ledger_path=ledger_path,
        )
        for generator in generator_specs
    ]
    return {
        "scenario": scenario,
        "snapshot_id": manifest["snapshot_id"],
        "synthetic_metadata": synthetic_metadata,
        "known_positive": scenario in LAGGED_SCENARIOS,
        "expected_relation": _expected_relation(scenario, true_lag),
        "generators": generator_results,
        "best_generator": _best_generator_name(generator_results),
        "fresh_snapshot_per_evaluation": True,
        "raw_blind_values_exposed": False,
    }


def _generator_specs(
    *,
    provider_name: str,
    manifest: Mapping[str, Any],
    exploration: Mapping[str, Any],
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    catalog_id = str(manifest["metric_catalog"]["catalog_id"])
    max_lag_seconds = int(manifest["max_lag_seconds"])
    brief = discovery_brief_from_exploration_view(
        exploration,
        metric_catalog=manifest["metric_catalog"],
    )
    llm_raw = _mock_provider_proposals(catalog_id, max_lag_seconds, seed, candidate_budget)
    provider = MockProvider(
        llm_raw,
        name=provider_name,
        request_config={"candidate_budget": candidate_budget, "blind_data_visible": False},
    )
    provider_run = run_provider(
        provider,
        brief,
        max_hypotheses=candidate_budget,
        seed=seed,
    )
    return [
        {
            "name": "llm_mock",
            "kind": "llm",
            "candidate_budget": candidate_budget,
            "hypotheses": list(provider_run.hypotheses),
            "invalid_proposals": len(provider_run.rejected_proposals),
            "provider_metadata": provider_run.metadata.model_dump(mode="json"),
            "cost_metadata": {
                "total_cost_usd": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "explanation": "MockProvider is deterministic and performs no billable network call.",
            },
        },
        _baseline_generator(
            "pairwise_lag",
            _pairwise_lag_hypotheses(catalog_id, max_lag_seconds, seed + 101, candidate_budget),
            candidate_budget,
        ),
        _baseline_generator(
            "random_dsl",
            _random_dsl_hypotheses(catalog_id, max_lag_seconds, seed + 202, candidate_budget),
            candidate_budget,
        ),
        _baseline_generator(
            "linear_combo",
            _linear_combo_hypotheses(catalog_id, max_lag_seconds, seed + 303, candidate_budget),
            candidate_budget,
        ),
        _baseline_generator(
            "persistence_zero_residual",
            _persistence_zero_hypotheses(catalog_id, max_lag_seconds, seed + 404, candidate_budget),
            candidate_budget,
        ),
    ]


def _baseline_generator(
    name: str,
    raw_hypotheses: Sequence[Mapping[str, Any]],
    candidate_budget: int,
) -> dict[str, Any]:
    accepted: list[HypothesisSpec] = []
    invalid = 0
    for proposal in raw_hypotheses:
        try:
            accepted.append(HypothesisSpec.model_validate(proposal))
        except ValidationError:
            invalid += 1
    return {
        "name": name,
        "kind": "baseline",
        "candidate_budget": candidate_budget,
        "hypotheses": accepted,
        "invalid_proposals": invalid,
        "provider_metadata": None,
        "cost_metadata": {
            "total_cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "explanation": "Deterministic baseline generator; no external provider cost.",
        },
    }


def _evaluate_generator(
    *,
    scenario: str,
    snapshot_id: str,
    generator: Mapping[str, Any],
    true_lag_seconds: int | None,
    exploration_frame: Any,
    artifact_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    fit_failures = 0
    for index, hypothesis in enumerate(generator["hypotheses"]):
        candidate_id = stable_hash(
            {
                "ablation_version": ABLATION_VERSION,
                "snapshot_id": snapshot_id,
                "generator": generator["name"],
                "index": index,
                "hypothesis_hash": hypothesis.hypothesis_hash(),
            }
        )
        try:
            fit_result = fit_hypothesis(hypothesis, exploration_frame)
        except FittingError:
            fit_failures += 1
            continue
        fit_artifact = _store_json_artifact(
            artifact_root,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "record_type": "ablation_fit_result",
                "ablation_version": ABLATION_VERSION,
                "generator": generator["name"],
                "snapshot_id": snapshot_id,
                "candidate_id": candidate_id,
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
                "fit_result": asdict(fit_result),
                "code_commit": current_code_commit(),
            },
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "hypothesis": hypothesis.model_dump(mode="json", exclude_none=True),
                "fitted_parameters": fit_result.fitted_parameters,
                "fit_result": {"fit_result_id": fit_artifact["sha256"]},
            }
        )
        fit_records.append(
            {
                "candidate_id": candidate_id,
                "fit_result_id": fit_artifact["sha256"],
                "hypothesis_hash": hypothesis.hypothesis_hash(),
                "exploration_metrics": fit_result.exploration_metrics,
                "complexity": fit_result.complexity,
                "recovered_expected_relation": _recovers_expected_relation(
                    scenario,
                    hypothesis,
                    true_lag_seconds,
                ),
                "artifact": fit_artifact,
            }
        )

    if candidates:
        selection = select_candidate(
            snapshot_id,
            candidates,
            artifact_root=artifact_root,
            record_artifact=True,
        )
        evaluations = [evaluation.to_dict() for evaluation in selection.evaluations]
        selected_id = selection.selected_candidate_id
        selected_evaluation = next(
            (evaluation for evaluation in evaluations if evaluation["candidate_id"] == selected_id),
            None,
        )
        selection_artifact = selection.artifact
        selected_hypothesis = next(
            (
                item["hypothesis"]
                for item in candidates
                if item["candidate_id"] == selected_id
            ),
            None,
        )
        selected_recovers = (
            _recovers_expected_relation(
                scenario,
                HypothesisSpec.model_validate(selected_hypothesis),
                true_lag_seconds,
            )
            if selected_hypothesis is not None
            else False
        )
        tuning_summary = _tuning_summary(selected_evaluation)
    else:
        evaluations = []
        selected_id = None
        selected_recovers = False
        tuning_summary = None
        selection_artifact = None

    result = {
        "scenario": scenario,
        "generator_name": generator["name"],
        "generator_kind": generator["kind"],
        "candidate_budget": int(generator["candidate_budget"]),
        "valid_candidates": len(generator["hypotheses"]),
        "fitted_candidates": len(candidates),
        "invalid_proposals": int(generator["invalid_proposals"]) + fit_failures,
        "selected_candidate_id": selected_id,
        "recovery": {
            "known_positive": scenario in LAGGED_SCENARIOS,
            "expected_relation_recovered": selected_recovers,
            "any_candidate_recovered_expected_relation": any(
                record["recovered_expected_relation"] for record in fit_records
            ),
        },
        "false_positive": bool(scenario not in LAGGED_SCENARIOS and selected_id is not None),
        "tuning_performance": tuning_summary,
        "blind_performance": None,
        "blind_performance_explanation": "Blind evaluation was not run in the default ablation path.",
        "complexity": _complexity_summary(evaluations),
        "cost_metadata": generator["cost_metadata"],
        "provider_metadata": generator["provider_metadata"],
        "fit_results": fit_records,
        "tuning_evaluations": evaluations,
        "selection_artifact": selection_artifact,
        "fitting_evaluator_version": FITTING_VERSION,
        "selection_evaluator_version": SELECTION_VERSION,
        "raw_blind_values_exposed": False,
    }
    artifact = _store_json_artifact(
        artifact_root,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "record_type": "ablation_generator_result",
            "ablation_version": ABLATION_VERSION,
            **result,
            "code_commit": current_code_commit(),
        },
    )
    append_event(
        "result_interpreted",
        {
            "interpretation_type": "ablation_generator_tuning_selection",
            "ablation_version": ABLATION_VERSION,
            "dataset_snapshot_id": snapshot_id,
            "scenario": scenario,
            "generator_name": generator["name"],
            "selected_candidate_id": selected_id,
            "false_positive": result["false_positive"],
            "expected_relation_recovered": selected_recovers,
            "invalid_proposals": result["invalid_proposals"],
            "raw_blind_values_exposed": False,
            "artifact_root": str(artifact_root.resolve()),
            "artifacts": {"ablation_generator_result": artifact},
        },
        artifact_hashes={"ablation_generator_result": artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
    )
    return {**result, "artifact": artifact}


def _mock_provider_proposals(
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    proposals = [
        _lagged_linear_hypothesis(
            title="Mock LLM lagged x predicts y",
            source="x",
            lag_seconds=max_lag_seconds,
            catalog_id=catalog_id,
            seed=seed,
            origin=Origin.LLM,
            rationale="Deterministic mock LLM proposal used for ablation comparison.",
        ),
        _lagged_linear_hypothesis(
            title="Mock LLM contemporaneous x predicts y",
            source="x",
            lag_seconds=0,
            catalog_id=catalog_id,
            seed=seed + 1,
            origin=Origin.LLM,
            rationale="Deterministic mock LLM alternative with no lag.",
        ),
        _lagged_linear_hypothesis(
            title="Mock LLM control predicts y",
            source="control",
            lag_seconds=max_lag_seconds,
            catalog_id=catalog_id,
            seed=seed + 2,
            origin=Origin.LLM,
            rationale="Deterministic mock LLM negative-control-like alternative.",
        ),
        {
            "schema_version": "1.0",
            "hypothesis_type": "observational_prediction",
            "title": "Invalid mock target leakage",
            "concise_claim": "Invalid proposal intentionally uses y as input.",
            "rationale": "Invalid proposal used to count provider rejections.",
            "target_metric": "y",
            "input_metrics": ["y"],
            "target_transform": "identity",
            "expression": {"node": "metric", "metric": "y"},
            "parameter_bounds": {},
            "expected_direction": "positive",
            "maximum_lag_seconds": 0,
            "fitting_metric": "rmse",
            "tuning_metric": "rmse",
            "blind_metrics": ["mae", "rmse", "spearman_r"],
            "minimum_blind_effect": 0.1,
            "minimum_baseline_improvement": 0.05,
            "negative_controls": [],
            "falsification_conditions": [{"description": "Invalid proposal must be rejected."}],
            "complexity_budget": {"max_ast_nodes": 8, "max_source_metrics": 1},
            "origin": Origin.LLM.value,
            "parent_hypothesis_ids": [],
            "snapshot_metric_catalog_id": catalog_id,
            "random_seed": seed + 3,
        },
    ]
    return proposals[:candidate_budget]


def _pairwise_lag_hypotheses(
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    lags = _lag_grid(max_lag_seconds)
    proposals = [
        _lagged_linear_hypothesis(
            title=f"Pairwise {source} lag {lag_seconds}s predicts y",
            source=source,
            lag_seconds=lag_seconds,
            catalog_id=catalog_id,
            seed=seed + index,
            origin=Origin.BASELINE,
            rationale="Pairwise lag baseline with fitted scale and offset.",
        )
        for index, (source, lag_seconds) in enumerate(
            (source, lag_seconds)
            for source in ("x", "control")
            for lag_seconds in lags
        )
    ]
    return proposals[:candidate_budget]


def _random_dsl_hypotheses(
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    proposals: list[dict[str, Any]] = []
    for index in range(candidate_budget):
        source = rng.choice(("x", "control"))
        lag_seconds = rng.choice(_lag_grid(max_lag_seconds))
        base: dict[str, Any] = {
            "node": "lag",
            "input": {"node": "metric", "metric": source},
            "lag_seconds": lag_seconds,
        }
        variant = index % 4
        if variant == 1:
            base = {
                "node": "rolling_mean",
                "input": base,
                "window_seconds": max(DEFAULT_SAMPLE_INTERVAL_SECONDS, max_lag_seconds // 3 or 300),
                "min_periods": 1,
            }
        elif variant == 2 and lag_seconds > 0:
            base = {
                "node": "difference",
                "input": {"node": "metric", "metric": source},
                "period_seconds": min(lag_seconds, max_lag_seconds),
            }
        elif variant == 3:
            base = {"node": "absolute_value", "input": base}
        expression = _scale_offset_expression(base)
        proposals.append(
            _hypothesis(
                title=f"Random DSL {index + 1} predicts y",
                concise_claim=f"A generated expression over {source} is associated with y.",
                rationale="Seeded random DSL baseline with controlled AST complexity.",
                target_metric="y",
                input_metrics=[source],
                expression=expression,
                parameter_bounds=_scale_offset_bounds(),
                maximum_lag_seconds=max_lag_seconds,
                catalog_id=catalog_id,
                seed=seed + index,
                origin=Origin.BASELINE,
                max_ast_nodes=8,
            )
        )
    return proposals


def _linear_combo_hypotheses(
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    proposals = [
        _linear_combination_hypothesis(
            title="Linear x predicts y",
            sources=("x",),
            catalog_id=catalog_id,
            max_lag_seconds=max_lag_seconds,
            seed=seed,
        ),
        _linear_combination_hypothesis(
            title="Linear control predicts y",
            sources=("control",),
            catalog_id=catalog_id,
            max_lag_seconds=max_lag_seconds,
            seed=seed + 1,
        ),
        _linear_combination_hypothesis(
            title="Linear x plus control predicts y",
            sources=("x", "control"),
            catalog_id=catalog_id,
            max_lag_seconds=max_lag_seconds,
            seed=seed + 2,
        ),
    ]
    return proposals[:candidate_budget]


def _persistence_zero_hypotheses(
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    proposals = [
        _hypothesis(
            title="Zero level baseline predicts y",
            concise_claim="The y level is predicted by zero.",
            rationale="Zero residual baseline candidate for ablation comparison.",
            target_metric="y",
            input_metrics=["x"],
            target_transform="identity",
            expression={"node": "numeric_constant", "value": 0.0},
            parameter_bounds={},
            expected_direction="nonzero",
            maximum_lag_seconds=max_lag_seconds,
            catalog_id=catalog_id,
            seed=seed,
            origin=Origin.BASELINE,
            max_ast_nodes=1,
        ),
        _hypothesis(
            title="Persistence zero residual predicts y",
            concise_claim="The y first difference is predicted by zero.",
            rationale="Persistence baseline expressed as a zero residual on the target difference.",
            target_metric="y",
            input_metrics=["x"],
            target_transform="difference",
            expression={"node": "numeric_constant", "value": 0.0},
            parameter_bounds={},
            expected_direction="nonzero",
            maximum_lag_seconds=max_lag_seconds,
            catalog_id=catalog_id,
            seed=seed + 1,
            origin=Origin.BASELINE,
            max_ast_nodes=1,
        ),
    ]
    return proposals[:candidate_budget]


def _lagged_linear_hypothesis(
    *,
    title: str,
    source: str,
    lag_seconds: int,
    catalog_id: str,
    seed: int,
    origin: Origin,
    rationale: str,
) -> dict[str, Any]:
    return _hypothesis(
        title=title,
        concise_claim=f"Lagged {source} is associated with y.",
        rationale=rationale,
        target_metric="y",
        input_metrics=[source],
        expression=_scale_offset_expression(
            {
                "node": "lag",
                "input": {"node": "metric", "metric": source},
                "lag_seconds": int(lag_seconds),
            }
        ),
        parameter_bounds=_scale_offset_bounds(),
        maximum_lag_seconds=lag_seconds,
        catalog_id=catalog_id,
        seed=seed,
        origin=origin,
        max_ast_nodes=8,
    )


def _linear_combination_hypothesis(
    *,
    title: str,
    sources: Sequence[str],
    catalog_id: str,
    max_lag_seconds: int,
    seed: int,
) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    bounds: dict[str, dict[str, float]] = {}
    for index, source in enumerate(sources, start=1):
        parameter = f"scale_{index}"
        terms.append(
            {
                "node": "multiply",
                "left": {"node": "fitted_parameter", "parameter": parameter},
                "right": {"node": "metric", "metric": source},
            }
        )
        bounds[parameter] = {"lower": -5.0, "upper": 5.0}
    expression = terms[0]
    for term in terms[1:]:
        expression = {"node": "add", "left": expression, "right": term}
    expression = {
        "node": "add",
        "left": expression,
        "right": {"node": "fitted_parameter", "parameter": "offset"},
    }
    bounds["offset"] = {"lower": -20.0, "upper": 20.0}
    return _hypothesis(
        title=title,
        concise_claim=f"A linear combination of {', '.join(sources)} is associated with y.",
        rationale="Simple linear-combination baseline with one or two source metrics.",
        target_metric="y",
        input_metrics=list(sources),
        expression=expression,
        parameter_bounds=bounds,
        maximum_lag_seconds=max_lag_seconds,
        catalog_id=catalog_id,
        seed=seed,
        origin=Origin.BASELINE,
        max_ast_nodes=10,
        max_source_metrics=2,
    )


def _hypothesis(
    *,
    title: str,
    concise_claim: str,
    rationale: str,
    target_metric: str,
    input_metrics: Sequence[str],
    expression: Mapping[str, Any],
    parameter_bounds: Mapping[str, Any],
    maximum_lag_seconds: int,
    catalog_id: str,
    seed: int,
    origin: Origin,
    target_transform: str = "identity",
    expected_direction: str = "positive",
    max_ast_nodes: int = 8,
    max_source_metrics: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "hypothesis_type": "observational_prediction",
        "title": title,
        "concise_claim": concise_claim,
        "rationale": rationale,
        "target_metric": target_metric,
        "input_metrics": list(input_metrics),
        "target_transform": target_transform,
        "expression": dict(expression),
        "parameter_bounds": dict(parameter_bounds),
        "expected_direction": expected_direction,
        "maximum_lag_seconds": int(maximum_lag_seconds),
        "fitting_metric": "rmse",
        "tuning_metric": "rmse",
        "blind_metrics": ["mae", "rmse", "spearman_r"],
        "minimum_blind_effect": 0.1,
        "minimum_baseline_improvement": 0.05,
        "negative_controls": [
            {"metric": "control", "rationale": "Independent control should not track the fitted prediction."}
        ]
        if "control" not in input_metrics
        else [],
        "falsification_conditions": [
            {"description": "Tuning performance does not improve over zero or persistence baselines."}
        ],
        "complexity_budget": {"max_ast_nodes": max_ast_nodes, "max_source_metrics": max_source_metrics},
        "origin": origin.value,
        "parent_hypothesis_ids": [],
        "snapshot_metric_catalog_id": catalog_id,
        "random_seed": int(seed),
    }


def _scale_offset_expression(input_expression: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "node": "add",
        "left": {
            "node": "multiply",
            "left": {"node": "fitted_parameter", "parameter": "scale"},
            "right": dict(input_expression),
        },
        "right": {"node": "fitted_parameter", "parameter": "offset"},
    }


def _scale_offset_bounds() -> dict[str, dict[str, float]]:
    return {
        "scale": {"lower": -5.0, "upper": 5.0},
        "offset": {"lower": -20.0, "upper": 20.0},
    }


def _lag_grid(max_lag_seconds: int) -> list[int]:
    if max_lag_seconds <= 0:
        return [0]
    middle = max(DEFAULT_SAMPLE_INTERVAL_SECONDS, int(round(max_lag_seconds / 2)))
    values = [0, middle, max_lag_seconds]
    return sorted({int(value) for value in values if 0 <= value <= max_lag_seconds})


def _recovers_expected_relation(
    scenario: str,
    hypothesis: HypothesisSpec,
    true_lag_seconds: int | None,
) -> bool:
    if scenario not in LAGGED_SCENARIOS or true_lag_seconds is None:
        return False
    if hypothesis.target_metric != "y" or "x" not in expression_metrics(hypothesis.expression):
        return False
    lags = expression_lag_seconds(hypothesis.expression)
    return any(abs(lag - true_lag_seconds) <= DEFAULT_SAMPLE_INTERVAL_SECONDS for lag in lags)


def _expected_relation(scenario: str, true_lag_seconds: int | None) -> dict[str, Any]:
    if scenario in LAGGED_SCENARIOS:
        return {
            "source_metric": "x",
            "target_metric": "y",
            "lag_seconds": true_lag_seconds,
            "direction": "positive",
        }
    return {
        "source_metric": None,
        "target_metric": "y",
        "lag_seconds": None,
        "direction": None,
    }


def _tuning_summary(evaluation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "candidate_id": evaluation["candidate_id"],
        "passed": all(evaluation["passed_gates"].values()),
        "score": evaluation["score"],
        "tuning_observations": evaluation["tuning_observations"],
        "tuning_mae": evaluation["tuning_mae"],
        "tuning_rmse": evaluation["tuning_rmse"],
        "tuning_spearman_rho": evaluation["tuning_spearman_rho"],
        "baseline_improvement": evaluation["baseline_improvement"],
        "passed_gates": evaluation["passed_gates"],
    }


def _complexity_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {"min_ast_nodes": None, "max_ast_nodes": None, "selected_ast_nodes": None}
    ast_nodes = [int(item["complexity"]["ast_nodes"]) for item in evaluations]
    selected = next((item for item in evaluations if item.get("default_winner")), None)
    return {
        "min_ast_nodes": min(ast_nodes),
        "max_ast_nodes": max(ast_nodes),
        "selected_ast_nodes": None if selected is None else selected["complexity"]["ast_nodes"],
    }


def _best_generator_name(generator_results: Sequence[Mapping[str, Any]]) -> str | None:
    ranked = sorted(
        generator_results,
        key=lambda result: (
            result["tuning_performance"] is None,
            -float(result["tuning_performance"]["score"]) if result["tuning_performance"] else math.inf,
            result["generator_name"],
        ),
    )
    return None if not ranked or ranked[0]["tuning_performance"] is None else str(ranked[0]["generator_name"])


def _value_conclusion(scenario_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    llm_wins = 0
    baseline_wins = 0
    for scenario in scenario_results:
        best = scenario.get("best_generator")
        if best == "llm_mock":
            llm_wins += 1
        elif best is not None:
            baseline_wins += 1
    if llm_wins > baseline_wins:
        return {
            "llm_adds_value": True,
            "conclusion": "Mock LLM outperformed baseline generators on this tuning-only ablation.",
        }
    return {
        "llm_adds_value": False,
        "conclusion": "No LLM value shown over the baseline generators in this tuning-only ablation.",
    }


def _write_synthetic_database(
    *,
    scenario: str,
    db_path: Path,
    duration_hours: float,
    sample_interval_seconds: int,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    dataset = generate_synthetic_series(
        scenario,
        sample_interval_seconds=sample_interval_seconds,
        duration_hours=duration_hours,
        noise=noise,
        seed=seed,
    )
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    control_rng = random.Random(seed + 9173)
    measurements: list[Measurement] = []
    for index, sample in enumerate(dataset.samples):
        if sample.x is not None and math.isfinite(sample.x):
            measurements.append(Measurement(sample.timestamp_utc, "x", sample.x, "synthetic", scenario))
        if sample.y is not None and math.isfinite(sample.y):
            measurements.append(Measurement(sample.timestamp_utc, "y", sample.y, "synthetic", scenario))
        control = 1.8 * math.sin(index * 1.913 + 0.2) + control_rng.gauss(0.0, max(noise, 0.2))
        measurements.append(Measurement(sample.timestamp_utc, "control", control, "synthetic", scenario))
    insert_measurements(conn, measurements)
    conn.close()
    return {
        **dataset.metadata,
        "database_path": str(db_path.resolve()),
        "inserted_measurement_count": len(measurements),
        "control_metric_seed": seed + 9173,
    }


def _store_json_artifact(root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise AblationError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


__all__ = [
    "ABLATION_VERSION",
    "AblationError",
    "run_ablation",
]
