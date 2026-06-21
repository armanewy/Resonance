from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from behavior_lab.benchmarks.splits import chronological_group_purged_split, group_disjoint_split
from behavior_lab.data_sources.registry import default_registry
from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.datasets.nber_best_offer.tasks import build_tasks
from behavior_lab.offerlab_models.calibration.calibration import (
    bootstrap_brier_uncertainty,
    calibration_by_slices,
    reliability_diagram,
    temporal_drift,
)
from behavior_lab.offerlab_models.formulas.formulas import evaluate_formula_candidates
from behavior_lab.offerlab_models.frontier.frontier import counteroffer_frontier
from behavior_lab.offerlab_models.predictive.models import RegularizedLogisticClassifier, predictive_suite
from behavior_lab.offerlab_models.transfer.ablation import evaluate_transfer_ablation


def run_sample_research_suite() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        normalized = root / "normalized"
        build_sample_dataset(raw)
        normalize_dataset(raw, normalized)
        return build_research_leaderboards(build_tasks(normalized))


def build_research_leaderboards(tasks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    leaderboards: dict[str, Any] = {}
    for task_name in ["seller_next_action", "buyer_response_to_counter", "agreement", "final_price_ratio"]:
        rows = list(tasks.get(task_name, []))
        if not rows:
            leaderboards[task_name] = {"chronological": {}, "seller_disjoint": {}}
            continue
        chronological = chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id")
        seller_disjoint = group_disjoint_split(rows, group_key="seller_id")
        leaderboards[task_name] = {
            "chronological": predictive_suite(task_name, chronological.train, chronological.development, chronological.hidden),
            "seller_disjoint": predictive_suite(task_name, seller_disjoint.train, seller_disjoint.development, seller_disjoint.hidden),
        }

    formula_report = _formula_report(tasks.get("seller_next_action", []), leaderboards)
    calibration_report = _calibration_report(tasks.get("seller_next_action", []))
    frontier_report = _frontier_report(tasks.get("buyer_response_to_counter", []))
    transfer_report = evaluate_transfer_ablation()
    registry = default_registry()
    return {
        "evidence_role": "OFFERLAB_RESEARCH_MODEL_SUITE",
        "production_export_allowed": False,
        "source_id": "nber_ebay_best_offer",
        "leaderboards": leaderboards,
        "formula_hypotheses": formula_report,
        "calibration": calibration_report,
        "counteroffer_frontier_quality": frontier_report,
        "transfer_ablation": transfer_report,
        "artifact_lineage": {
            "source_dataset_ids": ["nber_ebay_best_offer"],
            "production_export": registry.verify_lineage(["nber_ebay_best_offer"], "production_export"),
            "commercial_training": registry.verify_lineage(["nber_ebay_best_offer"], "commercial_training"),
        },
        "universal_winner": None,
        "production_warning": "Research artifact only. NBER-derived models are not exportable to OfferLab production.",
    }


def _formula_report(rows: list[dict[str, Any]], leaderboards: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"candidate_count": 0, "hidden_lockbox": None}
    split = chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id")
    black_box_hidden_loss = None
    black_box_model_id = None
    development_board = leaderboards.get("seller_next_action", {}).get("chronological", {}).get("leaderboards", {}).get("development", [])
    if development_board:
        black_box = next((row for row in development_board if row["model_id"] == "regularized_glm"), development_board[0])
        black_box_model_id = str(black_box["model_id"])
    return evaluate_formula_candidates(
        split.train,
        split.development,
        split.hidden,
        black_box_model_id=black_box_model_id,
        black_box_hidden_loss=black_box_hidden_loss,
    )


def _calibration_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {"rows": len(rows)}
    split = chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id")
    model = RegularizedLogisticClassifier().fit(split.train)
    predictions = model.predict(split.development).predictions
    return {
        "model_id": model.model_id,
        "split": "development",
        "hidden_rows_reserved": len(split.hidden),
        "reliability": reliability_diagram(predictions),
        "by_slice": calibration_by_slices(predictions, split.development),
        "temporal_drift": temporal_drift(predictions, split.development),
        "bootstrap_uncertainty": bootstrap_brier_uncertainty(predictions, samples=80),
        "action_level_sample_counts": {
            "train": {label: sum(1 for row in split.train if str(row["label"]) == label) for label in model.labels},
            "development": {label: sum(1 for row in split.development if str(row["label"]) == label) for label in model.labels},
        },
    }


def _frontier_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "frontier": []}
    context = rows[0]
    listing_price = float(context["features"]["listing_price"])
    candidates = [round(listing_price * ratio, 2) for ratio in [0.60, 0.75, 0.90, 1.10]]
    return counteroffer_frontier(context, rows, candidates)
