from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from behavior_lab.benchmarks.metrics import classification_accuracy, multiclass_log_loss, regression_rmse
from behavior_lab.benchmarks.splits import assert_disjoint_groups, chronological_group_purged_split, group_disjoint_split
from behavior_lab.datasets.nber_best_offer.baselines import CategoryMajorityClassifier, MajorityClassifier, MedianRegressor, OfferRatioThresholdClassifier
from behavior_lab.datasets.nber_best_offer.tasks import assert_no_future_leakage, build_tasks


@dataclass(frozen=True)
class NberAuditReport:
    dataset_dir: str
    tasks: dict[str, Any]
    leakage_checks: dict[str, bool]
    split_checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(normalized_dir: str | Path) -> dict[str, Any]:
    tasks = build_tasks(normalized_dir)
    leaderboards: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_name, rows in tasks.items():
        if not rows:
            leaderboards[task_name] = {"chronological": [], "seller_disjoint": []}
            continue
        leaderboards[task_name] = {
            "chronological": _evaluate_split(task_name, chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id"), split_type="chronological"),
            "seller_disjoint": _evaluate_split(task_name, group_disjoint_split(rows, group_key="seller_id"), split_type="seller_disjoint"),
        }
    return {"leaderboards": leaderboards}


def audit(normalized_dir: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    tasks = build_tasks(normalized_dir)
    leakage_checks = {task_name: assert_no_future_leakage(rows) for task_name, rows in tasks.items()}
    split_checks = {}
    split_details = {}
    for task_name, rows in tasks.items():
        group_split = group_disjoint_split(rows, group_key="seller_id") if rows else None
        split_checks[f"{task_name}_seller_disjoint"] = assert_disjoint_groups(group_split, group_key="seller_id") if group_split else True
        chrono_split = chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id") if rows else None
        split_checks[f"{task_name}_listing_disjoint"] = assert_disjoint_groups(chrono_split, group_key="listing_id") if chrono_split else True
        split_details[task_name] = {
            "chronological": chrono_split.sizes() if chrono_split else {"train": 0, "development": 0, "hidden": 0},
            "chronological_group_key": "listing_id",
            "chronological_purge": {
                "purged_group_ids": list(chrono_split.purged_group_ids),
                "purged_rows": chrono_split.purged_rows,
            } if chrono_split else {"purged_group_ids": [], "purged_rows": 0},
            "seller_disjoint": group_split.sizes() if group_split else {"train": 0, "development": 0, "hidden": 0},
        }
    report = NberAuditReport(
        dataset_dir=str(Path(normalized_dir).resolve()),
        tasks={task_name: {"rows": len(rows), "splits": split_details[task_name]} for task_name, rows in tasks.items()},
        leakage_checks=leakage_checks,
        split_checks=split_checks,
    ).to_dict()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_split(task_name: str, split: Any, *, split_type: str) -> list[dict[str, Any]]:
    evaluation_rows = split.hidden or split.development
    if task_name in {"final_price_ratio", "response_latency"}:
        model = MedianRegressor().fit(split.train)
        predictions = model.predict(evaluation_rows).predictions
        return [
            {
                "model_id": model.model_id,
                "split_type": split_type,
                "split": "hidden" if split.hidden else "development",
                "rmse": regression_rmse(predictions),
                "features_used": [],
            }
        ]
    models = [MajorityClassifier().fit(split.train), CategoryMajorityClassifier().fit(split.train)]
    if task_name == "seller_next_action":
        models.append(OfferRatioThresholdClassifier().fit(split.train))
    rows_out = []
    for model in models:
        result = model.predict(evaluation_rows)
        rows_out.append(
            {
                "model_id": result.model_id,
                "split_type": split_type,
                "split": "hidden" if split.hidden else "development",
                "accuracy": classification_accuracy(result.predictions),
                "log_loss": multiclass_log_loss(result.predictions),
                "features_used": result.features_used,
            }
        )
    return rows_out
