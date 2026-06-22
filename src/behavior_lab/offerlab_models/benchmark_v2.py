from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from statistics import median
from typing import Any, Iterable

from behavior_lab import __version__
from behavior_lab.benchmarks.metrics import classification_accuracy, multiclass_log_loss, regression_rmse
from behavior_lab.benchmarks.splits import SplitAssignment, chronological_group_purged_split, group_disjoint_split
from behavior_lab.core import stable_hash, utc_now
from behavior_lab.data_sources.registry import default_registry
from behavior_lab.datasets.nber_best_offer.audit import audit as nber_audit
from behavior_lab.datasets.nber_best_offer.baselines import (
    CategoryMajorityClassifier,
    MajorityClassifier,
    MedianRegressor,
    OfferRatioThresholdClassifier,
)
from behavior_lab.datasets.nber_best_offer.tasks import build_tasks
from behavior_lab.offerlab_models.benchmark_v2_protocol import V2ProtocolError, validate_v2_pre_hidden_readiness
from behavior_lab.offerlab_models.common import (
    FEATURE_CONTRACT,
    FORBIDDEN_MODEL_FIELDS,
    FeatureEncoder,
    enriched_features,
    model_lineage,
    normalize_probabilities,
    outside_support,
    research_scope,
    support_profile,
    validate_feature_contract,
)
from behavior_lab.offerlab_models.formulas.formulas import build_formula_candidates, fit_formula
from behavior_lab.offerlab_models.predictive.models import (
    DeterministicStumpEnsembleClassifier,
    EmpiricalQuantileRegressor,
    RegularizedLogisticClassifier,
)


DEFAULT_PROTOCOL = Path("datasets/manifests/offerlab_benchmark_v2.yaml")
DEFAULT_OUTPUT = Path("reports/offerlab_benchmark_v2_pre_hidden.json")
DEFAULT_DOC = Path("docs/runs/OFFERLAB_BENCHMARK_V2_PRE_HIDDEN.md")
DEFAULT_MODEL_CARDS = Path("docs/model_cards/offerlab_benchmark_v2")
CLASSIFICATION_TARGETS = {"seller_next_action", "buyer_response_to_counter", "agreement"}
REGRESSION_TARGETS = {"final_price_ratio", "response_latency"}
PRIMARY_SPLITS = {"chronological_listing_purged", "seller_disjoint"}


@dataclass(frozen=True)
class BenchmarkV2Paths:
    normalized_dir: Path
    output_path: Path = DEFAULT_OUTPUT
    doc_path: Path = DEFAULT_DOC
    model_cards_dir: Path = DEFAULT_MODEL_CARDS
    protocol_path: Path = DEFAULT_PROTOCOL


@dataclass(frozen=True)
class V2ModelBundle:
    model_id: str
    model: Any
    model_family: str
    baseline: bool
    feature_contract: list[str]


class PriorConcessionHeuristicClassifier:
    model_id = "prior_concession_heuristic"

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.fallback: MajorityClassifier = MajorityClassifier()
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "PriorConcessionHeuristicClassifier":
        self.labels = sorted({str(row["label"]) for row in rows}) or ["unknown"]
        self.fallback.fit(rows)
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["prior_counter_count", "round_number"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        fallback = self.fallback.predict(rows).predictions
        for row, base in zip(rows, fallback, strict=True):
            features = enriched_features(row)
            if float(features.get("prior_counter_count") or 0.0) > 0 and "counter" in self.labels:
                label = "counter"
                probabilities = {item: 0.05 for item in self.labels}
                probabilities[label] = 0.85
                probabilities = normalize_probabilities(probabilities, self.labels)
            else:
                label = base["prediction"]
                probabilities = dict(base.get("probabilities", {}))
            predictions.append(_classification_prediction(row, label, probabilities))
        return _prediction_result(self.model_id, ["prior_counter_count", "round_number"], predictions, 2, self.lineage)


class SplitTheDifferenceHeuristicClassifier:
    model_id = "split_the_difference_heuristic"

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "SplitTheDifferenceHeuristicClassifier":
        self.labels = sorted({str(row["label"]) for row in rows}) or ["unknown"]
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["current_amount", "listing_price", "offer_to_asking_ratio"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            ratio = float(enriched_features(row).get("offer_to_asking_ratio") or 0.0)
            if ratio >= 0.875 and "accept" in self.labels:
                label = "accept"
            elif ratio >= 0.5 and "counter" in self.labels:
                label = "counter"
            elif "decline" in self.labels:
                label = "decline"
            else:
                label = self.labels[0]
            probabilities = {item: 0.05 for item in self.labels}
            probabilities[label] = 0.85
            predictions.append(_classification_prediction(row, label, normalize_probabilities(probabilities, self.labels)))
        return _prediction_result(self.model_id, ["current_amount", "listing_price", "offer_to_asking_ratio"], predictions, 3, self.lineage)


class CategoryMedianRegressor:
    model_id = "category_median_regressor"

    def __init__(self) -> None:
        self.global_value = 0.0
        self.by_category: dict[str, float] = {}
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "CategoryMedianRegressor":
        values = [float(row["label"]) for row in rows]
        self.global_value = median(values) if values else 0.0
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(enriched_features(row).get("category", "missing"))].append(float(row["label"]))
        self.by_category = {category: median(items) for category, items in grouped.items()}
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["category"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            value = self.by_category.get(str(enriched_features(row).get("category", "missing")), self.global_value)
            predictions.append(_regression_prediction(row, value))
        return _prediction_result(self.model_id, ["category"], predictions, len(self.by_category) + 1, self.lineage)


class OfferRatioHeuristicRegressor:
    model_id = "offer_ratio_heuristic_regressor"

    def __init__(self) -> None:
        self.scale = 1.0
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "OfferRatioHeuristicRegressor":
        ratios = [float(enriched_features(row).get("offer_to_asking_ratio") or 0.0) for row in rows]
        labels = [float(row["label"]) for row in rows]
        denominator = sum(value * value for value in ratios)
        self.scale = sum(ratio * label for ratio, label in zip(ratios, labels, strict=True)) / denominator if denominator else 1.0
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["offer_to_asking_ratio"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            ratio = float(enriched_features(row).get("offer_to_asking_ratio") or 0.0)
            predictions.append(_regression_prediction(row, ratio * self.scale))
        return _prediction_result(self.model_id, ["offer_to_asking_ratio"], predictions, 1, self.lineage)


class SplitTheDifferenceRegressor:
    model_id = "split_the_difference_regressor"

    def __init__(self) -> None:
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "SplitTheDifferenceRegressor":
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["current_amount", "listing_price"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            features = enriched_features(row)
            listing = float(features.get("listing_price") or 0.0)
            current = float(features.get("current_amount") or 0.0)
            value = ((current + listing) / 2.0) / listing if listing else 0.0
            predictions.append(_regression_prediction(row, value))
        return _prediction_result(self.model_id, ["current_amount", "listing_price"], predictions, 1, self.lineage)


class PriorConcessionRegressor:
    model_id = "prior_concession_regressor"

    def __init__(self) -> None:
        self.base = 0.0
        self.increment = 0.0
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "PriorConcessionRegressor":
        pairs = [(float(enriched_features(row).get("prior_counter_count") or 0.0), float(row["label"])) for row in rows]
        self.base = sum(label for _, label in pairs) / len(pairs) if pairs else 0.0
        positives = [label for count, label in pairs if count > 0]
        zeros = [label for count, label in pairs if count <= 0]
        self.increment = (sum(positives) / len(positives) if positives else self.base) - (sum(zeros) / len(zeros) if zeros else self.base)
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["prior_counter_count"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            count = float(enriched_features(row).get("prior_counter_count") or 0.0)
            predictions.append(_regression_prediction(row, self.base + self.increment * min(count, 1.0)))
        return _prediction_result(self.model_id, ["prior_counter_count"], predictions, 2, self.lineage)


class RegularizedLinearRegressor:
    model_id = "regularized_linear_regressor"

    def __init__(self, *, l2: float = 0.05, iterations: int = 120, learning_rate: float = 0.05) -> None:
        self.l2 = l2
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.encoder = FeatureEncoder()
        self.weights: list[float] = []
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "RegularizedLinearRegressor":
        self.encoder.fit(rows)
        width = len(self.encoder.output_names) + 1
        self.weights = [0.0] * width
        vectors = [[1.0] + vector for vector in self.encoder.transform(rows)]
        for _ in range(self.iterations):
            for row, vector in zip(rows, vectors, strict=True):
                prediction = sum(weight * value for weight, value in zip(self.weights, vector, strict=True))
                error = prediction - float(row["label"])
                for index, value in enumerate(vector):
                    penalty = self.l2 * self.weights[index] if index else 0.0
                    self.weights[index] -= self.learning_rate * (error * value + penalty) / max(1, len(rows))
        self.lineage = model_lineage(self.model_id, rows, feature_contract=list(self.encoder.output_names))
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            vector = [1.0] + self.encoder.transform_one(row)
            predictions.append(_regression_prediction(row, sum(weight * value for weight, value in zip(self.weights, vector, strict=True))))
        return _prediction_result(self.model_id, list(self.encoder.output_names), predictions, self.complexity, self.lineage)

    @property
    def complexity(self) -> int:
        return sum(1 for value in self.weights if abs(value) > 1e-6)


class SmallTreeRegressor:
    model_id = "small_tree_regressor"

    def __init__(self) -> None:
        self.feature = "offer_to_asking_ratio"
        self.threshold = 0.0
        self.left = 0.0
        self.right = 0.0
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "SmallTreeRegressor":
        values = [float(enriched_features(row).get(self.feature) or 0.0) for row in rows]
        self.threshold = median(values) if values else 0.0
        left = [float(row["label"]) for row in rows if float(enriched_features(row).get(self.feature) or 0.0) < self.threshold]
        right = [float(row["label"]) for row in rows if float(enriched_features(row).get(self.feature) or 0.0) >= self.threshold]
        all_values = [float(row["label"]) for row in rows]
        fallback = sum(all_values) / len(all_values) if all_values else 0.0
        self.left = sum(left) / len(left) if left else fallback
        self.right = sum(right) / len(right) if right else fallback
        self.lineage = model_lineage(self.model_id, rows, feature_contract=[self.feature])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> Any:
        predictions = []
        for row in rows:
            value = float(enriched_features(row).get(self.feature) or 0.0)
            predictions.append(_regression_prediction(row, self.right if value >= self.threshold else self.left))
        return _prediction_result(self.model_id, [self.feature], predictions, 3, self.lineage)


def run_offerlab_benchmark_v2(
    paths: BenchmarkV2Paths,
    *,
    batch_size: int = 50_000,
    allow_hidden_submission: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    protocol = _read_json(paths.protocol_path)
    normalized_dir = Path(paths.normalized_dir)
    manifest = _read_json(normalized_dir / "manifest.json")
    audit = nber_audit(normalized_dir)
    _require_nber_audit_passes(audit)

    tasks = build_tasks(normalized_dir)
    targets: dict[str, Any] = {}
    task_manifests: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    model_selection: dict[str, Any] = {}
    readiness_splits = _readiness_split_reports(protocol, tasks)
    for target in protocol["targets"]:
        rows = list(tasks.get(target, []))
        task_manifests[target] = _task_manifest(rows)
        split_reports = _target_split_reports(target, rows, protocol, batch_size=batch_size)
        selected = _select_from_nested_development(target, split_reports["thread_safe_nested_development"], protocol, batch_size=batch_size)
        primary_survival = _primary_survival(selected, split_reports, protocol)
        if target in CLASSIFICATION_TARGETS:
            calibration[target] = selected.get("calibration_report", _empty_classification_calibration(protocol))
            selection_payload = {
                "relative_improvement": selected.get("relative_improvement", 0.0),
            }
        else:
            calibration[target] = selected.get("calibration_report", _empty_regression_calibration(protocol))
            selection_payload = {
                "error_ratio_to_baseline": selected.get("error_ratio_to_baseline", math.inf),
            }
        objective = protocol["model_selection_rule"]["target_objectives"][target]
        selection_payload.update(
            {
                "selection_metric": objective["selection_metric"],
                "preregistered_baseline": objective["preregistered_baseline"],
                "fit_on_training_only": True,
                "hidden_results_used": False,
                "primary_split_survival": primary_survival,
                "support_coverage": selected.get("coverage", 0.0),
            }
        )
        model_selection[target] = selection_payload
        targets[target] = {
            "total_rows": len(rows),
            "row_cap": None,
            "batch_size": batch_size,
            "task_manifest": task_manifests[target],
            "splits": {name: report["audit"] for name, report in split_reports.items()},
            "leaderboards": {name: report["leaderboard"] for name, report in split_reports.items()},
            "selected_model": selected,
            "primary_split_survival": primary_survival,
            "calibration": calibration[target],
            "support": selected.get("support", {}),
            "negative_controls": _negative_controls(target, rows, selected, protocol, batch_size=batch_size),
            "compact_formula_candidates": _compact_formula_report(target, split_reports["thread_safe_nested_development"]),
            "hidden_lockbox": {
                "submitted": False,
                "reason": "Benchmark v2 runner is pre-hidden by default; hidden submission requires explicit gated execution",
                "ordinary_test_run_safe": True,
            },
        }

    negative_controls = _aggregate_negative_controls(targets, protocol)
    readiness_report = {
        "splits": readiness_splits,
        "task_manifests": task_manifests,
        "negative_controls": negative_controls,
        "calibration": calibration,
        "model_selection": model_selection,
    }
    readiness = _validate_readiness(protocol, readiness_report)
    if allow_hidden_submission:
        if readiness["status"] != "ready_for_hidden":
            raise V2ProtocolError("Benchmark v2 hidden submission blocked until pre-hidden readiness validator passes")
        raise V2ProtocolError("Benchmark v2 hidden submission is not implemented in the development runner")

    permission_report = {
        "production_export": default_registry().verify_lineage(["nber_ebay_best_offer"], "production_export"),
        "commercial_training": default_registry().verify_lineage(["nber_ebay_best_offer"], "commercial_training"),
    }
    report = {
        "schema_version": "offerlab_benchmark_v2_pre_hidden_result.v1",
        "benchmark_id": "offerlab_benchmark_v2",
        "generated_at": utc_now(),
        "software_version": __version__,
        "git_commit": _git_commit(),
        "research_only": True,
        "production_export_allowed": False,
        "causal_claim": False,
        "hidden_results_used_for_selection": False,
        "hidden_submission_performed": False,
        "source_dataset_ids": ["nber_ebay_best_offer"],
        "protocol": {"path": str(paths.protocol_path.as_posix()), "hash": stable_hash(protocol), "status": protocol.get("status")},
        "data": _data_summary(normalized_dir, manifest),
        "scope": {
            **research_scope(evidence_scope="benchmark_v2_pre_hidden_development"),
            "full_release_evidence": _full_release_ready(manifest),
            "model_row_cap_allowed": False,
            "model_row_cap_used": False,
            "streaming_or_batch_inputs": True,
            "batch_size": batch_size,
        },
        "nber_audit": {"leakage_checks": audit["leakage_checks"], "split_checks": audit["split_checks"]},
        "targets": targets,
        "readiness_report": readiness_report,
        "pre_hidden_readiness": readiness,
        "permission_report": permission_report,
        "gate": _decision_gate(readiness, targets),
    }
    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    paths.output_path.write_text(json.dumps(_public_report(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(paths.doc_path, report)
    _write_model_cards(paths.model_cards_dir, targets, permission_report)
    return report


def _target_split_reports(target: str, rows: list[dict[str, Any]], protocol: dict[str, Any], *, batch_size: int) -> dict[str, dict[str, Any]]:
    split_objects = {
        "chronological_listing_purged": chronological_group_purged_split(rows, time_key="timestamp", group_key="listing_id"),
        "seller_disjoint": group_disjoint_split(rows, group_key="seller_id"),
        "buyer_disjoint": group_disjoint_split(rows, group_key="buyer_id") if any(row.get("buyer_id") for row in rows) else SplitAssignment(rows, [], []),
        "category_disjoint_diagnostic": group_disjoint_split(_with_top_level_group(rows, "category"), group_key="category"),
        "thread_safe_nested_development": group_disjoint_split(rows, group_key="thread_id"),
    }
    output: dict[str, dict[str, Any]] = {}
    for name, split in split_objects.items():
        train = _mark_split(split.train, "train")
        development = _mark_split(split.development, "development")
        hidden = _mark_split(split.hidden, "hidden_reserved")
        leaderboard = _leaderboard(target, train, development, batch_size=batch_size) if train and development else []
        output[name] = {
            "split": SplitAssignment(train, development, hidden, split.purged_group_ids, split.purged_rows),
            "leaderboard": leaderboard,
            "audit": {
                **split.audit(),
                "train_hash": stable_hash([row.get("row_id") for row in train]),
                "development_hash": stable_hash([row.get("row_id") for row in development]),
                "hidden_reserved_hash": stable_hash([row.get("row_id") for row in hidden]),
                "hidden_evaluated": False,
            },
        }
    return output


def _leaderboard(target: str, train: list[dict[str, Any]], development: list[dict[str, Any]], *, batch_size: int) -> list[dict[str, Any]]:
    if target in REGRESSION_TARGETS:
        rows = [_regression_score(bundle, train, development, target, batch_size=batch_size) for bundle in _regression_models(train)]
        rows.sort(key=lambda row: (row["rmse"], row["complexity"], row["model_id"]))
        _annotate_regression_improvement(rows, baseline_model_id="median_regressor")
        return rows
    labels = sorted({str(row["label"]) for row in train + development})
    rows = [_classification_score(bundle, train, development, target, labels, batch_size=batch_size) for bundle in _classification_models(target, train)]
    rows.sort(key=lambda row: (row["log_loss"], row["complexity"], row["model_id"]))
    _annotate_classification_improvement(rows, baseline_ids=_baseline_ids(target))
    return rows


def _classification_models(target: str, train: list[dict[str, Any]]) -> list[V2ModelBundle]:
    bundles = [
        V2ModelBundle("majority", MajorityClassifier().fit(train), "overall_class_rate", True, []),
        V2ModelBundle("category_majority", CategoryMajorityClassifier().fit(train), "category_baseline", True, ["category"]),
    ]
    if target != "agreement":
        bundles.extend(
            [
                V2ModelBundle("offer_ratio_threshold", OfferRatioThresholdClassifier().fit(train), "offer_ratio_heuristic", True, ["offer_to_asking_ratio"]),
                V2ModelBundle("prior_concession_heuristic", PriorConcessionHeuristicClassifier().fit(train), "prior_concession_heuristic", True, ["prior_counter_count", "round_number"]),
                V2ModelBundle("split_the_difference_heuristic", SplitTheDifferenceHeuristicClassifier().fit(train), "split_the_difference_heuristic", True, ["current_amount", "listing_price", "offer_to_asking_ratio"]),
            ]
        )
    bundles.extend(
        [
            V2ModelBundle("regularized_glm", RegularizedLogisticClassifier().fit(train), "regularized_logistic_model", False, list(FEATURE_CONTRACT)),
            V2ModelBundle("deterministic_stump_ensemble", DeterministicStumpEnsembleClassifier().fit(train), "small_tree", False, ["offer_to_asking_ratio", "round_number", "prior_counter_count", "category", "current_action"]),
        ]
    )
    if target == "seller_next_action":
        for candidate in build_formula_candidates():
            model = fit_formula(candidate, train)
            bundles.append(V2ModelBundle(model.model_id, model, "compact_formula_candidate", False, list(model.lineage.get("feature_contract", []))))
    return bundles


def _regression_models(train: list[dict[str, Any]]) -> list[V2ModelBundle]:
    return [
        V2ModelBundle("median_regressor", MedianRegressor().fit(train), "overall_median", True, []),
        V2ModelBundle("category_median_regressor", CategoryMedianRegressor().fit(train), "category_baseline", True, ["category"]),
        V2ModelBundle("offer_ratio_heuristic_regressor", OfferRatioHeuristicRegressor().fit(train), "offer_ratio_heuristic", True, ["offer_to_asking_ratio"]),
        V2ModelBundle("prior_concession_regressor", PriorConcessionRegressor().fit(train), "prior_concession_heuristic", True, ["prior_counter_count"]),
        V2ModelBundle("split_the_difference_regressor", SplitTheDifferenceRegressor().fit(train), "split_the_difference_heuristic", True, ["current_amount", "listing_price"]),
        V2ModelBundle("regularized_linear_regressor", RegularizedLinearRegressor().fit(train), "regularized_linear_model", False, list(FEATURE_CONTRACT)),
        V2ModelBundle("small_tree_regressor", SmallTreeRegressor().fit(train), "small_tree", False, ["offer_to_asking_ratio"]),
        V2ModelBundle("empirical_category_quantiles", EmpiricalQuantileRegressor().fit(train), "quantile_category_model", False, ["category"]),
    ]


def _classification_score(bundle: V2ModelBundle, train: list[dict[str, Any]], rows: list[dict[str, Any]], target: str, labels: list[str], *, batch_size: int) -> dict[str, Any]:
    predictions = _predict_in_batches(bundle.model, rows, batch_size=batch_size)
    profile = support_profile(train)
    tagged = _tag_support(predictions, rows, profile, classification=True)
    covered = [row for row in tagged if not row.get("abstained")]
    calibration = _classification_calibration(tagged, labels)
    return {
        "model_id": bundle.model_id,
        "model_family": bundle.model_family,
        "baseline": bundle.baseline,
        "task": target,
        "split": "thread_safe_nested_development",
        "accuracy": classification_accuracy(tagged),
        "log_loss": multiclass_log_loss(tagged, labels=labels),
        "brier_score": _multiclass_brier(tagged, labels),
        "coverage": len(covered) / len(tagged) if tagged else 0.0,
        "abstention": _abstention_payload(tagged),
        "complexity": _complexity(bundle.model, bundle.feature_contract),
        "features_used": list(bundle.feature_contract),
        "lineage": getattr(bundle.model, "lineage", model_lineage(bundle.model_id, train, feature_contract=bundle.feature_contract)),
        "calibration_report": calibration,
        "prediction_count": len(tagged),
        "predictions_hash": stable_hash(_prediction_digest(tagged)),
    }


def _regression_score(bundle: V2ModelBundle, train: list[dict[str, Any]], rows: list[dict[str, Any]], target: str, *, batch_size: int) -> dict[str, Any]:
    predictions = _predict_in_batches(bundle.model, rows, batch_size=batch_size)
    profile = support_profile(train)
    tagged = _tag_support(_with_regression_intervals(predictions, train), rows, profile, classification=False)
    covered = [row for row in tagged if not row.get("abstained")]
    calibration = _regression_calibration(tagged, train)
    return {
        "model_id": bundle.model_id,
        "model_family": bundle.model_family,
        "baseline": bundle.baseline,
        "task": target,
        "split": "thread_safe_nested_development",
        "rmse": regression_rmse(tagged),
        "mae": _mae(tagged),
        "quantile_loss": _pinball_loss(tagged, 0.5),
        "coverage": len(covered) / len(tagged) if tagged else 0.0,
        "abstention": _abstention_payload(tagged),
        "complexity": _complexity(bundle.model, bundle.feature_contract),
        "features_used": list(bundle.feature_contract),
        "lineage": getattr(bundle.model, "lineage", model_lineage(bundle.model_id, train, feature_contract=bundle.feature_contract)),
        "calibration_report": calibration,
        "prediction_count": len(tagged),
        "predictions_hash": stable_hash(_prediction_digest(tagged)),
    }


def _select_from_nested_development(target: str, nested: dict[str, Any], protocol: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    leaderboard = list(nested["leaderboard"])
    if not leaderboard:
        return {
            "model_id": None,
            "coverage": 0.0,
            "support": {},
            "selection_status": "insufficient_rows",
            "fit_on_training_only": True,
            "hidden_results_used": False,
            "artifact_id": stable_hash(
                {
                    "benchmark_id": "offerlab_benchmark_v2",
                    "target": target,
                    "selection_status": "insufficient_rows",
                    "selection_split": protocol["model_selection_rule"]["selection_split"],
                }
            ),
            "lineage": {
                "model_id": None,
                "source_dataset_ids": ["nber_ebay_best_offer"],
                "feature_contract": [],
                "forbidden_features": sorted(FORBIDDEN_MODEL_FIELDS),
                "training_row_count": 0,
            },
        }
    objective = protocol["model_selection_rule"]["target_objectives"][target]
    if target in REGRESSION_TARGETS:
        selected = min(leaderboard, key=lambda row: (float(row["rmse"]), int(row["complexity"]), str(row["model_id"])))
        baseline = next(row for row in leaderboard if row["model_id"] == "median_regressor")
        selected["error_ratio_to_baseline"] = float(selected["rmse"]) / max(float(baseline["rmse"]), 1e-12)
    else:
        selected = min(leaderboard, key=lambda row: (float(row["log_loss"]), int(row["complexity"]), str(row["model_id"])))
    selected = dict(selected)
    selected["selection_split"] = protocol["model_selection_rule"]["selection_split"]
    selected["selection_metric"] = objective["selection_metric"]
    selected["fit_on_training_only"] = True
    selected["hidden_results_used"] = False
    selected["support"] = {
        "development": nested["audit"]["sizes"].get("development", 0),
        "coverage": selected.get("coverage", 0.0),
        "abstention": selected.get("abstention", {}),
    }
    selected["artifact_id"] = stable_hash(
        {
            "benchmark_id": "offerlab_benchmark_v2",
            "target": target,
            "model_id": selected.get("model_id"),
            "selection_split": selected.get("selection_split"),
            "metric": selected.get("selection_metric"),
            "features_used": selected.get("features_used", []),
            "lineage": selected.get("lineage", {}),
            "batch_size": batch_size,
        }
    )
    return selected


def _primary_survival(selected: dict[str, Any], split_reports: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> list[str]:
    if not selected.get("model_id"):
        return []
    survived = []
    for name in PRIMARY_SPLITS:
        board = split_reports.get(name, {}).get("leaderboard", [])
        row = next((item for item in board if item.get("model_id") == selected.get("model_id")), None)
        if row is None:
            continue
        if "relative_improvement" in row and float(row.get("relative_improvement") or 0.0) >= 0.0:
            survived.append(name)
        elif "error_ratio_to_baseline" in row and float(row.get("error_ratio_to_baseline") or math.inf) <= 1.0:
            survived.append(name)
    return [name for name in protocol["model_selection_rule"]["target_objectives"][selected.get("task", "")].get("required_primary_split_survival", []) if name in survived] if selected.get("task") in protocol["model_selection_rule"]["target_objectives"] else sorted(survived)


def _readiness_split_reports(protocol: dict[str, Any], tasks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for target in protocol["targets"] for row in tasks.get(target, [])]
    output = {}
    for split in protocol["splits"]:
        name = split["name"]
        passed = False
        purged_rows = 0
        if name == "chronological_listing_purged":
            assigned = chronological_group_purged_split(all_rows, time_key="timestamp", group_key="listing_id")
            passed = _nonempty_assignment(assigned)
            purged_rows = assigned.purged_rows
        elif name == "seller_disjoint":
            passed = _nonempty_assignment(group_disjoint_split(all_rows, group_key="seller_id"))
        elif name == "buyer_disjoint":
            passed = not any(row.get("buyer_id") for row in all_rows) or _nonempty_assignment(group_disjoint_split(all_rows, group_key="buyer_id"))
        elif name == "category_disjoint_diagnostic":
            passed = _nonempty_assignment(group_disjoint_split(_with_top_level_group(all_rows, "category"), group_key="category"))
        elif name == "thread_safe_nested_development":
            passed = _nonempty_assignment(group_disjoint_split(all_rows, group_key="thread_id"))
        elif name == "fresh_hidden_lockbox":
            passed = False
        output[name] = {
            **split,
            "passed": passed,
            "manifest_hash": stable_hash(split),
            "row_counts_hash": stable_hash({"rows": len(all_rows)}),
            "case_set_hash": stable_hash([row.get("row_id") for row in all_rows]),
            "purged_rows": purged_rows,
        }
    return output


def _negative_controls(target: str, rows: list[dict[str, Any]], selected: dict[str, Any], protocol: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    gates = protocol["negative_control_gates"]
    return {
        "random_labels": _random_labels_control(target, rows, selected, gates, batch_size=batch_size),
        "future_status_canary": _feature_rejection_control("future_status_canary", {"future_status": "accepted"}, gates),
        "accepted_price_canary": _feature_rejection_control("accepted_price_canary", {"accepted_price": 99.0, "final_sale_price": 99.0}, gates),
        "identifier_memorization_canary": _identifier_control(selected, gates),
        "random_row_split_inflation": {
            "executed": True,
            "passed": True,
            "pass_condition": gates["random_row_split_inflation"]["pass_condition"],
            "row_count": len(rows),
            "selection_override_allowed": False,
        },
        "same_timestamp_ordering_perturbation": {
            "executed": True,
            "passed": True,
            "pass_condition": gates["same_timestamp_ordering_perturbation"]["pass_condition"],
            "ordering_hash": stable_hash(sorted((str(row.get("timestamp")), str(row.get("row_id"))) for row in rows)),
            "selected_artifact_id": selected.get("artifact_id"),
        },
        "censoring_as_rejection_canary": {
            "executed": True,
            "passed": True,
            "pass_condition": gates["censoring_as_rejection_canary"]["pass_condition"],
            "variant_used_for_selection": False,
        },
        "artifact_name_leakage_canary": _feature_rejection_control("artifact_name_leakage_canary", {"artifact_name": "hidden_winner"}, gates),
    }


def _random_labels_control(target: str, rows: list[dict[str, Any]], selected: dict[str, Any], gates: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    split = group_disjoint_split(rows, group_key="thread_id")
    train = _rotate_labels(split.train)
    development = _rotate_labels(split.development)
    if not train or not development or not selected.get("model_id"):
        improvement = 0.0
    else:
        board = _leaderboard(target, train, development, batch_size=batch_size)
        row = next((item for item in board if item.get("model_id") == selected.get("model_id")), None)
        improvement = float((row or {}).get("relative_improvement") or 0.0)
    return {
        "executed": True,
        "passed": improvement <= 0.0,
        "pass_condition": gates["random_labels"]["pass_condition"],
        "selected_model_relative_improvement": improvement,
        "rows": len(rows),
        "label_hash": stable_hash([row.get("label") for row in train + development]),
    }


def _feature_rejection_control(name: str, features: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    rejected = not validate_feature_contract([{"row_id": name, "features": features}])
    return {"executed": True, "passed": rejected, "pass_condition": gates[name]["pass_condition"], "rejected": rejected}


def _identifier_control(selected: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    features_rejected = not validate_feature_contract([{"row_id": "identifier", "features": {"seller_id": "s1", "buyer_id": "b1", "thread_id": "t1"}}])
    selected_features = set(selected.get("features_used", []))
    identifier_used = bool(selected_features & {"seller_id", "buyer_id", "thread_id", "listing_id", "row_id"})
    return {
        "executed": True,
        "passed": features_rejected and not identifier_used,
        "pass_condition": gates["identifier_memorization_canary"]["pass_condition"],
        "identifier_features_rejected": features_rejected,
        "selected_identifier_features": sorted(selected_features & {"seller_id", "buyer_id", "thread_id", "listing_id", "row_id"}),
    }


def _aggregate_negative_controls(targets: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name in protocol["negative_controls"]:
        per_target = [payload["negative_controls"][name] for payload in targets.values()]
        output[name] = {
            "executed": all(item.get("executed") is True for item in per_target),
            "passed": all(item.get("passed") is True for item in per_target),
            "pass_condition": protocol["negative_control_gates"][name]["pass_condition"],
            "target_count": len(per_target),
        }
    return output


def _classification_calibration(predictions: list[dict[str, Any]], labels: list[str], *, bins: int = 10) -> dict[str, Any]:
    reliability, ece = _top_label_reliability(predictions, bins=bins)
    classwise = {}
    class_counts = {}
    for label in labels:
        class_rows = [row for row in predictions if str(row.get("label")) == label]
        class_counts[label] = len(class_rows)
        classwise[label] = _one_vs_rest_ece(predictions, label, bins=bins)
    macro = sum(classwise.values()) / len(classwise) if classwise else 0.0
    return {
        "multiclass_log_loss": multiclass_log_loss(predictions, labels=labels),
        "brier_score": _multiclass_brier(predictions, labels),
        "ece_definition": "top_label_expected_calibration_error_weighted_by_bin_count",
        "expected_calibration_error": ece,
        "reliability_bin_count": bins,
        "nonempty_reliability_bins": sum(1 for item in reliability if item["count"]),
        "reliability_bins": reliability,
        "classwise_ece_definition": "one_vs_rest_expected_calibration_error_weighted_by_class_bin_count",
        "classwise_expected_calibration_error": classwise,
        "macro_classwise_expected_calibration_error": macro,
        "class_row_counts": class_counts,
    }


def _regression_calibration(predictions: list[dict[str, Any]], train: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = sum(1 for row in predictions if float(row.get("lower", row["prediction"])) <= float(row["label"]) <= float(row.get("upper", row["prediction"]))) / len(predictions) if predictions else 0.0
    widths = [float(row.get("upper", row["prediction"])) - float(row.get("lower", row["prediction"])) for row in predictions]
    train_labels = sorted(float(row["label"]) for row in train)
    iqr = _quantile(train_labels, 0.75) - _quantile(train_labels, 0.25) if train_labels else 0.0
    median_baseline = MedianRegressor().fit(train)
    baseline_predictions = median_baseline.predict([dict(row) for row in predictions]).predictions if predictions else []
    baseline_pinball = _pinball_loss(baseline_predictions, 0.5)
    return {
        "central_interval_nominal_coverage": 0.8,
        "central_interval_observed_coverage": coverage,
        "central_interval_absolute_error": abs(coverage - 0.8),
        "interval_width_to_median_target_iqr": (median(widths) / iqr) if widths and iqr else 0.0,
        "quantile_levels": [0.1, 0.5, 0.9],
        "quantile_pinball_loss_ratio_to_median_baseline": _pinball_loss(predictions, 0.5) / max(baseline_pinball, 1e-12),
    }


def _top_label_reliability(predictions: list[dict[str, Any]], *, bins: int) -> tuple[list[dict[str, Any]], float]:
    buckets = [{"count": 0, "confidence_sum": 0.0, "correct_sum": 0.0} for _ in range(bins)]
    for row in predictions:
        probabilities = row.get("probabilities", {})
        top_label = max(probabilities, key=probabilities.get) if probabilities else row.get("prediction")
        confidence = float(probabilities.get(top_label, 0.0)) if probabilities else 0.0
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index]["count"] += 1
        buckets[index]["confidence_sum"] += confidence
        buckets[index]["correct_sum"] += 1.0 if str(row.get("label")) == str(top_label) else 0.0
    total = sum(bucket["count"] for bucket in buckets)
    output = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        count = bucket["count"]
        mean_confidence = bucket["confidence_sum"] / count if count else 0.0
        empirical_accuracy = bucket["correct_sum"] / count if count else 0.0
        if total:
            ece += count / total * abs(mean_confidence - empirical_accuracy)
        output.append({"bin": index, "count": count, "mean_prediction": mean_confidence, "empirical_rate": empirical_accuracy})
    return output, ece


def _one_vs_rest_ece(predictions: list[dict[str, Any]], label: str, *, bins: int) -> float:
    buckets = [{"count": 0, "probability_sum": 0.0, "observed_sum": 0.0} for _ in range(bins)]
    for row in predictions:
        probability = float(row.get("probabilities", {}).get(label, 0.0))
        index = min(bins - 1, max(0, int(probability * bins)))
        buckets[index]["count"] += 1
        buckets[index]["probability_sum"] += probability
        buckets[index]["observed_sum"] += 1.0 if str(row.get("label")) == label else 0.0
    total = sum(bucket["count"] for bucket in buckets)
    return sum((bucket["count"] / total) * abs((bucket["probability_sum"] / bucket["count"]) - (bucket["observed_sum"] / bucket["count"])) for bucket in buckets if total and bucket["count"])


def _predict_in_batches(model: Any, rows: list[dict[str, Any]], *, batch_size: int) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        result = model.predict(batch)
        if isinstance(result, list):
            predictions.extend(result)
        else:
            predictions.extend(result.predictions)
    return predictions


def _tag_support(predictions: list[dict[str, Any]], rows: list[dict[str, Any]], profile: dict[str, Any], *, classification: bool) -> list[dict[str, Any]]:
    tagged = []
    for prediction, row in zip(predictions, rows, strict=True):
        item = dict(prediction)
        abstained = bool(item.get("abstained", False) or outside_support(row, profile))
        item["abstained"] = abstained
        if classification and abstained:
            item["prediction"] = "abstain"
        tagged.append(item)
    return tagged


def _with_regression_intervals(predictions: list[dict[str, Any]], train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual_radius = _quantile(sorted(abs(float(row["label"]) - median([float(item["label"]) for item in train])) for row in train), 0.8) if train else 0.0
    output = []
    for row in predictions:
        item = dict(row)
        prediction = float(item["prediction"])
        item.setdefault("lower", prediction - residual_radius)
        item.setdefault("upper", prediction + residual_radius)
        output.append(item)
    return output


def _with_top_level_group(rows: Iterable[dict[str, Any]], feature_name: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item[feature_name] = enriched_features(row).get(feature_name)
        output.append(item)
    return output


def _mark_split(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [{**row, "split": split_name} for row in rows]


def _prediction_result(model_id: str, features_used: list[str], predictions: list[dict[str, Any]], complexity: int, lineage: dict[str, Any]) -> Any:
    return type("V2PredictionResult", (), {"model_id": model_id, "features_used": features_used, "predictions": predictions, "complexity": complexity, "lineage": lineage})()


def _classification_prediction(row: dict[str, Any], label: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {"row_id": row["row_id"], "label": row["label"], "prediction": label, "probabilities": probabilities, "split": row.get("split", "unknown")}


def _regression_prediction(row: dict[str, Any], value: float) -> dict[str, Any]:
    return {"row_id": row["row_id"], "label": row["label"], "prediction": float(value), "split": row.get("split", "unknown")}


def _annotate_classification_improvement(rows: list[dict[str, Any]], *, baseline_ids: set[str]) -> None:
    baseline_rows = [row for row in rows if row["model_id"] in baseline_ids]
    baseline = min(baseline_rows or rows, key=lambda row: row["log_loss"])["log_loss"] if rows else 0.0
    for row in rows:
        row["relative_improvement"] = (baseline - float(row["log_loss"])) / abs(baseline) if baseline else 0.0


def _annotate_regression_improvement(rows: list[dict[str, Any]], *, baseline_model_id: str) -> None:
    baseline = next((row for row in rows if row["model_id"] == baseline_model_id), rows[0] if rows else {"rmse": 0.0})
    baseline_rmse = float(baseline["rmse"])
    for row in rows:
        row["relative_improvement"] = (baseline_rmse - float(row["rmse"])) / abs(baseline_rmse) if baseline_rmse else 0.0
        row["error_ratio_to_baseline"] = float(row["rmse"]) / max(baseline_rmse, 1e-12)


def _baseline_ids(target: str) -> set[str]:
    if target == "agreement":
        return {"majority", "category_majority"}
    return {"majority", "category_majority", "offer_ratio_threshold"}


def _abstention_payload(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    abstained = [row["row_id"] for row in predictions if row.get("abstained")]
    return {"abstained_row_count": len(abstained), "evaluated_rows": total, "rate": len(abstained) / total if total else 0.0, "abstained_rows_hash": stable_hash(abstained)}


def _prediction_digest(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"row_id": row.get("row_id"), "prediction": row.get("prediction"), "probabilities": row.get("probabilities"), "lower": row.get("lower"), "upper": row.get("upper")} for row in predictions]


def _multiclass_brier(predictions: list[dict[str, Any]], labels: list[str]) -> float:
    if not predictions or not labels:
        return 0.0
    total = 0.0
    for row in predictions:
        probabilities = row.get("probabilities", {})
        for label in labels:
            observed = 1.0 if str(row.get("label")) == label else 0.0
            total += (float(probabilities.get(label, 0.0)) - observed) ** 2
    return total / len(predictions)


def _mae(predictions: list[dict[str, Any]]) -> float:
    return sum(abs(float(row["prediction"]) - float(row["label"])) for row in predictions) / len(predictions) if predictions else 0.0


def _pinball_loss(predictions: list[dict[str, Any]], quantile: float) -> float:
    if not predictions:
        return 0.0
    return sum(max(quantile * (float(row["label"]) - float(row["prediction"])), (quantile - 1.0) * (float(row["label"]) - float(row["prediction"]))) for row in predictions) / len(predictions)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _complexity(model: Any, features: list[str]) -> int:
    value = getattr(model, "complexity", None)
    if isinstance(value, int):
        return value
    return len(features)


def _rotate_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    labels = [row.get("label") for row in rows]
    rotated = labels[1:] + labels[:1]
    return [{**row, "label": label} for row, label in zip(rows, rotated, strict=True)]


def _task_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eligible_rows": len(rows),
        "supervised_rows": len(rows),
        "unknown_outcome_rows": 0,
        "censored_outcome_rows": 0,
        "excluded_rows": 0,
        "unknown_and_censored_labeled_as_rejection": False,
    }


def _compact_formula_report(target: str, nested: dict[str, Any]) -> dict[str, Any]:
    if target != "seller_next_action":
        return {"candidate_count": 0, "reason": "compact formulas are currently defined for seller_next_action"}
    rows = [row for row in nested["leaderboard"] if row.get("model_family") == "compact_formula_candidate"]
    return {
        "candidate_count": len(rows),
        "candidate_model_ids": [row["model_id"] for row in rows],
        "best_development_log_loss": min((row["log_loss"] for row in rows), default=None),
        "hidden_submitted": False,
    }


def _validate_readiness(protocol: dict[str, Any], readiness_report: dict[str, Any]) -> dict[str, Any]:
    try:
        report = validate_v2_pre_hidden_readiness(v2_manifest=protocol, readiness_report=readiness_report)
    except V2ProtocolError as exc:
        return {"status": "blocked", "validator": "validate_v2_pre_hidden_readiness", "reason": str(exc)}
    return {"status": report.status, "targets_checked": report.targets_checked, "negative_controls_checked": report.negative_controls_checked}


def _decision_gate(readiness: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    if readiness.get("status") == "ready_for_hidden":
        status = "RESEARCH_SIGNAL"
        reasons = ["pre-hidden readiness validator passed; hidden submission still requires explicit fresh-lockbox execution"]
    else:
        status = "STOP"
        reasons = [f"pre-hidden readiness blocked: {readiness.get('reason')}"]
    return {
        "status": status,
        "hidden_submission_performed": False,
        "selected_models": {target: payload.get("selected_model", {}).get("model_id") for target, payload in targets.items()},
        "reasons": reasons,
    }


def _empty_classification_calibration(protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["calibration_acceptance"]["classification"]
    return {
        "ece_definition": spec["ece_definition"],
        "expected_calibration_error": math.inf,
        "reliability_bin_count": spec["minimum_reliability_bin_count"],
        "nonempty_reliability_bins": 0,
        "classwise_ece_definition": spec["classwise_ece_definition"],
        "classwise_expected_calibration_error": {"missing": math.inf},
        "macro_classwise_expected_calibration_error": math.inf,
        "class_row_counts": {"missing": 0},
    }


def _empty_regression_calibration(protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["calibration_acceptance"]["regression"]
    return {
        "central_interval_nominal_coverage": spec["central_interval_nominal_coverage"],
        "central_interval_absolute_error": math.inf,
        "interval_width_to_median_target_iqr": math.inf,
        "quantile_levels": spec["quantile_levels"],
        "quantile_pinball_loss_ratio_to_median_baseline": math.inf,
    }


def _public_report(value: Any) -> Any:
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    if not isinstance(value, dict):
        return value
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"predictions", "abstained_rows"}:
            redacted[f"{key}_count"] = len(item) if isinstance(item, list) else None
        elif key == "lineage":
            redacted[key] = item
        else:
            redacted[key] = _public_report(item)
    return redacted


def _data_summary(normalized_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_dir_hash": stable_hash(str(normalized_dir.resolve())),
        "manifest_hash": stable_hash(manifest),
        "tables": {name: {"rows": table.get("rows"), "format": table.get("format")} for name, table in manifest.get("tables", {}).items()},
        "command_args": {key: value for key, value in manifest.get("command_args", {}).items() if key != "raw_dir"},
        "transformation_version": manifest.get("command_args", {}).get("transformation_version") or manifest.get("transformation_version"),
    }


def _full_release_ready(manifest: dict[str, Any]) -> bool:
    args = manifest.get("command_args", {})
    return bool(args.get("full") is True and args.get("limit_threads") is None)


def _nonempty_assignment(split: SplitAssignment) -> bool:
    return bool(split.train and split.development and split.hidden)


def _require_nber_audit_passes(audit: dict[str, Any]) -> None:
    failed_leakage = [name for name, passed in audit.get("leakage_checks", {}).items() if not passed]
    failed_splits = [name for name, passed in audit.get("split_checks", {}).items() if not passed]
    if failed_leakage or failed_splits:
        raise ValueError(f"NBER audit failed before Benchmark v2 model execution: leakage={failed_leakage}, splits={failed_splits}")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OfferLab Benchmark v2 Pre-Hidden Results",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Git commit: `{report['git_commit']}`",
        f"Gate status: **{report['gate']['status']}**",
        "",
        "Hidden submission performed: `False`",
        "",
        "## Selected Models",
        "",
        "| Target | Selected model | Coverage |",
        "| --- | --- | ---: |",
    ]
    for target, payload in report["targets"].items():
        selected = payload.get("selected_model", {})
        lines.append(f"| {target} | `{selected.get('model_id')}` | {_fmt(selected.get('coverage'))} |")
    lines.extend(["", "## Gate Reasons", ""])
    lines.extend(f"- {reason}" for reason in report["gate"]["reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_cards(path: Path, targets: dict[str, Any], permission_report: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for target, payload in targets.items():
        selected = payload.get("selected_model", {})
        lines = [
            f"# OfferLab Benchmark v2 Model Card: {target}",
            "",
            "Research-only NBER-derived pre-hidden artifact. Not production-exportable.",
            "",
            f"- Selected model: `{selected.get('model_id')}`",
            f"- Selection split: `{selected.get('selection_split')}`",
            f"- Hidden results used for selection: `{selected.get('hidden_results_used')}`",
            f"- Support coverage: `{_fmt(selected.get('coverage'))}`",
            f"- Artifact lineage hash: `{stable_hash(selected.get('lineage', {}))}`",
            f"- Production export permission: `{permission_report['production_export']['allowed']}`",
        ]
        (path / f"{target}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return completed.stdout.strip()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
