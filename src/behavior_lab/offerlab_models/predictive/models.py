from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import median
from typing import Any

from behavior_lab.benchmarks.metrics import classification_accuracy, multiclass_log_loss, regression_rmse
from behavior_lab.datasets.nber_best_offer.baselines import CategoryMajorityClassifier, MajorityClassifier, MedianRegressor, OfferRatioThresholdClassifier
from behavior_lab.offerlab_models.common import (
    EVIDENCE_ROLE,
    FEATURE_CONTRACT,
    FORBIDDEN_MODEL_FIELDS,
    FeatureEncoder,
    PRODUCTION_EXPORT_ALLOWED,
    enriched_features,
    model_lineage,
    normalize_probabilities,
    outside_support,
    support_abstention_report,
    support_profile,
    validate_feature_contract,
)


@dataclass
class PredictionResult:
    model_id: str
    features_used: list[str]
    predictions: list[dict[str, Any]]
    complexity: int
    lineage: dict[str, Any]


class RegularizedLogisticClassifier:
    def __init__(self, *, l2: float = 0.05, iterations: int = 120, learning_rate: float = 0.15) -> None:
        self.model_id = "regularized_glm"
        self.l2 = l2
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.labels: list[str] = []
        self.encoder = FeatureEncoder()
        self.weights: dict[str, list[float]] = {}
        self.profile: dict[str, Any] = {}
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "RegularizedLogisticClassifier":
        self.labels = sorted({str(row["label"]) for row in rows})
        if not self.labels:
            self.labels = ["unknown"]
        self.encoder.fit(rows)
        self.profile = support_profile(rows)
        width = len(self.encoder.output_names) + 1
        self.weights = {label: [0.0] * width for label in self.labels}
        vectors = [[1.0] + vector for vector in self.encoder.transform(rows)]
        for _ in range(self.iterations):
            for row, vector in zip(rows, vectors):
                expected = self._softmax(vector)
                true_label = str(row["label"])
                for label in self.labels:
                    error = (1.0 if label == true_label else 0.0) - expected[label]
                    for index, value in enumerate(vector):
                        penalty = self.l2 * self.weights[label][index] if index else 0.0
                        self.weights[label][index] += self.learning_rate * (error * value - penalty) / max(1, len(rows))
        self.lineage = model_lineage(self.model_id, rows, feature_contract=self.encoder.output_names)
        return self

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResult:
        predictions = []
        for row in rows:
            abstained = outside_support(row, self.profile)
            probabilities = self._softmax([1.0] + self.encoder.transform_one(row))
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": "abstain" if abstained else prediction,
                    "probabilities": probabilities,
                    "split": row.get("split", "unknown"),
                    "abstained": abstained,
                }
            )
        return PredictionResult(self.model_id, list(self.encoder.output_names), predictions, self.complexity, self.lineage)

    @property
    def complexity(self) -> int:
        return sum(1 for values in self.weights.values() for value in values if abs(value) > 1e-6)

    def _softmax(self, vector: list[float]) -> dict[str, float]:
        scores = {}
        for label in self.labels:
            scores[label] = sum(weight * value for weight, value in zip(self.weights[label], vector))
        offset = max(scores.values()) if scores else 0.0
        exps = {label: math.exp(score - offset) for label, score in scores.items()}
        return normalize_probabilities(exps, self.labels)


class SmoothedOfferHistogramClassifier:
    def __init__(self, *, bins: int = 8, smoothing: float = 1.0) -> None:
        self.model_id = "smoothed_offer_histogram"
        self.bins = bins
        self.smoothing = smoothing
        self.labels: list[str] = []
        self.global_counts: Counter[str] = Counter()
        self.bucket_counts: dict[tuple[str, int], Counter[str]] = {}
        self.profile: dict[str, Any] = {}
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "SmoothedOfferHistogramClassifier":
        if not validate_feature_contract(rows):
            raise ValueError("rows contain forbidden participant or outcome fields inside features")
        self.labels = sorted({str(row["label"]) for row in rows}) or ["unknown"]
        self.global_counts = Counter(str(row["label"]) for row in rows)
        grouped: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        for row in rows:
            features = enriched_features(row)
            grouped[(str(features.get("category", "missing")), self._bin(features.get("offer_to_asking_ratio")))][str(row["label"])] += 1
        self.bucket_counts = dict(grouped)
        self.profile = support_profile(rows)
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["category", "offer_to_asking_ratio"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResult:
        predictions = []
        for row in rows:
            features = enriched_features(row)
            key = (str(features.get("category", "missing")), self._bin(features.get("offer_to_asking_ratio")))
            counts = self.bucket_counts.get(key, Counter())
            raw = {
                label: counts[label] + self.smoothing * (self.global_counts[label] + self.smoothing)
                for label in self.labels
            }
            probabilities = normalize_probabilities(raw, self.labels)
            abstained = outside_support(row, self.profile)
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": "abstain" if abstained else prediction,
                    "probabilities": probabilities,
                    "split": row.get("split", "unknown"),
                    "abstained": abstained,
                }
            )
        return PredictionResult(self.model_id, ["category", "offer_to_asking_ratio"], predictions, len(self.bucket_counts), self.lineage)

    def _bin(self, value: Any) -> int:
        ratio = max(0.0, min(1.5, float(value or 0.0)))
        return min(self.bins - 1, int(ratio / 1.5 * self.bins))


class DeterministicStumpEnsembleClassifier:
    def __init__(self) -> None:
        self.model_id = "deterministic_stump_ensemble"
        self.labels: list[str] = []
        self.stumps: list[dict[str, Any]] = []
        self.profile: dict[str, Any] = {}
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "DeterministicStumpEnsembleClassifier":
        if not validate_feature_contract(rows):
            raise ValueError("rows contain forbidden participant or outcome fields inside features")
        self.labels = sorted({str(row["label"]) for row in rows}) or ["unknown"]
        self.profile = support_profile(rows)
        self.stumps = []
        for feature in ["offer_to_asking_ratio", "round_number", "prior_counter_count"]:
            values = [float(enriched_features(row).get(feature) or 0.0) for row in rows]
            threshold = median(values) if values else 0.0
            branches = {"left": Counter(), "right": Counter()}
            for row in rows:
                branch = "right" if float(enriched_features(row).get(feature) or 0.0) >= threshold else "left"
                branches[branch][str(row["label"])] += 1
            self.stumps.append({"feature": feature, "threshold": threshold, "branches": branches})
        for feature in ["category", "current_action"]:
            branches: dict[str, Counter[str]] = defaultdict(Counter)
            for row in rows:
                branches[str(enriched_features(row).get(feature, "missing"))][str(row["label"])] += 1
            self.stumps.append({"feature": feature, "branches": dict(branches)})
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["offer_to_asking_ratio", "round_number", "prior_counter_count", "category", "current_action"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResult:
        predictions = []
        for row in rows:
            votes = Counter()
            features = enriched_features(row)
            for stump in self.stumps:
                if "threshold" in stump:
                    branch = "right" if float(features.get(stump["feature"]) or 0.0) >= stump["threshold"] else "left"
                    votes.update(stump["branches"][branch])
                else:
                    votes.update(stump["branches"].get(str(features.get(stump["feature"], "missing")), Counter()))
            probabilities = normalize_probabilities({label: votes[label] + 1.0 for label in self.labels}, self.labels)
            abstained = outside_support(row, self.profile)
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": "abstain" if abstained else prediction,
                    "probabilities": probabilities,
                    "split": row.get("split", "unknown"),
                    "abstained": abstained,
                }
            )
        return PredictionResult(self.model_id, ["offer_to_asking_ratio", "round_number", "prior_counter_count", "category", "current_action"], predictions, len(self.stumps), self.lineage)


class MonotonicOfferClassifier:
    def __init__(self, *, positive_label: str = "accept", bins: int = 8) -> None:
        self.model_id = "monotonic_offer_model"
        self.positive_label = positive_label
        self.bins = bins
        self.labels: list[str] = []
        self.accept_curve: list[float] = []
        self.non_accept_distribution: dict[str, float] = {}
        self.profile: dict[str, Any] = {}
        self.lineage: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "MonotonicOfferClassifier":
        self.labels = sorted({str(row["label"]) for row in rows}) or [self.positive_label]
        counts = [Counter() for _ in range(self.bins)]
        for row in rows:
            counts[self._bin(enriched_features(row).get("offer_to_asking_ratio"))][str(row["label"])] += 1
        rates = []
        weights = []
        for counter in counts:
            total = sum(counter.values())
            rates.append((counter[self.positive_label] + 0.5) / (total + 1.0))
            weights.append(max(total, 1))
        self.accept_curve = _pava(rates, weights)
        non_accept = Counter(str(row["label"]) for row in rows if str(row["label"]) != self.positive_label)
        total_non_accept = sum(non_accept.values())
        other_labels = [label for label in self.labels if label != self.positive_label]
        self.non_accept_distribution = {
            label: (non_accept[label] + 1.0) / (total_non_accept + len(other_labels))
            for label in other_labels
        }
        self.profile = support_profile(rows)
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["offer_to_asking_ratio"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResult:
        predictions = []
        for row in rows:
            positive = self.accept_curve[self._bin(enriched_features(row).get("offer_to_asking_ratio"))] if self.accept_curve else 0.5
            raw = {label: (1.0 - positive) * self.non_accept_distribution.get(label, 0.0) for label in self.labels}
            raw[self.positive_label] = positive
            probabilities = normalize_probabilities(raw, self.labels)
            abstained = outside_support(row, self.profile)
            prediction = max(probabilities, key=probabilities.get)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": "abstain" if abstained else prediction,
                    "probabilities": probabilities,
                    "split": row.get("split", "unknown"),
                    "abstained": abstained,
                }
            )
        return PredictionResult(self.model_id, ["offer_to_asking_ratio"], predictions, len(self.accept_curve), self.lineage)

    def _bin(self, value: Any) -> int:
        ratio = max(0.0, min(1.5, float(value or 0.0)))
        return min(self.bins - 1, int(ratio / 1.5 * self.bins))


class EmpiricalQuantileRegressor:
    def __init__(self, *, lower: float = 0.1, upper: float = 0.9) -> None:
        self.model_id = "empirical_category_quantiles"
        self.lower = lower
        self.upper = upper
        self.global_quantiles: dict[str, float] = {}
        self.by_category: dict[str, dict[str, float]] = {}
        self.lineage: dict[str, Any] = {}
        self.profile: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "EmpiricalQuantileRegressor":
        labels = [float(row["label"]) for row in rows]
        self.global_quantiles = self._quantiles(labels)
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(enriched_features(row).get("category", "missing"))].append(float(row["label"]))
        self.by_category = {category: self._quantiles(values) for category, values in grouped.items()}
        self.profile = support_profile(rows)
        self.lineage = model_lineage(self.model_id, rows, feature_contract=["category"])
        return self

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResult:
        predictions = []
        for row in rows:
            quantiles = self.by_category.get(str(enriched_features(row).get("category", "missing")), self.global_quantiles)
            abstained = outside_support(row, self.profile)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": quantiles.get("median", 0.0),
                    "lower": quantiles.get("lower", 0.0),
                    "upper": quantiles.get("upper", 0.0),
                    "split": row.get("split", "unknown"),
                    "abstained": abstained,
                }
            )
        return PredictionResult(self.model_id, ["category"], predictions, len(self.by_category) + 3, self.lineage)

    def _quantiles(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"lower": 0.0, "median": 0.0, "upper": 0.0}
        ordered = sorted(values)
        return {"lower": _pick_quantile(ordered, self.lower), "median": _pick_quantile(ordered, 0.5), "upper": _pick_quantile(ordered, self.upper)}


class PredictiveHiddenLockbox:
    _used_ids: set[str] = set()

    @classmethod
    def submit_once(cls, lockbox_id: str, rows: list[dict[str, Any]]) -> None:
        if not lockbox_id.strip():
            raise ValueError("lockbox_id is required")
        if lockbox_id in cls._used_ids:
            raise RuntimeError("predictive hidden lockbox already used")
        cls._used_ids.add(lockbox_id)


def predictive_suite(
    task_name: str,
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    *,
    hidden_lockbox_id: str | None = None,
) -> dict[str, Any]:
    if task_name in {"final_price_ratio", "response_latency"}:
        return _regression_suite(task_name, train, development, hidden, hidden_lockbox_id=hidden_lockbox_id)
    return _classification_suite(task_name, train, development, hidden, hidden_lockbox_id=hidden_lockbox_id)


def _classification_suite(
    task_name: str,
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    *,
    hidden_lockbox_id: str | None,
) -> dict[str, Any]:
    labels = sorted({str(row["label"]) for row in train + development + hidden})
    models: list[Any] = [MajorityClassifier().fit(train), CategoryMajorityClassifier().fit(train)]
    if task_name == "seller_next_action":
        models.append(OfferRatioThresholdClassifier().fit(train))
    models.extend(
        [
            RegularizedLogisticClassifier().fit(train),
            SmoothedOfferHistogramClassifier().fit(train),
            DeterministicStumpEnsembleClassifier().fit(train),
            MonotonicOfferClassifier(positive_label="accept").fit(train),
        ]
    )
    boards: dict[str, list[dict[str, Any]]] = {"development": [], "hidden": []}
    train_profile = support_profile(train)
    for model in models:
        if development:
            boards["development"].append(_classification_score(model, train, development, "development", labels, train_profile, task_name))
    boards["development"].sort(key=lambda item: (item["log_loss"], item["complexity"]))
    hidden_lockbox: dict[str, Any] = {
        "submitted": False,
        "hidden_rows_reserved": len(hidden),
        "reason": "hidden evaluation requires an explicit hidden_lockbox_id",
    }
    if hidden_lockbox_id is not None and hidden and boards["development"]:
        PredictiveHiddenLockbox.submit_once(f"{hidden_lockbox_id}:{task_name}", hidden)
        selected_model_id = boards["development"][0]["model_id"]
        selected_model = next(model for model in models if getattr(model, "model_id", "") == selected_model_id or getattr(model.predict([]), "model_id", "") == selected_model_id)
        boards["hidden"].append(_classification_score(selected_model, train, hidden, "hidden", labels, train_profile, task_name))
        hidden_lockbox = {
            "submitted": True,
            "hidden_submission_count": 1,
            "hidden_rows": len(hidden),
            "selected_model_id": selected_model_id,
            "lockbox_id": f"{hidden_lockbox_id}:{task_name}",
        }
    return {
        "task": task_name,
        "task_type": "classification",
        "evidence_role": EVIDENCE_ROLE,
        "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
        "feature_contract": FEATURE_CONTRACT,
        "forbidden_features": sorted(FORBIDDEN_MODEL_FIELDS),
        "participant_id_features_used": False,
        "support": {
            "development": support_abstention_report(train, development),
            "hidden_reserved": support_abstention_report(train, hidden),
        },
        "leaderboards": boards,
        "hidden_lockbox": hidden_lockbox,
        "universal_winner": None,
    }


def _regression_suite(
    task_name: str,
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    *,
    hidden_lockbox_id: str | None,
) -> dict[str, Any]:
    models: list[Any] = [MedianRegressor().fit(train), EmpiricalQuantileRegressor().fit(train)]
    boards: dict[str, list[dict[str, Any]]] = {"development": [], "hidden": []}
    train_profile = support_profile(train)
    for model in models:
        if development:
            boards["development"].append(_regression_score(model, train, development, "development", train_profile, task_name))
    boards["development"].sort(key=lambda item: (item["rmse"], item["complexity"]))
    hidden_lockbox: dict[str, Any] = {
        "submitted": False,
        "hidden_rows_reserved": len(hidden),
        "reason": "hidden evaluation requires an explicit hidden_lockbox_id",
    }
    if hidden_lockbox_id is not None and hidden and boards["development"]:
        PredictiveHiddenLockbox.submit_once(f"{hidden_lockbox_id}:{task_name}", hidden)
        selected_model_id = boards["development"][0]["model_id"]
        selected_model = next(model for model in models if getattr(model, "model_id", "") == selected_model_id)
        boards["hidden"].append(_regression_score(selected_model, train, hidden, "hidden", train_profile, task_name))
        hidden_lockbox = {
            "submitted": True,
            "hidden_submission_count": 1,
            "hidden_rows": len(hidden),
            "selected_model_id": selected_model_id,
            "lockbox_id": f"{hidden_lockbox_id}:{task_name}",
        }
    return {
        "task": task_name,
        "task_type": "regression",
        "evidence_role": EVIDENCE_ROLE,
        "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
        "feature_contract": FEATURE_CONTRACT,
        "forbidden_features": sorted(FORBIDDEN_MODEL_FIELDS),
        "participant_id_features_used": False,
        "support": {
            "development": support_abstention_report(train, development),
            "hidden_reserved": support_abstention_report(train, hidden),
        },
        "leaderboards": boards,
        "hidden_lockbox": hidden_lockbox,
        "universal_winner": None,
    }


def _classification_score(model: Any, train: list[dict[str, Any]], rows: list[dict[str, Any]], split_name: str, labels: list[str], train_profile: dict[str, Any], task_name: str) -> dict[str, Any]:
    result = model.predict(rows)
    predictions = _tag_support(getattr(result, "predictions", result.predictions), rows, train_profile, abstain_prediction=True)
    covered = [row for row in predictions if not row.get("abstained")]
    row = {
        "model_id": getattr(result, "model_id", getattr(model, "model_id", "unknown")),
        "task": task_name,
        "split": split_name,
        "accuracy": classification_accuracy(predictions),
        "log_loss": multiclass_log_loss(predictions, labels=labels),
        "coverage": len(covered) / len(predictions) if predictions else 0.0,
        "covered_accuracy": classification_accuracy(covered) if covered else None,
        "covered_log_loss": multiclass_log_loss(covered, labels=labels) if covered else None,
        "complexity": getattr(result, "complexity", len(getattr(result, "features_used", []))),
        "features_used": list(getattr(result, "features_used", [])),
        "lineage": getattr(result, "lineage", model_lineage(getattr(result, "model_id", "baseline"), train, feature_contract=list(getattr(result, "features_used", [])))),
    }
    return row


def _regression_score(model: Any, train: list[dict[str, Any]], rows: list[dict[str, Any]], split_name: str, train_profile: dict[str, Any], task_name: str) -> dict[str, Any]:
    result = model.predict(rows)
    predictions = _tag_support(getattr(result, "predictions", result.predictions), rows, train_profile, abstain_prediction=False)
    covered = [row for row in predictions if not row.get("abstained")]
    row = {
        "model_id": getattr(result, "model_id", getattr(model, "model_id", "unknown")),
        "task": task_name,
        "split": split_name,
        "rmse": regression_rmse(predictions),
        "coverage": len(covered) / len(predictions) if predictions else 0.0,
        "covered_rmse": regression_rmse(covered) if covered else None,
        "complexity": getattr(result, "complexity", len(getattr(result, "features_used", []))),
        "features_used": list(getattr(result, "features_used", [])),
        "lineage": getattr(result, "lineage", model_lineage(getattr(result, "model_id", "baseline"), train, feature_contract=list(getattr(result, "features_used", [])))),
    }
    if predictions and "lower" in predictions[0]:
        row["interval_coverage"] = sum(1 for item in predictions if item["lower"] <= float(item["label"]) <= item["upper"]) / len(predictions)
    return row


def _tag_support(predictions: list[dict[str, Any]], rows: list[dict[str, Any]], train_profile: dict[str, Any], *, abstain_prediction: bool) -> list[dict[str, Any]]:
    tagged = []
    for prediction, row in zip(predictions, rows):
        item = dict(prediction)
        abstained = bool(item.get("abstained", False) or outside_support(row, train_profile))
        item["abstained"] = abstained
        if abstained and abstain_prediction:
            item["prediction"] = "abstain"
        tagged.append(item)
    return tagged


def _pava(values: list[float], weights: list[int]) -> list[float]:
    blocks = [{"value": value, "weight": float(weight), "count": 1} for value, weight in zip(values, weights)]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index]["value"] <= blocks[index + 1]["value"]:
            index += 1
            continue
        total_weight = blocks[index]["weight"] + blocks[index + 1]["weight"]
        merged = {
            "value": (blocks[index]["value"] * blocks[index]["weight"] + blocks[index + 1]["value"] * blocks[index + 1]["weight"]) / total_weight,
            "weight": total_weight,
            "count": blocks[index]["count"] + blocks[index + 1]["count"],
        }
        blocks[index : index + 2] = [merged]
        index = max(index - 1, 0)
    output = []
    for block in blocks:
        output.extend([float(block["value"])] * int(block["count"]))
    return output


def _pick_quantile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])
