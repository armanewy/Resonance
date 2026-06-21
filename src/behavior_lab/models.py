from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Any

from behavior_lab.core import HypothesisSpec, new_id, stable_hash, to_jsonable
from behavior_lab.dsl import Formula
from behavior_lab.evaluation import evaluate_model
from behavior_lab.temporal import feature_catalog

MODEL_ARTIFACT_VERSION = 2
SOFTWARE_VERSION = "0.4.0"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(float(value), 50.0), -50.0)))


def numeric_features(rows: list[dict[str, Any]]) -> list[str]:
    return feature_catalog(rows)


def _smoothed_rate(targets: list[int], default: float = 0.5) -> float:
    if not targets:
        return default
    return (sum(targets) + 1.0) / (len(targets) + 2.0)


def _feature_stats(rows: list[dict[str, Any]], features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in features:
        values = [float(row["features"].get(name, 0.0)) for row in rows]
        avg = mean(values) if values else 0.0
        scale = pstdev(values) if len(values) > 1 else 1.0
        means[name] = avg
        scales[name] = scale if scale > 1e-9 else 1.0
    return means, scales


def _feature_target_association(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        return 0.0
    positives = [float(row["features"].get(name, 0.0)) for row in rows if int(row["target"]) == 1]
    negatives = [float(row["features"].get(name, 0.0)) for row in rows if int(row["target"]) == 0]
    if not positives or not negatives:
        return 0.0
    all_values = positives + negatives
    scale = pstdev(all_values) if len(all_values) > 1 else 1.0
    return abs(mean(positives) - mean(negatives)) / max(scale, 1e-9)


def _rank_features(rows: list[dict[str, Any]]) -> list[str]:
    names = numeric_features(rows)
    return sorted(names, key=lambda name: (-_feature_target_association(rows, name), name))


@dataclass
class BaseRateModel:
    model_id: str
    rate: float
    complexity: int = 1
    origin: str = "baseline"

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.rate


@dataclass
class RecentActionBaseline:
    model_id: str
    recent_rate: float
    complexity: int = 2
    origin: str = "baseline"

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.recent_rate


@dataclass
class NearestNeighborModel:
    model_id: str
    rows: list[dict[str, Any]]
    features: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    complexity: int = 8
    origin: str = "generic_model"

    def predict_proba(self, features: dict[str, Any]) -> float:
        if not self.rows:
            return 0.5
        best = min(self.rows, key=lambda row: self._distance(features, row["features"]))
        return 0.8 if best["target"] else 0.2

    def _distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        total = 0.0
        for name in self.features:
            scale = self.scales.get(name, 1.0)
            total += ((float(a.get(name, 0.0)) - float(b.get(name, 0.0))) / scale) ** 2
        return total


@dataclass
class FittedLogisticFormula:
    model_id: str
    hypothesis_id: str
    formula: Formula
    weights: list[float]
    complexity: int
    origin: str = "submitted"

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.formula.terms) + 1:
            raise ValueError("Formula artifact has a weight/term length mismatch")
        if any(not math.isfinite(float(weight)) for weight in self.weights):
            raise ValueError("Formula weights must be finite")

    def predict_proba(self, features: dict[str, Any]) -> float:
        vector = self.formula.vector(features)
        logit = sum(weight * value for weight, value in zip(self.weights, vector, strict=True))
        return sigmoid(logit)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "intercept": self.weights[0],
            "origin": self.origin,
            "terms": [
                {"expression": term.expression, "coefficient": self.weights[index + 1]}
                for index, term in enumerate(self.formula.terms)
            ],
        }


class LogisticFormulaHypothesis:
    def __init__(self, spec: HypothesisSpec):
        self.spec = spec
        self.formula = Formula.parse(list(spec.structure.get("terms", [])))
        self.origin = str(spec.structure.get("origin", "submitted"))

    def fit(
        self,
        rows: list[dict[str, Any]],
        *,
        learning_rate: float = 0.08,
        iterations: int = 220,
        l2: float = 0.01,
    ) -> FittedLogisticFormula:
        if not rows:
            weights = [0.0] * (len(self.formula.terms) + 1)
            return FittedLogisticFormula(
                new_id("m"), self.spec.hypothesis_id, self.formula, weights, self.formula.complexity, self.origin
            )
        vectors = [self.formula.vector(row["features"]) for row in rows]
        targets = [int(row["target"]) for row in rows]
        width = len(vectors[0])
        column_means = [0.0] * width
        column_scales = [1.0] * width
        for index in range(1, width):
            values = [vector[index] for vector in vectors]
            column_means[index] = mean(values)
            scale = pstdev(values) if len(values) > 1 else 1.0
            column_scales[index] = scale if scale > 1e-9 else 1.0
        standardized = [
            [1.0]
            + [
                (vector[index] - column_means[index]) / column_scales[index]
                for index in range(1, width)
            ]
            for vector in vectors
        ]
        standardized_weights = [0.0] * width
        previous_loss = float("inf")
        for iteration in range(iterations):
            gradients = [0.0] * width
            loss = 0.0
            for vector, target in zip(standardized, targets, strict=True):
                prediction = sigmoid(sum(weight * value for weight, value in zip(standardized_weights, vector, strict=True)))
                error = prediction - target
                loss += -(target * math.log(max(prediction, 1e-12)) + (1 - target) * math.log(max(1 - prediction, 1e-12)))
                for index, value in enumerate(vector):
                    gradients[index] += error * value
            scale = 1.0 / len(rows)
            for index in range(width):
                penalty = l2 * standardized_weights[index] if index > 0 else 0.0
                standardized_weights[index] -= learning_rate * (gradients[index] * scale + penalty)
            average_loss = loss * scale
            if iteration > 20 and abs(previous_loss - average_loss) < 1e-8:
                break
            previous_loss = average_loss

        # Convert back to the original formula scale so coefficients remain
        # directly interpretable and artifacts require no hidden preprocessing.
        weights = [0.0] * width
        weights[0] = standardized_weights[0]
        for index in range(1, width):
            weights[index] = standardized_weights[index] / column_scales[index]
            weights[0] -= standardized_weights[index] * column_means[index] / column_scales[index]
        return FittedLogisticFormula(
            model_id=new_id("m"),
            hypothesis_id=self.spec.hypothesis_id,
            formula=self.formula,
            weights=weights,
            complexity=self.formula.complexity,
            origin=self.origin,
        )


@dataclass
class ThresholdRuleModel:
    model_id: str
    variable: str
    threshold: float
    low_rate: float
    high_rate: float
    complexity: int = 4
    origin: str = "generic_model"

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.high_rate if float(features.get(self.variable, 0.0)) > self.threshold else self.low_rate


def _quantile_thresholds(values: list[float], limit: int = 25) -> list[float]:
    unique = sorted(set(values))
    if len(unique) < 2:
        return []
    boundaries = [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    if len(boundaries) <= limit:
        return boundaries
    indices = sorted({round(index * (len(boundaries) - 1) / (limit - 1)) for index in range(limit)})
    return [boundaries[index] for index in indices]


def fit_threshold_rule(rows: list[dict[str, Any]], variables: list[str] | None = None) -> ThresholdRuleModel:
    variables = variables or numeric_features(rows)
    if not rows:
        return ThresholdRuleModel(new_id("m"), "none", 0.0, 0.5, 0.5)
    best_model: ThresholdRuleModel | None = None
    best_loss = float("inf")
    for variable in variables:
        values = [float(row["features"].get(variable, 0.0)) for row in rows]
        for threshold in _quantile_thresholds(values):
            low = [int(row["target"]) for row in rows if float(row["features"].get(variable, 0.0)) <= threshold]
            high = [int(row["target"]) for row in rows if float(row["features"].get(variable, 0.0)) > threshold]
            if not low or not high:
                continue
            model = ThresholdRuleModel(
                new_id("m"), variable, threshold, _smoothed_rate(low), _smoothed_rate(high)
            )
            metrics = evaluate_model(model, rows, split="training")
            if metrics.log_loss < best_loss:
                best_loss = metrics.log_loss
                best_model = model
    return best_model or ThresholdRuleModel(new_id("m"), variables[0] if variables else "none", 0.0, 0.5, 0.5)


@dataclass
class DecisionStumpModel:
    model_id: str
    variable: str
    threshold: float
    left_rate: float
    right_rate: float
    complexity: int = 5
    origin: str = "generic_model"

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.right_rate if float(features.get(self.variable, 0.0)) > self.threshold else self.left_rate


def fit_small_tree(rows: list[dict[str, Any]]) -> DecisionStumpModel:
    rule = fit_threshold_rule(rows)
    return DecisionStumpModel(rule.model_id, rule.variable, rule.threshold, rule.low_rate, rule.high_rate)


@dataclass
class TwoStateModeModel:
    model_id: str
    features: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    centroid_a: dict[str, float]
    centroid_b: dict[str, float]
    state_a_rate: float
    state_b_rate: float
    complexity: int = 7
    origin: str = "generic_model"

    def predict_proba(self, features: dict[str, Any]) -> float:
        if not self.features:
            return (self.state_a_rate + self.state_b_rate) / 2
        da = self._distance(features, self.centroid_a)
        db = self._distance(features, self.centroid_b)
        # Soft assignment prevents hard discontinuities in sparse data.
        wa = math.exp(-min(da, 50.0))
        wb = math.exp(-min(db, 50.0))
        total = wa + wb
        if total <= 0:
            return (self.state_a_rate + self.state_b_rate) / 2
        return (wa * self.state_a_rate + wb * self.state_b_rate) / total

    def _distance(self, values: dict[str, Any], centroid: dict[str, float]) -> float:
        return sum(
            ((float(values.get(name, self.means.get(name, 0.0))) - centroid[name]) / self.scales.get(name, 1.0)) ** 2
            for name in self.features
        )


def fit_two_state_model(rows: list[dict[str, Any]], max_features: int = 6) -> TwoStateModeModel:
    if not rows:
        return TwoStateModeModel(new_id("m"), [], {}, {}, {}, {}, 0.5, 0.5)
    features = _rank_features(rows)[:max_features]
    means, scales = _feature_stats(rows, features)
    if not features:
        rate = _smoothed_rate([int(row["target"]) for row in rows])
        return TwoStateModeModel(new_id("m"), [], means, scales, {}, {}, rate, rate)

    def standardized(row: dict[str, Any]) -> list[float]:
        return [(float(row["features"].get(name, 0.0)) - means[name]) / scales[name] for name in features]

    vectors = [standardized(row) for row in rows]
    first_index = min(range(len(vectors)), key=lambda index: tuple(vectors[index]))
    second_index = max(
        range(len(vectors)),
        key=lambda index: sum((vectors[index][j] - vectors[first_index][j]) ** 2 for j in range(len(features))),
    )
    centroids = [list(vectors[first_index]), list(vectors[second_index])]
    assignments = [0] * len(vectors)
    for _ in range(20):
        changed = False
        for index, vector in enumerate(vectors):
            distances = [sum((value - center[j]) ** 2 for j, value in enumerate(vector)) for center in centroids]
            assignment = 0 if distances[0] <= distances[1] else 1
            changed = changed or assignment != assignments[index]
            assignments[index] = assignment
        for state in [0, 1]:
            members = [vectors[index] for index, assignment in enumerate(assignments) if assignment == state]
            if members:
                centroids[state] = [mean(member[j] for member in members) for j in range(len(features))]
        if not changed:
            break

    targets_a = [int(rows[index]["target"]) for index, state in enumerate(assignments) if state == 0]
    targets_b = [int(rows[index]["target"]) for index, state in enumerate(assignments) if state == 1]
    overall = _smoothed_rate([int(row["target"]) for row in rows])
    centroid_a = {name: centroids[0][index] * scales[name] + means[name] for index, name in enumerate(features)}
    centroid_b = {name: centroids[1][index] * scales[name] + means[name] for index, name in enumerate(features)}
    return TwoStateModeModel(
        new_id("m"),
        features,
        means,
        scales,
        centroid_a,
        centroid_b,
        _smoothed_rate(targets_a, overall),
        _smoothed_rate(targets_b, overall),
    )


class SymbolicSearch:
    """Small, generic symbolic-regression-style search over the safe DSL."""

    def __init__(self, max_terms: int = 5, candidate_limit: int = 18):
        self.max_terms = max_terms
        self.candidate_limit = candidate_limit

    def candidate_terms(self, rows: list[dict[str, Any]]) -> list[str]:
        ranked = _rank_features(rows)
        top = ranked[: min(8, len(ranked))]
        base_terms = list(top)
        threshold_terms: list[str] = []
        for name in top[:6]:
            values = sorted(float(row["features"].get(name, 0.0)) for row in rows)
            if values:
                median_value = values[len(values) // 2]
                threshold_terms.append(f"indicator({name} > {median_value:.6g})")
        interactions = [f"{left} * {right}" for index, left in enumerate(top[:5]) for right in top[index + 1 : 5]]
        candidates: list[str] = []
        for term in base_terms + threshold_terms + interactions:
            if term not in candidates:
                candidates.append(term)
        return candidates[: self.candidate_limit]

    def search(
        self,
        training_rows: list[dict[str, Any]],
        development_rows: list[dict[str, Any]],
        target_name: str,
    ) -> FittedLogisticFormula:
        terms = self.candidate_terms(training_rows)
        selected: list[str] = []
        best_model: FittedLogisticFormula | None = None
        best_loss = float("inf")
        for _ in range(min(self.max_terms, len(terms))):
            best_term: str | None = None
            best_round_model: FittedLogisticFormula | None = None
            best_round_loss = best_loss
            for term in terms:
                if term in selected:
                    continue
                spec = HypothesisSpec.formula(
                    new_id("h_symbolic"), target_name, selected + [term], origin="symbolic_search"
                )
                model = LogisticFormulaHypothesis(spec).fit(training_rows)
                score_rows = development_rows or training_rows
                loss = evaluate_model(model, score_rows, split="development").log_loss
                if loss < best_round_loss - 1e-8:
                    best_round_loss = loss
                    best_term = term
                    best_round_model = model
            if best_term is None or best_round_model is None:
                break
            selected.append(best_term)
            best_model = best_round_model
            best_loss = best_round_loss
        if best_model:
            return best_model
        fallback_terms = terms[:1]
        spec = HypothesisSpec.formula(new_id("h_symbolic"), target_name, fallback_terms, origin="symbolic_search")
        return LogisticFormulaHypothesis(spec).fit(training_rows)


class ModelFoundry:
    def fit_zoo(
        self,
        training_rows: list[dict[str, Any]],
        development_rows: list[dict[str, Any]],
        target_name: str,
    ) -> list[Any]:
        targets = [int(row["target"]) for row in training_rows]
        base_rate = _smoothed_rate(targets)
        recent = training_rows[-25:] if training_rows else []
        recent_rate = _smoothed_rate([int(row["target"]) for row in recent], base_rate)
        features = numeric_features(training_rows)
        means, scales = _feature_stats(training_rows, features)
        models: list[Any] = [
            BaseRateModel(new_id("m"), base_rate),
            RecentActionBaseline(new_id("m"), recent_rate),
            NearestNeighborModel(new_id("m"), list(training_rows), features, means, scales),
            fit_threshold_rule(training_rows),
            fit_small_tree(training_rows),
            fit_two_state_model(training_rows),
        ]
        # Generic all-linear reference. Unlike the old version, this does not
        # hard-code the known drivers of a particular synthetic world.
        linear_terms = _rank_features(training_rows)[:8]
        reference_spec = HypothesisSpec.formula(
            "generic_linear_reference_v2",
            target_name,
            linear_terms,
            origin="generic_reference",
            falsification_conditions=["does not outperform the base-rate baseline on development"],
        )
        models.append(LogisticFormulaHypothesis(reference_spec).fit(training_rows))
        models.append(SymbolicSearch(max_terms=5, candidate_limit=18).search(training_rows, development_rows, target_name))
        return models


def training_snapshot_hash(rows: list[dict[str, Any]]) -> str:
    snapshot = [
        {"case_id": row.get("case_id"), "features": row.get("features", {}), "target": row.get("target")}
        for row in rows
    ]
    return stable_hash(snapshot)


def model_to_artifact(model: Any, training_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    feature_schema = numeric_features(training_rows or [])
    common = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "class": type(model).__name__,
        "model_id": model.model_id,
        "complexity": getattr(model, "complexity", None),
        "origin": getattr(model, "origin", "unknown"),
        "feature_schema": feature_schema,
        "training_snapshot_hash": training_snapshot_hash(training_rows or []),
    }
    if isinstance(model, BaseRateModel):
        common.update({"family": "base_rate", "rate": model.rate})
    elif isinstance(model, RecentActionBaseline):
        common.update({"family": "recent_rate", "recent_rate": model.recent_rate})
    elif isinstance(model, NearestNeighborModel):
        common.update(
            {
                "family": "nearest_neighbor",
                "rows": to_jsonable(model.rows),
                "features": list(model.features),
                "means": model.means,
                "scales": model.scales,
            }
        )
    elif isinstance(model, FittedLogisticFormula):
        common.update(
            {
                "family": "logistic_formula",
                "hypothesis_id": model.hypothesis_id,
                "formula_terms": [term.expression for term in model.formula.terms],
                "weights": list(model.weights),
                "origin": model.origin,
                "parameters": model.parameters,
            }
        )
    elif isinstance(model, ThresholdRuleModel):
        common.update(
            {
                "family": "threshold_rule",
                "variable": model.variable,
                "threshold": model.threshold,
                "low_rate": model.low_rate,
                "high_rate": model.high_rate,
            }
        )
    elif isinstance(model, DecisionStumpModel):
        common.update(
            {
                "family": "decision_stump",
                "variable": model.variable,
                "threshold": model.threshold,
                "left_rate": model.left_rate,
                "right_rate": model.right_rate,
            }
        )
    elif isinstance(model, TwoStateModeModel):
        common.update(
            {
                "family": "two_state_mode",
                "features": model.features,
                "means": model.means,
                "scales": model.scales,
                "centroid_a": model.centroid_a,
                "centroid_b": model.centroid_b,
                "state_a_rate": model.state_a_rate,
                "state_b_rate": model.state_b_rate,
            }
        )
    else:
        raise TypeError(f"Unsupported model type for persistence: {type(model).__name__}")
    common["artifact_hash"] = stable_hash({key: value for key, value in common.items() if key != "artifact_hash"})
    return common


def validate_model_artifact(artifact: dict[str, Any]) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be an object")
    if int(artifact.get("artifact_version", -1)) != MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported model artifact version: {artifact.get('artifact_version')!r}")
    expected_hash = artifact.get("artifact_hash")
    if not expected_hash:
        raise ValueError("Model artifact is missing artifact_hash")
    body = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    if stable_hash(body) != expected_hash:
        raise ValueError("Model artifact hash mismatch")
    if not str(artifact.get("model_id", "")).strip():
        raise ValueError("Model artifact is missing model_id")
    if not str(artifact.get("training_snapshot_hash", "")).strip():
        raise ValueError("Model artifact is missing training_snapshot_hash")
    try:
        complexity = int(artifact.get("complexity", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Model artifact complexity must be an integer") from exc
    if complexity < 0:
        raise ValueError("Model artifact complexity may not be negative")
    feature_schema = artifact.get("feature_schema", [])
    if not isinstance(feature_schema, list) or any(not isinstance(item, str) for item in feature_schema):
        raise ValueError("Model artifact feature_schema must be a list of strings")
    if len(set(feature_schema)) != len(feature_schema):
        raise ValueError("Model artifact feature_schema contains duplicates")

    family = artifact.get("family")
    allowed = {
        "base_rate",
        "recent_rate",
        "nearest_neighbor",
        "logistic_formula",
        "threshold_rule",
        "decision_stump",
        "two_state_mode",
    }
    if family not in allowed:
        raise ValueError(f"Unknown model artifact family: {family!r}")

    def finite_number(key: str) -> float:
        try:
            value = float(artifact[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Model artifact field {key!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"Model artifact field {key!r} must be finite")
        return value

    def probability(key: str) -> float:
        value = finite_number(key)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Model artifact probability {key!r} must be in [0, 1]")
        return value

    def numeric_mapping(key: str, *, positive: bool = False) -> dict[str, float]:
        raw = artifact.get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"Model artifact field {key!r} must be an object")
        result: dict[str, float] = {}
        for name, item in raw.items():
            try:
                value = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Model artifact {key!r}[{name!r}] must be numeric") from exc
            if not math.isfinite(value) or (positive and value <= 0.0):
                qualifier = "positive and finite" if positive else "finite"
                raise ValueError(f"Model artifact {key!r}[{name!r}] must be {qualifier}")
            result[str(name)] = value
        return result

    if family == "base_rate":
        probability("rate")
    elif family == "recent_rate":
        probability("recent_rate")
    elif family == "nearest_neighbor":
        rows = artifact.get("rows")
        features = artifact.get("features")
        if not isinstance(rows, list) or not isinstance(features, list):
            raise ValueError("Nearest-neighbor artifact requires rows and features lists")
        if any(not isinstance(name, str) for name in features) or len(set(features)) != len(features):
            raise ValueError("Nearest-neighbor features must be unique strings")
        means = numeric_mapping("means")
        scales = numeric_mapping("scales", positive=True)
        if set(features) - set(means) or set(features) - set(scales):
            raise ValueError("Nearest-neighbor means/scales must cover every feature")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("features"), dict):
                raise ValueError("Nearest-neighbor rows must contain feature objects")
            if int(row.get("target", -1)) not in {0, 1}:
                raise ValueError("Nearest-neighbor rows must contain binary targets")
            for name, value in row["features"].items():
                if not isinstance(value, (int, float, bool)) or not math.isfinite(float(value)):
                    raise ValueError(f"Nearest-neighbor feature {name!r} must be finite and numeric")
    elif family == "logistic_formula":
        if not str(artifact.get("hypothesis_id", "")).strip():
            raise ValueError("Formula artifact is missing hypothesis_id")
        raw_terms = artifact.get("formula_terms", [])
        raw_weights = artifact.get("weights", [])
        if not isinstance(raw_terms, list) or any(not isinstance(term, str) for term in raw_terms):
            raise ValueError("Formula artifact terms must be a list of strings")
        if not isinstance(raw_weights, list):
            raise ValueError("Formula artifact weights must be a list")
        terms = list(raw_terms)
        weights = list(raw_weights)
        Formula.parse(terms)
        if len(weights) != len(terms) + 1:
            raise ValueError("Formula artifact weight count does not match terms")
        if any(not math.isfinite(float(value)) for value in weights):
            raise ValueError("Formula artifact contains non-finite weights")
    elif family == "threshold_rule":
        if not str(artifact.get("variable", "")).strip():
            raise ValueError("Threshold artifact is missing variable")
        finite_number("threshold")
        probability("low_rate")
        probability("high_rate")
    elif family == "decision_stump":
        if not str(artifact.get("variable", "")).strip():
            raise ValueError("Decision-stump artifact is missing variable")
        finite_number("threshold")
        probability("left_rate")
        probability("right_rate")
    elif family == "two_state_mode":
        features = artifact.get("features")
        if not isinstance(features, list) or any(not isinstance(name, str) for name in features):
            raise ValueError("Two-state artifact features must be a list of strings")
        if len(set(features)) != len(features):
            raise ValueError("Two-state artifact features contain duplicates")
        mappings = {
            "means": numeric_mapping("means"),
            "scales": numeric_mapping("scales", positive=True),
            "centroid_a": numeric_mapping("centroid_a"),
            "centroid_b": numeric_mapping("centroid_b"),
        }
        for key, mapping in mappings.items():
            if set(features) - set(mapping):
                raise ValueError(f"Two-state artifact {key} must cover every feature")
        probability("state_a_rate")
        probability("state_b_rate")


def model_from_artifact(artifact: dict[str, Any]) -> Any:
    validate_model_artifact(artifact)
    family = artifact.get("family")
    model_id = str(artifact["model_id"])
    origin = str(artifact.get("origin", "persisted"))
    if family == "base_rate":
        return BaseRateModel(model_id, float(artifact["rate"]), origin=origin)
    if family == "recent_rate":
        return RecentActionBaseline(model_id, float(artifact["recent_rate"]), origin=origin)
    if family == "nearest_neighbor":
        return NearestNeighborModel(
            model_id,
            list(artifact.get("rows", [])),
            list(artifact.get("features", [])),
            {str(k): float(v) for k, v in artifact.get("means", {}).items()},
            {str(k): float(v) for k, v in artifact.get("scales", {}).items()},
            origin=origin,
        )
    if family == "logistic_formula":
        terms = list(artifact.get("formula_terms", []))
        formula = Formula.parse(terms)
        return FittedLogisticFormula(
            model_id=model_id,
            hypothesis_id=str(artifact["hypothesis_id"]),
            formula=formula,
            weights=[float(value) for value in artifact.get("weights", [])],
            complexity=int(artifact.get("complexity", formula.complexity)),
            origin=origin,
        )
    if family == "threshold_rule":
        return ThresholdRuleModel(
            model_id,
            str(artifact["variable"]),
            float(artifact["threshold"]),
            float(artifact["low_rate"]),
            float(artifact["high_rate"]),
            origin=origin,
        )
    if family == "decision_stump":
        return DecisionStumpModel(
            model_id,
            str(artifact["variable"]),
            float(artifact["threshold"]),
            float(artifact["left_rate"]),
            float(artifact["right_rate"]),
            origin=origin,
        )
    if family == "two_state_mode":
        return TwoStateModeModel(
            model_id,
            list(artifact.get("features", [])),
            {str(k): float(v) for k, v in artifact.get("means", {}).items()},
            {str(k): float(v) for k, v in artifact.get("scales", {}).items()},
            {str(k): float(v) for k, v in artifact.get("centroid_a", {}).items()},
            {str(k): float(v) for k, v in artifact.get("centroid_b", {}).items()},
            float(artifact["state_a_rate"]),
            float(artifact["state_b_rate"]),
            origin=origin,
        )
    raise ValueError(f"Cannot reconstruct model artifact family {family!r}")
