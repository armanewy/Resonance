from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Any

from behavior_lab.core import HypothesisSpec, new_id, stable_hash, to_jsonable
from behavior_lab.dsl import Formula
from behavior_lab.evaluation import evaluate_model
from behavior_lab.temporal import feature_catalog

MODEL_ARTIFACT_VERSION = 1
SOFTWARE_VERSION = "0.1.0"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 50.0), -50.0)))


def numeric_features(rows: list[dict[str, Any]]) -> list[str]:
    return feature_catalog(rows)


@dataclass
class BaseRateModel:
    model_id: str
    rate: float
    complexity: int = 1

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.rate


@dataclass
class RecentActionBaseline:
    model_id: str
    recent_rate: float
    complexity: int = 2

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.recent_rate


@dataclass
class NearestNeighborModel:
    model_id: str
    rows: list[dict[str, Any]]
    features: list[str]
    complexity: int = 8

    def predict_proba(self, features: dict[str, Any]) -> float:
        if not self.rows:
            return 0.5
        best = min(self.rows, key=lambda row: self._distance(features, row["features"]))
        return 0.85 if best["target"] else 0.15

    def _distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        total = 0.0
        for name in self.features:
            total += (float(a.get(name, 0.0)) - float(b.get(name, 0.0))) ** 2
        return total


@dataclass
class FittedLogisticFormula:
    model_id: str
    hypothesis_id: str
    formula: Formula
    weights: list[float]
    complexity: int

    def predict_proba(self, features: dict[str, Any]) -> float:
        vector = self.formula.vector(features)
        logit = sum(weight * value for weight, value in zip(self.weights, vector, strict=True))
        return sigmoid(logit)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "intercept": self.weights[0],
            "terms": [
                {"expression": term.expression, "coefficient": self.weights[index + 1]}
                for index, term in enumerate(self.formula.terms)
            ],
        }


class LogisticFormulaHypothesis:
    def __init__(self, spec: HypothesisSpec):
        self.spec = spec
        self.formula = Formula.parse(list(spec.structure.get("terms", [])))

    def fit(
        self,
        rows: list[dict[str, Any]],
        *,
        learning_rate: float = 0.05,
        iterations: int = 160,
        l2: float = 0.01,
    ) -> FittedLogisticFormula:
        if not rows:
            weights = [0.0] * (len(self.formula.terms) + 1)
            return FittedLogisticFormula(new_id("m"), self.spec.hypothesis_id, self.formula, weights, self.formula.complexity)
        vectors = [self.formula.vector(row["features"]) for row in rows]
        targets = [int(row["target"]) for row in rows]
        weights = [0.0] * len(vectors[0])
        for _ in range(iterations):
            gradients = [0.0] * len(weights)
            for vector, target in zip(vectors, targets, strict=True):
                prediction = sigmoid(sum(weight * value for weight, value in zip(weights, vector, strict=True)))
                error = prediction - target
                for index, value in enumerate(vector):
                    gradients[index] += error * value
            scale = 1.0 / len(rows)
            for index in range(len(weights)):
                penalty = l2 * weights[index] if index > 0 else 0.0
                weights[index] -= learning_rate * (gradients[index] * scale + penalty)
        return FittedLogisticFormula(
            model_id=new_id("m"),
            hypothesis_id=self.spec.hypothesis_id,
            formula=self.formula,
            weights=weights,
            complexity=self.formula.complexity,
        )


@dataclass
class ThresholdRuleModel:
    model_id: str
    variable: str
    threshold: float
    low_rate: float
    high_rate: float
    complexity: int = 4

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.high_rate if float(features.get(self.variable, 0.0)) > self.threshold else self.low_rate


def fit_threshold_rule(rows: list[dict[str, Any]], variables: list[str] | None = None) -> ThresholdRuleModel:
    variables = variables or numeric_features(rows)
    if not rows:
        return ThresholdRuleModel(new_id("m"), "none", 0.0, 0.5, 0.5)
    best_model: ThresholdRuleModel | None = None
    best_loss = float("inf")
    for variable in variables:
        values = sorted({float(row["features"].get(variable, 0.0)) for row in rows})
        if len(values) < 2:
            continue
        thresholds = [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
        for threshold in thresholds[:25]:
            low = [row["target"] for row in rows if float(row["features"].get(variable, 0.0)) <= threshold]
            high = [row["target"] for row in rows if float(row["features"].get(variable, 0.0)) > threshold]
            if not low or not high:
                continue
            low_rate = (sum(low) + 1) / (len(low) + 2)
            high_rate = (sum(high) + 1) / (len(high) + 2)
            model = ThresholdRuleModel(new_id("m"), variable, threshold, low_rate, high_rate)
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

    def predict_proba(self, features: dict[str, Any]) -> float:
        return self.right_rate if float(features.get(self.variable, 0.0)) > self.threshold else self.left_rate


def fit_small_tree(rows: list[dict[str, Any]]) -> DecisionStumpModel:
    rule = fit_threshold_rule(rows)
    return DecisionStumpModel(rule.model_id, rule.variable, rule.threshold, rule.low_rate, rule.high_rate)


@dataclass
class TwoStateModeModel:
    model_id: str
    depleted_rate: float
    exploratory_rate: float
    complexity: int = 7

    def predict_proba(self, features: dict[str, Any]) -> float:
        depleted_probability = sigmoid(
            1.1 * (7.0 - float(features.get("sleep_hours", 7.0)))
            + 1.0 * (1.0 - float(features.get("previous_task_success", 0.5)))
            + 0.8 * float(features.get("fatigue", 0.0))
        )
        return depleted_probability * self.depleted_rate + (1.0 - depleted_probability) * self.exploratory_rate


def fit_two_state_model(rows: list[dict[str, Any]]) -> TwoStateModeModel:
    if not rows:
        return TwoStateModeModel(new_id("m"), 0.5, 0.5)
    depleted = []
    exploratory = []
    for row in rows:
        features = row["features"]
        if float(features.get("fatigue", 0.0)) > 0.6 or float(features.get("sleep_hours", 8.0)) < 6.2:
            depleted.append(row["target"])
        else:
            exploratory.append(row["target"])
    depleted_rate = (sum(depleted) + 1) / (len(depleted) + 2) if depleted else 0.45
    exploratory_rate = (sum(exploratory) + 1) / (len(exploratory) + 2) if exploratory else 0.55
    return TwoStateModeModel(new_id("m"), depleted_rate, exploratory_rate)


class SymbolicSearch:
    """Small symbolic-regression-style search over safe formula terms."""

    def __init__(self, max_terms: int = 5, candidate_limit: int = 14):
        self.max_terms = max_terms
        self.candidate_limit = candidate_limit

    def candidate_terms(self, rows: list[dict[str, Any]]) -> list[str]:
        variables = numeric_features(rows)
        base_terms = [name for name in variables]
        threshold_terms = [f"indicator({name} > 0.6)" for name in variables if name not in {"sleep_hours", "estimated_duration_minutes", "recent_context_switches"}]
        interactions: list[str] = []
        if "explicit_first_step" in variables and "ambiguity" in variables:
            interactions.append("explicit_first_step * indicator(ambiguity > 0.6)")
        if "fatigue" in variables and "ambiguity" in variables:
            interactions.append("fatigue * ambiguity")
        if "public_commitment" in variables and "deadline_near" in variables:
            interactions.append("public_commitment * deadline_near")
        return (base_terms + threshold_terms + interactions)[: self.candidate_limit]

    def search(self, training_rows: list[dict[str, Any]], development_rows: list[dict[str, Any]], target_name: str) -> FittedLogisticFormula:
        terms = self.candidate_terms(training_rows)
        selected: list[str] = []
        best_model: FittedLogisticFormula | None = None
        best_loss = float("inf")
        for _ in range(min(self.max_terms, len(terms))):
            improved = False
            best_term = None
            best_round_model = None
            for term in terms:
                if term in selected:
                    continue
                spec = HypothesisSpec.formula(new_id("h"), target_name, selected + [term])
                model = LogisticFormulaHypothesis(spec).fit(training_rows)
                score_rows = development_rows or training_rows
                loss = evaluate_model(model, score_rows, split="development").log_loss
                if loss < best_loss:
                    best_loss = loss
                    best_term = term
                    best_round_model = model
                    improved = True
            if not improved or best_term is None or best_round_model is None:
                break
            selected.append(best_term)
            best_model = best_round_model
        if best_model:
            return best_model
        fallback = HypothesisSpec.formula(new_id("h"), target_name, terms[:1] or ["bias"])
        return LogisticFormulaHypothesis(fallback).fit(training_rows)


class ModelFoundry:
    def fit_zoo(
        self,
        training_rows: list[dict[str, Any]],
        development_rows: list[dict[str, Any]],
        target_name: str,
    ) -> list[Any]:
        targets = [row["target"] for row in training_rows]
        base_rate = (sum(targets) + 1) / (len(targets) + 2) if targets else 0.5
        recent = training_rows[-25:] if training_rows else []
        recent_rate = (sum(row["target"] for row in recent) + 1) / (len(recent) + 2) if recent else base_rate
        models: list[Any] = [
            BaseRateModel(new_id("m"), base_rate),
            RecentActionBaseline(new_id("m"), recent_rate),
            NearestNeighborModel(new_id("m"), list(training_rows), numeric_features(training_rows)),
            fit_threshold_rule(training_rows),
            fit_small_tree(training_rows),
            fit_two_state_model(training_rows),
        ]
        hand_terms = [
            "explicit_first_step",
            "ambiguity",
            "fatigue",
            "public_commitment",
            "deadline_near",
            "recent_context_switches",
        ]
        available = set(numeric_features(training_rows))
        formula_terms = [term for term in hand_terms if term in available]
        if "explicit_first_step" in available and "ambiguity" in available:
            formula_terms.append("explicit_first_step * indicator(ambiguity > 0.6)")
        spec = HypothesisSpec.formula("task_start_default_override_v1", target_name, formula_terms)
        models.append(LogisticFormulaHypothesis(spec).fit(training_rows))
        models.append(SymbolicSearch(max_terms=5, candidate_limit=14).search(training_rows, development_rows, target_name))
        return models


def training_snapshot_hash(rows: list[dict[str, Any]]) -> str:
    snapshot = [
        {
            "case_id": row.get("case_id"),
            "features": row.get("features", {}),
            "target": row.get("target"),
        }
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
        "feature_schema": feature_schema,
        "training_snapshot_hash": training_snapshot_hash(training_rows or []),
    }
    if isinstance(model, BaseRateModel):
        common.update({"family": "base_rate", "rate": model.rate})
    elif isinstance(model, RecentActionBaseline):
        common.update({"family": "recent_rate", "recent_rate": model.recent_rate})
    elif isinstance(model, NearestNeighborModel):
        common.update({"family": "nearest_neighbor", "rows": to_jsonable(model.rows), "features": list(model.features)})
    elif isinstance(model, FittedLogisticFormula):
        common.update(
            {
                "family": "logistic_formula",
                "hypothesis_id": model.hypothesis_id,
                "formula_terms": [term.expression for term in model.formula.terms],
                "weights": list(model.weights),
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
                "depleted_rate": model.depleted_rate,
                "exploratory_rate": model.exploratory_rate,
            }
        )
    else:
        common.update({"family": "unknown"})
    return common


def model_from_artifact(artifact: dict[str, Any]) -> Any:
    family = artifact.get("family")
    model_id = str(artifact["model_id"])
    if family == "base_rate":
        return BaseRateModel(model_id, float(artifact["rate"]))
    if family == "recent_rate":
        return RecentActionBaseline(model_id, float(artifact["recent_rate"]))
    if family == "nearest_neighbor":
        return NearestNeighborModel(model_id, list(artifact.get("rows", [])), list(artifact.get("features", [])))
    if family == "logistic_formula":
        terms = list(artifact.get("formula_terms", []))
        weights = [float(value) for value in artifact.get("weights", [])]
        formula = Formula.parse(terms)
        return FittedLogisticFormula(
            model_id=model_id,
            hypothesis_id=str(artifact["hypothesis_id"]),
            formula=formula,
            weights=weights,
            complexity=int(artifact.get("complexity", formula.complexity)),
        )
    if family == "threshold_rule":
        return ThresholdRuleModel(
            model_id,
            str(artifact["variable"]),
            float(artifact["threshold"]),
            float(artifact["low_rate"]),
            float(artifact["high_rate"]),
        )
    if family == "decision_stump":
        return DecisionStumpModel(
            model_id,
            str(artifact["variable"]),
            float(artifact["threshold"]),
            float(artifact["left_rate"]),
            float(artifact["right_rate"]),
        )
    if family == "two_state_mode":
        return TwoStateModeModel(model_id, float(artifact["depleted_rate"]), float(artifact["exploratory_rate"]))
    raise ValueError(f"Cannot reconstruct model artifact family {family!r}")
