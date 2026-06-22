from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from behavior_lab.benchmarks.metrics import multiclass_log_loss
from behavior_lab.core import stable_hash
from behavior_lab.offerlab_models.common import (
    EVIDENCE_ROLE,
    FEATURE_CONTRACT,
    PRODUCTION_EXPORT_ALLOWED,
    enriched_features,
    model_lineage,
    normalize_probabilities,
    research_scope,
    reserve_hidden_submission,
)


@dataclass(frozen=True)
class FormulaCandidate:
    formula_id: str
    target_label: str
    terms: list[str]
    falsification_condition: str

    @property
    def complexity(self) -> int:
        return len(self.terms)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["complexity"] = self.complexity
        return payload


@dataclass
class FormulaModel:
    candidate: FormulaCandidate
    coefficients: dict[str, float]
    intercept: float
    labels: list[str]
    non_target_distribution: dict[str, float]
    lineage: dict[str, Any]

    @property
    def model_id(self) -> str:
        return self.candidate.formula_id

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictions = []
        for row in rows:
            score = self.intercept + sum(self.coefficients[term] * term_value(term, row) for term in self.candidate.terms)
            target_probability = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score))))
            probabilities = {label: (1.0 - target_probability) * self.non_target_distribution.get(label, 0.0) for label in self.labels}
            probabilities[self.candidate.target_label] = target_probability
            probabilities = normalize_probabilities(probabilities, self.labels)
            predictions.append(
                {
                    "row_id": row["row_id"],
                    "label": row["label"],
                    "prediction": max(probabilities, key=probabilities.get),
                    "probabilities": probabilities,
                    "split": row.get("split", "unknown"),
                }
            )
        return predictions


class FormulaHiddenLockbox:
    def __init__(self, hidden_rows: list[dict[str, Any]], *, lockbox_id: str, store_path: str | Path, target: str = "formula") -> None:
        if not lockbox_id.strip():
            raise ValueError("lockbox_id is required")
        self._hidden_rows = list(hidden_rows)
        self.lockbox_id = lockbox_id
        self.store_path = store_path
        self.target = target

    def submit_once(self, model: FormulaModel) -> dict[str, Any]:
        reservation = reserve_hidden_submission(
            store_path=self.store_path,
            namespace="formula_suite",
            requested_lockbox_id=self.lockbox_id,
            target=self.target,
            hidden_rows=self._hidden_rows,
            artifact_id=_formula_artifact_id(model),
        )
        predictions = model.predict(self._hidden_rows)
        return {
            "submitted": True,
            "formula_id": model.model_id,
            **reservation,
            "hidden_rows": len(self._hidden_rows),
            "hidden_log_loss": multiclass_log_loss(predictions, labels=model.labels),
            "hidden_submission_count": 1,
        }


def build_formula_candidates() -> list[FormulaCandidate]:
    return [
        FormulaCandidate(
            "formula_relative_offer_accept",
            "accept",
            ["relative_offer", "gap_to_listing", "round_number"],
            "Fails if higher relative offers do not reduce chronological development log loss versus base rate.",
        ),
        FormulaCandidate(
            "formula_low_offer_decline",
            "decline",
            ["relative_offer_below_70", "gap_to_listing", "seller_experience"],
            "Fails if low relative offers are not more likely to be declined out of sample.",
        ),
        FormulaCandidate(
            "formula_concession_counter",
            "counter",
            ["relative_offer", "prior_counter_count", "round_number"],
            "Fails if offer ratio and round context do not improve counter prediction.",
        ),
        FormulaCandidate(
            "formula_timing_category_interaction",
            "counter",
            ["timing_hour", "category_refurbished", "relative_offer_x_round"],
            "Fails if timing/category interaction does not beat simpler offer-ratio terms.",
        ),
        FormulaCandidate(
            "formula_listing_price_threshold",
            "accept",
            ["listing_price_ratio", "relative_offer_above_85", "round_number"],
            "Fails if listing-price closeness does not calibrate acceptance probabilities.",
        ),
    ]


def fit_formula(candidate: FormulaCandidate, rows: list[dict[str, Any]]) -> FormulaModel:
    labels = sorted({str(row["label"]) for row in rows}) or [candidate.target_label]
    target_count = sum(1 for row in rows if str(row["label"]) == candidate.target_label)
    base_rate = (target_count + 0.5) / (len(rows) + 1.0) if rows else 0.5
    intercept = math.log(base_rate / max(1.0 - base_rate, 1e-9))
    coefficients = {}
    for term in candidate.terms:
        positives = [term_value(term, row) for row in rows if str(row["label"]) == candidate.target_label]
        negatives = [term_value(term, row) for row in rows if str(row["label"]) != candidate.target_label]
        coefficients[term] = _mean(positives) - _mean(negatives)
    non_target_labels = [label for label in labels if label != candidate.target_label]
    non_target_counts = {label: sum(1 for row in rows if str(row["label"]) == label) for label in non_target_labels}
    total = sum(non_target_counts.values())
    non_target_distribution = {
        label: (non_target_counts[label] + 1.0) / (total + len(non_target_labels))
        for label in non_target_labels
    }
    return FormulaModel(
        candidate=candidate,
        coefficients=coefficients,
        intercept=intercept,
        labels=labels,
        non_target_distribution=non_target_distribution,
        lineage=model_lineage(candidate.formula_id, rows, feature_contract=_formula_feature_contract(candidate.terms)),
    )


def evaluate_formula_candidates(
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    *,
    black_box_model_id: str | None = None,
    black_box_hidden_loss: float | None = None,
    hidden_lockbox_id: str | None = None,
    hidden_lockbox_store_path: str | Path | None = None,
) -> dict[str, Any]:
    for split_name, rows in [("train", train), ("development", development), ("hidden", hidden)]:
        for row in rows:
            row["split"] = split_name
    labels = sorted({str(row["label"]) for row in train + development})
    baseline_predictions = _base_rate_predictions(development, train, labels)
    baseline_development_loss = multiclass_log_loss(baseline_predictions, labels=labels)
    dev_rows = []
    models = []
    for candidate in build_formula_candidates():
        model = fit_formula(candidate, train)
        models.append(model)
        predictions = model.predict(development)
        development_loss = multiclass_log_loss(predictions, labels=labels)
        passed = development_loss < baseline_development_loss
        dev_rows.append(
            {
                "formula_id": candidate.formula_id,
                "development_log_loss": development_loss,
                "baseline_development_log_loss": baseline_development_loss,
                "passed_falsification": passed,
                "complexity": candidate.complexity,
                "falsification_condition": candidate.falsification_condition,
                "terms": list(candidate.terms),
                "lineage": model.lineage,
            }
        )
    dev_rows.sort(key=lambda item: (item["development_log_loss"], item["complexity"]))
    eligible = [row for row in dev_rows if row["passed_falsification"]]
    chosen_id = eligible[0]["formula_id"] if eligible else None
    chosen_model = next((model for model in models if model.model_id == chosen_id), None)
    hidden_report = None
    if chosen_model is not None and hidden:
        if hidden_lockbox_id is None:
            hidden_report = {
                "submitted": False,
                "reason": "hidden submission requires an explicit hidden_lockbox_id",
                "hidden_rows_reserved": len(hidden),
            }
        else:
            if hidden_lockbox_store_path is None:
                raise ValueError("hidden_lockbox_store_path is required for hidden submission")
            hidden_report = FormulaHiddenLockbox(
                hidden,
                lockbox_id=hidden_lockbox_id,
                store_path=hidden_lockbox_store_path,
                target="seller_next_action_formula",
            ).submit_once(chosen_model)
    hidden_submitted = bool(hidden_report and hidden_report.get("submitted"))
    return {
        "evidence_role": EVIDENCE_ROLE,
        "research_only": True,
        "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
        "scope": research_scope(),
        "feature_contract": FEATURE_CONTRACT,
        "falsification_enforced": True,
        "baseline_development_log_loss": baseline_development_loss,
        "candidate_count": len(dev_rows),
        "development": dev_rows,
        "chosen_formula_id": chosen_id,
        "hidden_lockbox": hidden_report,
        "black_box_comparison": {
            "compared": hidden_submitted and black_box_model_id is not None and black_box_hidden_loss is not None,
            "black_box_model_id": black_box_model_id,
            "black_box_hidden_loss": black_box_hidden_loss,
            "claim": (
                "formula hidden metrics are compared only after a formula hidden-lockbox submission"
                if hidden_submitted
                else "formula candidates are retained only on chronological development falsification in this report"
            ),
        },
    }


def term_value(term: str, row: dict[str, Any]) -> float:
    features = enriched_features(row)
    ratio = float(features.get("offer_to_asking_ratio") or 0.0)
    current_amount = float(features.get("current_amount") or 0.0)
    listing_price = float(features.get("listing_price") or 0.0)
    if term == "relative_offer":
        return ratio
    if term == "gap_to_listing":
        return (listing_price - current_amount) / listing_price if listing_price else 0.0
    if term == "round_number":
        return float(features.get("round_number") or 0.0)
    if term == "timing_hour":
        return float(features.get("event_hour") or 0.0) / 24.0
    if term in {"seller_experience", "prior_counter_count"}:
        return float(features.get("prior_counter_count") or 0.0)
    if term == "listing_price_ratio":
        return current_amount / listing_price if listing_price else 0.0
    if term == "category_refurbished":
        return 1.0 if "refurbished" in str(features.get("category", "")).lower() else 0.0
    if term == "relative_offer_below_70":
        return 1.0 if ratio < 0.70 else 0.0
    if term == "relative_offer_above_85":
        return 1.0 if ratio >= 0.85 else 0.0
    if term == "relative_offer_x_round":
        return ratio * float(features.get("round_number") or 0.0)
    raise ValueError(f"Unknown formula term {term!r}")


def _formula_feature_contract(terms: list[str]) -> list[str]:
    mapping = {
        "relative_offer": ["offer_to_asking_ratio"],
        "gap_to_listing": ["current_amount", "listing_price"],
        "round_number": ["round_number"],
        "timing_hour": ["event_hour"],
        "seller_experience": ["prior_counter_count"],
        "prior_counter_count": ["prior_counter_count"],
        "listing_price_ratio": ["current_amount", "listing_price"],
        "category_refurbished": ["category"],
        "relative_offer_below_70": ["offer_to_asking_ratio"],
        "relative_offer_above_85": ["offer_to_asking_ratio"],
        "relative_offer_x_round": ["offer_to_asking_ratio", "round_number"],
    }
    output: list[str] = []
    for term in terms:
        output.extend(mapping.get(term, []))
    return sorted(set(output))


def _formula_artifact_id(model: FormulaModel) -> str:
    return stable_hash(
        {
            "suite": "formula_v1",
            "model_id": model.model_id,
            "candidate": model.candidate.to_dict(),
            "coefficients": model.coefficients,
            "intercept": model.intercept,
            "labels": model.labels,
            "non_target_distribution": model.non_target_distribution,
            "lineage": model.lineage,
        }
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _base_rate_predictions(rows: list[dict[str, Any]], train: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    counts = {label: sum(1 for row in train if str(row["label"]) == label) for label in labels}
    total = sum(counts.values()) + len(labels)
    probabilities = {label: (counts[label] + 1.0) / total for label in labels}
    return [
        {
            "row_id": row["row_id"],
            "label": row["label"],
            "prediction": max(probabilities, key=probabilities.get),
            "probabilities": probabilities,
            "split": row.get("split", "development"),
        }
        for row in rows
    ]
