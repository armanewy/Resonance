from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable

from behavior_lab.core import stable_hash
from behavior_lab.data_sources.registry import default_registry


SOURCE_ID = "nber_ebay_best_offer"
EVIDENCE_ROLE = "OFFERLAB_RESEARCH_MODEL"
PRODUCTION_EXPORT_ALLOWED = default_registry().check(SOURCE_ID, "production_export").allowed

FEATURE_CONTRACT = [
    "category",
    "condition",
    "listing_price",
    "reference_price",
    "current_actor",
    "current_action",
    "current_amount",
    "offer_to_asking_ratio",
    "round_number",
    "prior_turn_count",
    "prior_counter_count",
    "event_hour",
]

FORBIDDEN_MODEL_FIELDS = {
    "buyer_id",
    "seller_id",
    "listing_id",
    "thread_id",
    "row_id",
    "source_row_id",
    "status",
    "final_sale_price",
    "accepted_price",
    "outcome",
    "label",
}

CATEGORICAL_FEATURES = ["category", "condition", "current_actor", "current_action"]
NUMERIC_FEATURES = [
    "listing_price",
    "reference_price",
    "current_amount",
    "offer_to_asking_ratio",
    "round_number",
    "prior_turn_count",
    "prior_counter_count",
    "event_hour",
]


@dataclass(frozen=True)
class ModelLineage:
    model_id: str
    source_dataset_ids: list[str]
    feature_contract: list[str]
    forbidden_features: list[str]
    training_row_count: int
    training_rows_hash: str
    training_feature_values_hash: str
    evidence_role: str = EVIDENCE_ROLE
    production_export_allowed: bool = PRODUCTION_EXPORT_ALLOWED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_lineage(model_id: str, rows: Iterable[dict[str, Any]], *, feature_contract: list[str] | None = None) -> dict[str, Any]:
    items = list(rows)
    declared_contract = list(feature_contract or FEATURE_CONTRACT)
    raw_features = _raw_features_for_contract(declared_contract)
    feature_payload = [
        {
            name: enriched_features(row).get(name)
            for name in raw_features
        }
        for row in items
    ]
    payload = [
        {
            "row_id": row.get("row_id"),
            "label": row.get("label"),
            "timestamp": row.get("timestamp"),
            "features": feature_values,
        }
        for row, feature_values in zip(items, feature_payload, strict=True)
    ]
    return ModelLineage(
        model_id=model_id,
        source_dataset_ids=[SOURCE_ID],
        feature_contract=declared_contract,
        forbidden_features=sorted(FORBIDDEN_MODEL_FIELDS),
        training_row_count=len(items),
        training_rows_hash=stable_hash(payload),
        training_feature_values_hash=stable_hash(feature_payload),
    ).to_dict()


def _raw_features_for_contract(feature_contract: list[str]) -> list[str]:
    selected: list[str] = []
    for raw_name in FEATURE_CONTRACT:
        if raw_name in feature_contract or any(
            name.startswith(f"{raw_name}=") for name in feature_contract
        ):
            selected.append(raw_name)
    # Unknown derived names still require binding to all permitted raw inputs;
    # otherwise a caller could claim an opaque transform while hashing no data.
    return selected or list(FEATURE_CONTRACT)


def validate_feature_contract(rows: Iterable[dict[str, Any]]) -> bool:
    for row in rows:
        features = row.get("features", {})
        if set(features) & FORBIDDEN_MODEL_FIELDS:
            return False
    return True


def enriched_features(row: dict[str, Any]) -> dict[str, Any]:
    features = dict(row.get("features", {}))
    features["event_hour"] = _event_hour(features.get("event_time") or row.get("timestamp"))
    return features


class FeatureEncoder:
    def __init__(self, *, feature_contract: list[str] | None = None) -> None:
        self.feature_contract = list(feature_contract or FEATURE_CONTRACT)
        self.categorical_values: dict[str, list[str]] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.output_names: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> "FeatureEncoder":
        if not validate_feature_contract(rows):
            raise ValueError("rows contain forbidden participant or outcome fields inside features")
        feature_rows = [enriched_features(row) for row in rows]
        for feature in CATEGORICAL_FEATURES:
            if feature in self.feature_contract:
                values = sorted({str(items.get(feature, "missing")) for items in feature_rows})
                self.categorical_values[feature] = values or ["missing"]
        for feature in NUMERIC_FEATURES:
            if feature not in self.feature_contract:
                continue
            values = [_to_float(items.get(feature)) for items in feature_rows]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
            self.numeric_means[feature] = mean
            self.numeric_scales[feature] = math.sqrt(variance) or 1.0
        names = []
        for feature in NUMERIC_FEATURES:
            if feature in self.feature_contract:
                names.append(feature)
        for feature in CATEGORICAL_FEATURES:
            if feature in self.feature_contract:
                names.extend(f"{feature}={value}" for value in self.categorical_values.get(feature, []))
        self.output_names = names
        return self

    def transform_one(self, row: dict[str, Any]) -> list[float]:
        features = enriched_features(row)
        vector = []
        for feature in NUMERIC_FEATURES:
            if feature in self.feature_contract:
                vector.append((_to_float(features.get(feature)) - self.numeric_means.get(feature, 0.0)) / self.numeric_scales.get(feature, 1.0))
        for feature in CATEGORICAL_FEATURES:
            if feature in self.feature_contract:
                value = str(features.get(feature, "missing"))
                vector.extend(1.0 if value == known else 0.0 for known in self.categorical_values.get(feature, []))
        return vector

    def transform(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        return [self.transform_one(row) for row in rows]


def support_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_rows = [enriched_features(row) for row in rows]
    numeric: dict[str, dict[str, float]] = {}
    for feature in NUMERIC_FEATURES:
        values = [_to_float(items.get(feature)) for items in feature_rows if items.get(feature) is not None]
        if values:
            numeric[feature] = {"min": min(values), "max": max(values), "count": len(values)}
    categorical = {
        feature: sorted({str(items.get(feature, "missing")) for items in feature_rows})
        for feature in CATEGORICAL_FEATURES
    }
    return {"numeric": numeric, "categorical": categorical, "row_count": len(rows)}


def outside_support(row: dict[str, Any], profile: dict[str, Any], *, ignore: set[str] | None = None) -> bool:
    ignore = ignore or set()
    features = enriched_features(row)
    for feature, bounds in profile.get("numeric", {}).items():
        if feature in ignore:
            continue
        value = _to_float(features.get(feature))
        if value < bounds["min"] or value > bounds["max"]:
            return True
    for feature, values in profile.get("categorical", {}).items():
        if feature in ignore:
            continue
        if str(features.get(feature, "missing")) not in set(values):
            return True
    return False


def support_abstention_report(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile = support_profile(train_rows)
    outside = [row.get("row_id") for row in eval_rows if outside_support(row, profile)]
    total = len(eval_rows)
    return {
        "train_support_rows": len(train_rows),
        "evaluated_rows": total,
        "abstained_rows": outside,
        "abstention_rate": len(outside) / total if total else 0.0,
    }


def normalize_probabilities(probabilities: dict[str, float], labels: list[str]) -> dict[str, float]:
    cleaned = {label: max(0.0, float(probabilities.get(label, 0.0))) for label in labels}
    total = sum(cleaned.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels} if labels else {}
    return {label: value / total for label, value in cleaned.items()}


def _event_hour(value: Any) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.hour + parsed.minute / 60.0)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
