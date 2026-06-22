from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable

from behavior_lab.core import stable_hash
from behavior_lab.data_sources.registry import default_registry


SOURCE_ID = "nber_ebay_best_offer"
EVIDENCE_ROLE = "OFFERLAB_RESEARCH_MODEL"
EVIDENCE_SCOPE = "bounded_smoke_or_semantics"
PRODUCTION_EXPORT_ALLOWED = default_registry().check(SOURCE_ID, "production_export").allowed

FEATURE_CONTRACT = [
    "category",
    "condition",
    "listing_price",
    "current_actor",
    "current_action",
    "current_amount",
    "offer_to_asking_ratio",
    "round_number",
    "prior_turn_count",
    "prior_counter_count",
    "event_hour",
]
ALLOWED_MODEL_FEATURES = set(FEATURE_CONTRACT)

FORBIDDEN_MODEL_FIELDS = {
    "buyer_id",
    "seller_id",
    "listing_id",
    "thread_id",
    "row_id",
    "source_row_id",
    "status",
    "status_id",
    "event_time",
    "response_time",
    "reference_price",
    "ref_price4",
    "excluded_reference_price_ref_price4",
    "buyer_id_if_sold",
    "sold_by_best_offer",
    "bo_ck_yn",
    "item_price",
    "auto_accept_price",
    "auto_decline_price",
    "accept_price",
    "decline_price",
    "buyer_us_if_sold",
    "final_sale_price",
    "accepted_price",
    "outcome",
    "label",
}

CATEGORICAL_FEATURES = ["category", "condition", "current_actor", "current_action"]
NUMERIC_FEATURES = [
    "listing_price",
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
        feature_names = set(features)
        if feature_names & FORBIDDEN_MODEL_FIELDS:
            return False
        if not feature_names <= ALLOWED_MODEL_FEATURES:
            return False
    return True


def research_scope(*, evidence_scope: str = EVIDENCE_SCOPE) -> dict[str, Any]:
    return {
        "research_only": True,
        "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
        "commercial_training_allowed": False,
        "full_release_evidence": False,
        "evidence_scope": evidence_scope,
        "source_dataset_ids": [SOURCE_ID],
    }


def enriched_features(row: dict[str, Any]) -> dict[str, Any]:
    features = dict(row.get("features", {}))
    features["event_hour"] = _event_hour(row.get("timestamp"))
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


def reserve_hidden_submission(
    *,
    store_path: str | Path,
    namespace: str,
    requested_lockbox_id: str,
    target: str,
    hidden_rows: list[dict[str, Any]],
    artifact_id: str,
) -> dict[str, Any]:
    if not str(requested_lockbox_id).strip():
        raise ValueError("lockbox_id is required")
    if not str(namespace).strip():
        raise ValueError("namespace is required")
    if not str(target).strip():
        raise ValueError("target is required")
    if not str(artifact_id).strip():
        raise ValueError("artifact_id is required")
    from behavior_lab.offerlab_research.api import AppendOnlyResearchStore, ResearchBudgetError

    tokens = _hidden_case_tokens(hidden_rows)
    case_set_hash = stable_hash(tokens)
    legacy_case_set_hash = stable_hash(sorted({_hidden_content_case_token(row) for row in hidden_rows}))
    event_type = "offerlab_hidden_submission_reserved"
    store = AppendOnlyResearchStore(store_path)
    requested_tokens = set(tokens)

    def guard(events: list[dict[str, Any]]) -> None:
        for event in events:
            if event.get("event_type") not in {
                event_type,
                "hidden_submission_reserved",
                "hidden_submitted",
            }:
                continue
            payload = event.get("payload", {})
            result = payload.get("result", {})
            if not isinstance(result, dict):
                result = {}
            previous_lockbox_id = (
                payload.get("requested_lockbox_id")
                or result.get("lockbox_id")
                or result.get("requested_lockbox_id")
            )
            if previous_lockbox_id == requested_lockbox_id:
                raise ResearchBudgetError("hidden submission budget exhausted for this lockbox")
            previous_case_set = (
                payload.get("hidden_case_set_hash")
                or payload.get("canonical_lockbox_id")
                or result.get("hidden_case_set_hash")
                or result.get("canonical_lockbox_id")
            )
            if previous_case_set in {case_set_hash, legacy_case_set_hash}:
                raise ResearchBudgetError("hidden case set was already reserved")
            previous_tokens = set(payload.get("hidden_case_tokens", []) or result.get("hidden_case_tokens", []))
            if requested_tokens & previous_tokens:
                raise ResearchBudgetError("hidden case overlap detected with a previously reserved lockbox")

    event = store.append_guarded(
        event_type,
        {
            "namespace": namespace,
            "requested_lockbox_id": requested_lockbox_id,
            "target": target,
            "hidden_case_set_hash": case_set_hash,
            "hidden_case_tokens": tokens,
            "hidden_case_tokens_hash": stable_hash(tokens),
            "hidden_rows": len(hidden_rows),
            "artifact_id": artifact_id,
        },
        guard=guard,
    )
    return {
        "reservation_event_id": event["event_id"],
        "lockbox_id": requested_lockbox_id,
        "canonical_lockbox_id": case_set_hash,
        "hidden_case_set_hash": case_set_hash,
        "hidden_rows": len(hidden_rows),
        "artifact_id": artifact_id,
    }


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


def _hidden_case_tokens(rows: list[dict[str, Any]]) -> list[str]:
    tokens: set[str] = set()
    for row in rows:
        tokens.add(_hidden_content_case_token(row))
        source_token = _hidden_source_case_token(row)
        if source_token is not None:
            tokens.add(source_token)
    return sorted(tokens)


def _hidden_content_case_token(row: dict[str, Any]) -> str:
    return stable_hash(
        {
            "task": row.get("task"),
            "timestamp": row.get("timestamp"),
            "features": row.get("features", {}),
            "observed_history": row.get("observed_history", []),
        }
    )


def _hidden_source_case_token(row: dict[str, Any]) -> str | None:
    source_identity = {
        "task": row.get("task"),
        "row_id": row.get("row_id"),
        "thread_id": row.get("thread_id"),
        "listing_id": row.get("listing_id"),
        "source_row_id": row.get("source_row_id"),
    }
    if not any(value not in {None, ""} for value in source_identity.values()):
        return None
    return stable_hash(source_identity)
