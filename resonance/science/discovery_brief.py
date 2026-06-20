from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Any, Literal, Sequence

from pydantic import Field

from resonance.science.contracts import StrictModel, canonical_json, stable_hash


DISCOVERY_BRIEF_SCHEMA_VERSION = "science-discovery-brief-v1"


class MetricBrief(StrictModel):
    name: str = Field(min_length=1)
    units: tuple[str, ...] = Field(default_factory=tuple)
    sources: tuple[str, ...] = Field(default_factory=tuple)
    coverage: dict[str, Any] = Field(default_factory=dict)
    cadence: dict[str, Any] = Field(default_factory=dict)


class DiscoveryBrief(StrictModel):
    """Exploration-only context that can be sent to hypothesis providers."""

    schema_version: Literal[DISCOVERY_BRIEF_SCHEMA_VERSION] = DISCOVERY_BRIEF_SCHEMA_VERSION
    snapshot_id: str | None = None
    metric_catalog_id: str | None = None
    metrics: tuple[MetricBrief, ...]
    exploration_boundary: dict[str, Any] = Field(default_factory=dict)
    descriptive_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    correlations: dict[str, float | None] = Field(default_factory=dict)
    anomaly_summaries: dict[str, Any] = Field(default_factory=dict)
    compact_summaries: dict[str, Any] = Field(default_factory=dict)
    sample_plot_metadata: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    selected_memory_summaries: tuple[str, ...] = Field(default_factory=tuple)

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", exclude_none=True))

    def artifact_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude_none=True))


def discovery_brief_from_exploration_view(
    exploration_view: dict[str, Any],
    *,
    metric_catalog: dict[str, Any] | None = None,
    selected_memory_summaries: Sequence[str] = (),
) -> DiscoveryBrief:
    """Build a provider brief from exploration rows without carrying split metadata forward."""

    rows = tuple(exploration_view.get("rows") or ())
    metrics = _metric_briefs(rows, metric_catalog)
    return DiscoveryBrief(
        snapshot_id=exploration_view.get("snapshot_id"),
        metric_catalog_id=(metric_catalog or {}).get("catalog_id"),
        metrics=tuple(metrics),
        exploration_boundary=_exploration_boundary(rows),
        descriptive_stats=_descriptive_stats(rows, [metric.name for metric in metrics]),
        correlations=_correlations(rows, [metric.name for metric in metrics]),
        anomaly_summaries=_anomaly_summaries(rows, [metric.name for metric in metrics]),
        compact_summaries={
            "row_count": len(rows),
            "metric_count": len(metrics),
        },
        sample_plot_metadata=_sample_plot_metadata(rows, [metric.name for metric in metrics]),
        selected_memory_summaries=tuple(selected_memory_summaries),
    )


def serialize_discovery_brief(brief: DiscoveryBrief) -> str:
    return brief.canonical_json()


def hash_discovery_brief(brief: DiscoveryBrief) -> str:
    return brief.artifact_hash()


def _metric_briefs(
    rows: Sequence[dict[str, Any]],
    metric_catalog: dict[str, Any] | None,
) -> list[MetricBrief]:
    if metric_catalog is not None:
        return [
            MetricBrief(
                name=str(metric["name"]),
                units=tuple(str(unit) for unit in metric.get("units", ())),
                sources=tuple(str(source) for source in metric.get("sources", ())),
                coverage=dict(metric.get("coverage") or {}),
                cadence=dict(metric.get("cadence") or {}),
            )
            for metric in metric_catalog.get("metrics", ())
        ]

    names = sorted({name for row in rows for name in row.get("metrics", {})})
    briefs: list[MetricBrief] = []
    for name in names:
        observations = [
            observation
            for row in rows
            for observation in row.get("metrics", {}).get(name, ())
        ]
        briefs.append(
            MetricBrief(
                name=name,
                units=tuple(sorted({str(observation.get("unit")) for observation in observations})),
                sources=tuple(sorted({str(observation.get("source")) for observation in observations})),
                coverage={"sample_count": len(observations)},
                cadence={},
            )
        )
    return briefs


def _exploration_boundary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"start_utc": None, "end_utc": None, "row_count": 0}
    return {
        "start_utc": rows[0].get("timestamp_utc"),
        "end_utc": rows[-1].get("timestamp_utc"),
        "row_count": len(rows),
    }


def _descriptive_stats(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = _values_for_metric(rows, metric)
        if not values:
            result[metric] = {"count": 0}
            continue
        result[metric] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(mean(values), 6),
        }
    return result


def _correlations(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for left_index, left in enumerate(metrics):
        for right in metrics[left_index + 1 :]:
            pairs = _paired_values(rows, left, right)
            key = f"{left}|{right}"
            result[key] = _pearson(pairs) if len(pairs) >= 2 else None
    return result


def _anomaly_summaries(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for metric in metrics:
        values = _values_for_metric(rows, metric)
        if len(values) < 2:
            summaries[metric] = {"z_score_abs_gt_3_count": 0}
            continue
        center = mean(values)
        variance = sum((value - center) ** 2 for value in values) / len(values)
        stddev = sqrt(variance)
        summaries[metric] = {
            "z_score_abs_gt_3_count": (
                sum(1 for value in values if abs((value - center) / stddev) > 3)
                if stddev > 0
                else 0
            )
        }
    return summaries


def _sample_plot_metadata(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "kind": "line",
            "metric": metric,
            "x": "timestamp_utc",
            "y": "value",
            "point_count": len(_values_for_metric(rows, metric)),
        }
        for metric in metrics
    )


def _values_for_metric(rows: Sequence[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(observation["value"])
        for row in rows
        for observation in row.get("metrics", {}).get(metric, ())
        if "value" in observation
    ]


def _paired_values(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        left_values = row.get("metrics", {}).get(left, ())
        right_values = row.get("metrics", {}).get(right, ())
        if left_values and right_values:
            pairs.append((float(left_values[0]["value"]), float(right_values[0]["value"])))
    return pairs


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    left_values = [left for left, _ in pairs]
    right_values = [right for _, right in pairs]
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    numerator = sum(
        (left - left_mean) * (right - right_mean)
        for left, right in zip(left_values, right_values, strict=True)
    )
    left_denominator = sqrt(sum((left - left_mean) ** 2 for left in left_values))
    right_denominator = sqrt(sum((right - right_mean) ** 2 for right in right_values))
    denominator = left_denominator * right_denominator
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


__all__ = [
    "DISCOVERY_BRIEF_SCHEMA_VERSION",
    "DiscoveryBrief",
    "MetricBrief",
    "discovery_brief_from_exploration_view",
    "hash_discovery_brief",
    "serialize_discovery_brief",
]
