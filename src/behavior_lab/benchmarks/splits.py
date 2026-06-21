from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from behavior_lab.core import stable_hash


@dataclass(frozen=True)
class SplitAssignment:
    train: list[dict[str, Any]]
    development: list[dict[str, Any]]
    hidden: list[dict[str, Any]]
    purged_group_ids: tuple[str, ...] = field(default_factory=tuple)
    purged_rows: int = 0

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "development": len(self.development),
            "hidden": len(self.hidden),
        }

    def audit(self) -> dict[str, Any]:
        return {
            "sizes": self.sizes(),
            "purged_group_ids": list(self.purged_group_ids),
            "purged_rows": self.purged_rows,
        }


def chronological_split(
    rows: Iterable[dict[str, Any]],
    *,
    time_key: str,
    train_fraction: float = 0.6,
    development_fraction: float = 0.2,
) -> SplitAssignment:
    ordered = sorted(rows, key=lambda row: _chronological_key(row.get(time_key)))
    if not ordered:
        return SplitAssignment([], [], [])
    _validate_fractions(train_fraction, development_fraction)
    train_end, development_end = _split_boundaries(
        len(ordered), train_fraction, development_fraction
    )
    return SplitAssignment(
        ordered[:train_end],
        ordered[train_end:development_end],
        ordered[development_end:],
    )


def chronological_group_purged_split(
    rows: Iterable[dict[str, Any]],
    *,
    time_key: str,
    group_key: str,
    train_fraction: float = 0.6,
    development_fraction: float = 0.2,
) -> SplitAssignment:
    """Chronological split that never places one group in multiple regions.

    Boundaries are first determined from the decision rows themselves. Any
    group whose rows straddle a boundary is removed from every split. This is
    conservative by design: a negotiation thread may not partially train and
    partially evaluate a model.
    """

    ordered = sorted(rows, key=lambda row: _chronological_key(row.get(time_key)))
    if not ordered:
        return SplitAssignment([], [], [])
    _validate_fractions(train_fraction, development_fraction)
    train_end, development_end = _split_boundaries(
        len(ordered), train_fraction, development_fraction
    )
    provisional: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(ordered):
        if index < train_end:
            split_name = "train"
        elif index < development_end:
            split_name = "development"
        else:
            split_name = "hidden"
        provisional.append((split_name, row))

    group_regions: dict[str, set[str]] = {}
    for split_name, row in provisional:
        group = _group_value(row, group_key)
        group_regions.setdefault(group, set()).add(split_name)
    purged_groups = {
        group for group, regions in group_regions.items() if len(regions) > 1
    }

    train: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    purged_rows = 0
    for split_name, row in provisional:
        if _group_value(row, group_key) in purged_groups:
            purged_rows += 1
            continue
        if split_name == "train":
            train.append(row)
        elif split_name == "development":
            development.append(row)
        else:
            hidden.append(row)
    return SplitAssignment(
        train,
        development,
        hidden,
        purged_group_ids=tuple(sorted(purged_groups)),
        purged_rows=purged_rows,
    )


def group_disjoint_split(
    rows: Iterable[dict[str, Any]],
    *,
    group_key: str,
    train_fraction: float = 0.6,
    development_fraction: float = 0.2,
) -> SplitAssignment:
    materialized = list(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        groups.setdefault(_group_value(row, group_key), []).append(row)
    ordered_groups = sorted(groups)
    if not ordered_groups:
        return SplitAssignment([], [], [])
    _validate_fractions(train_fraction, development_fraction)
    train_end, development_end = _split_boundaries(
        len(ordered_groups), train_fraction, development_fraction
    )
    train_groups = set(ordered_groups[:train_end])
    dev_groups = set(ordered_groups[train_end:development_end])
    hidden_groups = set(ordered_groups[development_end:])
    return SplitAssignment(
        [row for row in materialized if _group_value(row, group_key) in train_groups],
        [row for row in materialized if _group_value(row, group_key) in dev_groups],
        [row for row in materialized if _group_value(row, group_key) in hidden_groups],
    )


def assert_disjoint_groups(split: SplitAssignment, *, group_key: str) -> bool:
    sets = []
    for rows in [split.train, split.development, split.hidden]:
        sets.append({_group_value(row, group_key) for row in rows})
    return not (
        sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
    )


def _validate_fractions(train_fraction: float, development_fraction: float) -> None:
    if (
        train_fraction <= 0
        or development_fraction < 0
        or train_fraction + development_fraction >= 1
    ):
        raise ValueError("fractions must leave a non-empty hidden region")


def _split_boundaries(
    count: int, train_fraction: float, development_fraction: float
) -> tuple[int, int]:
    train_end = max(1, int(count * train_fraction))
    development_end = max(
        train_end, int(count * (train_fraction + development_fraction))
    )
    if count >= 3:
        development_end = min(development_end, count - 1)
    return train_end, development_end


def _group_value(row: dict[str, Any], group_key: str) -> str:
    value = row.get(group_key)
    if value is None or str(value) == "":
        # Missing group IDs must not collapse unrelated rows into one group.
        return f"__missing__:{stable_hash(row)}"
    return str(value)


def _chronological_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return (1, text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, parsed.timestamp())
    return (1, str(value))
