from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from resonance.analysis.contracts import FrameLike, TableLike, ValidationResult


DEFAULT_RANDOM_SEED = 20260619
DEFAULT_MIN_OVERLAP = 8


def chronological_holdout_validation(
    frame: FrameLike,
    *,
    candidate_lag_steps: Iterable[int],
    holdout_fraction: float = 0.25,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> ValidationResult:
    """Select the best lag on discovery rows, then evaluate that lag on holdout rows."""

    rows = _coerce_rows(frame)
    lags = _candidate_lags(candidate_lag_steps)
    warnings: list[str] = []
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    split_index = max(1, min(len(rows) - 1, round(len(rows) * (1 - holdout_fraction))))
    discovery_rows = rows[:split_index]
    holdout_rows = rows[split_index:]

    best = _best_lag_score(discovery_rows, lags, min_overlap=min_overlap)
    if best is None:
        warnings.append("no discovery lag met the minimum overlap")
        return ValidationResult(
            permutation_p_value=None,
            holdout_rho=None,
            holdout_overlap=0,
            sign_stability=None,
            warnings=tuple(warnings),
        )

    holdout_rho, holdout_overlap = _rho_at_lag(holdout_rows, best["lag_steps"], min_overlap=min_overlap)
    if holdout_rho is None:
        warnings.append("holdout lag did not meet the minimum overlap or variance requirement")

    return ValidationResult(
        permutation_p_value=None,
        holdout_rho=holdout_rho,
        holdout_overlap=holdout_overlap,
        sign_stability=None,
        warnings=tuple(warnings),
    )


def max_lag_block_permutation_test(
    frame: FrameLike,
    *,
    candidate_lag_steps: Iterable[int],
    permutations: int = 199,
    block_size: int | None = None,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    seed: int = DEFAULT_RANDOM_SEED,
) -> float | None:
    """Estimate a p-value for the maximum absolute rho across the full lag search."""

    if permutations <= 0:
        raise ValueError("permutations must be positive")

    rows = _coerce_rows(frame)
    lags = _candidate_lags(candidate_lag_steps)
    observed = _max_abs_rho(rows, lags, min_overlap=min_overlap)
    if observed is None:
        return None

    resolved_block_size = block_size or _default_block_size(len(rows), lags)
    if resolved_block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = random.Random(seed)
    null_at_least_observed = 0
    for _ in range(permutations):
        permuted_rows = _permute_y_blocks(rows, block_size=resolved_block_size, rng=rng)
        permuted_max = _max_abs_rho(permuted_rows, lags, min_overlap=min_overlap)
        if permuted_max is not None and permuted_max >= observed:
            null_at_least_observed += 1

    return (null_at_least_observed + 1) / (permutations + 1)


def window_stability(
    frame: FrameLike,
    *,
    lag_steps: int,
    window_count: int = 4,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> ValidationResult:
    """Evaluate one frozen lag in chronological windows and report sign stability."""

    if window_count <= 0:
        raise ValueError("window_count must be positive")

    rows = _coerce_rows(frame)
    windows = _chronological_windows(rows, window_count)
    full_rho, _ = _rho_at_lag(rows, lag_steps, min_overlap=min_overlap)
    reference_sign = _sign(full_rho)
    window_scores = []
    matching_signs = 0
    evaluable_windows = 0

    for index, window_rows in enumerate(windows):
        rho, overlap = _rho_at_lag(window_rows, lag_steps, min_overlap=min_overlap)
        row: dict[str, Any] = {
            "window_index": index,
            "lag_steps": lag_steps,
            "rho": rho,
            "overlap_count": overlap,
        }
        if window_rows:
            row["start_utc"] = window_rows[0].get("timestamp_utc")
            row["end_utc"] = window_rows[-1].get("timestamp_utc")
        window_scores.append(row)

        window_sign = _sign(rho)
        if reference_sign is not None and window_sign is not None:
            evaluable_windows += 1
            if window_sign == reference_sign:
                matching_signs += 1

    sign_stability = None
    if reference_sign is not None and evaluable_windows:
        sign_stability = matching_signs / evaluable_windows

    warnings = ()
    if sign_stability is None:
        warnings = ("no windows met the minimum overlap and variance requirement",)

    return ValidationResult(
        permutation_p_value=None,
        holdout_rho=None,
        holdout_overlap=0,
        sign_stability=sign_stability,
        window_scores=tuple(window_scores),
        warnings=warnings,
    )


def _coerce_rows(frame: FrameLike) -> tuple[Mapping[str, Any], ...]:
    source = frame.frame if hasattr(frame, "frame") else frame
    rows = tuple(source)
    if not rows:
        raise ValueError("frame must contain at least one row")
    for row in rows:
        if "x" not in row or "y" not in row:
            raise ValueError("frame rows must contain x and y columns")
    if all("timestamp_utc" in row for row in rows):
        return tuple(sorted(rows, key=lambda row: row["timestamp_utc"]))
    return rows


def _candidate_lags(candidate_lag_steps: Iterable[int]) -> tuple[int, ...]:
    lags = tuple(dict.fromkeys(candidate_lag_steps))
    if not lags:
        raise ValueError("candidate_lag_steps must contain at least one lag")
    return lags


def _best_lag_score(
    rows: Sequence[Mapping[str, Any]],
    candidate_lag_steps: Sequence[int],
    *,
    min_overlap: int,
) -> dict[str, Any] | None:
    scores = []
    for lag_steps in candidate_lag_steps:
        rho, overlap = _rho_at_lag(rows, lag_steps, min_overlap=min_overlap)
        if rho is None:
            continue
        scores.append({"lag_steps": lag_steps, "rho": rho, "overlap_count": overlap})

    if not scores:
        return None
    return max(scores, key=lambda score: (abs(score["rho"]), score["overlap_count"], -abs(score["lag_steps"])))


def _max_abs_rho(
    rows: Sequence[Mapping[str, Any]],
    candidate_lag_steps: Sequence[int],
    *,
    min_overlap: int,
) -> float | None:
    best = _best_lag_score(rows, candidate_lag_steps, min_overlap=min_overlap)
    if best is None:
        return None
    return abs(best["rho"])


def _rho_at_lag(
    rows: Sequence[Mapping[str, Any]],
    lag_steps: int,
    *,
    min_overlap: int,
) -> tuple[float | None, int]:
    if abs(lag_steps) >= len(rows):
        return None, 0

    x_start = max(0, -lag_steps)
    y_start = max(0, lag_steps)
    pair_count = len(rows) - abs(lag_steps)
    left: list[float] = []
    right: list[float] = []

    for offset in range(pair_count):
        x_value = _optional_float(rows[x_start + offset]["x"])
        y_value = _optional_float(rows[y_start + offset]["y"])
        if x_value is not None and y_value is not None:
            left.append(x_value)
            right.append(y_value)

    overlap = len(left)
    if overlap < min_overlap:
        return None, overlap
    return _pearson(left, right), overlap


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0:
        return None
    return numerator / denominator


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return None
    return numeric


def _default_block_size(row_count: int, candidate_lag_steps: Sequence[int]) -> int:
    max_lag = max(abs(lag) for lag in candidate_lag_steps)
    return max(2, max_lag + 1, row_count // 20)


def _permute_y_blocks(
    rows: Sequence[Mapping[str, Any]],
    *,
    block_size: int,
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    y_values = [row["y"] for row in rows]
    blocks = [y_values[index : index + block_size] for index in range(0, len(y_values), block_size)]
    rng.shuffle(blocks)
    shuffled_y = [value for block in blocks for value in block]
    return tuple({**row, "y": shuffled_y[index]} for index, row in enumerate(rows))


def _chronological_windows(
    rows: Sequence[Mapping[str, Any]],
    window_count: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    windows = []
    for index in range(window_count):
        start = round(index * len(rows) / window_count)
        end = round((index + 1) * len(rows) / window_count)
        windows.append(tuple(rows[start:end]))
    return tuple(windows)


def _sign(value: float | None) -> int | None:
    if value is None or value == 0:
        return None
    return 1 if value > 0 else -1


__all__ = [
    "chronological_holdout_validation",
    "max_lag_block_permutation_test",
    "window_stability",
]
