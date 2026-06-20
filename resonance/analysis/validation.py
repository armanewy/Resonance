from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from resonance.analysis.contracts import FrameLike, ValidationResult
from resonance.analysis.correlation import numeric_array, spearman_at_lag


DEFAULT_RANDOM_SEED = 20260619
DEFAULT_MIN_OVERLAP = 8


def chronological_holdout_validation(
    frame: FrameLike,
    *,
    candidate_lag_steps: Iterable[int],
    holdout_fraction: float = 0.25,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> ValidationResult:
    """Select the best lag on discovery rows, then evaluate it with Spearman on holdout."""

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

    holdout_rho, holdout_overlap = _rho_at_lag(
        holdout_rows,
        best["lag_steps"],
        min_overlap=min_overlap,
    )
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
    """Estimate a max-over-lags Spearman p-value using contiguous Y blocks."""

    if permutations <= 0:
        raise ValueError("permutations must be positive")

    rows = _coerce_rows(frame)
    lags = _candidate_lags(candidate_lag_steps)
    x, y = _xy_arrays(rows)
    observed = _max_abs_rho_arrays(x, y, lags, min_overlap=min_overlap)
    if observed is None:
        return None

    resolved_block_size = block_size or _default_block_size(len(rows), lags)
    if resolved_block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = random.Random(seed)
    null_at_least_observed = 0
    for _ in range(permutations):
        permuted_y = _permute_blocks(y, block_size=resolved_block_size, rng=rng)
        permuted_max = _max_abs_rho_arrays(x, permuted_y, lags, min_overlap=min_overlap)
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
    """Evaluate one frozen Spearman lag in chronological windows."""

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
    if hasattr(source, "to_dict"):
        try:
            rows = tuple(source.reset_index().to_dict(orient="records"))
        except (AttributeError, TypeError, ValueError):
            rows = tuple(source)
    else:
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
    lags = tuple(dict.fromkeys(int(lag) for lag in candidate_lag_steps))
    if not lags:
        raise ValueError("candidate_lag_steps must contain at least one lag")
    return lags


def _best_lag_score(
    rows: Sequence[Mapping[str, Any]],
    candidate_lag_steps: Sequence[int],
    *,
    min_overlap: int,
) -> dict[str, Any] | None:
    x, y = _xy_arrays(rows)
    scores = []
    for lag_steps in candidate_lag_steps:
        rho, overlap = spearman_at_lag(x, y, lag_steps, min_overlap=min_overlap)
        if rho is None:
            continue
        scores.append({"lag_steps": lag_steps, "rho": rho, "overlap_count": overlap})

    if not scores:
        return None
    return max(
        scores,
        key=lambda score: (
            abs(score["rho"]),
            score["overlap_count"],
            -abs(score["lag_steps"]),
            score["lag_steps"],
        ),
    )


def _max_abs_rho_arrays(
    x: np.ndarray,
    y: np.ndarray,
    candidate_lag_steps: Sequence[int],
    *,
    min_overlap: int,
) -> float | None:
    best: float | None = None
    for lag in candidate_lag_steps:
        rho, _ = spearman_at_lag(x, y, lag, min_overlap=min_overlap)
        if rho is None:
            continue
        score = abs(rho)
        if best is None or score > best:
            best = score
    return best


def _rho_at_lag(
    rows: Sequence[Mapping[str, Any]],
    lag_steps: int,
    *,
    min_overlap: int,
) -> tuple[float | None, int]:
    x, y = _xy_arrays(rows)
    return spearman_at_lag(x, y, lag_steps, min_overlap=min_overlap)


def _xy_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        numeric_array([row.get("x") for row in rows]),
        numeric_array([row.get("y") for row in rows]),
    )


def _default_block_size(row_count: int, candidate_lag_steps: Sequence[int]) -> int:
    max_lag = max(abs(lag) for lag in candidate_lag_steps)
    return max(2, max_lag + 1, row_count // 20)


def _permute_blocks(values: np.ndarray, *, block_size: int, rng: random.Random) -> np.ndarray:
    blocks = [values[index : index + block_size].copy() for index in range(0, len(values), block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks) if blocks else values.copy()


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
