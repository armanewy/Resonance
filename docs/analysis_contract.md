# Pair Analysis Contract

This is a deliberately small contract for future pairwise time-series analysis.
It defines data shapes only; it does not define or implement statistical
algorithms.

## Structures

`AlignedPair` describes two metrics aligned to a common cadence:

- `x_metric`
- `y_metric`
- `cadence_seconds`
- `frame`, containing UTC-indexed `x` and `y` columns
- `x_coverage`
- `y_coverage`
- `start_utc`
- `end_utc`

`LagScanResult` describes discovery-period lag scores:

- `scores`, with `lag_steps`, `lag_seconds`, `rho`, and `overlap_count`
- `best_lag_steps`
- `best_lag_seconds`
- `best_rho`

`ValidationResult` describes validation-only checks:

- `permutation_p_value`
- `holdout_rho`
- `holdout_overlap`
- `sign_stability`
- `window_scores`
- `warnings`

`PairAnalysis` groups the aligned pair metadata, transform name, lag result, and
validation result.

## Invariants

- Time indexes and timestamp fields are UTC timezone-aware.
- Alignment never performs implicit forward filling.
- Missing observations remain missing.
- A positive lag means changes in X precede aligned changes in Y.
- The best lag must be selected using discovery data only.
- Validation data must never influence best-lag selection.
