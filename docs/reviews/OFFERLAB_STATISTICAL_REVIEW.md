# OfferLab Statistical Review

Verdict: do not advance to shadow mode or any paid experiment from the current evidence.

## Evidence Reviewed

- `reports/offerlab_benchmark_v1.json`
- `docs/runs/OFFERLAB_BENCHMARK_V1_RESULTS.md`
- `docs/runs/EBAY_PRODUCTION_FEASIBILITY.md`
- `docs/runs/EBAY_SANDBOX_ROLE_PROBE.md`

## Findings

1. Benchmark v1 is a bounded smoke run, not full-release evidence. The run uses the 100,000-thread bounded normalization and a deterministic 500-row per-target model cap.
2. The core `seller_next_action` hidden improvement is `0.0250`, below the preregistered `0.05` engineering threshold.
3. Support coverage is too low for a deployable claim. The selected core hidden row coverage is `0.37`, below the `0.80` gate.
4. Calibration quality is not validated. Calibration payloads exist for classification targets, but no declared multiclass calibration threshold passes.
5. Frozen Benchmark v1 negative controls and split diagnostics are incomplete. Missing controls and omitted splits correctly fail the gate.

## Recommendation

Do not repeat Benchmark v1. It is frozen and hidden-spent. Run Benchmark v2 or
later after full normalization, full protocol split execution, complete
negative controls, validated calibration, fresh hidden cases, and non-row-capped
model evaluation. Until then, treat all model metrics as diagnostic only.
