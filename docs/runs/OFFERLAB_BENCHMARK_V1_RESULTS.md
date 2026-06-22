# OfferLab Benchmark v1 Results

Generated at: `2026-06-22T00:19:06.600991+00:00`
Git commit: `53453484f102a98e1631073ace1b042d1f3821ed`

## Decision

Gate status: **STOP**

Reasons:
- gate predicate failed: full_release_evidence
- gate predicate failed: row_cap_disabled
- gate predicate failed: protocol_splits_complete
- gate predicate failed: hidden_improvement_at_least_5pct
- gate predicate failed: calibration_quality_validated
- gate predicate failed: hidden_support_coverage_at_least_80pct
- gate predicate failed: negative_controls_passed

## Scope

- Evidence scope: `bounded_smoke_or_semantics`
- Baseline scope: `bounded_100k_normalized_sample`
- Model scope: `deterministic_row_cap_500_per_target`
- Protocol splits complete: `False`
- Production export allowed: `False`

## Target Summary

| Target | Rows | Sampled | Hidden submitted | Best hidden improvement | Seller-disjoint improvement |
| --- | ---: | ---: | --- | ---: | ---: |
| seller_next_action | 333493 | 500 | True | 0.0250 | 0.0690 |
| buyer_response_to_counter | 91924 | 500 | True | 0.0000 | 0.0043 |
| agreement | 98372 | 500 | True | 0.0697 | 0.0074 |
| final_price_ratio | 51957 | 500 | True | 0.0000 | 0.0000 |
| response_latency | 425531 | 500 | True | 0.0000 | 0.0000 |

## Direct Answers

- Does any model beat the strongest simple baseline? Bounded smoke only; see per-target relative improvement in the JSON report.
- Does the gain survive seller-disjoint evaluation? Bounded smoke only; not a full evidence gate.
- Does the gain survive a later time block? Chronological bounded baselines ran; full-release later-period evidence is not available.
- Is calibration reported? Calibration payloads are emitted for classification rows; this is not a production calibration claim.
- What variables carry the gain? Feature lists and lineage are in the JSON report and model cards.
- Did compact formulas pass development falsification? Formula hypotheses ran for `seller_next_action` only without a hidden formula submission.
- Where does the model abstain? Abstention reports are emitted per model row.
- How much performance came from identifiers or history fields? Identifier fields are forbidden by the feature contract; frozen Benchmark v1 negative controls remain incomplete and are gate failures.
- Does the result remain after all canary and negative controls? Negative-control diagnostics report pass/fail fields, but full-release confirmation remains blocked.

## Limitations

- This run uses the 100,000-thread bounded normalization, not the full NBER release.
- Inspectable model runs use a deterministic per-target row cap because the current full model suite is not yet optimized for hundreds of thousands of rows.
- Hidden submissions are one-shot within the recorded lockbox store, but this is not a remote third-party lockbox.
- No production model is exported and no causal seller-profit claim is made.
