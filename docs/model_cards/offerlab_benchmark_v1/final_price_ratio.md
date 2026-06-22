# OfferLab Benchmark v1 Model Card: final_price_ratio

Research-only NBER-derived artifact. Not production-exportable.

- Total target rows: `51957`
- Sampled model rows: `500`
- Hidden submitted: `True`
- Overall Benchmark v1 gate: `STOP`
- Protocol splits complete: `False`
- Omitted protocol splits: `buyer_disjoint, category_disjoint_diagnostic, thread_safe_nested_development`
- Missing negative controls: `future_status_canary, accepted_price_canary, identifier_memorization_canary, censoring_as_rejection_canary`
- Production export permission: `False`
- Selected model: `median_regressor`
- Features used: ``
- Hidden relative improvement vs development-selected baseline: `0.0000`
- Hidden support coverage: `0.3100`
- Hidden abstention rate: `0.6900`
- Lineage hash: `e46ca97f51092fe85a979c9dcacb1e9c7e5bcd2c9c288b831aa3a9a05a568946`

Limitations:

- Bounded 100k normalization, not full-release evidence.
- Row-capped inspectable model run.
- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.
- No causal or seller-profit claim.
