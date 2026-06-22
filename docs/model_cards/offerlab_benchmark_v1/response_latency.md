# OfferLab Benchmark v1 Model Card: response_latency

Research-only NBER-derived artifact. Not production-exportable.

- Total target rows: `425531`
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
- Hidden support coverage: `0.4200`
- Hidden abstention rate: `0.5800`
- Lineage hash: `692faba512a92d10b6e45e9be6c7db428ee63f61e2eab79b884f5b7726eb0c69`

Limitations:

- Bounded 100k normalization, not full-release evidence.
- Row-capped inspectable model run.
- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.
- No causal or seller-profit claim.
