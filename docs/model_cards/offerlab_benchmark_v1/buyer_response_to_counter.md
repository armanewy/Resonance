# OfferLab Benchmark v1 Model Card: buyer_response_to_counter

Research-only NBER-derived artifact. Not production-exportable.

- Total target rows: `91924`
- Sampled model rows: `500`
- Hidden submitted: `True`
- Overall Benchmark v1 gate: `STOP`
- Protocol splits complete: `False`
- Omitted protocol splits: `buyer_disjoint, category_disjoint_diagnostic, thread_safe_nested_development`
- Missing negative controls: `future_status_canary, accepted_price_canary, identifier_memorization_canary, censoring_as_rejection_canary`
- Production export permission: `False`
- Selected model: `majority`
- Features used: ``
- Hidden relative improvement vs development-selected baseline: `0.0000`
- Hidden support coverage: `0.3400`
- Hidden abstention rate: `0.6600`
- Lineage hash: `1912eecb4d91b8a0d0ba122ccf17245552c5d1de15fa29218e04d0ff59e61fe4`

Limitations:

- Bounded 100k normalization, not full-release evidence.
- Row-capped inspectable model run.
- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.
- No causal or seller-profit claim.
