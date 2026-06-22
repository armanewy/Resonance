# OfferLab Benchmark v1 Model Card: agreement

Research-only NBER-derived artifact. Not production-exportable.

- Total target rows: `98372`
- Sampled model rows: `500`
- Hidden submitted: `True`
- Overall Benchmark v1 gate: `STOP`
- Protocol splits complete: `False`
- Omitted protocol splits: `buyer_disjoint, category_disjoint_diagnostic, thread_safe_nested_development`
- Missing negative controls: `future_status_canary, accepted_price_canary, identifier_memorization_canary, censoring_as_rejection_canary`
- Production export permission: `False`
- Selected model: `regularized_glm`
- Features used: `225 features, hash 8cc57b398082bfc9f48824fa0fbf3e22d975215122e4c625bded81da3108908d`
- Hidden relative improvement vs development-selected baseline: `0.0697`
- Hidden support coverage: `0.3300`
- Hidden abstention rate: `0.6700`
- Lineage hash: `ce8d75d83383b3ce86c369eb7981c6b9e5bcb62665662a9438f95d67a48836a0`

Limitations:

- Bounded 100k normalization, not full-release evidence.
- Row-capped inspectable model run.
- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.
- No causal or seller-profit claim.
