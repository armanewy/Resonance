# OfferLab Benchmark v1 Model Card: seller_next_action

Research-only NBER-derived artifact. Not production-exportable.

- Total target rows: `333493`
- Sampled model rows: `500`
- Hidden submitted: `True`
- Overall Benchmark v1 gate: `STOP`
- Protocol splits complete: `False`
- Omitted protocol splits: `buyer_disjoint, category_disjoint_diagnostic, thread_safe_nested_development`
- Missing negative controls: `future_status_canary, accepted_price_canary, identifier_memorization_canary, censoring_as_rejection_canary`
- Production export permission: `False`
- Selected model: `regularized_glm`
- Features used: `222 features, hash 5153dc96c5709e64795bf91a09a096adf60c30174372de62c5d58409dd634825`
- Hidden relative improvement vs development-selected baseline: `0.0250`
- Hidden support coverage: `0.3700`
- Hidden abstention rate: `0.6300`
- Lineage hash: `b2f64509d4e34a2c1875c4798fae1052f74868ac662ff554ad8c5a71b80c8167`

Limitations:

- Bounded 100k normalization, not full-release evidence.
- Row-capped inspectable model run.
- Standalone card inherits the overall STOP gate; hidden metrics are diagnostic only.
- No causal or seller-profit claim.
