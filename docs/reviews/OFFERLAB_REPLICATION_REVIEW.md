# OfferLab Replication Review

Verdict: the result is not yet independently replicable as a Benchmark v1 claim.

## Evidence Reviewed

- `datasets/manifests/offerlab_benchmark_v1.yaml`
- `reports/offerlab_benchmark_v1.json`
- generated model cards
- eBay blocked-run docs

## Findings

1. The protocol is frozen, but the runner is intentionally partial. Buyer-disjoint, category-disjoint diagnostic, and thread-safe nested development splits are omitted and gate-failing.
2. Required frozen negative controls are incomplete. The report marks future-status, accepted-price, identifier-memorization, and censoring-as-rejection controls missing.
3. Hidden submissions are locally one-shot and redacted, but not a remote third-party lockbox.
4. The eBay live-data side is not reproducible yet because required credentials and manual listing IDs are absent.

## Recommendation

Do not repeat Benchmark v1. It is frozen and hidden-spent. Run Benchmark v2 or
later only after the full protocol is implemented and the data/probe
prerequisites are available. Preserve the current v1 run as a diagnostic
baseline, not as evidence for deployment.
