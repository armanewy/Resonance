# NBER Best Offer Benchmark Audit

The NBER lane is the primary public evidence campaign for OfferLab. It must answer whether observable bargaining variables predict seller actions, buyer responses, agreement, final price ratio, and response latency under leakage-safe splits.

## Required Controls

- Chronological split with complete negotiation threads confined to one region; threads crossing a boundary are purged and reported.
- Seller-disjoint split where seller identifiers are available.
- Category breakdown.
- Future-round leakage check.
- Final-price and final-status leakage check.
- Random-label control before accepting any complex model.
- Identifier memorization check before using buyer or seller history.

## Current Implementation

The repository implements a fixture-sized NBER path:

- `nber-best-offer build-sample`
- `nber-best-offer normalize`
- `nber-best-offer benchmark`
- `nber-best-offer audit`

The original normalizer supports the repository fixture schema (`listings.csv`
and `turns.csv`) and emits JSONL partition tables as a standard-library
fallback. The official full NBER release now has a separate Wave 1 real-source
adapter and contract; it should remain separate from the fixture normalizer.

Wave 1 adds the first real-release contract and bounded adapter path:

- `docs/research/NBER_REAL_SCHEMA.md`
- `datasets/manifests/nber_best_offer_real_mapping.yaml`
- `docs/research/NBER_REPLICATION_CONTRACT.md`
- `datasets/manifests/nber_replication_targets.yaml`
- `docs/research/NBER_SCALE_PLAN.md`
- `nber-best-offer inspect-schema`
- `nber-best-offer source-inventory`
- `nber-best-offer normalize-real`
- `nber-best-offer replication-check`

The real-release path is separate from the fixture normalizer. A bounded fixture
run accepts the verified real headers, writes partitioned Parquet when PyArrow
is installed, records raw hashes and mapping lineage, keeps thread rows grouped
before partition output, extracts only thread-linked listings for the first
benchmark, and supports resume after the thread pass. The full official release
has not been normalized yet. The `--full` mode is intentionally blocked until
the real-source path replaces its bounded-sample in-memory state with
disk-backed listing and thread indexes, full-run checkpoints, and disk preflight
evidence.

## Interpretation

This benchmark can support a statement like:

> Observable Best Offer variables predict negotiation outcomes better than simple baselines under chronological and seller-disjoint tests.

It cannot support:

> OfferLab causally increases seller profit.

That requires seller cost basis, actual fees, returns, holding costs, and prospective randomized or shadow-mode evidence.
