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

The current normalizer supports the repository fixture schema (`listings.csv` and `turns.csv`) and emits JSONL partition tables as a standard-library fallback. It does **not** yet map the official full NBER release schema. Full-release normalization requires a codebook-driven adapter and should use an optional streaming/Parquet dependency group.

## Interpretation

This benchmark can support a statement like:

> Observable Best Offer variables predict negotiation outcomes better than simple baselines under chronological and seller-disjoint tests.

It cannot support:

> OfferLab causally increases seller profit.

That requires seller cost basis, actual fees, returns, holding costs, and prospective randomized or shadow-mode evidence.
