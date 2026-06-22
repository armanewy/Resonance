# NBER Best Offer Benchmark Audit

The NBER lane is the primary public evidence campaign for OfferLab. It must answer whether observable bargaining variables predict seller actions, buyer responses, agreement, final price ratio, and response latency under leakage-safe splits.

## Required Controls

- Chronological split with complete listings confined to one region; boundary-crossing listings are purged and reported. This is stricter than thread-only purging because all negotiation threads attached to one listing stay in the same region.
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
benchmark, and supports resume after the thread pass. The `--full` mode now
runs the same streaming/checkpoint path without a thread limit, using SQLite
membership indexing rather than a Python in-memory listing-ID set. A fixture
full-path smoke run verifies that implementation path, but the official full
release has not been normalized yet.

Full-release benchmark use remains gated. Task construction and benchmark scope
require verified official source hashes and byte sizes, current partition-file
hash checks, full-run checkpoint validation, published-stat replication, and an
independent audit. Verification rechecks source files from disk and requires
hash-bound JSON artifacts for the replication and independent audit results.
Hand-written manifest booleans are not sufficient evidence.

## Interpretation

After a completed scoped benchmark run, this benchmark can support a statement like:

> Observable Best Offer variables predict negotiation outcomes better than simple baselines under chronological and seller-disjoint tests.

The current fixture, bounded real-source smoke runs, and fixture full-path smoke
run do not yet support a full-release performance claim.

It cannot support:

> OfferLab causally increases seller profit.

That requires seller cost basis, actual fees, returns, holding costs, and prospective randomized or shadow-mode evidence.
