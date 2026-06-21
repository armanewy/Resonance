# Reviewed Integrity Patch — v0.4.0

This release hardens the research harness without claiming that the public-data models are production-ready.

## Fixed

- Development and hidden evaluation budgets persist across processes.
- Evaluation reservations are consumed before scoring begins, preventing crash-and-retry probing.
- Hidden submissions bind to the exact proposal artifact, training snapshot, and hidden case set.
- Campaign metadata changes and overlapping hidden-case reuse under renamed campaigns are rejected.
- Research-store writes are hash-chained and protected by a local interprocess lock.
- NBER chronological benchmark paths keep complete negotiation threads together and purge boundary-crossing threads.
- Censored negotiations are excluded from negative agreement labels.
- Transfer-ablation reports no longer contain invented default metrics; an unrun ablation is marked `not_run`.
- The eBay API probe reports unrelated-listing visibility as an observation rather than assuming denial.
- Current authorized seller data requires hashed authorization evidence tied to an immutable ledger record before commercial training, inference, or export is permitted.
- Model lineage hashes the actual feature values used for training.
- Content-addressed cache object and metadata writes are concurrency-safe.
- Package and artifact software versions are synchronized at `0.4.0`.

## Validation

- `python -m pytest -q`: 152 passed.
- `python -m unittest discover -s tests -q`: 149 tests passed.
- `python -m compileall -q src tests tools`: passed.
- `git diff --check`: passed.

## Remaining limitation

The NBER acquisition lane can cache official files, but the normalizer currently supports only the repository fixture schema. A codebook-driven streaming adapter is still required before the full 98-million-listing release can be backtested.

The lockboxes are scientific integrity controls inside a trusted local process. They are not a security sandbox for malicious generated code; an autonomous external researcher should receive typed RPC access from a separate process.
