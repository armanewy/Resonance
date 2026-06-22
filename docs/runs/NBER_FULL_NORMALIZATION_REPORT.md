# NBER Full Normalization Report

Status: implementation gate updated on 2026-06-21. The `--full` real-source
normalizer is no longer hard-blocked, but the official full NBER release was
not normalized during this patch pass.

## Implemented Command

```powershell
$env:OFFERLAB_DATA_ROOT = "<external_data_root>"
python -m behavior_lab nber-best-offer normalize-real --raw-dir $env:OFFERLAB_DATA_ROOT\raw\nber_best_offer --output-dir $env:OFFERLAB_DATA_ROOT\processed\nber_best_offer_full --full
```

The command now runs the same streaming/checkpoint path used by bounded real
runs, with `limit_threads = null` and `normalization_scope =
full_unbounded_source_scan`.

## Full-Run Safeguards

- Full mode and `--limit-threads` are mutually exclusive.
- The thread pass streams source rows into deterministic hash buckets.
- Thread/listing membership is held in SQLite, not a Python in-memory listing-ID set.
- The thread pass checkpoint is reused only when command args, source hashes,
  headers, mapping hash, bucket hashes, and SQLite index content hashes match.
- Listing extraction streams the listing source and retains only thread-linked
  listings for the first benchmark.
- Every output partition records a SHA-256 hash and the manifest verifies those
  hashes before setting `streaming_full_run_passed`.
- The manifest records disk preflight, official-source hash/byte checks, and a
  separate `audited_full_release_evidence` gate.
- Full-release evidence verification rechecks official source files from disk
  against the expected byte sizes and SHA-256 hashes.
- Replication and independent-audit claims must point to JSON artifacts with
  recorded artifact hashes; manifest booleans alone are not accepted.

## Evidence Gate

A completed `--full` run is not automatically full-release benchmark evidence.
The manifest field `audited_full_release_evidence.passed` remains false until
all of these are true:

- `streaming_full_run_passed`
- `official_sources_matched`
- `full_run_checkpoint_validated`
- `partition_hashes_verified`
- `replication_contract_passed`
- `independent_audit_passed`

Task generation and benchmark scope both require that stricter evidence gate
before an unbounded full manifest can be used as full-release evidence. The
gate verifies the source files, partition files, replication artifact, and
independent audit artifact at evaluation time.

## Data Checked

No raw source rows from the official full release were read in this patch pass.
Previously inventoried official source metadata remains the expected contract:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `anon_bo_lists.csv.gz` | 4,451,661,738 | `CEDA12755878304DAA4CA43B45C72EC98A7382A1EE646E66C33F6841E5D1A646` |
| `anon_bo_threads.csv.gz` | 1,374,076,192 | `F6FAEB797A8ED2F0C84D0E3C6E9B82F0AD2BD971DF354D57C902B478E757DEE9` |

## Fixture Audit

The exact-schema fixture full run verifies the implementation path without
claiming official evidence:

- `--full` completed with `limit_threads = null`.
- Negotiation-turn rows written: `4`.
- Thread-linked listing rows written: `3`.
- Unmatched listing IDs: `0`.
- Full-run preflight passed.
- Official-source contract did not match, as expected for fixtures.
- `streaming_full_run_passed` was true.
- `audited_full_release_evidence.passed` remained false.
- An idempotent rerun returned the existing manifest.
- A forged manifest with official-looking hashes, true booleans, missing source
  files, and missing evidence artifacts was rejected.

Validated by:

```powershell
python -m pytest tests\nber_real\test_real_nber_pipeline.py -q -p no:cacheprovider
python -m pytest tests\test_nber_best_offer_pipeline.py -q -p no:cacheprovider
```

## Rows

Official full release:

- Total source rows read: `0` in this patch pass.
- Normalized rows written: `0` in this patch pass.
- Quarantined rows: `0` in this patch pass.
- Distinct threads: not evaluated on official full data.
- Distinct listings: not evaluated on official full data.
- Duplicate identifiers: not evaluated on official full data.
- Unmatched listing IDs: not evaluated on official full data.

Fixture full-path smoke:

- Normalized negotiation-turn rows: `4`.
- Thread-linked listing rows: `3`.
- Quarantine counts: `{}`.

## Replication Status

The full published-stat replication contract has not passed because the
official full release has not been normalized in this patch pass. The bounded
100K run remains only structure and lineage evidence, not a full-release
performance claim.

## Leakage Risks

Existing risks remain active until a real full run and independent audit are
complete:

- Outcome/future fields must stay out of predictor-facing snapshots.
- Reference prices/counts remain excluded unless recomputed with as-of timing.
- Thread/listing boundaries must stay split-safe.
- Published moments require the authors' full sample restrictions before model
  comparison.
- Observational counteroffer comparisons must not be interpreted causally.

## Gate

Implementation gate: passed for the streaming/checkpoint `--full` code path on
fixtures.

Official evidence gate: not passed. The official full release still needs an
actual external-data run, replication check, and independent audit before Wave 2
benchmark integration can claim full-release evidence.
