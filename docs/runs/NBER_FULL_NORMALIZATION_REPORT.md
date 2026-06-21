# NBER Full Normalization Report

Status: blocked on 2026-06-21 before full-source normalization began.

## Command

```powershell
$env:OFFERLAB_DATA_ROOT = "C:\OfferLabData"
python -m behavior_lab nber-best-offer normalize-real --raw-dir C:\OfferLabData\raw\nber_best_offer --output-dir C:\OfferLabData\processed\nber_best_offer_full --full
```

Result: the command raised `NberRealNormalizeError` before reading source rows:

```text
Full NBER normalization is intentionally blocked: the current real-source normalizer is validated only for bounded --limit-threads runs. Implement disk-backed listing/thread indexes and full-run checkpointing before using --full.
```

## Data Checked

Raw source files remained outside the repository under `C:\OfferLabData\raw\nber_best_offer`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `anon_bo_lists.csv.gz` | 4,451,661,738 | `CEDA12755878304DAA4CA43B45C72EC98A7382A1EE646E66C33F6841E5D1A646` |
| `anon_bo_threads.csv.gz` | 1,374,076,192 | `F6FAEB797A8ED2F0C84D0E3C6E9B82F0AD2BD971DF354D57C902B478E757DEE9` |

Local disk at command time:

- Drive: `C:`
- Free bytes: `284638674944`
- Used bytes: `713837408256`

## Lineage

- Git commit: `ecdf9ab357588b240fe452347db54d4f24351935`
- Transformation version: `nber_best_offer_real_normalization.v1`
- Random seed: default `20240621`
- Split-manifest hash: not applicable; full normalization did not start.
- Output directory requested: `C:\OfferLabData\processed\nber_best_offer_full`
- Full normalized manifest: not produced.

## Rows

- Total source rows read: `0`
- Normalized rows written: `0`
- Quarantined rows: `0`
- Distinct threads: not evaluated.
- Distinct listings: not evaluated.
- Duplicate identifiers: not evaluated.
- Unmatched listing IDs: not evaluated.
- Date range: not evaluated.

## Replication Status

The prior bounded 100K check remains useful only as a structure and lineage gate:

```json
{
  "bounded_structure_passed": true,
  "full_replication_passed": false,
  "passed": false,
  "fatal_failures": 0,
  "fatal_unevaluated": 21,
  "manifest_hash": "6DE96D4DE6E1379D4D11E5B8E243697774345C737E007B066F3551EE0D43E288",
  "targets_hash": "2DC4CD2644E356B836F863B9C5C99ABB9FB06E5C4F51E0877039EDFB3B14676F"
}
```

Published descriptive moments and full-source structural targets are not replicated.

## Leakage Risks

No model training or task construction was run from a full normalized dataset. The current blocker prevents a misleading full-data claim before the execution path has full-run proof. Existing risks remain:

- Outcome/future fields must not enter predictor-facing snapshots.
- Reference prices/counts remain excluded unless recomputed with as-of timing.
- Thread/listing boundaries must stay split-safe.
- Published moments require the authors' full sample restrictions before model comparison.

## Gate

Wave 2 full normalization did not pass. Do not run Wave 2B, Wave 2C, or the Wave 2 benchmark integration against real NBER data until `--full` has a separately tested bounded-memory implementation with full-run checkpoints, disk preflight, partition verification, and published-stat replication.

