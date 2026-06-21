# NBER Wave 1 100K Gate Report

Status: bounded real-source Wave 1 gate run completed on 2026-06-21.

## Commands

```powershell
$env:OFFERLAB_DATA_ROOT = "C:\OfferLabData"
python -m behavior_lab nber-best-offer inspect-schema --raw-dir C:\OfferLabData\raw\nber_best_offer
python -m behavior_lab nber-best-offer normalize-real --raw-dir C:\OfferLabData\raw\nber_best_offer --output-dir C:\OfferLabData\processed\nber_best_offer_100k --limit-threads 100000 --bucket-count 64 --partition-rows 50000
python -m behavior_lab nber-best-offer normalize-real --raw-dir C:\OfferLabData\raw\nber_best_offer --output-dir C:\OfferLabData\processed\nber_best_offer_100k --limit-threads 100000 --bucket-count 64 --partition-rows 50000
python -m behavior_lab nber-best-offer replication-check --normalized-dir C:\OfferLabData\processed\nber_best_offer_100k
python -m behavior_lab nber-best-offer normalize-real --raw-dir C:\OfferLabData\raw\nber_best_offer --output-dir C:\OfferLabData\processed\nber_best_offer_100k_resume_probe --limit-threads 100000 --bucket-count 64 --partition-rows 50000 --stop-after-thread-pass
python -m behavior_lab nber-best-offer normalize-real --raw-dir C:\OfferLabData\raw\nber_best_offer --output-dir C:\OfferLabData\processed\nber_best_offer_100k_resume_probe --limit-threads 100000 --bucket-count 64 --partition-rows 50000
```

The second `normalize-real` command was an idempotency rerun and returned the existing manifest without duplicating output.

## Data Used

Raw data stayed outside the repository under `C:\OfferLabData\raw\nber_best_offer`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `anon_bo_lists.csv.gz` | 4,451,661,738 | `CEDA12755878304DAA4CA43B45C72EC98A7382A1EE646E66C33F6841E5D1A646` |
| `anon_bo_threads.csv.gz` | 1,374,076,192 | `F6FAEB797A8ED2F0C84D0E3C6E9B82F0AD2BD971DF354D57C902B478E757DEE9` |
| `Codebook.xlsx` | 15,824 | `3FA5E83046AC29E610CF2BCF02FD85682F93F3608C689C5A434D794C65BB6516` |
| Released code zip | 38,620 | `94790B8638A0BE5D96807A1D09E970BBE7C2A8282FC91B95838F1220CA6D882E` |

## Run Lineage

- Git commit: `eb1772c7abf485fc82e7db7981d43f3c063175e0`
- Transformation version: `nber_best_offer_real_normalization.v1`
- Mapping manifest hash: `0693436B12AFE72EE10B928B589073569BEF9BC17BF511DB350692C6752088DE`
- Replication target manifest hash: `2DC4CD2644E356B836F863B9C5C99ABB9FB06E5C4F51E0877039EDFB3B14676F`
- Random seed: `20240621`
- Split manifest hash: not applicable; this was normalization only.
- Normalized manifest path: `C:\OfferLabData\processed\nber_best_offer_100k\manifest.json`
- Normalized manifest file hash from replication check: `6DE96D4DE6E1379D4D11E5B8E243697774345C737E007B066F3551EE0D43E288`

## Output

| Table | Format | Rows | Partitions |
| --- | --- | ---: | ---: |
| `negotiation_turns` | Parquet | 430,628 | 9 |
| `listings` | Parquet | 231,922 | 5 |

Thread-linked listing extraction:

- Distinct listing IDs referenced by sampled threads: 231,922
- Matched listing rows: 231,922
- Unmatched listing IDs: 0
- Non-negotiated listings omitted: true

Source thread pass:

- Distinct threads: 100,000
- Accepted turn rows: 430,628
- Complete duplicate rows removed: 9
- Offer type counts: `0=313494`, `1=22732`, `2=94402`
- Status counts: `0=59975`, `1=104836`, `2=62975`, `6=73293`, `7=117093`, `8=5056`, `9=7400`

Quarantine:

- Counts: `{}`
- Bounded examples: none

## Runtime And Memory

- Normalization runtime: 700.141 seconds measured by wrapper; manifest runtime: 699.56 seconds.
- Peak RSS during normalization: 262,041,600 bytes.
- Output location: `C:\OfferLabData\processed\nber_best_offer_100k`
- Resume probe: stopped after thread pass in 163.312 seconds, then resumed to completion in 474.079 seconds under `C:\OfferLabData\processed\nber_best_offer_100k_resume_probe`.

## Gate Checks

Passed:

- Real headers accepted for both official files.
- 100,000 real threads normalized.
- Matching listings joined with zero unmatched listing IDs.
- Hash lineage includes raw source hashes, mapping hash, partition hashes, Git commit, command args, and random seed.
- Idempotent rerun returned the existing manifest without duplicate output.
- Real resume probe completed without duplicate output or row-count drift.
- Replication target contract was not changed after the run.
- Frozen replication check returned no fatal failures for currently evaluable bounded-sample checks.

Not yet passed:

- Published descriptive moments require full-source replication of the authors' sample restrictions and were not evaluated on this bounded sample.

## Leakage Risks

- `status_id`, `response_time`, `item_price`, `bo_ck_yn`, sold-listing `auct_end_dt`, and purchaser fields remain future/outcome fields and must not enter predictor-facing snapshots.
- `accept_price` and `decline_price` are seller-private policy fields and should remain excluded unless explicitly authorized for a policy-leak diagnostic.
- `ref_price*`, `count*`, `view_item_count`, `wtchr_count`, and participant-history fields still need timing audits before feature use.
- This run normalized records; it did not train models or query hidden benchmark lockboxes.

## Scientific Limitations

- NBER artifacts remain research-only and non-exportable.
- This was a bounded negotiation-first sample, not the full 98-million-listing normalization.
- Observational counteroffer comparisons remain descriptive and cannot support causal profit claims.
- Published-stat replication is the next gate before model comparison.
