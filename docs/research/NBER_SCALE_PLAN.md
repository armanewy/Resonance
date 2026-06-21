# NBER Full-Scale Execution Design

Wave 1 Prompt 1D is a scale design for the NBER eBay Best Offer lane. It does not expand OfferLab, does not create a production connector, and does not change the fixture normalizer. The path stays research-only under `datasets/manifests/index.yaml`: NBER is allowed for research and internal benchmarking, and blocked for commercial training and production export.

## Source Boundary

Official source: https://www.nber.org/research/data/best-offer-sequential-bargaining

NBER states that the release is public, research-purpose eBay Best Offer bargaining data with identifying information removed. The official page lists `anon_bo_threads.csv.gz`, `anon_bo_lists.csv.gz`, `Codebook.xlsx`, `bargaining_data.zip`, and code. It states that `anon_bo_lists.csv` contains 98 million listings from May 1, 2012 through June 1, 2013; the page does not state the row count for `anon_bo_threads.csv`.

No large raw CSV was downloaded for this plan. HTTP metadata checked on 2026-06-21:

| File | URL | Bytes checked without body download | Last modified | ETag |
| --- | --- | ---: | --- | --- |
| `anon_bo_threads.csv.gz` | `https://data.nber.org/bargaining/anon_bo_threads.csv.gz` | 1,374,076,192 | Thu, 24 May 2018 17:51:23 GMT | `"51e6bd20-56cf74cb1dcc0"` |
| `anon_bo_lists.csv.gz` | `https://data.nber.org/bargaining/anon_bo_lists.csv.gz` | 4,451,661,738 | Thu, 24 May 2018 17:43:59 GMT | `"10956f7aa-56cf7323af5c0"` |
| `Codebook.xlsx` | `https://data.nber.org/bargaining/Codebook.xlsx` | not range-probed | Wed, 08 May 2024 12:11:27 GMT | `"3dd0-617f033bd4c3b-gzip"` |

The byte values for the two CSV.GZ files came from one-byte range responses (`Content-Range` totals), not from downloading the files.

## Practical Engine Comparison

The repo has no declared optional data-engine dependencies. In this environment, `pyarrow==24.0.0` and `psutil==7.2.2` are installed; `duckdb` and `polars` are not installed. The harness at `scripts/benchmark_nber_io.py` therefore runs PyArrow and the Python CSV/gzip fallback here, while DuckDB and Polars are explicitly skipped unless the packages are added to the environment.

The harness generates deterministic synthetic `anon_bo_threads.csv.gz` and `anon_bo_lists.csv.gz`, records SHA-256 hashes for those generated files, streams the thread file first, extracts thread-linked listing IDs, then streams listings and retains only matching listing rows.

| Approach | Current availability | Fit for full execution | Main risk |
| --- | --- | --- | --- |
| DuckDB | Not installed | Strong candidate if added: external joins, direct CSV.GZ scan, Parquet writes, disk-backed execution. | Adds an undeclared dependency; must pin version and test Windows spill behavior. |
| PyArrow | Installed | Best immediate path for Parquet writing and streaming batch reads. Use it for chunked normalization and Parquet materialization. | Membership joins are not enough by themselves; use a disk-backed listing ID index or sharded ID partitions. |
| Polars streaming | Not installed | Candidate for fast projection/filtering if added. | Streaming joins and partitioned writes need validation against the official schema and Windows paths. |
| Python CSV/gzip fallback | Always available | Required safety fallback for inventory, codebook checks, and tiny/rescue runs. | Slowest, JSONL-only unless paired with PyArrow writer, and easy to overuse memory if IDs are held in one set. |

## Generated Benchmark Results

Commands run:

```powershell
python scripts\benchmark_nber_io.py --rows 100000 --listing-rows 200000 --engines python,pyarrow,duckdb,polars --partitions 64 --json-output runs\nber_io_bench_100k.json
python scripts\benchmark_nber_io.py --rows 1000000 --listing-rows 2000000 --engines python,pyarrow,duckdb,polars --partitions 128 --json-output runs\nber_io_bench_1m.json
```

Generated data was deleted after each run; only the ignored JSON summaries were retained under `runs/`. The harness uses deterministic gzip output, so the generated source hashes are stable for the script version.

| Run | Generated thread rows | Generated listing rows | Distinct negotiated listings | Threads CSV.GZ SHA-256 | Listings CSV.GZ SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| 100K | 100,000 | 200,000 | 33,333 | `2a610255512b1413ee694617370a6c4c1ae47ca69386c317b95bcd439235e7bc` | `eb524e82832391ca88b3ccfb7dcf4c6c56679afb8a94f3234ace7dc853d749f3` |
| 1M | 1,000,000 | 2,000,000 | 333,333 | `cb498944c60a8395d91bccbec685886498f4e156499e64f99ac510c98197a8eb` | `2f4e75d1e9298a719791f6584e1d61d92a97dc0dcce5f4b78a188b00efeec8e8` |

| Run | Engine | Status | Runtime | Peak RSS | Output | Rows/sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 100K | Python CSV/gzip | ok | 1.45 s | 33.6 MB | 35.0 MB JSONL | 206K |
| 100K | PyArrow | ok | 1.86 s | 185.1 MB | 4.7 MB Parquet | 162K |
| 100K | DuckDB | skipped | not installed | n/a | n/a | n/a |
| 100K | Polars streaming | skipped | not installed | n/a | n/a | n/a |
| 1M | Python CSV/gzip | ok | 24.31 s | 70.3 MB | 349.7 MB JSONL | 123K |
| 1M | PyArrow | ok | 20.36 s | 491.1 MB | 56.1 MB Parquet | 147K |
| 1M | DuckDB | skipped | not installed | n/a | n/a | n/a |
| 1M | Polars streaming | skipped | not installed | n/a | n/a | n/a |

Interpretation:
- PyArrow is the best available Parquet writer in this environment and overtakes the fallback at 1M rows, but the harness still keeps negotiated listing IDs in memory. The full pipeline must replace that with sharded IDs or a disk-backed index.
- The Python fallback is useful as a rescue path and exposes low RSS on generated data, but JSONL output is roughly 6x larger than PyArrow Parquet at 1M rows.
- DuckDB and Polars cannot be claimed as tested here until the packages are added and pinned; the script records them as skipped.

## Negotiation-First Pipeline

The first benchmark must be negotiation-first: it should normalize only listings that are referenced by negotiation threads and must write an atomic final manifest saying that non-negotiated listings omitted from the first benchmark remain available for a later full-listing analysis.

1. Preflight source files.
   - Require explicit local paths for `anon_bo_threads.csv.gz`, `anon_bo_lists.csv.gz`, and the codebook-derived schema map.
   - Check file existence, readable bytes, gzip header, available disk, and output directory emptiness or resumable checkpoint state.
   - Refuse to start if free disk is below the estimated high-water mark plus 20 percent.

2. Stream `anon_bo_threads.csv.gz`.
   - Read in bounded batches.
   - Normalize negotiation turns into a codebook-backed schema with no future outcome fields in model features.
   - Route rows by `partition_id = sha256(listing_id)[0:8] % N`.
   - Write immutable chunk files such as `negotiation_turns/part=0042/chunk-000001.parquet.tmp`, then atomically replace the final chunk path after footer verification.
   - Record checkpoint entries by chunk: input file validator, row range, compressed byte watermark where available, output paths, row counts, and SHA-256 of each emitted file.

3. Extract distinct listing IDs referenced by negotiation threads.
   - Emit listing ID shards while streaming turns: `listing_id_candidates/part=0042/chunk-000001.parquet`.
   - Deduplicate per shard into `listing_ids_distinct/part=0042/ids.parquet`.
   - Keep per-shard memory bounded; do not require a single in-memory set for full execution.
   - For the standard-library fallback, use a SQLite table with `listing_id TEXT PRIMARY KEY, partition_id INTEGER` and batch `INSERT OR IGNORE`.

4. Stream `anon_bo_lists.csv.gz`.
   - Normalize listing rows in bounded batches.
   - Retain only rows whose `listing_id` is present in the negotiated listing index for the first benchmark.
   - With PyArrow-only execution, use the SQLite index or sharded ID files as the membership source. With DuckDB available, load distinct IDs as a disk-backed table and run an external join.
   - Write `negotiated_listings/part=0042/chunk-000001.parquet` with the same deterministic partition function.

5. Write atomic final manifest.
   - First write `manifest.json.tmp` and `manifest.sha256.tmp`; then `os.replace` both final files on the same volume.
   - Include source paths, source byte sizes, source validators, codebook hash, transformation version, row counts, partition count, omitted non-negotiated listing count if measured, checkpoint log path, and an explicit note: "non-negotiated listings omitted from first benchmark."
   - Mark the complete 98M-listing normalization path as deferred, not performed.

6. Preserve later full-listing analysis.
   - The later path should stream all 98 million listings to a separate dataset such as `all_listings_normalized/`, never overwrite `negotiated_listings/`.
   - Its manifest must identify the expanded scope and must not be pooled into the first negotiation benchmark.

## Restart After Interruption

The pipeline must support restart after interruption without duplicate rows.

Restart must be chunk-idempotent. A resumed run should read the checkpoint log, verify final chunk files and hashes, skip only verified complete chunks, and rewrite incomplete `.tmp` chunks from the last committed row boundary. The writer must never append to existing Parquet files; every output file name is derived from table, partition, and chunk ordinal. That prevents duplicate output after resume and preserves deterministic partitions.

The listing pass depends on a finalized distinct-ID index. If the turn pass is incomplete, listing extraction must refuse to start. If listing extraction is interrupted, resume from its own checkpoint after verifying the negotiated ID index hash.

## Deterministic Partitions

Use a cross-platform hash rule rather than engine-native hash functions:

```text
partition_id = int.from_bytes(sha256(listing_id.encode("utf-8")).digest()[:8], "big") % partition_count
```

This keeps PyArrow, DuckDB, Polars, and Python fallback outputs aligned. Use fixed-width partition directories such as `part=0042` for stable lexical ordering.

## Operational Controls

Bounded memory:
- Target batch size: 64K to 512K input rows, tuned by row width.
- Target partitions: start at 1024 for full source files; lower counts are acceptable for 100K and 1M tests.
- Never keep all 98 million listing rows in memory.
- Do not keep all distinct listing IDs in memory for full execution; dedupe by shard or use SQLite/DuckDB disk indexing.

Checkpoints:
- Store append-only checkpoint JSONL under `_checkpoints/`.
- Include chunk row counts, input validator, schema version, output file hash, and completion timestamp.
- Treat a checkpoint as valid only when every referenced output file exists and hashes match.

Progress reporting:
- Emit machine-readable progress JSONL every fixed row interval and at chunk commit.
- Include table, pass, rows read, rows written, compressed input bytes if available, rows per second, RSS, free disk, and current partition.

Disk preflight:
- Check `shutil.disk_usage(output_root)`.
- Require same-volume temp and final output roots for atomic replace.
- Estimate high-water disk as compressed inputs already present plus normalized Parquet outputs, listing ID index, checkpoint logs, and temporary chunk files.

Windows-compatible paths:
- Use `pathlib.Path`.
- Avoid symlinks, hardlinks, shell-specific path expansion, and colon-bearing file names.
- Keep run roots short, for example `D:\bdl\nber_runs\wave1`.
- Use `os.replace` for atomic same-volume promotion.

## Scale Estimates

These estimates are for planning only. They combine the synthetic benchmark slope with official compressed file sizes and the NBER-stated 98 million listing rows. The official page does not state full thread row count; until the real thread file is inventoried, "full thread data" should be treated as a byte-scaled estimate. Official rows will be wider than the synthetic rows once the codebook adapter is applied, so the high end should be used for scheduling.

| Scope | Expected work | Disk high-water | Runtime estimate | Peak memory target |
| --- | --- | ---: | ---: | ---: |
| 100K thread rows | Stream 100K generated thread rows, extract IDs, scan 200K listing rows, write negotiated subset | 100 MB to 500 MB | Measured 1.45 s fallback, 1.86 s PyArrow after generation | Measured 34 MB fallback, 185 MB PyArrow |
| 1M thread rows | Stream 1M generated thread rows, extract IDs, scan 2M listing rows, write negotiated subset | 1 GB to 5 GB | Measured 24.31 s fallback, 20.36 s PyArrow after generation | Measured 70 MB fallback, 491 MB PyArrow |
| Full thread data | Stream 1.37 GB official compressed thread file, write partitioned turns and distinct-ID shards | 8 GB to 30 GB until measured | 30 minutes to 3 hours | Under 2 GB with sharded IDs |
| Thread-linked listing extraction | Stream 4.45 GB official compressed listings once, retain only negotiated listing IDs | 15 GB to 60 GB depending retained count and index | 1 to 4 hours | Under 2 GB with SQLite/DuckDB/sharded ID index |
| Complete 98M-listing normalization | Normalize every listing row, not just negotiated listings | 40 GB to 150 GB high-water | 3 to 12 hours | Under 2 GB if strictly chunked |

## Leakage Risks And Controls

Leakage risks:
- Final status or final price can leak into seller-next-action features.
- Future negotiation turns can leak into earlier-turn tasks.
- Thread boundaries can cross chronological split boundaries if split after row-level normalization.
- Seller and buyer identifiers can support memorization if used before identifier controls.
- Listing rows can include variables constructed after the decision point unless the codebook mapping tags timing explicitly.

Controls:
- Normalize immutable turns first, then build task-specific feature views that exclude forbidden future fields from `src/behavior_lab/datasets/nber_best_offer/schema.py`.
- Split complete negotiation threads, not individual rows.
- Purge boundary-crossing threads and report them.
- Run chronological, seller-disjoint, category breakdown, random-label, future-round, final-status, and identifier-memorization audits before accepting benchmark claims.

## Gate

Gate condition for Prompt 1D: pass if the repo contains a bounded-memory, restartable, deterministic, Windows-compatible full-scale execution design; includes a generated-data benchmark harness; records that the first benchmark omits non-negotiated listings; preserves the later complete 98M-listing normalization path; and does not download or commit large raw CSVs.
