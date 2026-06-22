from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Iterator

from behavior_lab.datasets.nber_best_offer.source_schema import (
    OFFER_TYPE_MAP,
    REAL_LISTING_COLUMNS,
    REAL_THREAD_COLUMNS,
    REAL_TRANSFORMATION_VERSION,
    STATUS_MAP,
    mapping_hash,
    sha256_file,
    validate_real_headers,
)

NBER_SOURCE_ID = "nber_ebay_best_offer"
PREDICTOR_ALLOWED_LISTING_FIELDS = [
    "category",
    "condition",
    "listing_price",
    "photo_count",
    "seller_us",
]
PROTECTED_LISTING_FIELDS = [
    "buyer_id_if_sold",
    "end_time",
    "final_sale_price",
    "sold_by_best_offer",
    "auto_decline_price",
    "auto_accept_price",
    "buyer_us_if_sold",
    "excluded_reference_price_ref_price4",
    "excluded_reference_count4",
]
OFFICIAL_FULL_SOURCE_EXPECTATIONS = {
    "anon_bo_lists": {
        "sha256": "CEDA12755878304DAA4CA43B45C72EC98A7382A1EE646E66C33F6841E5D1A646",
        "bytes": 4_451_661_738,
    },
    "anon_bo_threads": {
        "sha256": "F6FAEB797A8ED2F0C84D0E3C6E9B82F0AD2BD971DF354D57C902B478E757DEE9",
        "bytes": 1_374_076_192,
    },
}


class NberRealNormalizeError(ValueError):
    pass


@dataclass(frozen=True)
class Quarantine:
    counts: dict[str, int]
    examples: list[dict[str, Any]]

    def add(self, reason: str, row: dict[str, str], *, source_file: str, line_number: int) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if len(self.examples) < 25:
            self.examples.append(
                {
                    "source_file": source_file,
                    "line_number": line_number,
                    "reason": reason,
                    "row_hash": _row_hash(row),
                    "fields": sorted(row),
                }
            )

    def to_payload(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "examples": list(self.examples)}

    def merge_payload(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        for reason, count in dict(payload.get("counts", {})).items():
            self.counts[str(reason)] = self.counts.get(str(reason), 0) + int(count)
        for example in list(payload.get("examples", [])):
            if len(self.examples) >= 25:
                break
            if isinstance(example, dict):
                self.examples.append(example)


def normalize_real_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    limit_threads: int | None = None,
    full: bool = False,
    bucket_count: int = 32,
    partition_rows: int = 50_000,
    seed: int = 20240621,
    resume: bool = False,
    stop_after_thread_pass: bool = False,
    stop_after_turn_partitions: bool = False,
) -> dict[str, Any]:
    if not full and limit_threads is None:
        raise NberRealNormalizeError("Use --limit-threads or --full for real NBER normalization")
    if full and limit_threads is not None:
        raise NberRealNormalizeError("Use either --full or --limit-threads, not both")
    start = time.perf_counter()
    raw = Path(raw_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    partition_checkpoints = checkpoints / "partitions"
    partition_checkpoints.mkdir(parents=True, exist_ok=True)
    temp_dir = output / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    args_signature = {
        "raw_dir": str(raw.resolve()),
        "limit_threads": limit_threads,
        "full": full,
        "bucket_count": bucket_count,
        "partition_rows": partition_rows,
        "seed": seed,
        "transformation_version": REAL_TRANSFORMATION_VERSION,
        "normalizer_schema_revision": "wave1_audit_gate.v2",
    }
    lists_path = _find_source(raw, "anon_bo_lists.csv")
    threads_path = _find_source(raw, "anon_bo_threads.csv")
    source_hashes = {"anon_bo_lists": sha256_file(lists_path), "anon_bo_threads": sha256_file(threads_path)}
    source_bytes = {"anon_bo_lists": lists_path.stat().st_size, "anon_bo_threads": threads_path.stat().st_size}
    header_report = _validate_source_headers(lists_path, threads_path)
    if not header_report["valid"]:
        raise NberRealNormalizeError(json.dumps(header_report, sort_keys=True))
    full_preflight = _full_run_preflight(
        raw=raw,
        output=output,
        lists_path=lists_path,
        threads_path=threads_path,
        source_hashes=source_hashes,
        source_bytes=source_bytes,
        full=full,
        limit_threads=limit_threads,
        bucket_count=bucket_count,
        partition_rows=partition_rows,
    )
    if full and not full_preflight["passed"]:
        raise NberRealNormalizeError(json.dumps(full_preflight, sort_keys=True))
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches_current(
            current,
            args_signature=args_signature,
            source_hashes=source_hashes,
            header_report=header_report,
        ):
            current["idempotent_rerun"] = True
            return current

    quarantine = Quarantine(counts={}, examples=[])

    bucket_dir = temp_dir / "thread_buckets"
    id_index_path = temp_dir / "thread_listing_ids.sqlite"
    thread_checkpoint = checkpoints / "thread_pass.complete.json"
    thread_counts: dict[str, Any]
    checkpoint_signature = _thread_checkpoint_signature(
        args_signature=args_signature,
        source_hashes=source_hashes,
        header_report=header_report,
    )
    checkpoint = _load_valid_thread_checkpoint(
        thread_checkpoint,
        checkpoint_signature,
        bucket_dir=bucket_dir,
        id_index_path=id_index_path,
    )
    if checkpoint is not None:
        thread_counts = dict(checkpoint["thread_counts"])
        quarantine.merge_payload(checkpoint.get("quarantine"))
    else:
        if bucket_dir.exists():
            shutil.rmtree(bucket_dir)
        if id_index_path.exists():
            id_index_path.unlink()
        bucket_dir.mkdir(parents=True)
        thread_counts = _bucket_thread_rows(
            threads_path,
            bucket_dir,
            id_index_path,
            bucket_count=bucket_count,
            limit_threads=limit_threads,
            quarantine=quarantine,
        )
        _write_atomic_json(
            thread_checkpoint,
            {
                "schema_version": "nber_real_thread_pass_checkpoint.v2",
                "signature": checkpoint_signature,
                "thread_counts": thread_counts,
                "quarantine": quarantine.to_payload(),
            },
        )
    if stop_after_thread_pass:
        return {"status": "stopped_after_thread_pass", "thread_pass": thread_counts, "output_dir": str(output.resolve())}

    turn_table = tables / "negotiation_turns"
    listing_table = tables / "listings"
    for table_name, table_dir in [("negotiation_turns", turn_table), ("listings", listing_table)]:
        if table_dir.exists() and not resume:
            shutil.rmtree(table_dir)
            for checkpoint_file in partition_checkpoints.glob(f"{table_name}_*.json"):
                checkpoint_file.unlink()
        table_dir.mkdir(parents=True, exist_ok=True)

    partition_signature = _partition_checkpoint_signature(
        args_signature=args_signature,
        source_hashes=source_hashes,
        header_report=header_report,
    )
    turn_rows = _write_turn_partitions(
        bucket_dir,
        turn_table,
        checkpoint_dir=partition_checkpoints,
        signature=partition_signature,
        bucket_count=bucket_count,
        partition_rows=partition_rows,
        quarantine=quarantine,
        resume=resume,
    )
    if stop_after_turn_partitions:
        return {
            "status": "stopped_after_turn_partitions",
            "thread_pass": thread_counts,
            "turn_partitions": turn_rows,
            "output_dir": str(output.resolve()),
        }
    listing_rows = _write_listing_partitions(
        lists_path,
        id_index_path,
        listing_table,
        checkpoint_dir=partition_checkpoints,
        signature=partition_signature,
        partition_rows=partition_rows,
        quarantine=quarantine,
        resume=resume,
    )
    quarantine_path = output / "quarantine.json"
    quarantine_payload = quarantine.to_payload()
    _write_atomic_json(quarantine_path, quarantine_payload)
    manifest_sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest = {
        "status": "complete",
        "schema_version": "nber_real_normalized_manifest.v1",
        "transformation_version": REAL_TRANSFORMATION_VERSION,
        "source_dataset_ids": [NBER_SOURCE_ID],
        "research_only": True,
        "production_export_allowed": False,
        "commercial_training_allowed": False,
        "predictor_feature_policy": {
            "allowed_listing_fields": PREDICTOR_ALLOWED_LISTING_FIELDS,
            "protected_listing_fields": PROTECTED_LISTING_FIELDS,
            "reference_price_policy": "ref_price4 is preserved as excluded audit metadata and is not exported as predictor-facing reference_price.",
        },
        "git_commit": _git_commit(),
        "command_args": args_signature,
        "normalization_scope": "full_unbounded_source_scan" if full else "bounded_thread_limit",
        "random_seed": seed,
        "mapping_manifest_hash": mapping_hash(),
        "source_files": {
            "anon_bo_lists": {"path": str(lists_path.resolve()), "sha256": source_hashes["anon_bo_lists"], "bytes": source_bytes["anon_bo_lists"]},
            "anon_bo_threads": {"path": str(threads_path.resolve()), "sha256": source_hashes["anon_bo_threads"], "bytes": source_bytes["anon_bo_threads"]},
        },
        "header_validation": header_report,
        "full_release_preflight": full_preflight,
        "official_source_contract": _official_source_contract(source_hashes, source_bytes),
        "tables": {
            "negotiation_turns": {
                "path": str(turn_table.resolve()),
                "format": "parquet" if _pyarrow_available() else "jsonl",
                "rows": turn_rows["rows"],
                "partitions": turn_rows["partitions"],
                "schema": {
                    "schema_version": "nber_real_negotiation_turns.v1",
                    "transformation_version": REAL_TRANSFORMATION_VERSION,
                    "columns": _normalized_columns(turn_rows["partitions"]),
                },
            },
            "listings": {
                "path": str(listing_table.resolve()),
                "format": "parquet" if _pyarrow_available() else "jsonl",
                "rows": listing_rows["rows"],
                "partitions": listing_rows["partitions"],
                "schema": {
                    "schema_version": "nber_real_listings.v1",
                    "transformation_version": REAL_TRANSFORMATION_VERSION,
                    "columns": _normalized_columns(listing_rows["partitions"]),
                },
            },
        },
        "summary": _normalization_summary(thread_counts=thread_counts, turn_rows=turn_rows, listing_rows=listing_rows),
        "source_thread_pass": thread_counts,
        "thread_linked_listing_extraction": {
            "distinct_listing_ids": listing_rows["distinct_listing_ids"],
            "matched_listings": listing_rows["rows"],
            "unmatched_listing_ids": listing_rows["unmatched_listing_ids"],
            "unmatched_examples_hash": listing_rows["unmatched_examples_hash"],
            "non_negotiated_listings_omitted": True,
            "membership_index": "sqlite",
        },
        "quarantine": {"path": str(quarantine_path.resolve()), **quarantine_payload},
        "audited_full_release_evidence": _audited_full_release_evidence(
            full=full,
            limit_threads=limit_threads,
            full_preflight=full_preflight,
            official_source_contract=_official_source_contract(source_hashes, source_bytes),
            thread_checkpoint=thread_checkpoint,
            thread_counts=thread_counts,
            turn_rows=turn_rows,
            listing_rows=listing_rows,
        ),
        "lineage": {
            "raw_source_hashes": source_hashes,
            "split_manifest_hash": None,
            "normalization_manifest_hash": None,
            "normalization_manifest_payload_hash": None,
            "normalization_manifest_sha256_file": str(manifest_sha_path.resolve()),
        },
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    manifest_hash = _canonical_manifest_hash(manifest)
    manifest["lineage"]["normalization_manifest_hash"] = manifest_hash
    manifest["lineage"]["normalization_manifest_payload_hash"] = manifest_hash
    _write_atomic_json(manifest_path, manifest)
    replication_artifact = _run_replication_artifact(output, manifest_hash=manifest_hash)
    manifest["replication_checks"] = replication_artifact
    manifest["audited_full_release_evidence"]["replication_contract_artifact"] = {
        "path": replication_artifact["path"],
        "sha256": replication_artifact["sha256"],
    }
    manifest_hash = _canonical_manifest_hash(manifest)
    manifest["lineage"]["normalization_manifest_hash"] = manifest_hash
    manifest["lineage"]["normalization_manifest_payload_hash"] = manifest_hash
    _write_atomic_json(manifest_path, manifest)
    manifest_sha_path.write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
    return manifest


def full_normalization_status(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    checkpoints = output / "checkpoints"
    partition_checkpoint_dir = checkpoints / "partitions"
    checkpoint_files = sorted(partition_checkpoint_dir.glob("*.json")) if partition_checkpoint_dir.exists() else []
    partition_checkpoints = []
    for path in checkpoint_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            partition_checkpoints.append({"path": str(path.resolve()), "valid_json": False})
            continue
        partition_checkpoints.append(
            {
                "path": str(path.resolve()),
                "valid_json": True,
                "table": payload.get("table"),
                "partition_index": payload.get("partition", {}).get("partition_index"),
                "rows": payload.get("partition", {}).get("rows"),
                "sha256_verified": _partition_record_hash_matches(payload.get("partition", {})),
            }
        )
    if not manifest_path.exists():
        return {
            "schema_version": "nber_full_normalization_status.v1",
            "status": "incomplete",
            "output_dir": str(output.resolve()),
            "manifest_exists": False,
            "thread_checkpoint_exists": (checkpoints / "thread_pass.complete.json").exists(),
            "partition_checkpoints": partition_checkpoints,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition_integrity = _manifest_partition_integrity(manifest)
    evidence = verify_full_release_evidence(manifest)
    return {
        "schema_version": "nber_full_normalization_status.v1",
        "status": manifest.get("status", "unknown"),
        "output_dir": str(output.resolve()),
        "manifest_exists": True,
        "manifest_hash": sha256_file(manifest_path),
        "normalization_manifest_hash": manifest.get("lineage", {}).get("normalization_manifest_hash"),
        "source_files": manifest.get("source_files", {}),
        "summary": manifest.get("summary", {}),
        "partition_integrity": partition_integrity,
        "full_release_evidence": evidence,
        "thread_checkpoint_exists": (checkpoints / "thread_pass.complete.json").exists(),
        "partition_checkpoints": partition_checkpoints,
    }


def inspect_real_source_schema(raw_dir: str | Path) -> dict[str, Any]:
    raw = Path(raw_dir)
    lists_path = _find_source(raw, "anon_bo_lists.csv")
    threads_path = _find_source(raw, "anon_bo_threads.csv")
    return _validate_source_headers(lists_path, threads_path)


def _bucket_thread_rows(
    threads_path: Path,
    bucket_dir: Path,
    id_index_path: Path,
    *,
    bucket_count: int,
    limit_threads: int | None,
    quarantine: Quarantine,
) -> dict[str, Any]:
    accepted_rows = 0
    duplicate_rows = 0
    status_counts: Counter[str] = Counter()
    offer_type_counts: Counter[str] = Counter()
    index = _open_id_index(id_index_path, reset=True)
    distinct_threads = 0
    bucket_row_counts = [0 for _ in range(bucket_count)]
    bucket_handles = [(bucket_dir / f"bucket_{index:04d}.jsonl").open("w", encoding="utf-8", newline="\n") for index in range(bucket_count)]
    try:
        with _open_text(threads_path) as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                thread_id = row.get("anon_thread_id", "")
                listing_id = row.get("anon_item_id", "")
                if not thread_id or not listing_id or not row.get("anon_byr_id") or not row.get("anon_slr_id"):
                    quarantine.add("missing_required_thread_identifier", row, source_file=threads_path.name, line_number=line_number)
                    continue
                known_thread = _id_index_contains(index, "seen_threads", "thread_id", thread_id)
                if limit_threads is not None and not known_thread and distinct_threads >= limit_threads:
                    continue
                row_digest = _row_hash(row)
                if not _id_index_insert(index, "row_hashes", "row_hash", row_digest):
                    duplicate_rows += 1
                    continue
                if not known_thread:
                    _id_index_insert(index, "seen_threads", "thread_id", thread_id)
                    distinct_threads += 1
                status_counts[row.get("status_id", "")] += 1
                offer_type_counts[row.get("offr_type_id", "")] += 1
                if row.get("status_id", "") not in STATUS_MAP:
                    quarantine.add("unknown_status_id", row, source_file=threads_path.name, line_number=line_number)
                    continue
                if row.get("offr_type_id", "") not in OFFER_TYPE_MAP:
                    quarantine.add("unknown_offr_type_id", row, source_file=threads_path.name, line_number=line_number)
                    continue
                bucket = int(hashlib.sha256(thread_id.encode("utf-8")).hexdigest(), 16) % bucket_count
                bucket_handles[bucket].write(json.dumps(row, sort_keys=True) + "\n")
                bucket_row_counts[bucket] += 1
                _id_index_insert(index, "listing_ids", "listing_id", listing_id)
                _id_index_insert(index, "buyer_ids", "buyer_id", row.get("anon_byr_id", ""))
                _id_index_insert(index, "seller_ids", "seller_id", row.get("anon_slr_id", ""))
                accepted_rows += 1
                if accepted_rows % 10_000 == 0:
                    index.commit()
    finally:
        for handle in bucket_handles:
            handle.close()
        index.commit()
        id_index_stats = _id_index_checkpoint_stats(index)
        index.close()
    return {
        "source": str(threads_path.resolve()),
        "accepted_rows": accepted_rows,
        "distinct_threads": distinct_threads,
        "duplicate_full_rows_removed": duplicate_rows,
        "status_counts": dict(status_counts),
        "offer_type_counts": dict(offer_type_counts),
        "limit_threads": limit_threads,
        "bucket_manifest": _bucket_manifest(bucket_dir, bucket_count, row_counts=bucket_row_counts),
        "id_index_stats": id_index_stats,
    }


def _write_turn_partitions(
    bucket_dir: Path,
    table_dir: Path,
    *,
    checkpoint_dir: Path,
    signature: dict[str, Any],
    bucket_count: int,
    partition_rows: int,
    quarantine: Quarantine,
    resume: bool,
) -> dict[str, Any]:
    rows_out = []
    partitions = []
    total = 0
    part_index = 0
    stats = _new_turn_summary()
    for bucket in (bucket_dir / f"bucket_{index:04d}.jsonl" for index in range(bucket_count)):
        staging_path = bucket.with_suffix(".sqlite")
        if staging_path.exists():
            staging_path.unlink()
        conn = sqlite3.connect(staging_path)
        try:
            conn.execute(
                "CREATE TABLE rows (thread_id TEXT NOT NULL, sort_time TEXT NOT NULL, offer_type TEXT NOT NULL, amount TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            batch = []
            with bucket.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    batch.append(
                        (
                            str(row["anon_thread_id"]),
                            _parse_datetime_string(row.get("src_cre_date")) or str(row.get("src_cre_date", "")),
                            str(row.get("offr_type_id", "")),
                            str(row.get("offr_price", "")),
                            json.dumps(row, sort_keys=True),
                        )
                    )
                    if len(batch) >= 10_000:
                        conn.executemany("INSERT INTO rows VALUES (?, ?, ?, ?, ?)", batch)
                        batch = []
                if batch:
                    conn.executemany("INSERT INTO rows VALUES (?, ?, ?, ?, ?)", batch)
            conn.commit()
            for (thread_id,) in conn.execute("SELECT DISTINCT thread_id FROM rows ORDER BY thread_id"):
                cursor = conn.execute(
                    "SELECT payload FROM rows WHERE thread_id = ? ORDER BY sort_time, offer_type, amount",
                    (thread_id,),
                )
                for index, (payload,) in enumerate(cursor, start=1):
                    row = json.loads(payload)
                    try:
                        normalized = _normalize_thread_row(row, turn_index=index)
                    except Exception:
                        quarantine.add("thread_normalization_error", row, source_file=bucket.name, line_number=index)
                        continue
                    rows_out.append(normalized)
                    _update_turn_summary(stats, normalized)
                    if len(rows_out) >= partition_rows:
                        partitions.append(
                            _write_or_resume_partition(
                                table_dir,
                                "negotiation_turns",
                                "turns",
                                part_index,
                                rows_out,
                                checkpoint_dir=checkpoint_dir,
                                signature=signature,
                                resume=resume,
                            )
                        )
                        total += len(rows_out)
                        rows_out = []
                        part_index += 1
        finally:
            conn.close()
            if staging_path.exists():
                staging_path.unlink()
    if rows_out:
        partitions.append(
            _write_or_resume_partition(
                table_dir,
                "negotiation_turns",
                "turns",
                part_index,
                rows_out,
                checkpoint_dir=checkpoint_dir,
                signature=signature,
                resume=resume,
            )
        )
        total += len(rows_out)
    return {"rows": total, "partitions": partitions, "summary": _finalize_summary(stats)}


def _write_listing_partitions(
    lists_path: Path,
    id_index_path: Path,
    table_dir: Path,
    *,
    checkpoint_dir: Path,
    signature: dict[str, Any],
    partition_rows: int,
    quarantine: Quarantine,
    resume: bool,
) -> dict[str, Any]:
    rows_out = []
    partitions = []
    total = 0
    part_index = 0
    duplicate_listing_ids = 0
    index = _open_id_index(id_index_path, reset=False)
    stats = _open_listing_stats(table_dir / "_listing_stats.sqlite")
    try:
        with _open_text(lists_path) as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                listing_id = row.get("anon_item_id", "")
                if not _id_index_contains(index, "listing_ids", "listing_id", listing_id):
                    continue
                if not _stats_insert(stats, "listing_ids", "listing_id", listing_id):
                    duplicate_listing_ids += 1
                    quarantine.add("duplicate_listing_id", row, source_file=lists_path.name, line_number=line_number)
                    continue
                try:
                    normalized = _normalize_listing_row(row)
                except Exception:
                    quarantine.add("listing_normalization_error", row, source_file=lists_path.name, line_number=line_number)
                    continue
                rows_out.append(normalized)
                _update_listing_summary(stats, normalized)
                index.execute("UPDATE listing_ids SET matched = 1 WHERE listing_id = ?", (listing_id,))
                if len(rows_out) >= partition_rows:
                    partitions.append(
                        _write_or_resume_partition(
                            table_dir,
                            "listings",
                            "listings",
                            part_index,
                            rows_out,
                            checkpoint_dir=checkpoint_dir,
                            signature=signature,
                            resume=resume,
                        )
                    )
                    total += len(rows_out)
                    rows_out = []
                    part_index += 1
        if rows_out:
            partitions.append(
                _write_or_resume_partition(
                    table_dir,
                    "listings",
                    "listings",
                    part_index,
                    rows_out,
                    checkpoint_dir=checkpoint_dir,
                    signature=signature,
                    resume=resume,
                )
            )
            total += len(rows_out)
        index.commit()
        id_stats = _listing_id_stats(index)
        listing_summary = _listing_stats_summary(stats)
    finally:
        stats.close()
        stats_path = table_dir / "_listing_stats.sqlite"
        if stats_path.exists():
            stats_path.unlink()
        index.close()
    return {
        "rows": total,
        "partitions": partitions,
        "summary": listing_summary,
        "duplicate_listing_ids_quarantined": duplicate_listing_ids,
        **id_stats,
    }


def _normalize_thread_row(row: dict[str, str], *, turn_index: int) -> dict[str, Any]:
    offer_type = OFFER_TYPE_MAP[row["offr_type_id"]]
    raw_hash = _row_hash(row)
    return {
        "source_row_id": raw_hash,
        "raw_source_row_hash": raw_hash,
        "thread_id": row["anon_thread_id"],
        "listing_id": row["anon_item_id"],
        "buyer_id": row["anon_byr_id"],
        "seller_id": row["anon_slr_id"],
        "turn_index": turn_index,
        "actor": offer_type["actor"],
        "action": offer_type["action"],
        "amount": _float_or_none(row.get("offr_price")),
        "status": STATUS_MAP[row["status_id"]],
        "status_id": _int_or_none(row.get("status_id")),
        "event_date": _parse_date_string(row.get("src_cre_dt")),
        "event_time": _parse_datetime_string(row.get("src_cre_date")),
        "response_time": _parse_datetime_string(row.get("response_time")),
        "seller_feedback_score_at_offer": _int_or_none(row.get("fdbk_score_src")),
        "seller_feedback_positive_at_offer": _float_or_none(row.get("fdbk_pstv_src")),
        "seller_best_offer_thread_history": _int_or_none(row.get("slr_hist")),
        "buyer_best_offer_thread_history": _int_or_none(row.get("byr_hist")),
        "has_message": _bool_or_none(row.get("any_mssg")),
        "buyer_us": _bool_or_none(row.get("byr_us")),
        "transformation_version": REAL_TRANSFORMATION_VERSION,
    }


def _normalize_listing_row(row: dict[str, str]) -> dict[str, Any]:
    product_id = None if row.get("anon_product_id") in {"", "547957"} else row.get("anon_product_id")
    return {
        "source_row_id": row["anon_item_id"],
        "raw_source_row_hash": _row_hash(row),
        "listing_id": row["anon_item_id"],
        "seller_id": row["anon_slr_id"],
        "buyer_id_if_sold": row.get("anon_buyer_id") or None,
        "title_code": row.get("anon_title_code") or None,
        "product_id": product_id,
        "category": row.get("anon_leaf_categ_id") or None,
        "meta_category": row.get("meta_categ_id") or None,
        "condition": row.get("item_cndtn_id") or None,
        "listing_price": _float_or_none(row.get("start_price_usd")),
        "reference_price": None,
        "reference_count": None,
        "reference_price_unavailable_reason": "ref_price4 is excluded from predictor-facing features until its as-of semantics are proven.",
        "excluded_reference_price_ref_price4": _float_or_none(row.get("ref_price4")),
        "excluded_reference_count4": _int_or_none(row.get("count4")),
        "start_time": _parse_date_string(row.get("auct_start_dt")),
        "end_time": _parse_date_string(row.get("auct_end_dt")),
        "final_sale_price": _float_or_none(row.get("item_price")),
        "sold_by_best_offer": _bool_or_none(row.get("bo_ck_yn")),
        "photo_count": _int_or_none(row.get("photo_count")),
        "view_count": _int_or_none(row.get("view_item_count")),
        "watcher_count": _int_or_none(row.get("wtchr_count")),
        "auto_decline_price": _float_or_none(row.get("decline_price")),
        "auto_accept_price": _float_or_none(row.get("accept_price")),
        "seller_us": _bool_or_none(row.get("slr_us")),
        "buyer_us_if_sold": _bool_or_none(row.get("buyer_us")),
        "protected_outcome_fields_present": True,
        "transformation_version": REAL_TRANSFORMATION_VERSION,
    }


def _write_partition(table_dir: Path, stem: str, index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    partition_dir = table_dir / f"partition={index:05d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    if _pyarrow_available():
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = partition_dir / f"{stem}_{index:05d}.parquet"
        table = pa.Table.from_pylist(rows)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        pq.read_metadata(tmp)
        os.replace(tmp, path)
    else:
        path = partition_dir / f"{stem}_{index:05d}.jsonl"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=partition_dir, newline="\n") as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    partition = {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256_file(path), "partition_index": index}
    if not _partition_record_hash_matches(partition):
        raise NberRealNormalizeError(f"Partition hash verification failed for {path}")
    return partition


def _write_or_resume_partition(
    table_dir: Path,
    table_name: str,
    stem: str,
    index: int,
    rows: list[dict[str, Any]],
    *,
    checkpoint_dir: Path,
    signature: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    checkpoint_path = _partition_checkpoint_path(checkpoint_dir, table_name, index)
    if resume:
        checkpoint = _load_valid_partition_checkpoint(
            checkpoint_path,
            signature=signature,
            table_name=table_name,
            partition_index=index,
            expected_rows=len(rows),
        )
        if checkpoint is not None:
            return dict(checkpoint["partition"])
    partition = _write_partition(table_dir, stem, index, rows)
    _write_atomic_json(
        checkpoint_path,
        {
            "schema_version": "nber_partition_checkpoint.v1",
            "signature": signature,
            "table": table_name,
            "partition": partition,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )
    checkpoint = _load_valid_partition_checkpoint(
        checkpoint_path,
        signature=signature,
        table_name=table_name,
        partition_index=index,
        expected_rows=len(rows),
    )
    if checkpoint is None:
        raise NberRealNormalizeError(f"Partition checkpoint verification failed for {checkpoint_path}")
    return partition


def _partition_checkpoint_path(checkpoint_dir: Path, table_name: str, index: int) -> Path:
    return checkpoint_dir / f"{table_name}_{index:05d}.json"


def _load_valid_partition_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    table_name: str,
    partition_index: int,
    expected_rows: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("signature") != signature:
        return None
    if payload.get("table") != table_name:
        return None
    partition = payload.get("partition", {})
    if partition.get("partition_index") != partition_index:
        return None
    if partition.get("rows") != expected_rows:
        return None
    if not _partition_record_hash_matches(partition):
        return None
    return payload


def _partition_record_hash_matches(partition: dict[str, Any]) -> bool:
    path_text = str(partition.get("path", ""))
    expected_hash = partition.get("sha256")
    if not path_text or not expected_hash:
        return False
    path = Path(path_text)
    return path.exists() and sha256_file(path) == expected_hash


def _validate_source_headers(lists_path: Path, threads_path: Path) -> dict[str, Any]:
    listing_header = _read_header(lists_path)
    thread_header = _read_header(threads_path)
    return validate_real_headers(listings=listing_header, threads=thread_header)


def _new_turn_summary() -> dict[str, Any]:
    return {
        "status_counts": Counter(),
        "offer_type_counts": Counter(),
        "event_time_min": None,
        "event_time_max": None,
        "response_time_min": None,
        "response_time_max": None,
    }


def _update_turn_summary(summary: dict[str, Any], row: dict[str, Any]) -> None:
    status = row.get("status")
    action = row.get("action")
    if status is not None:
        summary["status_counts"][str(status)] += 1
    if action is not None:
        summary["offer_type_counts"][str(action)] += 1
    _summary_range(summary, "event_time", row.get("event_time"))
    _summary_range(summary, "response_time", row.get("response_time"))


def _summary_range(summary: dict[str, Any], prefix: str, value: Any) -> None:
    if value in {None, ""}:
        return
    text = str(value)
    min_key = f"{prefix}_min"
    max_key = f"{prefix}_max"
    if summary.get(min_key) is None or text < summary[min_key]:
        summary[min_key] = text
    if summary.get(max_key) is None or text > summary[max_key]:
        summary[max_key] = text


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = {}
    for key, value in summary.items():
        if isinstance(value, Counter):
            finalized[key] = dict(sorted(value.items()))
        else:
            finalized[key] = value
    return finalized


def _open_listing_stats(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE listing_ids (listing_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE seller_ids (seller_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE buyer_ids (buyer_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE category_counts (category TEXT PRIMARY KEY, rows INTEGER NOT NULL)")
    conn.execute("CREATE TABLE ranges (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn


def _update_listing_summary(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    _stats_insert(conn, "seller_ids", "seller_id", row.get("seller_id"))
    _stats_insert(conn, "buyer_ids", "buyer_id", row.get("buyer_id_if_sold"))
    category = row.get("category")
    if category not in {None, ""}:
        conn.execute("INSERT OR IGNORE INTO category_counts (category, rows) VALUES (?, 0)", (str(category),))
        conn.execute("UPDATE category_counts SET rows = rows + 1 WHERE category = ?", (str(category),))
    _stats_range(conn, "listing_start_min", row.get("start_time"), minimum=True)
    _stats_range(conn, "listing_start_max", row.get("start_time"), minimum=False)
    _stats_range(conn, "listing_end_min", row.get("end_time"), minimum=True)
    _stats_range(conn, "listing_end_max", row.get("end_time"), minimum=False)


def _stats_insert(conn: sqlite3.Connection, table: str, column: str, value: Any) -> bool:
    _validate_stats_table_column(table, column)
    if value in {None, ""}:
        return False
    cursor = conn.execute(f"INSERT OR IGNORE INTO {table} ({column}) VALUES (?)", (str(value),))
    return cursor.rowcount > 0


def _stats_range(conn: sqlite3.Connection, key: str, value: Any, *, minimum: bool) -> None:
    if value in {None, ""}:
        return
    text = str(value)
    existing = conn.execute("SELECT value FROM ranges WHERE key = ?", (key,)).fetchone()
    if existing is None:
        conn.execute("INSERT INTO ranges (key, value) VALUES (?, ?)", (key, text))
        return
    old = str(existing[0])
    if (minimum and text < old) or ((not minimum) and text > old):
        conn.execute("UPDATE ranges SET value = ? WHERE key = ?", (text, key))


def _listing_stats_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.commit()
    ranges = {key: value for key, value in conn.execute("SELECT key, value FROM ranges ORDER BY key")}
    top_categories = [
        {"category": category, "rows": rows}
        for category, rows in conn.execute("SELECT category, rows FROM category_counts ORDER BY rows DESC, category LIMIT 25")
    ]
    return {
        "distinct_listings": int(conn.execute("SELECT COUNT(*) FROM listing_ids").fetchone()[0]),
        "distinct_sellers": int(conn.execute("SELECT COUNT(*) FROM seller_ids").fetchone()[0]),
        "distinct_buyers_if_sold": int(conn.execute("SELECT COUNT(*) FROM buyer_ids").fetchone()[0]),
        "distinct_categories": int(conn.execute("SELECT COUNT(*) FROM category_counts").fetchone()[0]),
        "top_categories": top_categories,
        "date_ranges": ranges,
    }


def _validate_stats_table_column(table: str, column: str) -> None:
    allowed = {"listing_ids": "listing_id", "seller_ids": "seller_id", "buyer_ids": "buyer_id"}
    if allowed.get(table) != column:
        raise NberRealNormalizeError(f"Invalid internal stats reference {table}.{column}")


def _normalization_summary(*, thread_counts: dict[str, Any], turn_rows: dict[str, Any], listing_rows: dict[str, Any]) -> dict[str, Any]:
    index_stats = thread_counts.get("id_index_stats", {})
    return {
        "schema_version": "nber_real_normalization_summary.v1",
        "row_counts": {
            "negotiation_turns": turn_rows.get("rows", 0),
            "listings": listing_rows.get("rows", 0),
        },
        "date_ranges": {
            "event_time": {
                "min": turn_rows.get("summary", {}).get("event_time_min"),
                "max": turn_rows.get("summary", {}).get("event_time_max"),
            },
            "response_time": {
                "min": turn_rows.get("summary", {}).get("response_time_min"),
                "max": turn_rows.get("summary", {}).get("response_time_max"),
            },
            "listing_start": {
                "min": listing_rows.get("summary", {}).get("date_ranges", {}).get("listing_start_min"),
                "max": listing_rows.get("summary", {}).get("date_ranges", {}).get("listing_start_max"),
            },
            "listing_end": {
                "min": listing_rows.get("summary", {}).get("date_ranges", {}).get("listing_end_min"),
                "max": listing_rows.get("summary", {}).get("date_ranges", {}).get("listing_end_max"),
            },
        },
        "categories": {
            "distinct": listing_rows.get("summary", {}).get("distinct_categories", 0),
            "top_counts": listing_rows.get("summary", {}).get("top_categories", []),
        },
        "sellers": {
            "distinct_in_threads": index_stats.get("seller_ids"),
            "distinct_in_matched_listings": listing_rows.get("summary", {}).get("distinct_sellers", 0),
        },
        "buyers": {
            "distinct_in_threads": index_stats.get("buyer_ids"),
            "distinct_if_sold_in_matched_listings": listing_rows.get("summary", {}).get("distinct_buyers_if_sold", 0),
        },
        "listings": {
            "referenced_distinct": listing_rows.get("distinct_listing_ids"),
            "matched": listing_rows.get("matched_listing_ids"),
            "unmatched": listing_rows.get("unmatched_listing_ids"),
        },
        "threads": {
            "distinct": thread_counts.get("distinct_threads"),
        },
        "turns": {
            "rows": turn_rows.get("rows", 0),
            "status_counts": thread_counts.get("status_counts", {}),
            "offer_type_counts": thread_counts.get("offer_type_counts", {}),
        },
        "duplicates": {
            "duplicate_thread_rows_removed": thread_counts.get("duplicate_full_rows_removed", 0),
            "duplicate_listing_ids_quarantined": listing_rows.get("duplicate_listing_ids_quarantined", 0),
        },
    }


def _normalized_columns(partitions: list[dict[str, Any]]) -> list[str]:
    for partition in partitions:
        path = Path(partition["path"])
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return sorted(json.loads(line))
    return []


def _run_replication_artifact(output: Path, *, manifest_hash: str) -> dict[str, Any]:
    artifact_path = output / "replication_check.json"
    try:
        from behavior_lab.datasets.nber_best_offer.replication import replication_check

        payload = replication_check(output)
    except Exception as exc:
        payload = {
            "schema_version": "nber_replication_check.v1",
            "passed": False,
            "full_replication_passed": False,
            "fatal_failures": [],
            "fatal_unevaluated": ["replication_check_error"],
            "error": str(exc),
        }
    payload["normalization_manifest_hash"] = manifest_hash
    _write_atomic_json(artifact_path, payload)
    return {
        "schema_version": "nber_replication_artifact.v1",
        "path": str(artifact_path.resolve()),
        "sha256": sha256_file(artifact_path),
        "passed": bool(payload.get("passed")),
        "full_replication_passed": bool(payload.get("full_replication_passed")),
        "fatal_failures": len(payload.get("fatal_failures", [])),
        "fatal_unevaluated": len(payload.get("fatal_unevaluated", [])),
    }


def _read_header(path: Path) -> list[str]:
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        return next(reader)


def _find_source(root: Path, name: str) -> Path:
    candidates = [root / name, root / f"{name}.gz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise NberRealNormalizeError(f"Missing {name} or {name}.gz in {root}")


def _full_run_preflight(
    *,
    raw: Path,
    output: Path,
    lists_path: Path,
    threads_path: Path,
    source_hashes: dict[str, str],
    source_bytes: dict[str, int],
    full: bool,
    limit_threads: int | None,
    bucket_count: int,
    partition_rows: int,
) -> dict[str, Any]:
    official_contract = _official_source_contract(source_hashes, source_bytes)
    checks = {
        "mode_valid": (full and limit_threads is None) or ((not full) and limit_threads is not None),
        "bucket_count_positive": bucket_count > 0,
        "partition_rows_positive": partition_rows > 0,
        "raw_dir_exists": raw.exists() and raw.is_dir(),
        "output_dir_exists": output.exists() and output.is_dir(),
        "listing_source_readable": lists_path.exists() and lists_path.stat().st_size > 0,
        "thread_source_readable": threads_path.exists() and threads_path.stat().st_size > 0,
        "thread_limited_when_bounded": full or (limit_threads is not None and limit_threads > 0),
        "uses_sqlite_membership_index": True,
        "uses_deterministic_thread_buckets": True,
        "avoids_in_memory_listing_id_set": True,
    }
    disk = _disk_preflight(output, source_bytes=source_bytes, full=full)
    checks["disk_preflight_passed"] = bool(disk["passed"])
    if full:
        checks["official_source_contract_checked"] = True
    return {
        "schema_version": "nber_full_run_preflight.v1",
        "scope": "full_unbounded_source_scan" if full else "bounded_thread_limit",
        "passed": all(checks.values()),
        "checks": checks,
        "disk": disk,
        "official_source_contract": official_contract,
        "checkpoint_strategy": {
            "thread_pass_checkpoint": "checkpoints/thread_pass.complete.json",
            "membership_index": "_tmp/thread_listing_ids.sqlite",
            "thread_buckets": "_tmp/thread_buckets/bucket_####.jsonl",
            "resume_rule": "reuse only when source hashes, headers, mapping hash, command args, bucket hashes, and SQLite index content hashes match",
        },
    }


def _disk_preflight(output: Path, *, source_bytes: dict[str, int], full: bool) -> dict[str, Any]:
    usage = shutil.disk_usage(output.resolve().anchor or output.resolve())
    compressed_bytes = int(sum(source_bytes.values()))
    estimated_required = compressed_bytes * (4 if full else 1)
    # Keep fixture tests practical while still recording a real disk check.
    required_free = max(estimated_required, 64 * 1024 * 1024)
    return {
        "passed": usage.free >= required_free,
        "free_bytes": usage.free,
        "required_free_bytes": required_free,
        "compressed_source_bytes": compressed_bytes,
        "estimation_rule": "full requires 4x compressed source bytes for deterministic buckets, SQLite index, partition output, and temporary files; bounded smoke requires at least 64MiB",
    }


def _official_source_contract(source_hashes: dict[str, str], source_bytes: dict[str, int]) -> dict[str, Any]:
    files = {}
    for logical_name, expected in OFFICIAL_FULL_SOURCE_EXPECTATIONS.items():
        actual_hash = source_hashes.get(logical_name)
        actual_bytes = source_bytes.get(logical_name)
        files[logical_name] = {
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_hash,
            "sha256_matches": actual_hash == expected["sha256"],
            "expected_bytes": expected["bytes"],
            "actual_bytes": actual_bytes,
            "bytes_match": actual_bytes == expected["bytes"],
        }
    return {
        "schema_version": "nber_official_source_contract.v1",
        "matches_expected_official_sources": all(item["sha256_matches"] and item["bytes_match"] for item in files.values()),
        "files": files,
        "research_only": True,
        "production_export_allowed": False,
    }


def _audited_full_release_evidence(
    *,
    full: bool,
    limit_threads: int | None,
    full_preflight: dict[str, Any],
    official_source_contract: dict[str, Any],
    thread_checkpoint: Path,
    thread_counts: dict[str, Any],
    turn_rows: dict[str, Any],
    listing_rows: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_validated = thread_checkpoint.exists() and bool(thread_counts.get("bucket_manifest", {}).get("valid")) and bool(thread_counts.get("id_index_stats"))
    partition_integrity = _partition_output_integrity(
        turn_rows=turn_rows,
        listing_rows=listing_rows,
        thread_counts=thread_counts,
    )
    partition_hashes_verified = bool(partition_integrity["passed"])
    streaming_full_run_passed = bool(
        full
        and limit_threads is None
        and full_preflight.get("passed")
        and checkpoint_validated
        and partition_hashes_verified
        and listing_rows.get("unmatched_listing_ids") == 0
    )
    official_sources_matched = bool(official_source_contract.get("matches_expected_official_sources"))
    return {
        "schema_version": "nber_full_release_evidence_gate.v1",
        "passed": False,
        "streaming_full_run_passed": streaming_full_run_passed,
        "official_sources_matched": official_sources_matched,
        "full_run_checkpoint_validated": checkpoint_validated,
        "partition_hashes_verified": partition_hashes_verified,
        "partition_integrity": partition_integrity,
        "replication_contract_passed": False,
        "replication_contract_artifact": None,
        "independent_audit_passed": False,
        "independent_audit_artifact": None,
        "reason": "Normalization can prove the streaming full-run path, but replication and independent audit must be recorded by separate Wave 2 checks before this becomes full-release benchmark evidence.",
    }


def _partition_output_integrity(*, turn_rows: dict[str, Any], listing_rows: dict[str, Any], thread_counts: dict[str, Any]) -> dict[str, Any]:
    manifest_like = {
        "tables": {
            "negotiation_turns": {"rows": turn_rows.get("rows", 0), "partitions": turn_rows.get("partitions", [])},
            "listings": {"rows": listing_rows.get("rows", 0), "partitions": listing_rows.get("partitions", [])},
        },
        "source_thread_pass": {"accepted_rows": thread_counts.get("accepted_rows")},
        "thread_linked_listing_extraction": {
            "matched_listings": listing_rows.get("matched_listing_ids", listing_rows.get("rows", 0)),
            "unmatched_listing_ids": listing_rows.get("unmatched_listing_ids"),
        },
    }
    return _manifest_partition_integrity(manifest_like)


def _manifest_matches_current(
    manifest: dict[str, Any],
    *,
    args_signature: dict[str, Any],
    source_hashes: dict[str, str],
    header_report: dict[str, Any],
) -> bool:
    if manifest.get("status") != "complete":
        return False
    if manifest.get("command_args") != args_signature:
        return False
    if manifest.get("transformation_version") != REAL_TRANSFORMATION_VERSION:
        return False
    if manifest.get("mapping_manifest_hash") != mapping_hash():
        return False
    if manifest.get("header_validation") != header_report:
        return False
    source_files = manifest.get("source_files", {})
    for logical_name, current_hash in source_hashes.items():
        if source_files.get(logical_name, {}).get("sha256") != current_hash:
            return False
    if manifest.get("lineage", {}).get("raw_source_hashes") != source_hashes:
        return False
    if not _manifest_partition_integrity(manifest)["passed"]:
        return False
    return True


def verify_full_release_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = manifest.get("audited_full_release_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    official_contract = manifest.get("official_source_contract", {})
    if not isinstance(official_contract, dict):
        official_contract = {}
    command_args = manifest.get("command_args", {})
    if not isinstance(command_args, dict):
        command_args = {}
    preflight = manifest.get("full_release_preflight", {})
    if not isinstance(preflight, dict):
        preflight = {}
    partition_integrity = _manifest_partition_integrity(manifest)
    source_file_integrity = _source_files_verify_now(manifest)
    replication_artifact = _replication_artifact_verification(manifest, evidence)
    independent_audit_artifact = _independent_audit_artifact_verification(manifest, evidence)
    checks = {
        "command_full_unbounded": command_args.get("full") is True and command_args.get("limit_threads") is None,
        "preflight_passed": preflight.get("passed") is True,
        "official_contract_matches": official_contract.get("matches_expected_official_sources") is True,
        "source_files_match_official_contract": _source_files_match_official_contract(manifest),
        "source_files_verified_now": source_file_integrity["passed"],
        "streaming_full_run_passed": evidence.get("streaming_full_run_passed") is True,
        "official_sources_matched": evidence.get("official_sources_matched") is True,
        "full_run_checkpoint_validated": evidence.get("full_run_checkpoint_validated") is True,
        "partition_hashes_verified": evidence.get("partition_hashes_verified") is True,
        "partition_integrity_verified_now": partition_integrity["passed"],
        "replication_contract_passed": evidence.get("replication_contract_passed") is True,
        "replication_artifact_verified": replication_artifact["passed"],
        "independent_audit_passed": evidence.get("independent_audit_passed") is True,
        "independent_audit_artifact_verified": independent_audit_artifact["passed"],
        "declared_gate_passed": evidence.get("passed") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "nber_full_release_evidence_verification.v1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "source_file_integrity": source_file_integrity,
        "partition_integrity": partition_integrity,
        "replication_artifact": replication_artifact,
        "independent_audit_artifact": independent_audit_artifact,
    }


def _source_files_match_official_contract(manifest: dict[str, Any]) -> bool:
    source_files = manifest.get("source_files", {})
    contract_files = manifest.get("official_source_contract", {}).get("files", {})
    for logical_name, expected in OFFICIAL_FULL_SOURCE_EXPECTATIONS.items():
        source_record = source_files.get(logical_name, {})
        contract_record = contract_files.get(logical_name, {})
        if source_record.get("sha256") != expected["sha256"]:
            return False
        if source_record.get("bytes") != expected["bytes"]:
            return False
        if contract_record.get("actual_sha256") != expected["sha256"] or contract_record.get("sha256_matches") is not True:
            return False
        if contract_record.get("actual_bytes") != expected["bytes"] or contract_record.get("bytes_match") is not True:
            return False
    return True


def _source_files_verify_now(manifest: dict[str, Any]) -> dict[str, Any]:
    source_files = manifest.get("source_files", {})
    files: dict[str, Any] = {}
    failures: list[str] = []
    for logical_name, expected in OFFICIAL_FULL_SOURCE_EXPECTATIONS.items():
        source_record = source_files.get(logical_name, {})
        path_text = str(source_record.get("path", ""))
        file_report: dict[str, Any] = {
            "path": path_text or None,
            "exists": False,
            "bytes_match": False,
            "sha256_match": False,
        }
        if not path_text:
            failures.append(f"{logical_name}:missing_path")
            files[logical_name] = file_report
            continue
        path = Path(path_text)
        if not path.exists():
            failures.append(f"{logical_name}:missing_file")
            files[logical_name] = file_report
            continue
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        file_report.update(
            {
                "exists": True,
                "actual_bytes": actual_bytes,
                "expected_bytes": expected["bytes"],
                "bytes_match": actual_bytes == expected["bytes"],
                "actual_sha256": actual_sha,
                "expected_sha256": expected["sha256"],
                "sha256_match": actual_sha == expected["sha256"],
            }
        )
        if actual_bytes != expected["bytes"]:
            failures.append(f"{logical_name}:bytes_mismatch")
        if actual_sha != expected["sha256"]:
            failures.append(f"{logical_name}:sha256_mismatch")
        files[logical_name] = file_report
    return {"schema_version": "nber_source_file_integrity.v1", "passed": not failures, "files": files, "failures": failures}


def _replication_artifact_verification(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    return _json_artifact_verification(
        manifest,
        evidence.get("replication_contract_artifact"),
        artifact_name="replication_contract_artifact",
        required_true_fields=["passed", "full_replication_passed"],
        required_zero_fields=["fatal_failures", "fatal_unevaluated"],
    )


def _independent_audit_artifact_verification(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    report = _json_artifact_verification(
        manifest,
        evidence.get("independent_audit_artifact"),
        artifact_name="independent_audit_artifact",
        required_true_fields=["passed", "independent_audit_passed"],
        required_zero_fields=[],
    )
    payload = report.get("payload", {})
    if not isinstance(payload, dict) or payload.get("scope") != "full_release":
        report["passed"] = False
        report.setdefault("failures", []).append("scope_not_full_release")
    report.pop("payload", None)
    return report


def _json_artifact_verification(
    manifest: dict[str, Any],
    artifact: Any,
    *,
    artifact_name: str,
    required_true_fields: list[str],
    required_zero_fields: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(artifact, dict):
        return {"schema_version": "nber_evidence_artifact_verification.v1", "artifact_name": artifact_name, "passed": False, "failures": [f"{artifact_name}:missing_or_not_object"]}
    path_text = str(artifact.get("path", ""))
    expected_sha = str(artifact.get("sha256", ""))
    if not path_text:
        failures.append("missing_path")
        return {"schema_version": "nber_evidence_artifact_verification.v1", "artifact_name": artifact_name, "passed": False, "failures": failures}
    path = Path(path_text)
    if not path.exists():
        failures.append("missing_file")
        return {"schema_version": "nber_evidence_artifact_verification.v1", "artifact_name": artifact_name, "path": str(path), "passed": False, "failures": failures}
    actual_sha = sha256_file(path)
    if not expected_sha:
        failures.append("missing_sha256")
    elif actual_sha != expected_sha:
        failures.append("sha256_mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures.append("invalid_json")
        payload = {}
    if not isinstance(payload, dict):
        failures.append("payload_not_object")
        payload = {}
    for field in required_true_fields:
        if payload.get(field) is not True:
            failures.append(f"{field}_not_true")
    for field in required_zero_fields:
        if payload.get(field) != 0:
            failures.append(f"{field}_not_zero")
    if not _artifact_binds_to_manifest(manifest, payload):
        failures.append("artifact_not_bound_to_manifest")
    return {
        "schema_version": "nber_evidence_artifact_verification.v1",
        "artifact_name": artifact_name,
        "path": str(path),
        "sha256": actual_sha,
        "passed": not failures,
        "failures": failures,
        "payload": payload,
    }


def _artifact_binds_to_manifest(manifest: dict[str, Any], payload: dict[str, Any]) -> bool:
    manifest_hash = manifest.get("lineage", {}).get("normalization_manifest_hash")
    if manifest_hash and payload.get("normalization_manifest_hash") == manifest_hash:
        return True
    return False


def _manifest_partition_integrity(manifest: dict[str, Any]) -> dict[str, Any]:
    tables = manifest.get("tables", {})
    checks: dict[str, Any] = {}
    failures: list[str] = []
    seen_paths: set[str] = set()
    for table_name, table in tables.items():
        partitions = list(table.get("partitions", []))
        table_rows = int(table.get("rows", 0))
        partition_rows = 0
        table_failures: list[str] = []
        for partition in partitions:
            path_text = str(partition.get("path", ""))
            if not path_text:
                table_failures.append("missing_partition_path")
                continue
            if path_text in seen_paths:
                table_failures.append(f"duplicate_partition_path:{path_text}")
            seen_paths.add(path_text)
            path = Path(path_text)
            if not path.exists():
                table_failures.append(f"missing_partition:{path.name}")
                continue
            expected_hash = partition.get("sha256")
            if not expected_hash or sha256_file(path) != expected_hash:
                table_failures.append(f"partition_hash_mismatch:{path.name}")
            try:
                partition_rows += int(partition.get("rows", 0))
            except (TypeError, ValueError):
                table_failures.append(f"invalid_partition_rows:{path.name}")
        if partition_rows != table_rows:
            table_failures.append(f"partition_row_sum_mismatch:{partition_rows}!={table_rows}")
        checks[table_name] = {
            "passed": not table_failures,
            "table_rows": table_rows,
            "partition_row_sum": partition_rows,
            "partitions": len(partitions),
            "failures": table_failures,
        }
        failures.extend(f"{table_name}:{failure}" for failure in table_failures)
    source_pass = manifest.get("source_thread_pass", {})
    turn_rows = tables.get("negotiation_turns", {}).get("rows")
    if turn_rows is not None and source_pass.get("accepted_rows") is not None and int(turn_rows) != int(source_pass["accepted_rows"]):
        failures.append("negotiation_turns:accepted_rows_mismatch")
    extraction = manifest.get("thread_linked_listing_extraction", {})
    listing_rows = tables.get("listings", {}).get("rows")
    if listing_rows is not None and extraction.get("matched_listings") is not None and int(listing_rows) != int(extraction["matched_listings"]):
        failures.append("listings:matched_listings_mismatch")
    if extraction.get("unmatched_listing_ids") not in {None, 0}:
        failures.append("listings:unmatched_listing_ids_nonzero")
    return {
        "schema_version": "nber_partition_integrity.v1",
        "passed": not failures and bool(tables),
        "tables": checks,
        "failures": failures,
    }


def _thread_checkpoint_signature(
    *,
    args_signature: dict[str, Any],
    source_hashes: dict[str, str],
    header_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "nber_real_thread_pass_signature.v1",
        "command_args": args_signature,
        "source_hashes": source_hashes,
        "header_validation": header_report,
        "mapping_manifest_hash": mapping_hash(),
    }


def _partition_checkpoint_signature(
    *,
    args_signature: dict[str, Any],
    source_hashes: dict[str, str],
    header_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "nber_real_partition_signature.v1",
        "command_args": args_signature,
        "source_hashes": source_hashes,
        "header_validation": header_report,
        "mapping_manifest_hash": mapping_hash(),
    }


def _bucket_manifest(bucket_dir: Path, bucket_count: int, *, row_counts: list[int] | None = None) -> dict[str, Any]:
    expected_names = {f"bucket_{index:04d}.jsonl" for index in range(bucket_count)}
    actual_names = {path.name for path in bucket_dir.glob("bucket_*.jsonl")}
    if actual_names != expected_names:
        return {"valid": False, "bucket_count": bucket_count, "buckets": [], "unexpected_buckets": sorted(actual_names - expected_names), "missing_buckets": sorted(expected_names - actual_names)}
    buckets = []
    total_rows = 0
    for index in range(bucket_count):
        path = bucket_dir / f"bucket_{index:04d}.jsonl"
        if not path.exists():
            return {"valid": False, "bucket_count": bucket_count, "buckets": []}
        rows = row_counts[index] if row_counts is not None else _line_count(path)
        total_rows += rows
        buckets.append(
            {
                "name": path.name,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"valid": True, "bucket_count": bucket_count, "total_rows": total_rows, "buckets": buckets}


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_valid_thread_checkpoint(
    path: Path,
    signature: dict[str, Any],
    *,
    bucket_dir: Path,
    id_index_path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("signature") != signature:
        return None
    if not bucket_dir.exists() or not any(bucket_dir.glob("bucket_*.jsonl")):
        return None
    if not _id_index_is_valid(id_index_path):
        return None
    if "thread_counts" not in payload:
        return None
    thread_counts = payload["thread_counts"]
    expected_bucket_manifest = thread_counts.get("bucket_manifest")
    if not expected_bucket_manifest:
        return None
    expected_bucket_count = payload.get("signature", {}).get("command_args", {}).get("bucket_count")
    if not isinstance(expected_bucket_count, int):
        return None
    if _bucket_manifest(bucket_dir, expected_bucket_count) != expected_bucket_manifest:
        return None
    expected_index_stats = thread_counts.get("id_index_stats")
    if not expected_index_stats:
        return None
    conn = _open_id_index(id_index_path, reset=False)
    try:
        if _id_index_checkpoint_stats(conn) != expected_index_stats:
            return None
    finally:
        conn.close()
    return payload


def _open_id_index(path: Path, *, reset: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()
    if not reset and not path.exists():
        raise NberRealNormalizeError(f"Missing thread/listing index at {path}")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS seen_threads (thread_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS row_hashes (row_hash TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS listing_ids (listing_id TEXT PRIMARY KEY, matched INTEGER NOT NULL DEFAULT 0)")
    conn.execute("CREATE TABLE IF NOT EXISTS buyer_ids (buyer_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS seller_ids (seller_id TEXT PRIMARY KEY)")
    return conn


def _id_index_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        conn = _open_id_index(path, reset=False)
        try:
            conn.execute("SELECT COUNT(*) FROM listing_ids").fetchone()
            conn.execute("SELECT COUNT(*) FROM seen_threads").fetchone()
            conn.execute("SELECT COUNT(*) FROM row_hashes").fetchone()
            conn.execute("SELECT COUNT(*) FROM buyer_ids").fetchone()
            conn.execute("SELECT COUNT(*) FROM seller_ids").fetchone()
            return True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def _id_index_checkpoint_stats(conn: sqlite3.Connection) -> dict[str, int | str]:
    return {
        "seen_threads": int(conn.execute("SELECT COUNT(*) FROM seen_threads").fetchone()[0]),
        "row_hashes": int(conn.execute("SELECT COUNT(*) FROM row_hashes").fetchone()[0]),
        "listing_ids": int(conn.execute("SELECT COUNT(*) FROM listing_ids").fetchone()[0]),
        "buyer_ids": int(conn.execute("SELECT COUNT(*) FROM buyer_ids").fetchone()[0]),
        "seller_ids": int(conn.execute("SELECT COUNT(*) FROM seller_ids").fetchone()[0]),
        "seen_threads_hash": _sqlite_column_hash(conn, "seen_threads", "thread_id"),
        "row_hashes_hash": _sqlite_column_hash(conn, "row_hashes", "row_hash"),
        "listing_ids_hash": _sqlite_column_hash(conn, "listing_ids", "listing_id"),
        "buyer_ids_hash": _sqlite_column_hash(conn, "buyer_ids", "buyer_id"),
        "seller_ids_hash": _sqlite_column_hash(conn, "seller_ids", "seller_id"),
    }


def _sqlite_column_hash(conn: sqlite3.Connection, table: str, column: str) -> str:
    _validate_index_table_column(table, column)
    digest = hashlib.sha256()
    for (value,) in conn.execute(f"SELECT {column} FROM {table} ORDER BY {column}"):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _id_index_contains(conn: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    _validate_index_table_column(table, column)
    if not value:
        return False
    return conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (value,)).fetchone() is not None


def _id_index_insert(conn: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    _validate_index_table_column(table, column)
    if not value:
        return False
    cursor = conn.execute(f"INSERT OR IGNORE INTO {table} ({column}) VALUES (?)", (value,))
    return cursor.rowcount > 0


def _validate_index_table_column(table: str, column: str) -> None:
    allowed = {
        "seen_threads": "thread_id",
        "row_hashes": "row_hash",
        "listing_ids": "listing_id",
        "buyer_ids": "buyer_id",
        "seller_ids": "seller_id",
    }
    if allowed.get(table) != column:
        raise NberRealNormalizeError(f"Invalid internal index reference {table}.{column}")


def _listing_id_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    distinct = int(conn.execute("SELECT COUNT(*) FROM listing_ids").fetchone()[0])
    matched = int(conn.execute("SELECT COUNT(*) FROM listing_ids WHERE matched = 1").fetchone()[0])
    unmatched_examples = [
        row[0]
        for row in conn.execute("SELECT listing_id FROM listing_ids WHERE matched = 0 ORDER BY listing_id LIMIT 25")
    ]
    return {
        "distinct_listing_ids": distinct,
        "matched_listing_ids": matched,
        "unmatched_listing_ids": distinct - matched,
        "unmatched_examples_hash": [_hash_value(value) for value in unmatched_examples],
    }


def _canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(manifest, sort_keys=True))
    lineage = canonical.setdefault("lineage", {})
    lineage["normalization_manifest_hash"] = None
    lineage["normalization_manifest_payload_hash"] = None
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_date_string(value: str | None) -> str | None:
    parsed = _parse_date(value)
    return parsed.date().isoformat() if parsed else None


def _parse_datetime_string(value: str | None) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed else None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%d%b%Y")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%d%b%Y %H:%M:%S")


def _float_or_none(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


def _bool_or_none(value: str | None) -> bool | None:
    if value in {None, ""}:
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"Expected 0/1 boolean, got {value!r}")


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, newline="\n") as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except Exception:
        return None


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


def _row_hash(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
