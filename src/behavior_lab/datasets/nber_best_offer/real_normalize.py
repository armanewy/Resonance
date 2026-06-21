from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
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
    stop_after_thread_pass: bool = False,
) -> dict[str, Any]:
    if not full and limit_threads is None:
        raise NberRealNormalizeError("Use --limit-threads or --full for real NBER normalization")
    if full:
        raise NberRealNormalizeError(
            "Full NBER normalization is intentionally blocked: the current real-source "
            "normalizer is validated only for bounded --limit-threads runs. Implement "
            "disk-backed listing/thread indexes and full-run checkpointing before using --full."
        )
    start = time.perf_counter()
    raw = Path(raw_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
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
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("command_args") == args_signature and current.get("status") == "complete":
            current["idempotent_rerun"] = True
            return current

    lists_path = _find_source(raw, "anon_bo_lists.csv")
    threads_path = _find_source(raw, "anon_bo_threads.csv")
    quarantine = Quarantine(counts={}, examples=[])
    source_hashes = {"anon_bo_lists": sha256_file(lists_path), "anon_bo_threads": sha256_file(threads_path)}
    header_report = _validate_source_headers(lists_path, threads_path)
    if not header_report["valid"]:
        raise NberRealNormalizeError(json.dumps(header_report, sort_keys=True))

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
    for table_dir in [turn_table, listing_table]:
        if table_dir.exists():
            shutil.rmtree(table_dir)
        table_dir.mkdir(parents=True)

    turn_rows = _write_turn_partitions(bucket_dir, turn_table, partition_rows=partition_rows, quarantine=quarantine)
    listing_rows = _write_listing_partitions(lists_path, id_index_path, listing_table, partition_rows=partition_rows, quarantine=quarantine)
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
        "random_seed": seed,
        "mapping_manifest_hash": mapping_hash(),
        "source_files": {
            "anon_bo_lists": {"path": str(lists_path.resolve()), "sha256": source_hashes["anon_bo_lists"], "bytes": lists_path.stat().st_size},
            "anon_bo_threads": {"path": str(threads_path.resolve()), "sha256": source_hashes["anon_bo_threads"], "bytes": threads_path.stat().st_size},
        },
        "header_validation": header_report,
        "tables": {
            "negotiation_turns": {"path": str(turn_table.resolve()), "format": "parquet" if _pyarrow_available() else "jsonl", "rows": turn_rows["rows"], "partitions": turn_rows["partitions"]},
            "listings": {"path": str(listing_table.resolve()), "format": "parquet" if _pyarrow_available() else "jsonl", "rows": listing_rows["rows"], "partitions": listing_rows["partitions"]},
        },
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
    manifest_sha_path.write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
    return manifest


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
                _id_index_insert(index, "listing_ids", "listing_id", listing_id)
                accepted_rows += 1
                if accepted_rows % 10_000 == 0:
                    index.commit()
    finally:
        for handle in bucket_handles:
            handle.close()
        index.commit()
        index.close()
    return {
        "source": str(threads_path.resolve()),
        "accepted_rows": accepted_rows,
        "distinct_threads": distinct_threads,
        "duplicate_full_rows_removed": duplicate_rows,
        "status_counts": dict(status_counts),
        "offer_type_counts": dict(offer_type_counts),
        "limit_threads": limit_threads,
    }


def _write_turn_partitions(bucket_dir: Path, table_dir: Path, *, partition_rows: int, quarantine: Quarantine) -> dict[str, Any]:
    rows_out = []
    partitions = []
    total = 0
    part_index = 0
    for bucket in sorted(bucket_dir.glob("bucket_*.jsonl")):
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
                        rows_out.append(_normalize_thread_row(row, turn_index=index))
                    except Exception:
                        quarantine.add("thread_normalization_error", row, source_file=bucket.name, line_number=index)
                        continue
                    if len(rows_out) >= partition_rows:
                        partitions.append(_write_partition(table_dir, "turns", part_index, rows_out))
                        total += len(rows_out)
                        rows_out = []
                        part_index += 1
        finally:
            conn.close()
            if staging_path.exists():
                staging_path.unlink()
    if rows_out:
        partitions.append(_write_partition(table_dir, "turns", part_index, rows_out))
        total += len(rows_out)
    return {"rows": total, "partitions": partitions}


def _write_listing_partitions(
    lists_path: Path,
    id_index_path: Path,
    table_dir: Path,
    *,
    partition_rows: int,
    quarantine: Quarantine,
) -> dict[str, Any]:
    rows_out = []
    partitions = []
    total = 0
    part_index = 0
    index = _open_id_index(id_index_path, reset=False)
    try:
        with _open_text(lists_path) as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                listing_id = row.get("anon_item_id", "")
                if not _id_index_contains(index, "listing_ids", "listing_id", listing_id):
                    continue
                try:
                    rows_out.append(_normalize_listing_row(row))
                except Exception:
                    quarantine.add("listing_normalization_error", row, source_file=lists_path.name, line_number=line_number)
                    continue
                index.execute("UPDATE listing_ids SET matched = 1 WHERE listing_id = ?", (listing_id,))
                if len(rows_out) >= partition_rows:
                    partitions.append(_write_partition(table_dir, "listings", part_index, rows_out))
                    total += len(rows_out)
                    rows_out = []
                    part_index += 1
        if rows_out:
            partitions.append(_write_partition(table_dir, "listings", part_index, rows_out))
            total += len(rows_out)
        index.commit()
        id_stats = _listing_id_stats(index)
    finally:
        index.close()
    return {"rows": total, "partitions": partitions, **id_stats}


def _normalize_thread_row(row: dict[str, str], *, turn_index: int) -> dict[str, Any]:
    offer_type = OFFER_TYPE_MAP[row["offr_type_id"]]
    return {
        "source_row_id": _row_hash(row),
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
    if _pyarrow_available():
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = table_dir / f"{stem}_{index:05d}.parquet"
        table = pa.Table.from_pylist(rows)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    else:
        path = table_dir / f"{stem}_{index:05d}.jsonl"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=table_dir, newline="\n") as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    return {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256_file(path)}


def _validate_source_headers(lists_path: Path, threads_path: Path) -> dict[str, Any]:
    listing_header = _read_header(lists_path)
    thread_header = _read_header(threads_path)
    return validate_real_headers(listings=listing_header, threads=thread_header)


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
            return True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


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
    allowed = {"seen_threads": "thread_id", "row_hashes": "row_hash", "listing_ids": "listing_id"}
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
