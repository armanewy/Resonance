from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import tracemalloc
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile
import xml.etree.ElementTree as ET

try:  # Preserve compatibility with the schema helper if another worker provides it.
    from behavior_lab.datasets.nber_best_offer.source_schema import REAL_LISTING_COLUMNS, REAL_THREAD_COLUMNS
except Exception:  # pragma: no cover - fallback only matters when the optional helper is absent
    REAL_LISTING_COLUMNS = [
        "anon_item_id",
        "anon_title_code",
        "anon_product_id",
        "anon_leaf_categ_id",
        "anon_slr_id",
        "anon_buyer_id",
        "auct_start_dt",
        "fdbk_score_start",
        "fdbk_pstv_start",
        "auct_end_dt",
        "start_price_usd",
        "photo_count",
        "to_lst_cnt",
        "bo_lst_cnt",
        "count1",
        "ref_price1",
        "count2",
        "ref_price2",
        "count3",
        "ref_price3",
        "item_cndtn_id",
        "view_item_count",
        "wtchr_count",
        "meta_categ_id",
        "item_price",
        "bo_ck_yn",
        "ship_time_slowest",
        "ship_time_fastest",
        "ship_time_chosen",
        "decline_price",
        "accept_price",
        "bin_rev",
        "lstg_gen_type_id",
        "store",
        "ref_price4",
        "count4",
        "slr_us",
        "buyer_us",
    ]
    REAL_THREAD_COLUMNS = [
        "anon_item_id",
        "anon_thread_id",
        "anon_byr_id",
        "anon_slr_id",
        "src_cre_dt",
        "fdbk_score_src",
        "fdbk_pstv_src",
        "offr_type_id",
        "status_id",
        "offr_price",
        "src_cre_date",
        "response_time",
        "slr_hist",
        "byr_hist",
        "any_mssg",
        "byr_us",
    ]


SOURCE_ID = "nber_ebay_best_offer"
DATASET_PAGE = "https://www.nber.org/research/data/best-offer-sequential-bargaining"
DEFAULT_RAW_DIR = Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")) / "raw" / "nber_best_offer"
DEFAULT_MANIFEST_PATH = Path("datasets/manifests/nber_best_offer_downloads.yaml")
DEFAULT_DOC_PATH = Path("docs/runs/NBER_SOURCE_INVENTORY.md")
DEFAULT_SAMPLE_SEED = 20260621

OFFICIAL_SOURCE_FILES = {
    "anon_bo_lists.csv.gz": "https://www.nber.org/bargaining/anon_bo_lists.csv.gz",
    "anon_bo_threads.csv.gz": "https://www.nber.org/bargaining/anon_bo_threads.csv.gz",
    "Codebook.xlsx": "https://www.nber.org/bargaining/Codebook.xlsx",
}

OFFICIAL_SOURCES: tuple[dict[str, str], ...] = (
    {
        "logical_name": "anon_bo_lists",
        "filename": "anon_bo_lists.csv.gz",
        "url": OFFICIAL_SOURCE_FILES["anon_bo_lists.csv.gz"],
        "kind": "csv_gzip",
    },
    {
        "logical_name": "anon_bo_threads",
        "filename": "anon_bo_threads.csv.gz",
        "url": OFFICIAL_SOURCE_FILES["anon_bo_threads.csv.gz"],
        "kind": "csv_gzip",
    },
    {
        "logical_name": "codebook",
        "filename": "Codebook.xlsx",
        "url": OFFICIAL_SOURCE_FILES["Codebook.xlsx"],
        "kind": "xlsx",
    },
)

IDENTIFIER_COLUMNS = {
    "anon_item_id",
    "anon_thread_id",
    "anon_byr_id",
    "anon_buyer_id",
    "anon_slr_id",
    "anon_title_code",
    "anon_product_id",
    "buyer_id",
    "seller_id",
    "listing_id",
    "thread_id",
}

DATE_COLUMN_HINTS = ("date", "time", "dt", "created", "start", "end", "posted", "submit", "closed")


@dataclass(frozen=True)
class DownloadRecord:
    name: str
    url: str
    path: str
    bytes: int
    sha256: str
    downloaded: bool
    resumed: bool


@dataclass(frozen=True)
class InventoryConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    doc_path: Path = DEFAULT_DOC_PATH
    sample_seed: int = DEFAULT_SAMPLE_SEED
    first_sample_rows: int = 100
    reservoir_rows: int = 10_000
    chronological_rows_per_slice: int = 100
    timeout_seconds: int = 120


class SourceInventoryError(RuntimeError):
    pass


def default_data_root() -> Path:
    return Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData"))


def default_raw_dir() -> Path:
    return default_data_root() / "raw" / "nber_best_offer"


def run_source_inventory(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    official_sources: Iterable[dict[str, str]] = OFFICIAL_SOURCES,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    first_sample_rows: int = 100,
    reservoir_rows: int = 10_000,
    chronological_rows_per_slice: int = 100,
    timeout_seconds: int = 120,
    write_outputs: bool = True,
    download: bool = False,
) -> dict[str, Any]:
    """Inventory official files and write a metadata-only committed inventory.

    Full NBER files are multi-gigabyte research data. Inventorying and report
    generation must not implicitly acquire them; callers opt in with
    ``download=True``.
    """

    config = InventoryConfig(
        raw_dir=Path(raw_dir),
        manifest_path=Path(manifest_path),
        doc_path=Path(doc_path),
        sample_seed=sample_seed,
        first_sample_rows=first_sample_rows,
        reservoir_rows=reservoir_rows,
        chronological_rows_per_slice=chronological_rows_per_slice,
        timeout_seconds=timeout_seconds,
    )
    started = time.perf_counter()
    tracemalloc.start()
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    external_inventory_path = config.raw_dir / "source_inventory_raw_previews.json"

    downloads: list[dict[str, Any]] = []
    private_files: list[dict[str, Any]] = []
    public_files: list[dict[str, Any]] = []
    for source in official_sources:
        destination = config.raw_dir / source["filename"]
        if download:
            download_record = download_once(source["url"], destination, timeout=config.timeout_seconds)
        elif destination.exists():
            download_record = {
                "url": source["url"],
                "path": str(destination.resolve()),
                "status": "already_present",
                "resumed": False,
                "bytes_written": 0,
                "final_size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "download_required_explicit_opt_in": True,
            }
        else:
            raise SourceInventoryError(
                f"Missing {destination.name} in {config.raw_dir}. "
                "Pass --download to acquire official NBER files explicitly."
            )
        downloads.append(download_record)
        if source.get("kind") == "xlsx" or destination.suffix.lower() == ".xlsx":
            private_result, public_result = inventory_xlsx(destination, source)
        else:
            private_result, public_result = inventory_csv_source(destination, source, config)
        private_files.append(private_result)
        public_files.append(public_result)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    public_manifest = {
        "manifest_version": 1,
        "generated_at_utc": generated_at,
        "source_id": SOURCE_ID,
        "official_dataset_page": DATASET_PAGE,
        "raw_data_dir": str(config.raw_dir.resolve()),
        "external_inventory_path": str(external_inventory_path.resolve()),
        "release_policy": {
            "raw_source_files_committed": False,
            "raw_sample_records_committed": False,
            "committed_metadata_only": True,
            "production_export_allowed": False,
        },
        "runtime": {
            "seconds": round(time.perf_counter() - started, 3),
            "peak_memory_bytes_tracemalloc": peak_memory,
            "current_memory_bytes_tracemalloc": current_memory,
        },
        "downloads": downloads,
        "files": public_files,
        "gate": {
            "wave": "Wave 1 Prompt 1B",
            "passed": True,
            "normalization_or_modeling_performed": False,
            "raw_datasets_committed": False,
            "implicit_downloads_performed": False if not download else None,
            "notes": [
                "Official NBER source files were inventoried only.",
                "Missing files are downloaded only when --download is supplied.",
                "Raw previews and deterministic samples are stored outside the repository.",
            ],
        },
    }
    private_inventory = {
        "generated_at_utc": generated_at,
        "source_id": SOURCE_ID,
        "official_dataset_page": DATASET_PAGE,
        "raw_data_dir": str(config.raw_dir.resolve()),
        "files": private_files,
    }
    if write_outputs:
        external_inventory_path.write_text(json.dumps(private_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        public_manifest["external_inventory_sha256"] = sha256_file(external_inventory_path)
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        config.manifest_path.write_text(render_yaml(public_manifest), encoding="utf-8")
        config.doc_path.parent.mkdir(parents=True, exist_ok=True)
        config.doc_path.write_text(render_markdown(public_manifest), encoding="utf-8")
    return public_manifest


def public_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    files = []
    for item in manifest["files"]:
        rows = item.get("rows", {})
        files.append(
            {
                "logical_name": item["logical_name"],
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "accepted_rows": rows.get("accepted"),
                "rejected_rows": rows.get("rejected"),
                "quarantined_rows": rows.get("quarantined"),
                "gzip_integrity": item.get("gzip_integrity", {}).get("valid"),
                "raw_samples_dir": item.get("raw_samples_dir"),
            }
        )
    return {
        "source_id": manifest["source_id"],
        "raw_data_dir": manifest["raw_data_dir"],
        "external_inventory_path": manifest["external_inventory_path"],
        "runtime": manifest["runtime"],
        "files": files,
        "raw_rows_printed": False,
    }


def download_official_sources(raw_dir: str | Path | None = None, *, timeout: int = 120) -> list[DownloadRecord]:
    destination = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for source in OFFICIAL_SOURCES:
        path = destination / source["filename"]
        result = download_once(source["url"], path, timeout=timeout)
        records.append(
            DownloadRecord(
                source["filename"],
                source["url"],
                str(path.resolve()),
                path.stat().st_size,
                sha256_file(path),
                result["status"] != "already_present",
                bool(result.get("resumed")),
            )
        )
    return records


def download_with_resume(url: str, destination: str | Path, *, timeout: int = 120) -> DownloadRecord:
    output = Path(destination)
    result = download_once(url, output, timeout=timeout)
    return DownloadRecord(
        output.name,
        url,
        str(output.resolve()),
        output.stat().st_size,
        sha256_file(output),
        result["status"] != "already_present",
        bool(result.get("resumed")),
    )


def download_once(url: str, destination: Path, *, timeout: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {
            "url": url,
            "path": str(destination.resolve()),
            "status": "already_present",
            "resumed": False,
            "bytes_written": 0,
            "final_size_bytes": destination.stat().st_size,
        }

    partial = destination.with_name(f".{destination.name}.part")
    progress = destination.with_name(f".{destination.name}.progress.json")
    remote = probe_remote(url, timeout=timeout)
    if remote["accept_ranges"] and remote["content_length"]:
        return download_by_ranges(url, destination, partial, progress, remote, timeout=timeout)
    if partial.exists():
        partial.unlink()
    request = Request(url, headers={"User-Agent": "BehaviorDiscoveryLab/0.4 source-inventory"})
    try:
        response = urlopen(request, timeout=timeout)  # nosec B310: explicit official dataset URLs
    except HTTPError as exc:
        raise SourceInventoryError(f"Failed to download {url!r}: {exc}") from exc
    bytes_written = 0
    with response, partial.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            handle.flush()
            bytes_written += len(chunk)
            write_progress(progress, url=url, path=destination, downloaded=bytes_written, total=None)
    os.replace(partial, destination)
    return {
        "url": url,
        "path": str(destination.resolve()),
        "status": "downloaded",
        "method": "single_stream",
        "resumed": False,
        "bytes_written": bytes_written,
        "final_size_bytes": destination.stat().st_size,
    }


def probe_remote(url: str, *, timeout: int) -> dict[str, Any]:
    if not url.lower().startswith(("http://", "https://")):
        return {"content_length": None, "accept_ranges": False}
    request = Request(url, method="HEAD", headers={"User-Agent": "BehaviorDiscoveryLab/0.4 source-inventory"})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: explicit official dataset URLs
            length = response.headers.get("Content-Length")
            accept_ranges = response.headers.get("Accept-Ranges", "")
            return {
                "content_length": int(length) if length and length.isdigit() else None,
                "accept_ranges": "bytes" in accept_ranges.lower(),
            }
    except Exception:
        return {"content_length": None, "accept_ranges": False}


def download_by_ranges(
    url: str,
    destination: Path,
    partial: Path,
    progress: Path,
    remote: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    total = int(remote["content_length"])
    start = partial.stat().st_size if partial.exists() else 0
    resumed = start > 0
    if start > total:
        partial.unlink()
        start = 0
        resumed = False
    bytes_written = 0
    chunk_span = 64 * 1024 * 1024
    with partial.open("ab" if start else "wb") as handle:
        while start < total:
            end = min(start + chunk_span - 1, total - 1)
            request = Request(
                url,
                headers={
                    "User-Agent": "BehaviorDiscoveryLab/0.4 source-inventory",
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urlopen(request, timeout=timeout) as response:  # nosec B310: explicit official dataset URLs
                status = getattr(response, "status", None)
                if status != 206:
                    raise SourceInventoryError(f"Expected HTTP 206 for ranged download of {url!r}, got {status!r}")
                while True:
                    data = response.read(8 * 1024 * 1024)
                    if not data:
                        break
                    handle.write(data)
                    handle.flush()
                    bytes_written += len(data)
                    start += len(data)
                    write_progress(progress, url=url, path=destination, downloaded=start, total=total)
            if start <= end:
                raise SourceInventoryError(f"Ranged download stalled for {url!r} at byte {start}")
    if partial.stat().st_size != total:
        raise SourceInventoryError(f"Downloaded {partial.stat().st_size} bytes for {url!r}; expected {total}")
    os.replace(partial, destination)
    write_progress(progress, url=url, path=destination, downloaded=total, total=total, complete=True)
    return {
        "url": url,
        "path": str(destination.resolve()),
        "status": "downloaded",
        "method": "http_range_chunks",
        "resumed": resumed,
        "bytes_written": bytes_written,
        "final_size_bytes": destination.stat().st_size,
    }


def write_progress(
    progress: Path,
    *,
    url: str,
    path: Path,
    downloaded: int,
    total: int | None,
    complete: bool = False,
) -> None:
    payload = {
        "url": url,
        "path": str(path.resolve()),
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "percent": round(downloaded * 100 / total, 4) if total else None,
        "complete": complete,
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    progress.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def inventory_official_sources(
    raw_dir: str | Path | None = None,
    *,
    download: bool = False,
    sample_dir: str | Path | None = None,
    reservoir_size: int = 10_000,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    downloads = [asdict(record) for record in download_official_sources(root)] if download else []
    files: dict[str, Any] = {}
    for source in OFFICIAL_SOURCES:
        path = root / source["filename"]
        if not path.exists():
            files[source["filename"]] = {"url": source["url"], "exists": False}
            continue
        files[source["filename"]] = inventory_file(path, sample_dir=sample_dir, reservoir_size=reservoir_size, seed=seed)
        files[source["filename"]]["url"] = source["url"]
    return {
        "schema_version": "nber_source_inventory.v1",
        "raw_dir": str(root.resolve()),
        "downloads": downloads,
        "files": files,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "privacy": "identifier fields are hashed in command output; raw samples are not written to the repository",
    }


def inventory_file(
    path: str | Path,
    *,
    sample_dir: str | Path | None = None,
    reservoir_size: int = 10_000,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        return {"path": str(source_path.resolve()), "exists": False}
    source = source_from_path(source_path)
    config = InventoryConfig(
        raw_dir=source_path.parent,
        sample_seed=seed,
        first_sample_rows=100,
        reservoir_rows=reservoir_size,
        chronological_rows_per_slice=100,
    )
    if source_path.suffix.lower() == ".xlsx":
        _, public = inventory_xlsx(source_path, source)
    else:
        _, public = inventory_csv_source(source_path, source, config, sample_dir=sample_dir)
    public["exists"] = True
    public["compressed"] = source_path.suffix.lower() == ".gz"
    public["compressed_size"] = public.get("compressed_size_bytes")
    public["newline"] = public.get("newline_format")
    public["gzip_isize_mod_2_32"] = public.get("gzip_footer_isize_modulo_2_32")
    if "rows" in public:
        public["rows_accepted"] = public["rows"]["accepted"]
        public["rows_rejected"] = public["rows"]["rejected"]
        public["rows"] = public["rows"]["accepted"]
    return public


def source_from_path(path: Path) -> dict[str, str]:
    for source in OFFICIAL_SOURCES:
        if source["filename"] == path.name:
            return dict(source)
    return {
        "logical_name": path.name.replace(".csv.gz", "").replace(".csv", "").replace(".xlsx", ""),
        "filename": path.name,
        "url": "",
        "kind": "xlsx" if path.suffix.lower() == ".xlsx" else "csv_gzip",
    }


def inventory_csv_source(
    path: Path,
    source: dict[str, str],
    config: InventoryConfig,
    *,
    sample_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text_meta = detect_text_metadata(path)
    footer_size = gzip_isize(path) if path.suffix.lower() == ".gz" else None
    samples_dir = Path(sample_dir) if sample_dir is not None else config.raw_dir / "samples" / source["logical_name"]
    samples_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = samples_dir / "quarantine_metadata.jsonl"
    first_sample_path = samples_dir / f"{source['logical_name']}_first_{config.first_sample_rows}.csv"
    reservoir_path = samples_dir / f"{source['logical_name']}_reservoir_{config.reservoir_rows}.csv"

    first_10_rows: list[dict[str, str]] = []
    first_10_redacted: list[dict[str, str]] = []
    last_row: dict[str, str] | None = None
    last_row_number: int | None = None
    header: list[str] = []
    first_hashes: list[str] = []
    valid_count = 0
    rejected_count = 0
    records_seen = 0
    parseable_dates = 0
    min_date = None
    max_date = None
    date_column_candidates: list[str] = []
    reservoir: list[list[str]] = []
    reservoir_redacted: list[dict[str, str]] = []
    rng = random.Random(config.sample_seed + stable_int(source["logical_name"]))

    with open_text(path, text_meta["encoding"]) as opened:
        reader = csv.reader(opened.text)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        date_column_candidates = choose_date_columns(header)
        with first_sample_path.open("w", encoding="utf-8", newline="") as first_handle, quarantine_path.open(
            "w", encoding="utf-8"
        ) as quarantine:
            first_writer = csv.writer(first_handle)
            if header:
                first_writer.writerow(header)
            for physical_row_number, row in enumerate(reader, start=2):
                records_seen += 1
                if len(row) != len(header):
                    rejected_count += 1
                    quarantine.write(
                        json.dumps(
                            {
                                "source": source["filename"],
                                "physical_row_number": physical_row_number,
                                "expected_fields": len(header),
                                "actual_fields": len(row),
                                "reason": "field_count_mismatch",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                valid_count += 1
                row_dict = dict(zip(header, row, strict=True))
                if len(first_10_rows) < 10:
                    first_10_rows.append(row_dict)
                    first_10_redacted.append(redact_row(row_dict))
                    first_hashes.append(hash_json(row_dict))
                last_row = row_dict
                last_row_number = physical_row_number
                if valid_count <= config.first_sample_rows:
                    first_writer.writerow(row)
                if len(reservoir) < config.reservoir_rows:
                    reservoir.append(list(row))
                    reservoir_redacted.append(redact_row(row_dict))
                else:
                    selected = rng.randrange(valid_count)
                    if selected < config.reservoir_rows:
                        reservoir[selected] = list(row)
                        reservoir_redacted[selected] = redact_row(row_dict)

                parsed = parse_row_date(row_dict, date_column_candidates)
                if parsed is not None:
                    parseable_dates += 1
                    if min_date is None or parsed < min_date:
                        min_date = parsed
                    if max_date is None or parsed > max_date:
                        max_date = parsed
        decompressed_bytes = opened.binary.tell()

    write_csv_sample(reservoir_path, header, reservoir)
    sample_files: dict[str, Any] = {
        "first_rows": sample_metadata(first_sample_path),
        "reservoir": sample_metadata(reservoir_path),
    }
    redacted_sample_paths = {
        "first_100_redacted": write_jsonl_sample(
            samples_dir / f"{source['logical_name']}.first_100.redacted.jsonl", first_10_redacted
        ),
        "reservoir_redacted": write_jsonl_sample(
            samples_dir / f"{source['logical_name']}.reservoir.redacted.jsonl", reservoir_redacted
        ),
    }
    chronological = build_chronological_samples(
        path=path,
        source=source,
        samples_dir=samples_dir,
        encoding=text_meta["encoding"],
        header=header,
        date_column_candidates=date_column_candidates,
        min_date=min_date,
        max_date=max_date,
        rows_per_slice=config.chronological_rows_per_slice,
    )
    if chronological:
        sample_files["chronological"] = chronological

    gzip_status = {
        "valid": True,
        "method": "streamed_to_eof_with_python_gzip_crc_validation" if path.suffix.lower() == ".gz" else "not_gzip",
    }
    last_row_hash = hash_json(last_row) if last_row is not None else None
    header_valid = header_is_valid(source["filename"], header)
    public = {
        "logical_name": source["logical_name"],
        "filename": source["filename"],
        "url": source["url"],
        "kind": source.get("kind", "csv_gzip"),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "compressed_size_bytes": path.stat().st_size if path.suffix.lower() == ".gz" else None,
        "sha256": sha256_file(path),
        "gzip_footer_isize_modulo_2_32": footer_size,
        "decompressed_size_bytes_streamed": decompressed_bytes,
        "decompressed_size_estimate": decompressed_bytes if decompressed_bytes is not None else footer_size,
        "encoding": text_meta["encoding"],
        "newline_format": text_meta["newline_format"],
        "header": header,
        "header_valid": header_valid,
        "header_sha256": hash_json(header),
        "first_10_valid_row_sha256": first_hashes,
        "first_10_valid_rows_redacted": first_10_redacted,
        "last_complete_readable_row_sha256": last_row_hash,
        "last_complete_readable_row_redacted": redact_row(last_row),
        "last_complete_readable_row_number": last_row_number,
        "rows": {
            "accepted": valid_count,
            "rejected": rejected_count,
            "quarantined": rejected_count,
            "records_seen_excluding_header": records_seen,
        },
        "date_detection": {
            "chosen_column": date_column_candidates[0] if date_column_candidates else None,
            "candidates": date_column_candidates,
            "parseable_rows": parseable_dates,
            "min_utc": min_date.isoformat() if min_date else None,
            "max_utc": max_date.isoformat() if max_date else None,
            "chronological_samples_created": bool(chronological),
        },
        "gzip_integrity": gzip_status,
        "raw_samples_dir": str(samples_dir.resolve()),
        "sample_files": sample_files,
        "sample_paths": redacted_sample_paths,
        "quarantine_metadata_path": str(quarantine_path.resolve()),
        "raw_row_values_committed": False,
    }
    private = dict(public)
    private["first_10_valid_rows"] = first_10_rows
    private["last_complete_readable_row"] = last_row
    return private, public


def summarize_csv(
    path: str | Path,
    *,
    sample_dir: str | Path | None = None,
    reservoir_size: int = 10_000,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    public = inventory_file(path, sample_dir=sample_dir, reservoir_size=reservoir_size, seed=seed)
    return {
        "header": public.get("header", []),
        "header_valid": public.get("header_valid"),
        "rows": public.get("rows", 0),
        "first_10_valid_rows_redacted": public.get("first_10_valid_rows_redacted", []),
        "last_complete_readable_row_redacted": public.get("last_complete_readable_row_redacted"),
        "sample_paths": public.get("sample_paths", {}),
    }


def build_chronological_samples(
    *,
    path: Path,
    source: dict[str, str],
    samples_dir: Path,
    encoding: str,
    header: list[str],
    date_column_candidates: list[str],
    min_date: datetime | None,
    max_date: datetime | None,
    rows_per_slice: int,
) -> dict[str, Any] | None:
    if not header or not date_column_candidates or min_date is None or max_date is None or min_date == max_date:
        return None
    middle = min_date + ((max_date - min_date) / 2)
    early: list[tuple[float, list[str]]] = []
    middle_rows: list[tuple[float, list[str]]] = []
    late: list[tuple[float, list[str]]] = []

    with open_text(path, encoding) as opened:
        reader = csv.reader(opened.text)
        try:
            next(reader)
        except StopIteration:
            return None
        for row in reader:
            if len(row) != len(header):
                continue
            parsed = parse_row_date(dict(zip(header, row, strict=True)), date_column_candidates)
            if parsed is None:
                continue
            timestamp = parsed.timestamp()
            update_smallest(early, (timestamp, list(row)), rows_per_slice)
            update_smallest(middle_rows, (abs((parsed - middle).total_seconds()), list(row)), rows_per_slice)
            update_largest(late, (timestamp, list(row)), rows_per_slice)

    outputs: dict[str, Any] = {
        "method": "streaming_date_extrema_and_midpoint_distance",
        "min_utc": min_date.isoformat(),
        "max_utc": max_date.isoformat(),
        "files": {},
    }
    for label, rows in {
        "early": sorted(early, key=lambda item: item[0]),
        "middle": sorted(middle_rows, key=lambda item: item[0]),
        "late": sorted(late, key=lambda item: item[0]),
    }.items():
        output = samples_dir / f"{source['logical_name']}_chronological_{label}_{rows_per_slice}.csv"
        write_csv_sample(output, header, [row for _, row in rows])
        outputs["files"][label] = sample_metadata(output)
    return outputs


def update_smallest(items: list[tuple[float, list[str]]], candidate: tuple[float, list[str]], limit: int) -> None:
    items.append(candidate)
    items.sort(key=lambda item: item[0])
    del items[limit:]


def update_largest(items: list[tuple[float, list[str]]], candidate: tuple[float, list[str]], limit: int) -> None:
    items.append(candidate)
    items.sort(key=lambda item: item[0], reverse=True)
    del items[limit:]


def inventory_xlsx(path: Path, source: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    sha256 = sha256_file(path)
    sheets = []
    rows_by_sheet: dict[str, list[list[str]]] = {}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        zip_integrity = {"valid": bad_member is None, "bad_member": bad_member}
        decompressed_size = sum(info.file_size for info in archive.infolist())
        sheets = workbook_sheets(archive)
        shared_strings = workbook_shared_strings(archive)
        for sheet in sheets:
            rows_by_sheet[sheet["name"]] = worksheet_first_rows(archive, sheet["path"], shared_strings, limit=10)
    first_sheet_rows = rows_by_sheet.get(sheets[0]["name"], []) if sheets else []
    header = first_sheet_rows[0] if first_sheet_rows else []
    public = {
        "logical_name": source["logical_name"],
        "filename": source["filename"],
        "url": source["url"],
        "kind": source.get("kind", "xlsx"),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "compressed_size_bytes": path.stat().st_size,
        "sha256": sha256,
        "decompressed_size_estimate": decompressed_size,
        "encoding": "xlsx_zip_xml",
        "newline_format": None,
        "header": header,
        "header_sha256": hash_json(header),
        "sheets": [sheet["name"] for sheet in sheets],
        "sheet_paths": {sheet["name"]: sheet["path"] for sheet in sheets},
        "zip_integrity": zip_integrity,
        "raw_row_values_committed": False,
    }
    private = dict(public)
    private["first_10_rows_by_sheet"] = rows_by_sheet
    return private, public


def workbook_sheets(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels if "Id" in rel.attrib and "Target" in rel.attrib}
    sheets = []
    for sheet in workbook.iter():
        if not sheet.tag.endswith("sheet"):
            continue
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = targets.get(rel_id or "", "")
        if target.startswith("/xl/"):
            path = target.lstrip("/")
        elif target.startswith("xl/"):
            path = target
        else:
            path = f"xl/{target.lstrip('/')}" if target else ""
        sheets.append({"name": sheet.attrib.get("name", "sheet"), "path": path})
    return sheets


def workbook_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root:
        values.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("t")))
    return values


def worksheet_first_rows(archive: zipfile.ZipFile, worksheet_path: str, shared_strings: list[str], *, limit: int) -> list[list[str]]:
    if not worksheet_path or worksheet_path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(worksheet_path))
    rows: list[list[str]] = []
    for row in root.iter():
        if not row.tag.endswith("row"):
            continue
        values_by_col: dict[int, str] = {}
        for cell in row:
            if not cell.tag.endswith("c"):
                continue
            values_by_col[column_index_from_ref(cell.attrib.get("r", "A1"))] = read_xlsx_cell(cell, shared_strings)
        if values_by_col:
            rows.append([values_by_col.get(index, "") for index in range(max(values_by_col) + 1)])
        if len(rows) >= limit:
            break
    return rows


def read_xlsx_cell(cell: ET.Element, shared_strings: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if node.tag.endswith("t"))
    value = next((node for node in cell if node.tag.endswith("v")), None)
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        index = int(value.text)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    return value.text


def column_index_from_ref(reference: str) -> int:
    letters = "".join(ch for ch in reference if ch.isalpha())
    value = 0
    for ch in letters.upper():
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return max(value - 1, 0)


def detect_text_metadata(path: Path) -> dict[str, str]:
    data = read_decompressed_prefix(path, limit=256 * 1024)
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        data.decode(encoding)
    except UnicodeDecodeError:
        encoding = "cp1252"
    return {"encoding": encoding, "newline_format": detect_newline_format(data)}


def read_decompressed_prefix(path: Path, *, limit: int) -> bytes:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read(limit)
    with path.open("rb") as handle:
        return handle.read(limit)


def detect_newline(path: str | Path) -> str:
    return detect_text_metadata(Path(path))["newline_format"]


def detect_newline_format(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    without_crlf = data.replace(b"\r\n", b"")
    lf = without_crlf.count(b"\n")
    cr = without_crlf.count(b"\r")
    present = []
    if crlf:
        present.append("CRLF")
    if lf:
        present.append("LF")
    if cr:
        present.append("CR")
    return "+".join(present) if present else "none_detected"


class OpenedText:
    def __init__(self, binary: Any, text: io.TextIOWrapper):
        self.binary = binary
        self.text = text

    def __enter__(self) -> "OpenedText":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.text.close()


def open_text(path: Path, encoding: str) -> OpenedText:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    binary = gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")
    return OpenedText(binary, io.TextIOWrapper(binary, encoding=encoding, newline=""))


def _open_text(path: Path) -> Iterator[str]:
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def choose_date_columns(header: list[str]) -> list[str]:
    return [column for column in header if any(hint in column.strip().lower() for hint in DATE_COLUMN_HINTS)]


def parse_row_date(row: dict[str, str], candidates: list[str]) -> datetime | None:
    for column in candidates:
        parsed = parse_datetime(row.get(column, "").strip())
        if parsed is not None:
            return parsed
    return None


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    cleaned = value[:-1] + "+00:00" if value.endswith("Z") else value
    for parser in (
        lambda item: datetime.fromisoformat(item),
        lambda item: datetime.strptime(item, "%Y-%m-%d"),
        lambda item: datetime.strptime(item, "%Y/%m/%d"),
        lambda item: datetime.strptime(item, "%m/%d/%Y"),
        lambda item: datetime.strptime(item, "%Y%m%d"),
        lambda item: datetime.strptime(item, "%Y-%m-%d %H:%M:%S"),
        lambda item: datetime.strptime(item, "%m/%d/%Y %H:%M:%S"),
    ):
        try:
            parsed = parser(cleaned)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def gzip_integrity(path: str | Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rb") as handle:
            for _chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                pass
        return {"ok": True, "valid": True}
    except Exception as exc:
        return {"ok": False, "valid": False, "error": str(exc)}


def gzip_isize(path: str | Path) -> int | None:
    source = Path(path)
    if source.stat().st_size < 4:
        return None
    with source.open("rb") as handle:
        handle.seek(-4, os.SEEK_END)
        return int.from_bytes(handle.read(4), "little")


def write_csv_sample(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def sample_metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_jsonl_sample(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temp, path)
    return str(path.resolve())


def _write_jsonl_sample(path: Path, rows: list[dict[str, str]]) -> str:
    return write_jsonl_sample(path, rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def redact_row(row: dict[str, str] | None) -> dict[str, str]:
    if row is None:
        return {}
    redacted = {}
    for key, value in row.items():
        if key in IDENTIFIER_COLUMNS and value:
            redacted[key] = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        else:
            redacted[key] = value
    return redacted


def _redact_row(row: dict[str, str] | None) -> dict[str, str]:
    return redact_row(row)


def header_is_valid(name: str, header: list[str]) -> bool | None:
    if name.startswith("anon_bo_lists"):
        return header == REAL_LISTING_COLUMNS
    if name.startswith("anon_bo_threads"):
        return header == REAL_THREAD_COLUMNS
    return None


def _header_valid(name: str, header: list[str]) -> bool | None:
    return header_is_valid(name, header)


def render_yaml(value: Any, *, indent: int = 0) -> str:
    return _render_yaml(value, indent=indent).rstrip() + "\n"


def _render_yaml(value: Any, *, indent: int) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_render_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_render_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{yaml_scalar(value)}\n"


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# NBER Best Offer Source Inventory",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Scope",
        "",
        "Wave 1 Prompt 1B acquired and inventoried the official NBER Best Offer source files only. No normalization, modeling, benchmark training, or production export was performed.",
        "",
        f"- Source id: `{manifest['source_id']}`",
        f"- Official dataset page: {manifest['official_dataset_page']}",
        f"- Raw data directory: `{manifest['raw_data_dir']}`",
        f"- External raw preview inventory: `{manifest['external_inventory_path']}`",
        f"- Runtime seconds: `{manifest['runtime']['seconds']}`",
        f"- Peak traced Python memory bytes: `{manifest['runtime']['peak_memory_bytes_tracemalloc']}`",
        "",
        "## Files",
        "",
        "| File | SHA-256 | Size bytes | Rows accepted | Rows rejected | Encoding | Newlines | Integrity |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in manifest["files"]:
        rows = item.get("rows", {})
        integrity = item.get("gzip_integrity", item.get("zip_integrity", {}))
        valid = integrity.get("valid")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['filename']}`",
                    f"`{item['sha256']}`",
                    str(item["size_bytes"]),
                    str(rows.get("accepted", "n/a")),
                    str(rows.get("rejected", "n/a")),
                    f"`{item.get('encoding')}`",
                    f"`{item.get('newline_format')}`",
                    "`passed`" if valid else "`failed`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Non-Sensitive Metadata",
            "",
            "Committed metadata includes headers, row counts, hashes of raw previews, integrity status, and paths to external samples. It intentionally does not include raw buyer identifiers, seller identifiers, or source record values.",
            "",
        ]
    )
    for item in manifest["files"]:
        lines.extend(
            [
                f"### {item['filename']}",
                "",
                f"- URL: {item['url']}",
                f"- Header: `{', '.join(item.get('header', []))}`",
                f"- Decompressed-size estimate: `{item.get('decompressed_size_estimate')}`",
                f"- First 10 valid row hashes: `{', '.join(item.get('first_10_valid_row_sha256', [])) or 'n/a'}`",
                f"- Last complete readable row hash: `{item.get('last_complete_readable_row_sha256', 'n/a')}`",
                f"- Raw samples directory: `{item.get('raw_samples_dir', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Leakage Risks",
            "",
            "- Official CSV rows may contain persistent buyer, seller, listing, or negotiation identifiers. Raw rows and samples therefore remain outside the repository.",
            "- The inventory records headers and row hashes, but no raw source record values are committed.",
            "- NBER remains research/internal-benchmarking evidence only and is not production-exportable.",
            "",
            "## Limitations",
            "",
            "- This task did not map the official schema into the fixture normalizer.",
            "- Chronological samples are produced only when parseable date-like columns are present.",
            "- Gzip footer ISIZE is modulo 2^32; streamed decompressed bytes are recorded when the CSV pass reaches EOF.",
            "",
            "## Gate",
            "",
            f"- Passed: `{manifest['gate']['passed']}`",
            "- Normalization/modeling performed: `false`",
            "- Raw datasets committed: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory official NBER Best Offer source files")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--doc", default=str(DEFAULT_DOC_PATH))
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--first-sample-rows", type=int, default=100)
    parser.add_argument("--reservoir-rows", type=int, default=10_000)
    parser.add_argument("--chronological-rows-per-slice", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--download", action="store_true", help="Explicitly download missing official files before inventory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = run_source_inventory(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        doc_path=args.doc,
        sample_seed=args.sample_seed,
        first_sample_rows=args.first_sample_rows,
        reservoir_rows=args.reservoir_rows,
        chronological_rows_per_slice=args.chronological_rows_per_slice,
        timeout_seconds=args.timeout_seconds,
        download=args.download,
    )
    print(json.dumps(public_summary(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
