from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable


THREAD_COLUMNS = [
    "thread_id",
    "listing_id",
    "buyer_id",
    "seller_id",
    "turn_index",
    "actor",
    "action",
    "amount",
    "status",
    "event_time",
]

LISTING_COLUMNS = [
    "listing_id",
    "seller_id",
    "category",
    "condition",
    "listing_price",
    "reference_price",
    "start_time",
    "end_time",
]

DEFAULT_ENGINES = ["duckdb", "pyarrow", "polars", "python"]


def stable_partition(value: str, partitions: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % partitions


def current_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.start_rss_bytes = current_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemorySampler":
        self.start_rss_bytes = current_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss_bytes())

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss_bytes())

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def open_gzip_csv(path: Path, columns: list[str]) -> tuple[io.TextIOWrapper, csv.DictWriter[str]]:
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    return text, writer


def generate_synthetic_inputs(root: Path, thread_rows: int, listing_rows: int, progress_interval: int) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    threads_path = root / "anon_bo_threads.csv.gz"
    listings_path = root / "anon_bo_lists.csv.gz"
    negotiated_listing_count = max(1, min(listing_rows, max(1, thread_rows // 3)))
    actions = [
        ("buyer", "offer", "submitted"),
        ("seller", "counter", "countered"),
        ("buyer", "counter", "countered"),
        ("seller", "accept", "accepted"),
        ("seller", "decline", "declined"),
    ]
    categories = ["electronics", "cameras", "collectibles", "home", "motors", "fashion", "media", "tools"]

    text, writer = open_gzip_csv(threads_path, THREAD_COLUMNS)
    with text:
        for index in range(thread_rows):
            listing_index = index % negotiated_listing_count
            actor, action, status = actions[index % len(actions)]
            writer.writerow(
                {
                    "thread_id": f"T{index // 3:012d}",
                    "listing_id": f"L{listing_index:012d}",
                    "buyer_id": f"B{index % max(1, negotiated_listing_count // 2):012d}",
                    "seller_id": f"S{listing_index % 100_000:012d}",
                    "turn_index": str(index % 5 + 1),
                    "actor": actor,
                    "action": action,
                    "amount": "" if action == "decline" else f"{50 + (index % 450) * 0.2:.2f}",
                    "status": status,
                    "event_time": f"2012-05-{index % 28 + 1:02d}T{index % 24:02d}:{index % 60:02d}:00",
                }
            )
            if progress_interval and index and index % progress_interval == 0:
                print(f"generated thread rows={index}", file=sys.stderr, flush=True)

    text, writer = open_gzip_csv(listings_path, LISTING_COLUMNS)
    with text:
        for index in range(listing_rows):
            writer.writerow(
                {
                    "listing_id": f"L{index:012d}",
                    "seller_id": f"S{index % 100_000:012d}",
                    "category": categories[index % len(categories)],
                    "condition": "used" if index % 3 else "new",
                    "listing_price": f"{75 + (index % 1000) * 0.55:.2f}",
                    "reference_price": f"{80 + (index % 1000) * 0.50:.2f}",
                    "start_time": f"2012-05-{index % 28 + 1:02d}T00:00:00",
                    "end_time": f"2012-06-{index % 28 + 1:02d}T00:00:00",
                }
            )
            if progress_interval and index and index % progress_interval == 0:
                print(f"generated listing rows={index}", file=sys.stderr, flush=True)

    return {
        "thread_rows": thread_rows,
        "listing_rows": listing_rows,
        "negotiated_listing_count": negotiated_listing_count,
        "threads_path": str(threads_path),
        "listings_path": str(listings_path),
        "threads_bytes": threads_path.stat().st_size,
        "listings_bytes": listings_path.stat().st_size,
        "threads_sha256": sha256_file(threads_path),
        "listings_sha256": sha256_file(listings_path),
    }


def normalize_turn(row: dict[str, str]) -> dict[str, object]:
    return {
        "source_row_id": f"{row['thread_id']}:{row['turn_index']}",
        "thread_id": row["thread_id"],
        "listing_id": row["listing_id"],
        "buyer_id": row["buyer_id"],
        "seller_id": row["seller_id"],
        "turn_index": int(row["turn_index"]),
        "actor": row["actor"],
        "action": row["action"],
        "amount": float(row["amount"]) if row["amount"] else None,
        "status": row["status"],
        "event_time": row["event_time"],
    }


def normalize_listing(row: dict[str, str]) -> dict[str, object]:
    return {
        "source_row_id": row["listing_id"],
        "listing_id": row["listing_id"],
        "seller_id": row["seller_id"],
        "category": row["category"],
        "condition": row["condition"],
        "listing_price": float(row["listing_price"]),
        "reference_price": float(row["reference_price"]) if row["reference_price"] else None,
        "start_time": row["start_time"],
        "end_time": row["end_time"],
    }


def jsonl_writer_for(writers: dict[int, io.TextIOWrapper], root: Path, partition_id: int) -> io.TextIOWrapper:
    writer = writers.get(partition_id)
    if writer is not None:
        return writer
    part_dir = root / f"part={partition_id:04d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    writer = (part_dir / "rows.jsonl").open("w", encoding="utf-8", newline="")
    writers[partition_id] = writer
    return writer


def run_python_csv_gzip(ctx: dict[str, object]) -> dict[str, object]:
    output = Path(ctx["output_dir"]) / "python"
    if output.exists():
        shutil.rmtree(output)
    turns_dir = output / "negotiation_turns"
    listings_dir = output / "negotiated_listings"
    turns_dir.mkdir(parents=True)
    listings_dir.mkdir(parents=True)
    partitions = int(ctx["partitions"])
    progress_interval = int(ctx["progress_interval"])

    listing_ids: set[str] = set()
    turn_writers: dict[int, io.TextIOWrapper] = {}
    listing_writers: dict[int, io.TextIOWrapper] = {}
    turn_rows = 0
    thread_started = time.perf_counter()
    try:
        with gzip.open(Path(ctx["threads_path"]), "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                listing_id = row["listing_id"]
                listing_ids.add(listing_id)
                out = jsonl_writer_for(turn_writers, turns_dir, stable_partition(listing_id, partitions))
                out.write(json.dumps(normalize_turn(row), sort_keys=True, separators=(",", ":")) + "\n")
                turn_rows += 1
                if progress_interval and turn_rows % progress_interval == 0:
                    print(f"python threads rows={turn_rows}", file=sys.stderr, flush=True)
    finally:
        for writer in turn_writers.values():
            writer.close()

    listing_started = time.perf_counter()
    listing_rows = 0
    matched_listing_rows = 0
    try:
        with gzip.open(Path(ctx["listings_path"]), "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                listing_rows += 1
                listing_id = row["listing_id"]
                if listing_id not in listing_ids:
                    continue
                out = jsonl_writer_for(listing_writers, listings_dir, stable_partition(listing_id, partitions))
                out.write(json.dumps(normalize_listing(row), sort_keys=True, separators=(",", ":")) + "\n")
                matched_listing_rows += 1
                if progress_interval and listing_rows % progress_interval == 0:
                    print(f"python listings rows={listing_rows}", file=sys.stderr, flush=True)
    finally:
        for writer in listing_writers.values():
            writer.close()

    finished = time.perf_counter()
    return {
        "thread_rows": turn_rows,
        "listing_rows": listing_rows,
        "matched_listing_rows": matched_listing_rows,
        "distinct_thread_listing_ids": len(listing_ids),
        "thread_elapsed_seconds": listing_started - thread_started,
        "listing_elapsed_seconds": finished - listing_started,
        "output_bytes": directory_size(output),
        "output_format": "partitioned_jsonl",
    }


def write_partitioned_table(table: object, root: Path, prefix: str, batch_index: int, partitions: list[int]) -> None:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    if not partitions:
        return
    for partition_id in sorted(set(partitions)):
        mask = pa.array([value == partition_id for value in partitions])
        part_dir = root / f"part={partition_id:04d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.filter(mask), part_dir / f"{prefix}-{batch_index:08d}.parquet", compression="snappy")


def run_pyarrow(ctx: dict[str, object]) -> dict[str, object]:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.csv as pacsv  # type: ignore[import-not-found]

    output = Path(ctx["output_dir"]) / "pyarrow"
    if output.exists():
        shutil.rmtree(output)
    turns_dir = output / "negotiation_turns"
    listings_dir = output / "negotiated_listings"
    turns_dir.mkdir(parents=True)
    listings_dir.mkdir(parents=True)
    partitions = int(ctx["partitions"])
    progress_interval = int(ctx["progress_interval"])
    read_options = pacsv.ReadOptions(block_size=8 * 1024 * 1024)

    listing_ids: set[str] = set()
    turn_rows = 0
    batch_index = 0
    thread_started = time.perf_counter()
    for batch in pacsv.open_csv(Path(ctx["threads_path"]), read_options=read_options):
        table = pa.Table.from_batches([batch])
        ids = [str(value) for value in table["listing_id"].to_pylist()]
        listing_ids.update(ids)
        part_values = [stable_partition(value, partitions) for value in ids]
        table = table.append_column("partition_id", pa.array(part_values, type=pa.int16()))
        write_partitioned_table(table, turns_dir, "thread", batch_index, part_values)
        turn_rows += batch.num_rows
        batch_index += 1
        if progress_interval and turn_rows % progress_interval < batch.num_rows:
            print(f"pyarrow threads rows={turn_rows}", file=sys.stderr, flush=True)

    listing_started = time.perf_counter()
    listing_rows = 0
    matched_listing_rows = 0
    batch_index = 0
    for batch in pacsv.open_csv(Path(ctx["listings_path"]), read_options=read_options):
        table = pa.Table.from_batches([batch])
        ids = [str(value) for value in table["listing_id"].to_pylist()]
        keep = [value in listing_ids for value in ids]
        listing_rows += batch.num_rows
        if any(keep):
            kept_ids = [value for value, include in zip(ids, keep) if include]
            part_values = [stable_partition(value, partitions) for value in kept_ids]
            kept = table.filter(pa.array(keep))
            kept = kept.append_column("partition_id", pa.array(part_values, type=pa.int16()))
            write_partitioned_table(kept, listings_dir, "listing", batch_index, part_values)
            matched_listing_rows += kept.num_rows
        batch_index += 1
        if progress_interval and listing_rows % progress_interval < batch.num_rows:
            print(f"pyarrow listings rows={listing_rows}", file=sys.stderr, flush=True)

    finished = time.perf_counter()
    return {
        "thread_rows": turn_rows,
        "listing_rows": listing_rows,
        "matched_listing_rows": matched_listing_rows,
        "distinct_thread_listing_ids": len(listing_ids),
        "thread_elapsed_seconds": listing_started - thread_started,
        "listing_elapsed_seconds": finished - listing_started,
        "output_bytes": directory_size(output),
        "output_format": "partitioned_parquet",
    }


def run_duckdb(ctx: dict[str, object]) -> dict[str, object]:
    import duckdb  # type: ignore[import-not-found]

    output = Path(ctx["output_dir"]) / "duckdb"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    con = duckdb.connect(str(output / "scratch.duckdb"))
    thread_started = time.perf_counter()
    con.execute("CREATE TEMP TABLE negotiated_ids AS SELECT DISTINCT listing_id FROM read_csv_auto(?)", [str(ctx["threads_path"])])
    turn_rows = con.execute("SELECT count(*) FROM read_csv_auto(?)", [str(ctx["threads_path"])]).fetchone()[0]
    listing_started = time.perf_counter()
    listing_rows = con.execute("SELECT count(*) FROM read_csv_auto(?)", [str(ctx["listings_path"])]).fetchone()[0]
    matched_listing_rows = con.execute(
        "SELECT count(*) FROM read_csv_auto(?) lists INNER JOIN negotiated_ids ids USING (listing_id)",
        [str(ctx["listings_path"])],
    ).fetchone()[0]
    distinct_ids = con.execute("SELECT count(*) FROM negotiated_ids").fetchone()[0]
    con.close()
    finished = time.perf_counter()
    return {
        "thread_rows": int(turn_rows),
        "listing_rows": int(listing_rows),
        "matched_listing_rows": int(matched_listing_rows),
        "distinct_thread_listing_ids": int(distinct_ids),
        "thread_elapsed_seconds": listing_started - thread_started,
        "listing_elapsed_seconds": finished - listing_started,
        "output_bytes": directory_size(output),
        "output_format": "duckdb_temp_index",
    }


def run_polars(ctx: dict[str, object]) -> dict[str, object]:
    import polars as pl  # type: ignore[import-not-found]

    output = Path(ctx["output_dir"]) / "polars"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    thread_started = time.perf_counter()
    threads = pl.scan_csv(Path(ctx["threads_path"]))
    listing_ids = threads.select("listing_id").unique().collect(streaming=True)["listing_id"].to_list()
    turn_rows = threads.select(pl.len()).collect(streaming=True).item()
    listing_started = time.perf_counter()
    ids = set(str(value) for value in listing_ids)
    listing_rows = 0
    matched_listing_rows = 0
    with gzip.open(Path(ctx["listings_path"]), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            listing_rows += 1
            if row["listing_id"] in ids:
                matched_listing_rows += 1
    finished = time.perf_counter()
    return {
        "thread_rows": int(turn_rows),
        "listing_rows": listing_rows,
        "matched_listing_rows": matched_listing_rows,
        "distinct_thread_listing_ids": len(ids),
        "thread_elapsed_seconds": listing_started - thread_started,
        "listing_elapsed_seconds": finished - listing_started,
        "output_bytes": directory_size(output),
        "output_format": "polars_streaming_scan_plus_python_membership",
    }


RUNNERS = {
    "duckdb": ("duckdb", run_duckdb),
    "pyarrow": ("pyarrow", run_pyarrow),
    "polars": ("polars", run_polars),
    "python": (None, run_python_csv_gzip),
}


def run_engine(name: str, ctx: dict[str, object]) -> dict[str, object]:
    package, runner = RUNNERS[name]
    if package and importlib.util.find_spec(package) is None:
        return {"engine": name, "available": False, "status": "skipped", "reason": f"Python package '{package}' is not installed"}
    gc.collect()
    started = time.perf_counter()
    with PeakMemorySampler() as sampler:
        try:
            result = runner(ctx)
            status = "ok"
            error = None
        except Exception as exc:
            result = {}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    output = {
        "engine": name,
        "available": True,
        "status": status,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "peak_rss_delta_bytes": sampler.peak_delta_bytes,
    }
    output.update(result)
    if error is not None:
        output["error"] = error
    if output.get("thread_rows") and elapsed > 0:
        output["total_rows_per_second"] = (
            int(output.get("thread_rows", 0)) + int(output.get("listing_rows", 0))
        ) / elapsed
    return output


def parse_engines(value: str) -> list[str]:
    engines = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(engines) - set(RUNNERS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown engines: {', '.join(unknown)}")
    return engines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark NBER-scale CSV.GZ ingestion approaches on generated data.")
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic anon_bo_threads.csv.gz rows.")
    parser.add_argument("--listing-rows", type=int, help="Synthetic anon_bo_lists.csv.gz rows.")
    parser.add_argument("--engines", type=parse_engines, default=DEFAULT_ENGINES, help="Comma-separated engines to run.")
    parser.add_argument("--partitions", type=int, default=64, help="Stable hash partition count.")
    parser.add_argument("--work-dir", type=Path, help="Directory for generated inputs and temporary outputs.")
    parser.add_argument("--keep-data", action="store_true", help="Keep generated inputs and outputs after the run.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print progress every N rows to stderr.")
    parser.add_argument("--json-output", type=Path, help="Optional path for a JSON copy of the summary.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    listing_rows = args.listing_rows if args.listing_rows is not None else args.rows * 2
    if listing_rows <= 0:
        raise SystemExit("--listing-rows must be positive")
    if args.partitions <= 0:
        raise SystemExit("--partitions must be positive")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="nber_io_bench_")
        work_dir = Path(temp_dir.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    data_dir = work_dir / "input"
    output_dir = work_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = generate_synthetic_inputs(data_dir, args.rows, listing_rows, args.progress_interval)
    ctx = {
        "threads_path": data["threads_path"],
        "listings_path": data["listings_path"],
        "output_dir": str(output_dir),
        "partitions": args.partitions,
        "progress_interval": args.progress_interval,
    }
    results = [run_engine(engine, ctx) for engine in args.engines]
    summary = {
        "benchmark": "nber_io_generated_csv_gzip",
        "schema": {
            "threads_file": "anon_bo_threads.csv.gz",
            "listings_file": "anon_bo_lists.csv.gz",
            "thread_columns": THREAD_COLUMNS,
            "listing_columns": LISTING_COLUMNS,
        },
        "data": data,
        "partitions": args.partitions,
        "engines": results,
        "work_dir": str(work_dir.resolve()) if args.keep_data else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
        tmp.write_text(encoded + "\n", encoding="utf-8")
        os.replace(tmp, args.json_output)

    if temp_dir is not None and not args.keep_data:
        temp_dir.cleanup()
    elif not args.keep_data:
        shutil.rmtree(work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
