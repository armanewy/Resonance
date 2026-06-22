from pathlib import Path
import gzip
import json
import os
import subprocess
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import _bootstrap  # noqa: F401,E402

from behavior_lab.datasets.nber_best_offer.real_normalize import (
    OFFICIAL_FULL_SOURCE_EXPECTATIONS,
    _artifact_binds_to_manifest,
    _independent_audit_artifact_verification,
    full_normalization_status,
    _normalize_listing_row,
    normalize_real_dataset,
    sha256_file,
    verify_full_release_evidence,
)
from behavior_lab.datasets.nber_best_offer.replication import replication_check, validate_replication_targets
from behavior_lab.datasets.nber_best_offer.source_schema import (
    REAL_LISTING_COLUMNS,
    REAL_THREAD_COLUMNS,
    inspect_schema,
    read_csv_header,
    load_real_mapping,
    validate_real_headers,
)
from behavior_lab.datasets.nber_best_offer.tasks import NberTaskError, assert_no_future_leakage, build_real_tasks_from_records, build_tasks
from behavior_lab.offerlab_models.common import validate_feature_contract


FIXTURES = ROOT / "tests" / "fixtures" / "nber_real_schema"


class RealNberPipelineTests(unittest.TestCase):
    def _write_gzip_fixture(self, source: Path, destination: Path) -> None:
        with source.open("rb") as input_handle, gzip.open(destination, "wb") as output_handle:
            output_handle.write(input_handle.read())

    def _read_first_partition_row(self, manifest: dict, table_name: str) -> dict:
        path = Path(manifest["tables"][table_name]["partitions"][0]["path"])
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()[0]
        with path.open("r", encoding="utf-8") as handle:
            return json.loads(next(line for line in handle if line.strip()))

    def test_fixture_headers_match_real_contract(self) -> None:
        report = validate_real_headers(
            listings=read_csv_header(FIXTURES / "anon_bo_lists.csv"),
            threads=read_csv_header(FIXTURES / "anon_bo_threads.csv"),
        )
        self.assertTrue(report["valid"])
        self.assertEqual(read_csv_header(FIXTURES / "anon_bo_lists.csv"), REAL_LISTING_COLUMNS)
        self.assertEqual(read_csv_header(FIXTURES / "anon_bo_threads.csv"), REAL_THREAD_COLUMNS)
        schema = inspect_schema()
        self.assertIn("mapping_hash", schema)
        self.assertIn("anon_bo_lists.csv", schema["expected_headers"])

    def test_real_normalize_limit_resume_and_replication_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            stopped = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2, stop_after_thread_pass=True)
            self.assertEqual(stopped["status"], "stopped_after_thread_pass")

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)
            self.assertEqual(manifest["tables"]["listings"]["rows"], 3)
            self.assertEqual(manifest["thread_linked_listing_extraction"]["unmatched_listing_ids"], 0)
            self.assertEqual(manifest["thread_linked_listing_extraction"]["membership_index"], "sqlite")
            self.assertTrue(manifest["research_only"])
            self.assertFalse(manifest["production_export_allowed"])
            self.assertFalse(manifest["commercial_training_allowed"])
            self.assertEqual(manifest["lineage"]["raw_source_hashes"]["anon_bo_lists"], manifest["source_files"]["anon_bo_lists"]["sha256"])
            self.assertIn("normalization_manifest_payload_hash", manifest["lineage"])
            self.assertIn("normalization_manifest_sha256_file", manifest["lineage"])
            self.assertTrue(Path(manifest["lineage"]["normalization_manifest_sha256_file"]).exists())
            self.assertTrue(Path(manifest["tables"]["negotiation_turns"]["partitions"][0]["path"]).exists())

            rerun = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertTrue(rerun["idempotent_rerun"])

            check = replication_check(output)
            self.assertIn("results", check)
            self.assertTrue(check["bounded_structure_passed"])
            self.assertFalse(check["full_replication_passed"])
            self.assertFalse(check["passed"])
            self.assertTrue(check["fatal_unevaluated"])

    def test_full_resume_reuses_completed_partitions_and_reports_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            self._write_gzip_fixture(FIXTURES / "anon_bo_lists.csv", raw / "anon_bo_lists.csv.gz")
            self._write_gzip_fixture(FIXTURES / "anon_bo_threads.csv", raw / "anon_bo_threads.csv.gz")

            stopped = normalize_real_dataset(
                raw,
                output,
                full=True,
                bucket_count=3,
                partition_rows=2,
                stop_after_turn_partitions=True,
            )
            turn_partition = Path(stopped["turn_partitions"]["partitions"][0]["path"])
            before_mtime = turn_partition.stat().st_mtime_ns
            before_hash = sha256_file(turn_partition)

            manifest = normalize_real_dataset(raw, output, full=True, bucket_count=3, partition_rows=2, resume=True)

            self.assertEqual(turn_partition.stat().st_mtime_ns, before_mtime)
            self.assertEqual(sha256_file(turn_partition), before_hash)
            self.assertEqual(manifest["tables"]["negotiation_turns"]["format"], "parquet" if turn_partition.suffix == ".parquet" else "jsonl")
            self.assertEqual(manifest["summary"]["row_counts"]["negotiation_turns"], 4)
            self.assertEqual(manifest["summary"]["date_ranges"]["event_time"]["min"], "2012-05-01T11:00:00")
            self.assertEqual(manifest["summary"]["date_ranges"]["response_time"]["max"], "2012-05-04T11:00:00")
            self.assertEqual(manifest["summary"]["categories"]["distinct"], 3)
            self.assertEqual(manifest["summary"]["sellers"]["distinct_in_threads"], 3)
            self.assertEqual(manifest["summary"]["buyers"]["distinct_in_threads"], 3)
            self.assertIn("replication_checks", manifest)
            self.assertTrue(Path(manifest["replication_checks"]["path"]).exists())
            self.assertIn("raw_source_row_hash", self._read_first_partition_row(manifest, "listings"))

            status = full_normalization_status(output)
            self.assertEqual(status["status"], "complete")
            self.assertTrue(status["partition_integrity"]["passed"])
            self.assertTrue(status["partition_checkpoints"])
            self.assertTrue(all(item["sha256_verified"] for item in status["partition_checkpoints"]))

    def test_cli_full_resume_default_paths_and_status_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            raw = data_root / "raw" / "nber_best_offer"
            raw.mkdir(parents=True)
            self._write_gzip_fixture(FIXTURES / "anon_bo_lists.csv", raw / "anon_bo_lists.csv.gz")
            self._write_gzip_fixture(FIXTURES / "anon_bo_threads.csv", raw / "anon_bo_threads.csv.gz")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            env["OFFERLAB_DATA_ROOT"] = str(data_root)

            normalize = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "behavior_lab",
                    "nber-best-offer",
                    "normalize-real",
                    "--full",
                    "--resume",
                    "--bucket-count",
                    "3",
                    "--partition-rows",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            manifest = json.loads(normalize.stdout)
            self.assertEqual(manifest["status"], "complete")

            status = subprocess.run(
                [sys.executable, "-m", "behavior_lab", "nber-best-offer", "full-status"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            payload = json.loads(status.stdout)
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["partition_integrity"]["passed"])

    def test_complete_manifest_rerun_revalidates_raw_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            threads_path = raw / "anon_bo_threads.csv"
            threads_path.write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            first = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            with threads_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write("1002,9999,2009,502,03may2012,35,98.5,0,0,75.00,03may2012 10:00:00,,0,0,0,1\n")

            rebuilt = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertNotIn("idempotent_rerun", rebuilt)
            self.assertNotEqual(first["source_files"]["anon_bo_threads"]["sha256"], rebuilt["source_files"]["anon_bo_threads"]["sha256"])
            self.assertEqual(rebuilt["tables"]["negotiation_turns"]["rows"], 5)

    def test_complete_manifest_rerun_rebuilds_deleted_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            partition = Path(manifest["tables"]["negotiation_turns"]["partitions"][0]["path"])
            partition.unlink()

            rebuilt = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertNotIn("idempotent_rerun", rebuilt)
            self.assertEqual(rebuilt["tables"]["negotiation_turns"]["rows"], 4)
            self.assertTrue(Path(rebuilt["tables"]["negotiation_turns"]["partitions"][0]["path"]).exists())

    def test_full_normalization_runs_unbounded_path_without_false_official_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            manifest = normalize_real_dataset(raw, output, full=True, bucket_count=3, partition_rows=2)
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["command_args"]["full"])
            self.assertIsNone(manifest["command_args"]["limit_threads"])
            self.assertEqual(manifest["normalization_scope"], "full_unbounded_source_scan")
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)
            self.assertEqual(manifest["tables"]["listings"]["rows"], 3)
            self.assertTrue(manifest["full_release_preflight"]["passed"])
            self.assertFalse(manifest["official_source_contract"]["matches_expected_official_sources"])
            self.assertTrue(manifest["audited_full_release_evidence"]["streaming_full_run_passed"])
            self.assertTrue(manifest["audited_full_release_evidence"]["full_run_checkpoint_validated"])
            self.assertTrue(manifest["audited_full_release_evidence"]["partition_hashes_verified"])
            self.assertFalse(manifest["audited_full_release_evidence"]["official_sources_matched"])
            self.assertFalse(manifest["audited_full_release_evidence"]["passed"])

            rerun = normalize_real_dataset(raw, output, full=True, bucket_count=3, partition_rows=2)
            self.assertTrue(rerun["idempotent_rerun"])

    def test_full_release_evidence_rejects_manifest_boolean_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            manifest = normalize_real_dataset(raw, output, full=True, bucket_count=3, partition_rows=2)
            manifest["source_files"] = {
                logical_name: {
                    "path": str(Path(tmp) / f"missing_{logical_name}.csv.gz"),
                    "sha256": expected["sha256"],
                    "bytes": expected["bytes"],
                }
                for logical_name, expected in OFFICIAL_FULL_SOURCE_EXPECTATIONS.items()
            }
            manifest["official_source_contract"] = {
                "matches_expected_official_sources": True,
                "files": {
                    logical_name: {
                        "actual_sha256": expected["sha256"],
                        "sha256_matches": True,
                        "actual_bytes": expected["bytes"],
                        "bytes_match": True,
                    }
                    for logical_name, expected in OFFICIAL_FULL_SOURCE_EXPECTATIONS.items()
                },
            }
            manifest["audited_full_release_evidence"] = {
                "passed": True,
                "streaming_full_run_passed": True,
                "official_sources_matched": True,
                "full_run_checkpoint_validated": True,
                "partition_hashes_verified": True,
                "replication_contract_passed": True,
                "independent_audit_passed": True,
            }

            report = verify_full_release_evidence(manifest)
            self.assertFalse(report["passed"])
            self.assertIn("source_files_verified_now", report["failures"])
            self.assertIn("replication_artifact_verified", report["failures"])
            self.assertIn("independent_audit_artifact_verified", report["failures"])

    def test_full_release_artifacts_require_exact_manifest_binding_and_scope(self) -> None:
        manifest = {
            "lineage": {
                "normalization_manifest_hash": "manifest-hash",
                "raw_source_hashes": {"anon_bo_lists": "lists", "anon_bo_threads": "threads"},
            }
        }
        self.assertFalse(_artifact_binds_to_manifest(manifest, {"raw_source_hashes": manifest["lineage"]["raw_source_hashes"]}))
        self.assertFalse(_artifact_binds_to_manifest(manifest, {"source_hashes": manifest["lineage"]["raw_source_hashes"]}))
        self.assertTrue(_artifact_binds_to_manifest(manifest, {"normalization_manifest_hash": "manifest-hash"}))

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "audit.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "independent_audit_passed": True,
                        "normalization_manifest_hash": "manifest-hash",
                    }
                ),
                encoding="utf-8",
            )
            missing_scope = _independent_audit_artifact_verification(
                manifest,
                {"independent_audit_artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}},
            )
            self.assertFalse(missing_scope["passed"])
            self.assertIn("scope_not_full_release", missing_scope["failures"])

            artifact_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "independent_audit_passed": True,
                        "scope": "full_release",
                        "normalization_manifest_hash": "manifest-hash",
                    }
                ),
                encoding="utf-8",
            )
            scoped = _independent_audit_artifact_verification(
                manifest,
                {"independent_audit_artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}},
            )
            self.assertTrue(scoped["passed"], scoped["failures"])

    def test_thread_checkpoint_mismatch_rebuilds_thread_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            normalize_real_dataset(raw, output, limit_threads=10, bucket_count=2, partition_rows=2, stop_after_thread_pass=True)
            checkpoint_path = output / "checkpoints" / "thread_pass.complete.json"
            first_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(first_checkpoint["signature"]["command_args"]["bucket_count"], 2)

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=3, partition_rows=2)
            second_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["command_args"]["bucket_count"], 3)
            self.assertEqual(second_checkpoint["signature"]["command_args"]["bucket_count"], 3)
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)

    def test_corrupted_thread_bucket_checkpoint_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2, stop_after_thread_pass=True)
            checkpoint = json.loads((output / "checkpoints" / "thread_pass.complete.json").read_text(encoding="utf-8"))
            bucket = next(item for item in checkpoint["thread_counts"]["bucket_manifest"]["buckets"] if item["rows"] > 0)
            (output / "_tmp" / "thread_buckets" / bucket["name"]).write_text("", encoding="utf-8")

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertEqual(manifest["source_thread_pass"]["accepted_rows"], 4)
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)

    def test_corrupted_thread_index_checkpoint_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2, stop_after_thread_pass=True)
            index_path = output / "_tmp" / "thread_listing_ids.sqlite"
            conn = sqlite3.connect(index_path)
            try:
                conn.execute("DELETE FROM row_hashes")
                conn.commit()
            finally:
                conn.close()

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertEqual(manifest["source_thread_pass"]["id_index_stats"]["row_hashes"], 4)
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)

    def test_same_count_thread_index_content_corruption_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2, stop_after_thread_pass=True)
            index_path = output / "_tmp" / "thread_listing_ids.sqlite"
            conn = sqlite3.connect(index_path)
            try:
                conn.execute("DELETE FROM listing_ids")
                conn.executemany(
                    "INSERT INTO listing_ids (listing_id, matched) VALUES (?, 0)",
                    [("missing-a",), ("missing-b",), ("missing-c",)],
                )
                conn.commit()
            finally:
                conn.close()

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertEqual(manifest["tables"]["listings"]["rows"], 3)
            self.assertEqual(manifest["thread_linked_listing_extraction"]["matched_listings"], 3)

    def test_extra_thread_bucket_checkpoint_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2, stop_after_thread_pass=True)
            bucket_dir = output / "_tmp" / "thread_buckets"
            source_bucket = next(path for path in bucket_dir.glob("bucket_*.jsonl") if path.stat().st_size > 0)
            (bucket_dir / "bucket_9999.jsonl").write_text(source_bucket.read_text(encoding="utf-8"), encoding="utf-8")

            manifest = normalize_real_dataset(raw, output, limit_threads=10, bucket_count=4, partition_rows=2)
            self.assertEqual(manifest["source_thread_pass"]["accepted_rows"], 4)
            self.assertEqual(manifest["tables"]["negotiation_turns"]["rows"], 4)

    def test_real_task_status_semantics_and_reference_price_exclusion(self) -> None:
        listing = {
            "listing_id": "l1",
            "seller_id": "s1",
            "category": "parts",
            "condition": "used",
            "listing_price": 100.0,
            "reference_price": None,
            "reference_price_unavailable_reason": "excluded",
            "excluded_reference_price_ref_price4": 95.0,
            "final_sale_price": 88.0,
            "sold_by_best_offer": True,
        }
        base_turn = {
            "listing_id": "l1",
            "buyer_id": "b1",
            "seller_id": "s1",
            "amount": 70.0,
            "event_time": "2020-01-01T00:00:00",
            "response_time": "2020-01-02T00:00:00",
        }
        turns = [
            {**base_turn, "thread_id": "accepted", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 1, "status": "accepted"},
            {**base_turn, "thread_id": "auto_accepted", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 9, "status": "auto_accepted"},
            {**base_turn, "thread_id": "declined", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 2, "status": "declined"},
            {**base_turn, "thread_id": "expired", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 0, "status": "expired"},
            {**base_turn, "thread_id": "countered", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 7, "status": "countered"},
            {**base_turn, "thread_id": "countered", "turn_index": 2, "actor": "seller", "action": "counter", "status_id": 1, "status": "accepted", "amount": 90.0},
            {**base_turn, "thread_id": "censored", "turn_index": 1, "actor": "buyer", "action": "offer", "status_id": 8, "status": "declined_other_buyer_accepted"},
        ]
        tasks = build_real_tasks_from_records([listing], turns)
        seller_labels = [row["label"] for row in tasks["seller_next_action"]]
        self.assertEqual(seller_labels.count("accept"), 2)
        self.assertIn("decline", seller_labels)
        self.assertIn("expire", seller_labels)
        self.assertIn("counter", seller_labels)
        self.assertEqual(len(seller_labels), 5)
        self.assertEqual(tasks["buyer_response_to_counter"][0]["label"], "accept")
        self.assertTrue(tasks["final_price_ratio"])
        self.assertTrue(all(row["label"] == 0.88 for row in tasks["final_price_ratio"]))
        self.assertTrue(any(row["label"] == 86400.0 for row in tasks["response_latency"]))
        self.assertNotIn("reference_price", tasks["seller_next_action"][0]["features"])
        self.assertNotIn("event_time", tasks["seller_next_action"][0]["features"])
        self.assertNotIn("status_id", tasks["seller_next_action"][0]["observed_history"][0])
        self.assertNotIn("response_time", tasks["seller_next_action"][0]["observed_history"][0])

    def test_real_leakage_guards_reject_forbidden_feature_aliases(self) -> None:
        row = {
            "row_id": "r1",
            "features": {
                "status_id": 1,
                "response_time": "2020-01-01T00:00:00",
                "ref_price4": 95.0,
                "excluded_reference_price_ref_price4": 95.0,
                "item_price": 88.0,
                "bo_ck_yn": 1,
                "final_sale_price": 88.0,
                "buyer_id_if_sold": "b1",
                "sold_by_best_offer": True,
                "auto_accept_price": 99.0,
                "auto_decline_price": 50.0,
                "accept_price": 99.0,
                "decline_price": 50.0,
                "buyer_us_if_sold": True,
            },
            "observed_history": [],
        }
        self.assertFalse(assert_no_future_leakage([row]))
        self.assertFalse(validate_feature_contract([row]))

    def test_build_tasks_refuses_unbounded_real_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tables").mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "command_args": {"full": True, "limit_threads": None},
                        "tables": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(NberTaskError):
                build_tasks(root)

    def test_ref_price4_is_excluded_from_predictor_reference_price(self) -> None:
        row = dict(zip(REAL_LISTING_COLUMNS, ["" for _ in REAL_LISTING_COLUMNS], strict=True))
        row.update(
            {
                "anon_item_id": "l1",
                "anon_slr_id": "s1",
                "start_price_usd": "100",
                "ref_price4": "95",
                "count4": "12",
                "bo_ck_yn": "1",
                "slr_us": "1",
                "buyer_us": "0",
            }
        )
        normalized = _normalize_listing_row(row)
        self.assertIsNone(normalized["reference_price"])
        self.assertEqual(normalized["excluded_reference_price_ref_price4"], 95.0)
        self.assertTrue(normalized["protected_outcome_fields_present"])

    def test_replication_targets_are_valid(self) -> None:
        report = validate_replication_targets()
        self.assertTrue(report["valid"], report["errors"])
        self.assertGreaterEqual(report["target_count"], 12)
        self.assertIn("published_descriptive_moment", report["level_counts"])

    def test_real_mapping_manifest_is_json_subset_yaml(self) -> None:
        manifest = load_real_mapping(ROOT / "datasets" / "manifests" / "nber_best_offer_real_mapping.yaml")
        self.assertEqual(manifest["files"]["anon_bo_threads.csv"]["header"], REAL_THREAD_COLUMNS)
        self.assertEqual(manifest["files"]["anon_bo_lists.csv"]["header"], REAL_LISTING_COLUMNS)


if __name__ == "__main__":
    unittest.main()
