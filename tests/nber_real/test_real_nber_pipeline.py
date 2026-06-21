from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import _bootstrap  # noqa: F401,E402

from behavior_lab.datasets.nber_best_offer.real_normalize import _normalize_listing_row, normalize_real_dataset
from behavior_lab.datasets.nber_best_offer.replication import replication_check, validate_replication_targets
from behavior_lab.datasets.nber_best_offer.source_schema import (
    REAL_LISTING_COLUMNS,
    REAL_THREAD_COLUMNS,
    inspect_schema,
    read_csv_header,
    load_real_mapping,
    validate_real_headers,
)
from behavior_lab.datasets.nber_best_offer.tasks import build_real_tasks_from_records


FIXTURES = ROOT / "tests" / "fixtures" / "nber_real_schema"


class RealNberPipelineTests(unittest.TestCase):
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

    def test_full_normalization_is_blocked_until_full_run_proof_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            output = Path(tmp) / "normalized"
            raw.mkdir()
            (raw / "anon_bo_lists.csv").write_text((FIXTURES / "anon_bo_lists.csv").read_text(encoding="utf-8"), encoding="utf-8")
            (raw / "anon_bo_threads.csv").write_text((FIXTURES / "anon_bo_threads.csv").read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(Exception, "Full NBER normalization is intentionally blocked"):
                normalize_real_dataset(raw, output, full=True, bucket_count=3, partition_rows=2)

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
        self.assertIsNone(tasks["seller_next_action"][0]["features"]["reference_price"])
        self.assertNotIn("status_id", tasks["seller_next_action"][0]["observed_history"][0])
        self.assertNotIn("response_time", tasks["seller_next_action"][0]["observed_history"][0])

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
