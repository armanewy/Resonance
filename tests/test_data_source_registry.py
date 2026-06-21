from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path
import tempfile
import unittest

from behavior_lab.benchmarks.contracts import ArtifactLineage, BenchmarkManifest, validate_manifest
from behavior_lab.benchmarks.contracts import BenchmarkContractError
from behavior_lab.data_sources.cache import ContentAddressedCache
from behavior_lab.data_sources.registry import AuthorizationEvidence, default_registry


class DataSourceRegistryTests(unittest.TestCase):
    def test_restricted_research_sources_cannot_export_production_artifacts(self) -> None:
        registry = default_registry()
        self.assertFalse(registry.check("nber_ebay_best_offer", "production_export").allowed)
        self.assertFalse(registry.check("criteo_uplift", "production_export").allowed)
        self.assertFalse(registry.check("current_ebay_authorized_data", "production_export").allowed)
        evidence = AuthorizationEvidence.create(
            source_id="current_ebay_authorized_data",
            authorization_id="auth-test",
            owner_subject_hash="owner-hash",
            authorized_at="2026-06-21T12:00:00+00:00",
            scopes=["sell.fulfillment.readonly"],
            ledger_record_hash="ledger-hash",
        )
        self.assertTrue(
            registry.check(
                "current_ebay_authorized_data",
                "production_export",
                authorization_evidence=evidence,
            ).allowed
        )

    def test_lineage_verification_blocks_mixed_restricted_sources(self) -> None:
        result = default_registry().verify_lineage(["nber_ebay_best_offer", "current_ebay_authorized_data"], "production_export")
        self.assertFalse(result["allowed"])
        self.assertEqual(len(result["checks"]), 2)

    def test_empty_lineage_is_not_allowed(self) -> None:
        result = default_registry().verify_lineage([], "production_export")
        self.assertFalse(result["allowed"])
        self.assertIn("at least one", result["reason"])

    def test_content_addressed_cache_deduplicates_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.txt"
            source.write_text("same bytes", encoding="utf-8")
            cache = ContentAddressedCache(Path(tmp) / "cache")
            first = cache.add_file(source)
            second = cache.add_file(source)
            self.assertEqual(first.sha256, second.sha256)
            self.assertTrue(Path(first.path).exists())

    def test_benchmark_manifest_validates_and_reports_permissions(self) -> None:
        lineage = ArtifactLineage(
            artifact_id="artifact_001",
            source_dataset_ids=["nber_ebay_best_offer"],
            transformation_ids=["sample"],
            allowed_uses={"research": True, "production_export": False},
            license_status="uncertain",
        )
        manifest = BenchmarkManifest(
            benchmark_id="nber_seller_next_action",
            source_dataset_ids=["nber_ebay_best_offer"],
            task_type="classification",
            target_name="seller_next_action",
            feature_contract=["offer_to_asking_ratio"],
            forbidden_features=["final_price"],
            split_contract={"type": "chronological"},
            lineage=lineage,
        )
        result = validate_manifest(manifest)
        self.assertTrue(result["valid"])
        self.assertFalse(result["production_export_permission"]["allowed"])

    def test_manifest_rejects_lineage_that_hides_restricted_sources(self) -> None:
        lineage = ArtifactLineage(
            artifact_id="artifact_001",
            source_dataset_ids=["nber_ebay_best_offer"],
            transformation_ids=["sample"],
            allowed_uses={"research": True, "production_export": False},
            license_status="uncertain",
        )
        manifest = BenchmarkManifest(
            benchmark_id="bad_manifest",
            source_dataset_ids=["current_ebay_authorized_data"],
            task_type="classification",
            target_name="seller_next_action",
            feature_contract=["offer_to_asking_ratio"],
            forbidden_features=["final_price"],
            split_contract={"type": "chronological"},
            lineage=lineage,
        )
        with self.assertRaises(BenchmarkContractError):
            validate_manifest(manifest)

    def test_manifest_rejects_disallowed_lineage_use_claims(self) -> None:
        lineage = ArtifactLineage(
            artifact_id="artifact_001",
            source_dataset_ids=["nber_ebay_best_offer"],
            transformation_ids=["sample"],
            allowed_uses={"production_export": True},
            license_status="uncertain",
        )
        manifest = BenchmarkManifest(
            benchmark_id="bad_manifest",
            source_dataset_ids=["nber_ebay_best_offer"],
            task_type="classification",
            target_name="seller_next_action",
            feature_contract=["offer_to_asking_ratio"],
            forbidden_features=["final_price"],
            split_contract={"type": "chronological"},
            lineage=lineage,
        )
        with self.assertRaises(BenchmarkContractError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
