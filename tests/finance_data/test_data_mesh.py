from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.cli import main
from behavior_lab.finance_data.data_mesh import FinancialDataMesh, SUPPORTED_ADAPTER_TYPES


class FinancialDataMeshTests(unittest.TestCase):
    def test_supported_manifest_adapter_types_cover_required_generic_shapes(self) -> None:
        self.assertTrue(
            {
                "json_api",
                "csv_api",
                "static_timestamped_public_file",
                "socrata",
                "ckan",
                "arcgis_feature_server",
                "sdmx",
                "rss_atom",
                "geojson",
                "gtfs",
                "gtfs_realtime",
            }.issubset(SUPPORTED_ADAPTER_TYPES)
        )

    def test_manifest_only_source_activation_stays_experimental(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = FinancialDataMesh(tmp)

            activated = mesh.activate_manifest(_manifest(), fixture_payload=_fixture())
            catalog = mesh.catalog()

            self.assertEqual(activated["status"], "experimental")
            self.assertEqual(activated["activation_scope"], "experimental_catalog")
            self.assertFalse(activated["production_source_activation"])
            self.assertEqual(activated["trial"]["status"], "passed")
            self.assertEqual(catalog["sources"][0]["source_id"], "official_json_cost_source")
            self.assertFalse(catalog["production_state_mutated"])

    def test_acquire_missing_source_family_activates_manifest_candidate(self) -> None:
        contract = {
            "proposal_id": "compute_contract",
            "missing_sources": ["billing_export"],
            "required_source_families": ["billing_export"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).acquire(
                contract_proposals=[contract],
                manifests=[_manifest()],
                fixtures_by_source={"official_json_cost_source": _fixture()},
                search_budget=4,
                llm_budget_usd=0.0,
            )

            self.assertEqual(len(result["activated_experimental_sources"]), 1)
            self.assertEqual(result["missing_source_families"], [])
            self.assertFalse(result["generated_connector_code"])
            self.assertFalse(result["production_source_activation"])

    def test_schema_drift_repair_switches_experimental_version_and_preserves_old_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = FinancialDataMesh(tmp)
            mesh.activate_manifest(_manifest(), fixture_payload=_fixture())
            candidate = _manifest(version="v2", value_field="new_cost", endpoint="https://official.example.invalid/v2/costs")
            fixture = {"records": [{"published_at": "2026-06-22T12:00:00+00:00", "available_at": "2026-06-22T12:01:00+00:00", "new_cost": 17.5}]}

            repaired = mesh.repair_source(
                "official_json_cost_source",
                failure={"error": "schema field changed from cost to new_cost"},
                candidate_manifest=candidate,
                fixture_payload=fixture,
            )

            self.assertEqual(repaired["repair_status"], "switched_experimental_version")
            self.assertTrue(repaired["old_source_preserved"])
            self.assertTrue(repaired["new_source_version_preserved"])
            self.assertEqual(repaired["isolated_canary"]["status"], "passed")
            self.assertFalse(repaired["production_state_mutated"])

    def test_dead_source_substitution_preserves_versions_without_production_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replacement = _manifest(
                source_id="official_substitute_cost_source",
                version="v1",
                endpoint="https://official-substitute.example.invalid/costs",
            )
            repaired = FinancialDataMesh(tmp).repair_source(
                "dead_cost_source",
                failure={"http_status": 404, "message": "gone"},
                candidate_manifest=replacement,
                fixture_payload=_fixture(),
            )

            self.assertEqual(repaired["previous_source_id"], "dead_cost_source")
            self.assertEqual(repaired["replacement_source_id"], "official_substitute_cost_source")
            self.assertEqual(repaired["repair_status"], "switched_experimental_version")
            self.assertFalse(repaired["production_source_activation"])

    def test_progressive_backfill_is_resumable_and_deduplicates_completed_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = FinancialDataMesh(tmp)
            first = mesh.backfill_plan(source_id="official_json_cost_source", start_date="2026-06-01", end_date="2026-06-10", chunk_days=3)
            completed = [first["chunks"][0]["chunk_id"]]

            second = mesh.backfill_plan(
                source_id="official_json_cost_source",
                start_date="2026-06-01",
                end_date="2026-06-10",
                chunk_days=3,
                completed_chunk_ids=completed,
            )

            self.assertTrue(second["resumable"])
            self.assertTrue(second["chunks"][0]["completed"])
            self.assertEqual(second["next_chunk"]["chunk_id"], second["chunks"][1]["chunk_id"])

    def test_unbounded_rate_limit_blocks_activation(self) -> None:
        manifest = _manifest(rate_limits={"bounded": False, "requests_per_minute": 10000})

        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).activate_manifest(manifest, fixture_payload=_fixture())

            self.assertEqual(result["validation"]["status"], "rejected")
            self.assertIn("unbounded_rate_limit", result["validation"]["reasons"])
            self.assertEqual(result["trial"]["status"], "blocked")

    def test_revision_leakage_and_ambiguous_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = FinancialDataMesh(tmp)
            revision = mesh.validate_manifest(_manifest(revision_behavior={"mode": "current_only", "uses_current_revision_only": True}))
            timestamps = mesh.validate_manifest(_manifest(event_timestamp={"field": "published_at", "semantics": "unknown"}))

            self.assertIn("revision_leakage_risk", revision["validation"]["reasons"])
            self.assertIn("ambiguous_timestamps", timestamps["validation"]["reasons"])

    def test_unclear_license_enters_approval_not_experimental_catalog(self) -> None:
        manifest = _manifest(license={"status": "unclear", "summary": "Terms need review", "url": "https://official.example.invalid/terms"})

        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).activate_manifest(manifest, fixture_payload=_fixture())

            self.assertEqual(result["validation"]["status"], "approval_required")
            self.assertIn("unclear_license", result["validation"]["approval_required"])
            self.assertEqual(result["trial"]["status"], "blocked")
            self.assertEqual(FinancialDataMesh(tmp).catalog()["sources"], [])

    def test_authority_fields_and_secret_values_are_rejected_before_coercion(self) -> None:
        manifest = _manifest()
        manifest["activation_status"] = "activated"
        manifest["request_parameters"]["money_allocation"] = {"amount": 10}
        manifest["endpoint"] = "https://official.example.invalid/data?token=secret-token"

        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).validate_manifest(manifest)

            self.assertEqual(result["validation"]["status"], "rejected")
            self.assertIn("production_activation_requested", result["validation"]["reasons"])
            self.assertIn("secret_exposure", result["validation"]["reasons"])
            serialized = json.dumps(result, sort_keys=True)
            self.assertNotIn("secret-token", serialized)

    def test_secret_values_under_secret_shaped_keys_are_redacted(self) -> None:
        manifest = _manifest()
        manifest["request_parameters"]["api_key"] = "plain-secret-value"

        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).validate_manifest(manifest)

            self.assertEqual(result["validation"]["status"], "rejected")
            self.assertIn("secret_exposure", result["validation"]["reasons"])
            serialized = json.dumps(result, sort_keys=True)
            self.assertIn("[REDACTED]", serialized)
            self.assertNotIn("plain-secret-value", serialized)

    def test_repair_rejects_nested_authority_fields_before_manifest_coercion(self) -> None:
        candidate = _manifest(source_id="replacement_with_nested_authority", version="v2")
        candidate["quality_checks"].append({"name": "looks_safe", "activation_status": "production"})

        with tempfile.TemporaryDirectory() as tmp:
            repaired = FinancialDataMesh(tmp).repair_source(
                "official_json_cost_source",
                failure={"error": "schema field changed"},
                candidate_manifest=candidate,
                fixture_payload=_fixture(),
            )

            self.assertEqual(repaired["repair_status"], "repair_blocked")
            self.assertEqual(repaired["replacement_source_id"], "replacement_with_nested_authority")
            self.assertIn("production_activation_requested", repaired["trial"]["validation"]["reasons"])
            self.assertEqual(FinancialDataMesh(tmp).catalog()["sources"], [])

    def test_malformed_manifest_trial_is_appended_as_blocked_not_raised(self) -> None:
        manifest = _manifest(normalized_series={"not": "a list"})

        with tempfile.TemporaryDirectory() as tmp:
            mesh = FinancialDataMesh(tmp)
            result = mesh.activate_manifest(manifest, fixture_payload=_fixture())

            self.assertEqual(result["validation"]["status"], "rejected")
            self.assertEqual(result["trial"]["status"], "blocked")
            self.assertIn("malformed_manifest", result["validation"]["reasons"])
            self.assertTrue(mesh.verify())

    def test_generic_adapter_parser_fails_closed_on_malformed_feed_with_provenance(self) -> None:
        manifest = _manifest(
            adapter_type="rss_atom",
            event_timestamp={"field": "pubDate", "semantics": "provider_event_time"},
            availability_timestamp={"field": "published", "semantics": "provider_publication_time"},
            normalized_series=[
                {
                    "series_id": "official_json_cost_source.hourly_cost",
                    "display_name": "Hourly Cost",
                    "observation_kind": "economic_release",
                    "value_field": "cost",
                    "event_time_field": "pubDate",
                    "availability_time_field": "published",
                    "unit": "USD",
                    "geography": {"type": "global", "id": "001"},
                    "contract_usage": ["compute_cost_avoidance"],
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).activate_manifest(
                manifest,
                fixture_payload="<rss><channel><item><cost>12.5</cost>",
                fixture_name="C:\\Users\\audit\\secret_fixture.xml",
            )

            self.assertEqual(result["trial"]["status"], "failed")
            self.assertEqual(result["trial"]["parser_result"], "parse_failed")
            self.assertEqual(result["trial"]["fixture_name"], "secret_fixture.xml")
            self.assertFalse(result["trial"]["parser_provenance"]["executed_generated_code"])
            self.assertIn("source_artifact_hash", result["trial"]["retrieval_provenance"])
            self.assertNotIn("C:\\Users\\audit", json.dumps(result))

    def test_malicious_generated_connector_is_sandbox_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).audit_generated_connector(
                source_id="official_json_cost_source",
                manifest_hash="abc123",
                code="import os\nos.environ['TOKEN']\nbroker.place_order()\n",
            )

            self.assertFalse(result["accepted"])
            self.assertTrue(result["sandboxed"])
            self.assertFalse(result["inherits_parent_environment"])
            self.assertFalse(result["production_database_writes"])
            self.assertIn("malicious_generated_connector", result["reasons"])

    def test_generated_connector_with_file_db_and_network_access_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).audit_generated_connector(
                source_id="official_json_cost_source",
                manifest_hash="abc123",
                code="import sqlite3, socket\nopen('production.db', 'w').write('x')\nsocket.create_connection(('example.com', 443))\n",
            )

            self.assertFalse(result["accepted"])
            self.assertTrue(result["sandboxed"])
            self.assertFalse(result["inherits_parent_environment"])
            self.assertFalse(result["production_database_writes"])
            self.assertIn("malicious_generated_connector", result["reasons"])

    def test_source_value_classification_prunes_redundant_without_erasing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = FinancialDataMesh(tmp).classify_source_value(
                source_id="redundant_cost_source",
                metrics={
                    "target_and_contract_usage": 1,
                    "predictive_lift": 0.0,
                    "prospective_survival": 0.0,
                    "economic_decision_lift": 0.0,
                    "reliability": 0.99,
                    "freshness": 0.95,
                    "maintenance_incidents": 0,
                    "api_and_llm_cost": 3.0,
                    "redundancy": 0.95,
                },
            )

            self.assertEqual(result["classification"], "redundant")
            self.assertEqual(result["budget_action"], "reduce_future_budget")
            self.assertFalse(result["source_erased"])

    def test_cli_activates_manifest_with_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            fixture = root / "fixture.json"
            manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
            fixture.write_text(json.dumps(_fixture()), encoding="utf-8")
            stream = io.StringIO()

            with redirect_stdout(stream):
                main(["money", "data-mesh", "activate", "--state-dir", str(root / "state"), "--manifest", str(manifest), "--fixture", str(fixture)])

            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["status"], "experimental")
            self.assertEqual(payload["trial"]["status"], "passed")


def _fixture() -> dict:
    return {
        "records": [
            {"published_at": "2026-06-22T12:00:00+00:00", "available_at": "2026-06-22T12:01:00+00:00", "cost": 12.5},
            {"published_at": "2026-06-22T13:00:00+00:00", "available_at": "2026-06-22T13:01:00+00:00", "cost": 13.25},
        ]
    }


def _manifest(**overrides: object) -> dict:
    value_field = str(overrides.pop("value_field", "cost"))
    payload = {
        "source_id": "official_json_cost_source",
        "version": "v1",
        "source_family": "billing_export",
        "display_name": "Official JSON Cost Source",
        "official_publisher": "Official Example Publisher",
        "adapter_type": "json_api",
        "endpoint": "https://official.example.invalid/costs",
        "request_parameters": {"records_path": "records"},
        "pagination": {"mode": "none"},
        "event_timestamp": {"field": "published_at", "semantics": "provider_event_time"},
        "availability_timestamp": {"field": "available_at", "semantics": "provider_publication_time"},
        "timezone": "UTC",
        "units": {"cost": "USD"},
        "geography": {"type": "global", "id": "001"},
        "cadence": {"seconds": 3600},
        "revision_behavior": {"mode": "all_available", "revision_id_field": "revision_id"},
        "missing_value_behavior": {"mode": "drop"},
        "license": {"status": "documented", "summary": "Official public terms permit research use", "url": "https://official.example.invalid/terms"},
        "rate_limits": {"bounded": True, "requests_per_minute": 30},
        "normalized_series": [
            {
                "series_id": "official_json_cost_source.hourly_cost",
                "display_name": "Hourly Cost",
                "observation_kind": "economic_release",
                "value_field": value_field,
                "event_time_field": "published_at",
                "availability_time_field": "available_at",
                "unit": "USD",
                "geography": {"type": "global", "id": "001"},
                "contract_usage": ["compute_cost_avoidance"],
            }
        ],
        "quality_checks": [{"name": "nonempty"}, {"name": "timestamp_parseable"}, {"name": "series_value_numeric"}],
        "documentation_urls": ["https://official.example.invalid/docs"],
        "credential_requirements": [],
        "generated_connector_required": False,
        "production_activation_requested": False,
    }
    payload.update(overrides)
    return payload


if __name__ == "__main__":
    unittest.main()
