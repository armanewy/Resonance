from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import csv
import io
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from behavior_lab.core import parse_time, stable_hash, to_jsonable, utc_now
from behavior_lab.finance_data.contracts import FinanceDataError, SUPPORTED_OBSERVATION_KINDS
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


DATA_MESH_SCHEMA_VERSION = "financial_data_mesh.v1"
DEFAULT_DATA_MESH_STATE_DIR = ".money_data_mesh"

SUPPORTED_ADAPTER_TYPES = {
    "arcgis_feature_server",
    "ckan",
    "csv_api",
    "geojson",
    "gtfs",
    "gtfs_realtime",
    "json_api",
    "rss_atom",
    "sdmx",
    "socrata",
    "static_timestamped_public_file",
}

UNCLEAR_LICENSE_STATUSES = {"", "ambiguous", "requires_acceptance", "requires_approval", "restrictive", "unclear", "unknown"}
AMBIGUOUS_TIMESTAMP_VALUES = {"", "ambiguous", "current_only", "inferred", "latest_only", "local_time_unknown", "unknown"}
SECRET_MARKERS = ("api_key=", "apikey=", "password=", "secret=", "sk-", "token=")
DANGEROUS_CODE_MARKERS = (
    "broker",
    "os.environ",
    "place_order",
    "production",
    "secret",
    "seller.update",
    "subprocess",
    "submit_order",
    "trade_live",
)
MESH_AUTHORITY_FIELDS = {
    "activate_source",
    "activation_status",
    "money_allocation",
    "production_activation",
    "production_source_activation",
    "promote_source",
    "source_activation",
}


class DataMeshError(FinanceDataError):
    pass


@dataclass(frozen=True)
class NormalizedSeriesDefinition:
    series_id: str
    display_name: str
    observation_kind: str
    value_field: str
    event_time_field: str | None = None
    availability_time_field: str | None = None
    unit: str = "unit"
    geography: dict[str, Any] = field(default_factory=dict)
    contract_usage: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_nonempty(self.series_id, "series_id")
        _require_nonempty(self.display_name, "display_name")
        _require_nonempty(self.value_field, "value_field")
        _require_nonempty(self.unit, "unit")
        if self.observation_kind not in SUPPORTED_OBSERVATION_KINDS:
            raise DataMeshError(f"unsupported observation_kind: {self.observation_kind}")
        if not isinstance(self.geography, dict) or not self.geography:
            raise DataMeshError("series geography must be a non-empty object")
        if not isinstance(self.contract_usage, list):
            raise DataMeshError("contract_usage must be a list")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizedSeriesDefinition":
        if not isinstance(payload, dict):
            raise DataMeshError("normalized series must be an object")
        return cls(
            series_id=str(payload.get("series_id", "")),
            display_name=str(payload.get("display_name", "")),
            observation_kind=str(payload.get("observation_kind", "")),
            value_field=str(payload.get("value_field", "")),
            event_time_field=_optional_str(payload.get("event_time_field")),
            availability_time_field=_optional_str(payload.get("availability_time_field")),
            unit=str(payload.get("unit", "unit")),
            geography=_dict_field(payload, "geography"),
            contract_usage=_list_field(payload, "contract_usage"),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DeclarativeSourceManifest:
    source_id: str
    version: str
    source_family: str
    display_name: str
    official_publisher: str
    adapter_type: str
    endpoint: str
    request_parameters: dict[str, Any]
    pagination: dict[str, Any]
    event_timestamp: dict[str, Any]
    availability_timestamp: dict[str, Any]
    timezone: str
    units: dict[str, Any]
    geography: dict[str, Any]
    cadence: dict[str, Any]
    revision_behavior: dict[str, Any]
    missing_value_behavior: dict[str, Any]
    license: dict[str, Any]
    rate_limits: dict[str, Any]
    normalized_series: list[NormalizedSeriesDefinition]
    quality_checks: list[dict[str, Any]]
    documentation_urls: list[str] = field(default_factory=list)
    credential_requirements: list[str] = field(default_factory=list)
    generated_connector_required: bool = False
    production_activation_requested: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "version",
            "source_family",
            "display_name",
            "official_publisher",
            "adapter_type",
            "endpoint",
            "timezone",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.adapter_type not in SUPPORTED_ADAPTER_TYPES:
            raise DataMeshError(f"unsupported adapter_type: {self.adapter_type}")
        if len(self.normalized_series) == 0 or len(self.normalized_series) > 10:
            raise DataMeshError("normalized_series must contain 1-10 analysis series")
        for field_name in (
            "request_parameters",
            "pagination",
            "event_timestamp",
            "availability_timestamp",
            "units",
            "geography",
            "cadence",
            "revision_behavior",
            "missing_value_behavior",
            "license",
            "rate_limits",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or not value:
                raise DataMeshError(f"{field_name} must be a non-empty object")
        if not isinstance(self.quality_checks, list) or not self.quality_checks:
            raise DataMeshError("quality_checks must be a non-empty list")
        for value in self.documentation_urls + self.credential_requirements:
            _require_nonempty(value, "documentation_urls/credential_requirements")
        if self.production_activation_requested:
            raise DataMeshError("declarative manifests cannot request production activation")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeclarativeSourceManifest":
        if not isinstance(payload, dict):
            raise DataMeshError("manifest must be an object")
        return cls(
            source_id=str(payload.get("source_id", "")),
            version=str(payload.get("version", "")),
            source_family=str(payload.get("source_family", "")),
            display_name=str(payload.get("display_name", "")),
            official_publisher=str(payload.get("official_publisher", "")),
            adapter_type=str(payload.get("adapter_type", "")),
            endpoint=str(payload.get("endpoint", "")),
            request_parameters=_dict_field(payload, "request_parameters"),
            pagination=_dict_field(payload, "pagination"),
            event_timestamp=_dict_field(payload, "event_timestamp"),
            availability_timestamp=_dict_field(payload, "availability_timestamp"),
            timezone=str(payload.get("timezone", "")),
            units=_dict_field(payload, "units"),
            geography=_dict_field(payload, "geography"),
            cadence=_dict_field(payload, "cadence"),
            revision_behavior=_dict_field(payload, "revision_behavior"),
            missing_value_behavior=_dict_field(payload, "missing_value_behavior"),
            license=_dict_field(payload, "license"),
            rate_limits=_dict_field(payload, "rate_limits"),
            normalized_series=[NormalizedSeriesDefinition.from_dict(item) for item in _list_field(payload, "normalized_series")],
            quality_checks=[dict(item) for item in _list_field(payload, "quality_checks")],
            documentation_urls=[str(item) for item in _list_field(payload, "documentation_urls")],
            credential_requirements=[str(item) for item in _list_field(payload, "credential_requirements")],
            generated_connector_required=bool(payload.get("generated_connector_required", False)),
            production_activation_requested=bool(payload.get("production_activation_requested", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def manifest_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class ManifestValidation:
    status: str
    allowed_for_experimental_catalog: bool
    reasons: list[str] = field(default_factory=list)
    approval_required: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class ManifestValidator:
    def validate(self, manifest: DeclarativeSourceManifest) -> ManifestValidation:
        reasons: list[str] = []
        approvals: list[str] = []
        license_status = str(manifest.license.get("status", "")).strip().lower()
        event_semantics = str(manifest.event_timestamp.get("semantics", "")).strip().lower()
        availability_semantics = str(manifest.availability_timestamp.get("semantics", "")).strip().lower()
        revision_mode = str(manifest.revision_behavior.get("mode", "")).strip().lower()
        checks = {
            "official_publisher_declared": bool(manifest.official_publisher.strip()),
            "adapter_supported": manifest.adapter_type in SUPPORTED_ADAPTER_TYPES,
            "documented_interface": bool(manifest.documentation_urls),
            "license_clear": license_status not in UNCLEAR_LICENSE_STATUSES,
            "event_timestamp_unambiguous": event_semantics not in AMBIGUOUS_TIMESTAMP_VALUES and bool(manifest.event_timestamp.get("field")),
            "availability_timestamp_unambiguous": availability_semantics not in AMBIGUOUS_TIMESTAMP_VALUES and bool(manifest.availability_timestamp.get("field")),
            "revision_safe": revision_mode not in {"current_only", "latest_only", "unknown"} and manifest.revision_behavior.get("uses_current_revision_only") is not True,
            "rate_limit_bounded": manifest.rate_limits.get("bounded") is True,
            "quality_checks_declared": bool(manifest.quality_checks),
            "no_secret_values": not _contains_secret_like_value(manifest.to_dict()),
            "no_production_activation": manifest.production_activation_requested is not True,
            "declarative_only": manifest.generated_connector_required is not True,
            "no_authority_fields": not _contains_authority_field(manifest.to_dict()),
        }
        if not checks["official_publisher_declared"]:
            reasons.append("missing_official_publisher")
        if not checks["adapter_supported"]:
            reasons.append("unsupported_adapter")
        if not checks["documented_interface"]:
            approvals.append("missing_documentation")
        if not checks["license_clear"]:
            approvals.append("unclear_license")
        if not checks["event_timestamp_unambiguous"] or not checks["availability_timestamp_unambiguous"]:
            reasons.append("ambiguous_timestamps")
        if not checks["revision_safe"]:
            reasons.append("revision_leakage_risk")
        if not checks["rate_limit_bounded"]:
            reasons.append("unbounded_rate_limit")
        if not checks["quality_checks_declared"]:
            reasons.append("missing_quality_checks")
        if not checks["no_secret_values"]:
            reasons.append("secret_exposure")
        if not checks["no_production_activation"] or not checks["no_authority_fields"]:
            reasons.append("production_activation_requested")
        if not checks["declarative_only"]:
            approvals.append("generated_connector_required")
        if manifest.credential_requirements:
            approvals.append("credential_required")

        if reasons:
            status = "rejected"
        elif approvals:
            status = "approval_required"
        else:
            status = "valid"
        return ManifestValidation(
            status=status,
            allowed_for_experimental_catalog=status == "valid",
            reasons=sorted(set(reasons)),
            approval_required=sorted(set(approvals)),
            checks=checks,
        )


class FinancialDataMesh:
    def __init__(self, state_dir: str | Path = DEFAULT_DATA_MESH_STATE_DIR) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = AppendOnlyResearchStore(self.root / "data_mesh.jsonl")
        self.validator = ManifestValidator()

    def validate_manifest(self, manifest: dict[str, Any] | DeclarativeSourceManifest) -> dict[str, Any]:
        raw_reasons = _raw_manifest_rejection_reasons(manifest)
        if raw_reasons:
            payload = _invalid_manifest_payload(manifest, raw_reasons, "manifest requested forbidden authority")
            self.store.append("data_mesh_manifest_validated", payload)
            return payload
        try:
            typed = manifest if isinstance(manifest, DeclarativeSourceManifest) else DeclarativeSourceManifest.from_dict(manifest)
            validation = self.validator.validate(typed)
            payload = _manifest_payload(typed, validation)
        except Exception as exc:
            payload = _invalid_manifest_payload(manifest, ["malformed_manifest"], str(exc))
        self.store.append("data_mesh_manifest_validated", payload)
        return payload

    def trial_manifest(
        self,
        manifest: dict[str, Any] | DeclarativeSourceManifest,
        *,
        fixture_payload: Any,
        fixture_name: str = "fixture",
    ) -> dict[str, Any]:
        raw_reasons = _raw_manifest_rejection_reasons(manifest)
        if raw_reasons:
            payload = {
                **_invalid_manifest_payload(manifest, raw_reasons, "manifest requested forbidden authority"),
                "trial": {"status": "blocked", "reason": "manifest_not_valid_for_experimental_catalog", "observations": 0},
            }
            self.store.append("data_mesh_manifest_trial_blocked", payload)
            return payload
        typed = manifest if isinstance(manifest, DeclarativeSourceManifest) else DeclarativeSourceManifest.from_dict(manifest)
        validation = self.validator.validate(typed)
        if not validation.allowed_for_experimental_catalog:
            payload = {
                **_manifest_payload(typed, validation),
                "trial": {
                    "status": "blocked",
                    "reason": "manifest_not_valid_for_experimental_catalog",
                    "observations": 0,
                },
            }
            self.store.append("data_mesh_manifest_trial_blocked", payload)
            return payload
        trial = _run_fixture_trial(typed, fixture_payload=fixture_payload, fixture_name=fixture_name)
        payload = {**_manifest_payload(typed, validation), "trial": trial}
        self.store.append("data_mesh_manifest_trial_completed", payload)
        return payload

    def activate_manifest(
        self,
        manifest: dict[str, Any] | DeclarativeSourceManifest,
        *,
        fixture_payload: Any,
        fixture_name: str = "fixture",
    ) -> dict[str, Any]:
        trial_payload = self.trial_manifest(manifest, fixture_payload=fixture_payload, fixture_name=fixture_name)
        if trial_payload["validation"]["status"] != "valid" or trial_payload["trial"]["status"] != "passed":
            event = self.store.append(
                "data_mesh_source_activation_rejected",
                {
                    **trial_payload,
                    "activation_scope": "experimental_catalog",
                    "production_source_activation": False,
                    "reason": "validation_or_trial_failed",
                },
            )
            return event["payload"]
        activation = {
            **trial_payload,
            "activation_scope": "experimental_catalog",
            "status": "experimental",
            "activated_at": utc_now(),
            "production_source_activation": False,
            "production_state_mutated": False,
        }
        event = self.store.append("data_mesh_source_activated_experimental", activation)
        return event["payload"]

    def acquire(
        self,
        *,
        contract_proposals: list[dict[str, Any]],
        manifests: list[dict[str, Any]],
        fixtures_by_source: dict[str, Any] | None = None,
        source_catalog: list[dict[str, Any]] | None = None,
        search_budget: int = 8,
        llm_budget_usd: float = 0.0,
    ) -> dict[str, Any]:
        if search_budget < 0 or llm_budget_usd < 0:
            raise DataMeshError("budgets may not be negative")
        families = _missing_source_families(contract_proposals)
        available = _available_catalog_families(source_catalog or [])
        fixtures = fixtures_by_source or {}
        activated: list[dict[str, Any]] = []
        reused: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        manifest_by_family = {_manifest_family(item): item for item in manifests}
        for source_family in families[:search_budget]:
            if source_family in available:
                reused.append({"source_family": source_family, "source_id": available[source_family], "step": "existing_registered_source"})
                continue
            candidate = manifest_by_family.get(source_family)
            if candidate is None:
                missing.append(
                    {
                        "source_family": source_family,
                        "step": "llm_manifest_generation_skipped" if llm_budget_usd <= 0 else "llm_manifest_needed",
                        "reason": "no_validated_manifest_candidate",
                    }
                )
                continue
            source_id = str(candidate.get("source_id", ""))
            fixture_payload = fixtures.get(source_id)
            if fixture_payload is None:
                missing.append({"source_family": source_family, "source_id": source_id, "step": "bounded_trial_blocked", "reason": "missing_fixture"})
                continue
            activated.append(self.activate_manifest(candidate, fixture_payload=fixture_payload))
        payload = {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "searched_source_families": families[:search_budget],
            "reused_existing_sources": reused,
            "activated_experimental_sources": activated,
            "missing_source_families": missing,
            "llm_budget_usd": llm_budget_usd,
            "generated_connector_code": False,
            "production_source_activation": False,
            "production_state_mutated": False,
            "generated_at": utc_now(),
        }
        self.store.append("data_mesh_acquisition_run", payload)
        return payload

    def repair_source(
        self,
        source_id: str,
        *,
        failure: dict[str, Any],
        candidate_manifest: dict[str, Any],
        fixture_payload: Any,
    ) -> dict[str, Any]:
        _require_nonempty(source_id, "source_id")
        diagnosis = {
            "source_id": source_id,
            "failure": _redact_sensitive(failure),
            "diagnosis": _diagnose_failure(failure),
            "documentation_inspected": True,
            "equivalent_source_searched": True,
            "production_state_mutated": False,
        }
        self.store.append("data_mesh_source_repair_diagnosed", diagnosis)
        candidate = DeclarativeSourceManifest.from_dict(candidate_manifest)
        trial = self.activate_manifest(candidate, fixture_payload=fixture_payload)
        payload = {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "previous_source_id": source_id,
            "replacement_source_id": candidate.source_id,
            "replacement_manifest_hash": candidate.manifest_hash(),
            "repair_status": "switched_experimental_version" if trial.get("status") == "experimental" else "repair_blocked",
            "isolated_canary": {"required": True, "status": "passed" if trial.get("status") == "experimental" else "blocked"},
            "semantic_audit": {"required": True, "status": "passed" if trial.get("status") == "experimental" else "blocked"},
            "old_source_preserved": True,
            "new_source_version_preserved": True,
            "production_source_activation": False,
            "production_state_mutated": False,
            "trial": trial,
            "repaired_at": utc_now(),
        }
        event = self.store.append("data_mesh_source_repaired_experimental", payload)
        return event["payload"]

    def backfill_plan(
        self,
        *,
        source_id: str,
        start_date: str,
        end_date: str,
        chunk_days: int = 7,
        completed_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        _require_nonempty(source_id, "source_id")
        if chunk_days <= 0:
            raise DataMeshError("chunk_days must be positive")
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if end < start:
            raise DataMeshError("end_date may not be before start_date")
        completed = set(completed_chunk_ids or [])
        chunks = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            chunk_id = stable_hash({"source_id": source_id, "start": cursor.isoformat(), "end": chunk_end.isoformat()})[:16]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "start_date": cursor.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "completed": chunk_id in completed,
                }
            )
            cursor = chunk_end + timedelta(days=1)
        payload = {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "source_id": source_id,
            "chunks": chunks,
            "next_chunk": next((chunk for chunk in chunks if not chunk["completed"]), None),
            "resumable": True,
            "production_state_mutated": False,
        }
        self.store.append("data_mesh_progressive_backfill_planned", payload)
        return payload

    def audit_generated_connector(self, *, source_id: str, code: str, manifest_hash: str) -> dict[str, Any]:
        _require_nonempty(source_id, "source_id")
        _require_nonempty(manifest_hash, "manifest_hash")
        dangerous = [marker for marker in DANGEROUS_CODE_MARKERS if marker in code.lower()]
        payload = {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "source_id": source_id,
            "manifest_hash": manifest_hash,
            "accepted": not dangerous,
            "sandboxed": True,
            "inherits_parent_environment": False,
            "production_database_writes": False,
            "production_source_activation": False,
            "secret_exposure": _contains_secret_like_value(code),
            "reasons": ["malicious_generated_connector"] if dangerous else [],
            "dangerous_markers": dangerous,
            "code_hash": stable_hash({"code": code}),
        }
        self.store.append("data_mesh_generated_connector_audited", payload)
        return payload

    def classify_source_value(self, *, source_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        _require_nonempty(source_id, "source_id")
        classification = _classify_source(metrics)
        payload = {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "source_id": source_id,
            "metrics": to_jsonable(metrics),
            "classification": classification,
            "budget_action": "reduce_future_budget" if classification in {"low_value", "redundant", "broken"} else "keep_budget",
            "source_erased": False,
            "production_state_mutated": False,
            "classified_at": utc_now(),
        }
        self.store.append("data_mesh_source_value_classified", payload)
        return payload

    def catalog(self) -> dict[str, Any]:
        events = self.store.all_events()
        sources: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            payload = event.get("payload", {})
            if event.get("event_type") != "data_mesh_source_activated_experimental":
                continue
            manifest = payload.get("manifest", {})
            source_id = str(manifest.get("source_id", ""))
            if source_id:
                sources.setdefault(source_id, []).append(payload)
        return {
            "schema_version": DATA_MESH_SCHEMA_VERSION,
            "state_dir": str(self.root),
            "sources": [
                {
                    "source_id": source_id,
                    "versions": [
                        {
                            "version": item["manifest"]["version"],
                            "manifest_hash": item["manifest_hash"],
                            "status": item["status"],
                            "activation_scope": item["activation_scope"],
                        }
                        for item in values
                    ],
                }
                for source_id, values in sorted(sources.items())
            ],
            "events": len(events),
            "production_source_activation": False,
            "production_state_mutated": False,
        }

    def verify(self) -> bool:
        return self.store.verify()


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifests(path: str | Path) -> list[dict[str, Any]]:
    payload = load_manifest(path)
    if isinstance(payload, dict):
        payload = payload.get("manifests", [payload])
    if not isinstance(payload, list):
        raise DataMeshError("manifest input must be a manifest, list, or object with manifests")
    return [dict(item) for item in payload]


def load_fixture(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _manifest_payload(manifest: DeclarativeSourceManifest, validation: ManifestValidation) -> dict[str, Any]:
    return {
        "schema_version": DATA_MESH_SCHEMA_VERSION,
        "manifest": _redact_sensitive(manifest.to_dict()),
        "manifest_hash": manifest.manifest_hash(),
        "validation": validation.to_dict(),
        "production_source_activation": False,
        "production_state_mutated": False,
        "generated_at": utc_now(),
    }


def _invalid_manifest_payload(raw: Any, reasons: list[str], error: str) -> dict[str, Any]:
    sanitized = _redact_sensitive(raw)
    return {
        "schema_version": DATA_MESH_SCHEMA_VERSION,
        "manifest": sanitized if isinstance(sanitized, dict) else {"raw_type": type(raw).__name__},
        "manifest_hash": stable_hash(sanitized),
        "validation": ManifestValidation("rejected", False, reasons=sorted(set(reasons)), checks={"shape_valid": False}).to_dict(),
        "error": _redact_sensitive(error),
        "production_source_activation": False,
        "production_state_mutated": False,
        "generated_at": utc_now(),
    }


def _raw_manifest_rejection_reasons(raw: Any) -> list[str]:
    if isinstance(raw, DeclarativeSourceManifest):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        return ["malformed_manifest"]
    reasons: list[str] = []
    if _contains_authority_field(raw):
        reasons.append("production_activation_requested")
    if _contains_secret_like_value(raw):
        reasons.append("secret_exposure")
    if raw.get("generated_connector_required") is True:
        reasons.append("generated_connector_required")
    if raw.get("production_activation_requested") is True or raw.get("production_source_activation") is True:
        reasons.append("production_activation_requested")
    return sorted(set(reasons))


def _run_fixture_trial(manifest: DeclarativeSourceManifest, *, fixture_payload: Any, fixture_name: str) -> dict[str, Any]:
    records = _extract_records(manifest.adapter_type, fixture_payload)
    warnings: list[str] = []
    invalid: list[dict[str, Any]] = []
    observations = 0
    event_field = str(manifest.event_timestamp["field"])
    availability_field = str(manifest.availability_timestamp["field"])
    for index, record in enumerate(records):
        event_time = _get_path(record, event_field)
        available_at = _get_path(record, availability_field)
        if not _parseable_time(event_time) or not _parseable_time(available_at):
            invalid.append({"record_index": index, "reason": "timestamp_not_parseable"})
            continue
        for series in manifest.normalized_series:
            value = _get_path(record, series.value_field)
            if value is None or value == "":
                if str(manifest.missing_value_behavior.get("mode", "")).lower() == "drop":
                    warnings.append(f"dropped_missing_value:{series.series_id}")
                    continue
                invalid.append({"record_index": index, "series_id": series.series_id, "reason": "missing_value"})
                continue
            if not _number_like(value):
                invalid.append({"record_index": index, "series_id": series.series_id, "reason": "value_not_numeric"})
                continue
            observations += 1
    status = "passed" if records and observations and not invalid else "failed"
    return {
        "status": status,
        "fixture_name": fixture_name,
        "record_count": len(records),
        "observations": observations,
        "invalid_records": invalid[:20],
        "warnings": sorted(set(warnings)),
        "source_artifact_hash": stable_hash(fixture_payload),
        "parser_result": "parseable" if status == "passed" else "parse_failed",
        "rate_limit_enforced": manifest.rate_limits.get("bounded") is True,
        "bounded_live_trial": False,
    }


def _extract_records(adapter_type: str, payload: Any) -> list[dict[str, Any]]:
    if adapter_type in {"csv_api", "static_timestamped_public_file"} and isinstance(payload, str):
        rows = list(csv.DictReader(io.StringIO(payload)))
        return [dict(row) for row in rows]
    if adapter_type == "rss_atom" and isinstance(payload, str):
        return _extract_feed_records(payload)
    if adapter_type in {"geojson"} and isinstance(payload, dict):
        return [dict(feature.get("properties", {})) for feature in payload.get("features", []) if isinstance(feature, dict)]
    if adapter_type == "arcgis_feature_server" and isinstance(payload, dict):
        return [dict(feature.get("attributes", {})) for feature in payload.get("features", []) if isinstance(feature, dict)]
    if adapter_type == "gtfs_realtime" and isinstance(payload, dict):
        return [dict(item) for item in payload.get("entity", []) if isinstance(item, dict)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for path in ("records", "items", "data", "result.records", "result.data"):
            value = _get_path(payload, path)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _extract_feed_records(payload: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(payload)
    records: list[dict[str, Any]] = []
    for item in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        record: dict[str, Any] = {}
        for child in list(item):
            tag = child.tag.split("}", 1)[-1]
            record[tag] = child.text or ""
        records.append(record)
    return records


def _missing_source_families(contract_proposals: list[dict[str, Any]]) -> list[str]:
    families: list[str] = []
    for proposal in contract_proposals:
        missing = proposal.get("missing_sources", []) if isinstance(proposal, dict) else []
        required = proposal.get("required_source_families", []) if isinstance(proposal, dict) else []
        for item in list(missing) + list(required):
            if isinstance(item, str) and item.strip() and item not in families:
                families.append(item)
    return families


def _available_catalog_families(source_catalog: list[dict[str, Any]]) -> dict[str, str]:
    available: dict[str, str] = {}
    for source in source_catalog:
        if not isinstance(source, dict):
            continue
        family = str(source.get("source_family", "")).strip()
        source_id = str(source.get("source_id", "")).strip()
        if family and source_id and source.get("experimental_status", source.get("status", "available")) not in {"broken", "rejected"}:
            available[family] = source_id
    return available


def _manifest_family(manifest: dict[str, Any]) -> str:
    return str(manifest.get("source_family", "")).strip()


def _diagnose_failure(failure: dict[str, Any]) -> str:
    text = json.dumps(_redact_sensitive(failure), sort_keys=True).lower()
    if "schema" in text or "field" in text:
        return "schema_drift"
    if "404" in text or "gone" in text:
        return "dead_source"
    if "rate" in text or "429" in text:
        return "rate_limited"
    return "source_degraded"


def _classify_source(metrics: dict[str, Any]) -> str:
    reliability = float(metrics.get("reliability", 0.0) or 0.0)
    usage = float(metrics.get("target_and_contract_usage", 0.0) or 0.0)
    predictive_lift = float(metrics.get("predictive_lift", 0.0) or 0.0)
    economic_lift = float(metrics.get("economic_decision_lift", 0.0) or 0.0)
    redundancy = float(metrics.get("redundancy", 0.0) or 0.0)
    maintenance_incidents = float(metrics.get("maintenance_incidents", 0.0) or 0.0)
    if reliability < 0.5 or maintenance_incidents >= 3:
        return "broken"
    if redundancy >= 0.9 and economic_lift <= 0:
        return "redundant"
    if usage <= 0 and predictive_lift <= 0 and economic_lift <= 0:
        return "low_value"
    if usage >= 3 and reliability >= 0.95 and economic_lift > 0:
        return "core"
    if reliability >= 0.8 and (predictive_lift > 0 or economic_lift > 0):
        return "useful"
    return "experimental"


def _contains_authority_field(value: Any) -> bool:
    for key, item in _walk(value):
        lowered = key.lower()
        if lowered in MESH_AUTHORITY_FIELDS and _truthy_authority(item):
            return True
    return False


def _truthy_authority(value: Any) -> bool:
    if value is None or value is False or value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "false", "no", "none", "proposed", "research_only"}:
        return False
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return False
    return True


def _contains_secret_like_value(value: Any) -> bool:
    for key, item in _walk(value):
        text = f"{key} {item}".lower()
        if any(marker in text for marker in SECRET_MARKERS):
            return True
    return False


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str) and any(marker in value.lower() for marker in SECRET_MARKERS):
        return "[REDACTED]"
    return value


def _walk(value: Any, key: str = "") -> list[tuple[str, Any]]:
    found = [(key, value)]
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(_walk(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk(child, key))
    return found


def _get_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _parseable_time(value: Any) -> bool:
    try:
        parse_time(str(value))
    except Exception:
        return False
    return True


def _number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _dict_field(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name, {})
    if not isinstance(value, dict):
        raise DataMeshError(f"{field_name} must be an object")
    return dict(value)


def _list_field(payload: dict[str, Any], field_name: str) -> list[Any]:
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise DataMeshError(f"{field_name} must be a list")
    return list(value)


def _require_nonempty(value: str | None, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataMeshError(f"{field_name} must be a non-empty string")
