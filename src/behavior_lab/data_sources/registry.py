from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from behavior_lab.core import parse_time, stable_hash


class DataSourceError(ValueError):
    pass


@dataclass(frozen=True)
class DataSource:
    source_id: str
    name: str
    role: str
    license_status: str
    license_url: str | None
    source_url: str
    allowed_uses: dict[str, bool]
    evidence_class: str
    integration_requirements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    requires_authorization_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PermissionCheck:
    source_id: str
    use: str
    allowed: bool
    reason: str
    authorization_evidence_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthorizationEvidence:
    source_id: str
    authorization_id: str
    owner_subject_hash: str
    authorized_at: str
    scopes: list[str]
    ledger_record_hash: str
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        authorization_id: str,
        owner_subject_hash: str,
        authorized_at: str,
        scopes: list[str],
        ledger_record_hash: str,
    ) -> "AuthorizationEvidence":
        body = {
            "source_id": source_id,
            "authorization_id": authorization_id,
            "owner_subject_hash": owner_subject_hash,
            "authorized_at": authorized_at,
            "scopes": sorted(set(scopes)),
            "ledger_record_hash": ledger_record_hash,
        }
        return cls(**body, evidence_hash=stable_hash(body))


def validate_authorization_evidence(
    evidence: AuthorizationEvidence | dict[str, Any],
    *,
    expected_source_id: str,
) -> AuthorizationEvidence:
    payload = evidence.to_dict() if isinstance(evidence, AuthorizationEvidence) else dict(evidence)
    required = {
        "source_id",
        "authorization_id",
        "owner_subject_hash",
        "authorized_at",
        "scopes",
        "ledger_record_hash",
        "evidence_hash",
    }
    missing = required - set(payload)
    if missing:
        raise DataSourceError(f"authorization evidence is missing fields: {sorted(missing)}")
    if payload["source_id"] != expected_source_id:
        raise DataSourceError("authorization evidence source_id does not match the data source")
    for key in ["authorization_id", "owner_subject_hash", "ledger_record_hash"]:
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise DataSourceError(f"authorization evidence {key} must be non-empty")
    parse_time(str(payload["authorized_at"]))
    scopes = payload["scopes"]
    if not isinstance(scopes, list) or not scopes or any(not isinstance(item, str) or not item.strip() for item in scopes):
        raise DataSourceError("authorization evidence scopes must be a non-empty list of strings")
    normalized = {
        "source_id": str(payload["source_id"]),
        "authorization_id": str(payload["authorization_id"]),
        "owner_subject_hash": str(payload["owner_subject_hash"]),
        "authorized_at": str(payload["authorized_at"]),
        "scopes": sorted(set(str(item) for item in scopes)),
        "ledger_record_hash": str(payload["ledger_record_hash"]),
    }
    if stable_hash(normalized) != str(payload["evidence_hash"]):
        raise DataSourceError("authorization evidence hash mismatch")
    return AuthorizationEvidence(**normalized, evidence_hash=str(payload["evidence_hash"]))


class SourceRegistry:
    def __init__(self, sources: list[DataSource]) -> None:
        by_id: dict[str, DataSource] = {}
        for source in sources:
            if source.source_id in by_id:
                raise DataSourceError(f"Duplicate data source {source.source_id!r}")
            by_id[source.source_id] = source
        self._sources = by_id

    def list(self) -> list[dict[str, Any]]:
        return [source.to_dict() for source in sorted(self._sources.values(), key=lambda item: item.source_id)]

    def get(self, source_id: str) -> DataSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise DataSourceError(f"Unknown data source {source_id!r}") from exc

    def inspect(self, source_id: str) -> dict[str, Any]:
        return self.get(source_id).to_dict()

    def permissions(
        self,
        source_id: str,
        *,
        authorization_evidence: AuthorizationEvidence | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = self.get(source_id)
        return {
            "source_id": source.source_id,
            "license_status": source.license_status,
            "requires_authorization_evidence": source.requires_authorization_evidence,
            "allowed_uses": dict(sorted(source.allowed_uses.items())),
            "production_export_allowed": self.check(
                source_id,
                "production_export",
                authorization_evidence=authorization_evidence,
            ).allowed,
        }

    def check(
        self,
        source_id: str,
        use: str,
        *,
        authorization_evidence: AuthorizationEvidence | dict[str, Any] | None = None,
    ) -> PermissionCheck:
        source = self.get(source_id)
        allowed = bool(source.allowed_uses.get(use, False))
        commercial_uses = {"commercial_training", "production_inference", "production_export"}
        if not allowed:
            return PermissionCheck(source_id, use, False, f"use {use!r} is not allowed for {source_id!r}")
        if use in commercial_uses and source.license_status != "confirmed":
            return PermissionCheck(source_id, use, False, f"license status is {source.license_status!r}, not confirmed")
        evidence_verified = False
        if use in commercial_uses and source.requires_authorization_evidence:
            if authorization_evidence is None:
                return PermissionCheck(
                    source_id,
                    use,
                    False,
                    "commercial use requires immutable account-authorization evidence",
                )
            try:
                validate_authorization_evidence(
                    authorization_evidence,
                    expected_source_id=source_id,
                )
            except (DataSourceError, ValueError) as exc:
                return PermissionCheck(source_id, use, False, f"invalid authorization evidence: {exc}")
            evidence_verified = True
        return PermissionCheck(
            source_id,
            use,
            True,
            "use allowed by registered source policy"
            + (" and verified authorization evidence" if evidence_verified else ""),
            authorization_evidence_verified=evidence_verified,
        )

    def verify_lineage(
        self,
        source_ids: list[str],
        requested_use: str,
        *,
        authorization_evidence_by_source: dict[str, AuthorizationEvidence | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not source_ids:
            return {
                "requested_use": requested_use,
                "allowed": False,
                "checks": [],
                "reason": "lineage must contain at least one source dataset",
            }
        evidence_map = authorization_evidence_by_source or {}
        checks = [
            self.check(
                source_id,
                requested_use,
                authorization_evidence=evidence_map.get(source_id),
            )
            for source_id in source_ids
        ]
        return {
            "requested_use": requested_use,
            "allowed": all(check.allowed for check in checks),
            "checks": [check.to_dict() for check in checks],
        }

    def verify_manifest_file(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        source_ids = payload.get("source_dataset_ids")
        if not isinstance(source_ids, list) or not all(isinstance(item, str) for item in source_ids):
            raise DataSourceError("manifest must contain source_dataset_ids as a list of strings")
        if not source_ids:
            raise DataSourceError("manifest source_dataset_ids may not be empty")
        requested_use = str(payload.get("requested_use", "internal_benchmarking"))
        evidence = payload.get("authorization_evidence_by_source")
        if evidence is not None and not isinstance(evidence, dict):
            raise DataSourceError("authorization_evidence_by_source must be an object")
        return self.verify_lineage(
            source_ids,
            requested_use,
            authorization_evidence_by_source=evidence,
        )


def default_sources() -> list[DataSource]:
    research_only = {
        "research": True,
        "internal_benchmarking": True,
        "commercial_training": False,
        "production_inference": False,
        "production_export": False,
    }
    simulation_only = {
        "research": True,
        "internal_benchmarking": True,
        "commercial_training": False,
        "production_inference": False,
        "production_export": False,
    }
    authorized_commercial = {
        "research": True,
        "internal_benchmarking": True,
        "commercial_training": True,
        "production_inference": True,
        "production_export": True,
    }
    return [
        DataSource(
            source_id="nber_ebay_best_offer",
            name="NBER Best Offer Sequential Bargaining",
            role="direct_evidence",
            license_status="uncertain",
            license_url="https://www.nber.org/research/data/best-offer-sequential-bargaining",
            source_url="https://www.nber.org/research/data/best-offer-sequential-bargaining",
            allowed_uses=research_only,
            evidence_class="direct eBay bargaining behavior",
            integration_requirements=["explicit download", "dataset citation", "commercial-use legal review"],
            notes=["Use for leakage-safe research benchmarks, not production model export."],
        ),
        DataSource(
            source_id="open_bandit_dataset",
            name="Open Bandit Dataset",
            role="evaluator_validation",
            license_status="confirmed",
            license_url="https://github.com/st-tech/zr-obp",
            source_url="https://zr-obp.readthedocs.io/en/latest/about.html",
            allowed_uses=research_only,
            evidence_class="off-policy evaluation benchmark",
            integration_requirements=["logged propensities", "policy support checks"],
            notes=["Use to validate OPE estimators; do not transfer e-commerce embeddings into OfferLab."],
        ),
        DataSource(
            source_id="criteo_uplift",
            name="Criteo Uplift Prediction Dataset",
            role="causal_validation",
            license_status="confirmed",
            license_url="https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
            source_url="https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
            allowed_uses=research_only,
            evidence_class="randomized uplift benchmark",
            integration_requirements=["noncommercial restriction", "negative controls"],
            notes=["Use to validate heterogeneous treatment-effect machinery only."],
        ),
        DataSource(
            source_id="auctionnet",
            name="AuctionNet",
            role="simulation",
            license_status="confirmed",
            license_url="https://github.com/alimama-tech/AuctionNet",
            source_url="https://github.com/alimama-tech/AuctionNet",
            allowed_uses=simulation_only,
            evidence_class="strategic ad-auction simulation",
            integration_requirements=["optional dependency", "simulation labeling"],
            notes=["Do not use AuctionNet as evidence about real eBay buyers."],
        ),
        DataSource(
            source_id="craigslist_bargain",
            name="CraigslistBargain",
            role="language_extraction",
            license_status="uncertain",
            license_url="https://github.com/stanfordnlp/cocoa/tree/master/craigslistbargain",
            source_url="https://huggingface.co/datasets/stanfordnlp/craigslist_bargains",
            allowed_uses=research_only,
            evidence_class="crowdworker negotiation dialogue",
            integration_requirements=["dialogue-act parser only", "commercial-use legal review"],
            notes=["Useful for text extraction; not acceptance-rate calibration."],
        ),
        DataSource(
            source_id="current_ebay_authorized_data",
            name="Current authorized eBay seller data",
            role="commercial_calibration",
            license_status="confirmed",
            license_url="https://developer.ebay.com/develop/guides-v2/authorization",
            source_url="https://developer.ebay.com/",
            allowed_uses=authorized_commercial,
            evidence_class="authorized seller production data",
            integration_requirements=[
                "OAuth consent",
                "official APIs only",
                "seller cost-basis import",
                "immutable authorization evidence",
            ],
            notes=["Production permission is granted only when an immutable authorization record is supplied."],
            requires_authorization_evidence=True,
        ),
    ]


def default_registry() -> SourceRegistry:
    return SourceRegistry(default_sources())
