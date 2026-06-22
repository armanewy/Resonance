from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
import subprocess
from typing import Any

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.money.accounting import maximum_drawdown
from behavior_lab.money.canary import CanaryOptions, MoneyCanaryError, MoneyCanaryManager, REAL_ACTION_FLAGS
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


OPERATIONS_SCHEMA_VERSION = "money_operations_release.v1"
DEFAULT_STATE_DIR = ".money_operations"
DEFAULT_RELEASE_COMMIT = "283a51b8000a2165f648b6aa4a5f3291e1206b60"
LAB_CONTRACTS = {
    "weather_edge": "operations-weather-edge",
    "etf_risk": "operations-etf-risk",
    "offerlab_seller_pilot": "operations-offerlab-seller",
}
LAB_CADENCE_DAYS = {"weather_edge": 1, "etf_risk": 7, "offerlab_seller_pilot": 1}
PAPER_NOTICE = "PAPER/SHADOW ONLY - NO REAL ACTION EXECUTED"


class MoneyOperationsError(ValueError):
    pass


@dataclass(frozen=True)
class OperationsOptions:
    state_dir: str | Path = DEFAULT_STATE_DIR
    as_of: str | None = None
    seller_readiness_report: str | Path | None = None
    release_commit: str | None = None


class MoneyOperations:
    def __init__(self, state_dir: str | Path = DEFAULT_STATE_DIR) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = AppendOnlyResearchStore(self.root / "operations.jsonl")
        self.canary_dir = self.root / "canaries"
        self.manifest_path = self.root / "release_manifest.json"
        self.lock_path = self.root / "operations.lock.json"
        self.source_health_path = self.root / "source_health.json"

    def start(self, *, as_of: str | None = None, seller_readiness_report: str | Path | None = None, release_commit: str | None = None) -> dict[str, Any]:
        timestamp = as_of or utc_now()
        parse_time(timestamp)
        if self._lock_exists():
            raise MoneyOperationsError("money operations already running; use status, stop, or recover")
        seller_ready = self._seller_ready(seller_readiness_report)
        manager = MoneyCanaryManager(self.canary_dir)
        canaries = []
        for lab, contract_id in LAB_CONTRACTS.items():
            options = CanaryOptions(
                lab=lab,
                as_of=timestamp,
                seller_pilot_ready=lab != "offerlab_seller_pilot" or seller_ready["passed"],
            )
            try:
                canaries.append(manager.start(contract_id, options))
            except MoneyCanaryError as exc:
                canaries.append(
                    {
                        "schema_version": "money_canary.v1",
                        "contract_id": contract_id,
                        "lab": lab,
                        "status": "blocked",
                        "reason": str(exc),
                        "paper_only": True,
                        "production_state": dict(REAL_ACTION_FLAGS),
                    }
                )
        manifest = _release_manifest(
            root=self.root,
            canaries=canaries,
            release_commit=release_commit or _git_commit() or DEFAULT_RELEASE_COMMIT,
            seller_readiness=seller_ready,
        )
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("release_hash") != manifest["release_hash"]:
                raise MoneyOperationsError("existing release manifest differs; material changes require a new state directory and new canaries")
            manifest = existing
        else:
            self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock = {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "started_at": timestamp,
            "state_dir": str(self.root),
            "release_hash": manifest["release_hash"],
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        self.lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.store.append("operations_started", _base_payload({"lock": lock, "release_hash": manifest["release_hash"]}))
        return {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "status": "running",
            "state_dir": str(self.root),
            "manifest": manifest,
            "lock": lock,
            "canaries": canaries,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def status(self) -> dict[str, Any]:
        manifest = self._manifest()
        canary_reports = self._canary_reports(manifest)
        source_health = self._source_health()
        missed = self._missed_cycles(canary_reports)
        return {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "status": "running" if self._lock_exists() else "stopped",
            "state_dir": str(self.root),
            "state_dir_is_absolute": self.root.is_absolute(),
            "release_hash": manifest.get("release_hash"),
            "manifest": manifest,
            "canaries": canary_reports,
            "missed_scheduled_cycles": missed,
            "source_health": source_health,
            "target_freshness": _target_freshness(canary_reports),
            "ledger_valid": self.store.verify(),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def doctor(self) -> dict[str, Any]:
        status = self.status()
        issues: list[dict[str, Any]] = []
        manifest = status["manifest"]
        if not status["state_dir_is_absolute"]:
            issues.append({"severity": "error", "code": "state_dir_not_absolute"})
        if not self.manifest_path.exists():
            issues.append({"severity": "error", "code": "missing_release_manifest"})
        if not self.store.verify():
            issues.append({"severity": "error", "code": "operations_store_hash_chain_invalid"})
        manager = MoneyCanaryManager(self.canary_dir)
        if not manager.verify():
            issues.append({"severity": "error", "code": "canary_store_hash_chain_invalid"})
        for lab, expected in manifest.get("canary_hashes", {}).items():
            report = status["canaries"].get(lab)
            if not report:
                issues.append({"severity": "error", "code": "missing_canary", "lab": lab})
                continue
            actual = stable_hash(report["protocol"]) if report.get("protocol") else None
            if actual != expected.get("protocol_hash"):
                issues.append({"severity": "error", "code": "canary_hash_mismatch", "lab": lab})
            if report.get("final_evidence_report", {}).get("real_money_allowed") is not False:
                issues.append({"severity": "error", "code": "real_money_allowed", "lab": lab})
        for lab, health in status["source_health"].items():
            if health.get("healthy") is not True:
                issues.append({"severity": "warning", "code": "source_unhealthy", "lab": lab})
            if health.get("stale") is True:
                issues.append({"severity": "warning", "code": "source_stale", "lab": lab})
        return {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "healthy": not any(issue["severity"] == "error" for issue in issues),
            "issues": issues,
            "status": status,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def weekly_report(self) -> dict[str, Any]:
        status = self.status()
        canary_reports = status["canaries"]
        resolved_decisions = 0
        unresolved_decisions = 0
        paper_value = 0.0
        shadow_value = 0.0
        capital = 0.0
        values = []
        no_action = 0
        decisions = 0
        calibration = {}
        for lab, report in canary_reports.items():
            for item in report.get("decision_history", []):
                decisions += 1
                if str(item.get("selected_action")) in {"cash", "no_action", "no_trade", "paper_buy_yes_or_no_trade"}:
                    no_action += 1
                capital += float(item.get("capital_required") or 0.0)
            for value in report.get("pnl_savings_history", []):
                pnl = float(value.get("paper_pnl") or 0.0)
                savings = float(value.get("seller_shadow_savings") or 0.0)
                paper_value += pnl
                shadow_value += savings
                values.append(pnl + savings)
            unresolved_decisions += len(report.get("decision_history", []))
            resolved_decisions += len(report.get("resolution_history", []))
            calibration[lab] = _calibration_summary(report)
        payload = {
            "schema_version": "money_operations_weekly_report.v1",
            "notice": PAPER_NOTICE,
            "paper_or_shadow_value": round(paper_value + shadow_value, 2),
            "paper_value": round(paper_value, 2),
            "seller_shadow_value": round(shadow_value, 2),
            "resolved_decisions": resolved_decisions,
            "unresolved_decisions": unresolved_decisions,
            "no_action_rate": round(no_action / decisions, 6) if decisions else 0.0,
            "capital_hypothetically_at_risk": round(capital, 2),
            "drawdown": maximum_drawdown(_cumulative(values))["maximum_drawdown"],
            "calibration": calibration,
            "source_health": status["source_health"],
            "research_and_api_cost": 0.0,
            "canary_comparability": _canary_comparability(canary_reports),
            "seller_data_readiness": status["manifest"].get("seller_readiness"),
            "approvals_required": _approvals_required(status),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        report_hash = stable_hash(payload)
        if not any(event["payload"].get("report_hash") == report_hash for event in self.store.all_events()):
            self.store.append("operations_weekly_report", _base_payload({**payload, "report_hash": report_hash}))
        return payload

    def stop(self) -> dict[str, Any]:
        was_running = self._lock_exists()
        if was_running:
            self.lock_path.unlink()
        payload = {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "status": "stopped",
            "was_running": was_running,
            "stopped_at": utc_now(),
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        self.store.append("operations_stopped", _base_payload(payload))
        return payload

    def recover(self, *, as_of: str | None = None, strategy_versions: dict[str, str] | None = None) -> dict[str, Any]:
        timestamp = as_of or utc_now()
        parse_time(timestamp)
        manifest = self._manifest()
        if not self._lock_exists():
            self.lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": OPERATIONS_SCHEMA_VERSION,
                        "recovered_at": timestamp,
                        "state_dir": str(self.root),
                        "release_hash": manifest.get("release_hash"),
                        "paper_only": True,
                        "production_state": dict(REAL_ACTION_FLAGS),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        manager = MoneyCanaryManager(self.canary_dir)
        resumed = []
        skipped = []
        for lab, item in manifest.get("canary_hashes", {}).items():
            canary_id = item.get("canary_id")
            if not canary_id:
                skipped.append({"lab": lab, "reason": "canary_not_started"})
                continue
            health = self._source_health().get(lab, {"healthy": True, "stale": False})
            if health.get("healthy") is not True or health.get("stale") is True:
                skipped.append({"lab": lab, "reason": "stale_or_unhealthy_source"})
                continue
            report = manager.report(canary_id)
            if not _cycle_due(lab, report["metrics"], timestamp, report["protocol"]["start_at"]):
                skipped.append({"lab": lab, "reason": "not_due"})
                continue
            try:
                resumed.append(manager.resume(canary_id, as_of=timestamp, strategy_version=(strategy_versions or {}).get(lab)))
            except MoneyCanaryError as exc:
                skipped.append({"lab": lab, "reason": "frozen_canary_rejected_change", "error": str(exc)})
        payload = {
            "schema_version": OPERATIONS_SCHEMA_VERSION,
            "status": "recovered",
            "as_of": timestamp,
            "resumed": resumed,
            "skipped": skipped,
            "blind_evaluation_repeated": False,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }
        self.store.append("operations_recovered", _base_payload(payload))
        return payload

    def record_source_health(self, lab: str, *, healthy: bool, stale: bool = False, reason: str = "") -> dict[str, Any]:
        if lab not in LAB_CONTRACTS:
            raise MoneyOperationsError(f"unsupported lab: {lab}")
        payload = self._source_health()
        payload[lab] = {"healthy": bool(healthy), "stale": bool(stale), "reason": reason, "checked_at": utc_now()}
        self.source_health_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload[lab]

    def _lock_exists(self) -> bool:
        return self.lock_path.exists()

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise MoneyOperationsError("release manifest not found; run operations start first")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _canary_reports(self, manifest: dict[str, Any]) -> dict[str, Any]:
        manager = MoneyCanaryManager(self.canary_dir)
        reports = {}
        for lab, item in manifest.get("canary_hashes", {}).items():
            canary_id = item.get("canary_id")
            if canary_id:
                reports[lab] = manager.report(canary_id)
        return reports

    def _source_health(self) -> dict[str, Any]:
        if self.source_health_path.exists():
            return json.loads(self.source_health_path.read_text(encoding="utf-8"))
        return {
            lab: {"healthy": True, "stale": False, "reason": "default_fixture_source", "checked_at": None}
            for lab in LAB_CONTRACTS
        }

    def _missed_cycles(self, reports: dict[str, Any]) -> list[dict[str, Any]]:
        now = utc_now()
        missed = []
        for lab, report in reports.items():
            metrics = report["metrics"]
            if _cycle_due(lab, metrics, now, report["protocol"]["start_at"]):
                missed.append({"lab": lab, "last_snapshot_at": _last_observed_at(report), "cadence_days": LAB_CADENCE_DAYS[lab]})
        return missed

    def _seller_ready(self, report_path: str | Path | None) -> dict[str, Any]:
        if report_path is None:
            return {"passed": False, "reason": "no_seller_readiness_report", "canary_start_allowed": False}
        path = Path(report_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate = payload.get("readiness_gate") or payload.get("data_readiness", {}).get("readiness_gate") or {}
        passed = bool(gate.get("passed"))
        return {
            "passed": passed,
            "report_hash": stable_hash(payload),
            "report_path_name": path.name,
            "canary_start_allowed": passed,
            "reason": "readiness_gate_passed" if passed else "readiness_gate_failed",
        }


def _release_manifest(*, root: Path, canaries: list[dict[str, Any]], release_commit: str, seller_readiness: dict[str, Any]) -> dict[str, Any]:
    canary_hashes = {}
    contract_hashes = {}
    for item in canaries:
        lab = str(item.get("lab"))
        protocol = item.get("protocol")
        if not protocol:
            canary_hashes[lab] = {"status": item.get("status"), "reason": item.get("reason")}
            continue
        canary_hashes[lab] = {
            "canary_id": protocol["canary_id"],
            "protocol_hash": stable_hash(protocol),
            "material_hash": protocol["material_hash"],
            "program_hash": protocol["frozen_strategy"]["program_hash"],
        }
        contract_hashes[lab] = stable_hash({"contract_id": protocol["contract_id"], "lab": lab})
    manifest = {
        "schema_version": OPERATIONS_SCHEMA_VERSION,
        "release_commit": release_commit,
        "state_dir": str(root),
        "contract_hashes": contract_hashes,
        "canary_hashes": canary_hashes,
        "model_program_hashes": {
            lab: item.get("program_hash")
            for lab, item in canary_hashes.items()
            if isinstance(item, dict) and item.get("program_hash")
        },
        "source_versions": _collect_protocol_field(canaries, "source_versions"),
        "evidence_thresholds": _collect_protocol_field(canaries, "prospective_gates"),
        "cost_assumptions": _collect_protocol_field(canaries, "cost_assumptions"),
        "fee_slippage_assumptions": _fee_slippage(canaries),
        "start_dates": _collect_protocol_field(canaries, "start_at"),
        "scheduled_end_dates": _collect_protocol_field(canaries, "end_at"),
        "seller_readiness": seller_readiness,
        "paper_only": True,
        "production_state": dict(REAL_ACTION_FLAGS),
    }
    manifest["release_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "release_hash"})
    return manifest


def _collect_protocol_field(canaries: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    output = {}
    for item in canaries:
        protocol = item.get("protocol")
        if protocol:
            output[str(protocol["lab"])] = protocol.get(field_name)
    return output


def _fee_slippage(canaries: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for item in canaries:
        protocol = item.get("protocol")
        if protocol:
            assumptions = protocol.get("cost_assumptions", {})
            output[str(protocol["lab"])] = {
                "fees": assumptions.get("fees_included"),
                "slippage": assumptions.get("slippage_included"),
                "turnover_costs": assumptions.get("turnover_costs_included"),
            }
    return output


def _base_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": OPERATIONS_SCHEMA_VERSION, **payload}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _last_observed_at(report: dict[str, Any]) -> str:
    history = report.get("source_health_history", [])
    if not history:
        return str(report["protocol"]["start_at"])
    return str(history[-1]["checked_at"])


def _cycle_due(lab: str, metrics: dict[str, Any], as_of: str, start_at: str) -> bool:
    last = parse_time(start_at) + timedelta(days=max(0, int(metrics.get("elapsed_days", 1)) - 1))
    return (parse_time(as_of).date() - last.date()).days >= LAB_CADENCE_DAYS[lab]


def _target_freshness(reports: dict[str, Any]) -> dict[str, Any]:
    return {
        lab: {
            "last_observed_at": _last_observed_at(report),
            "stale": _cycle_due(lab, report["metrics"], utc_now(), report["protocol"]["start_at"]),
        }
        for lab, report in reports.items()
    }


def _calibration_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report["lab"] == "weather_edge":
        return {"tracked": True, "metrics": ["brier_score", "log_loss"]}
    if report["lab"] == "etf_risk":
        return {"tracked": True, "metrics": ["realized_volatility_forecasts"]}
    return {"tracked": False, "reason": "seller_shadow_requires_mature_outcomes"}


def _canary_comparability(reports: dict[str, Any]) -> dict[str, Any]:
    return {
        lab: {
            "material_hash": report["protocol"]["material_hash"],
            "program_hash": report["protocol"]["frozen_strategy"]["program_hash"],
            "strategy_version": report["protocol"]["frozen_strategy"]["strategy_version"],
            "final_evidence_available": report["final_evidence_report"]["available"],
        }
        for lab, report in reports.items()
    }


def _approvals_required(status: dict[str, Any]) -> list[dict[str, Any]]:
    approvals = []
    for lab, health in status["source_health"].items():
        if health.get("healthy") is not True or health.get("stale") is True:
            approvals.append({"lab": lab, "reason": health.get("reason", "source_health_issue")})
    seller = status["manifest"].get("seller_readiness", {})
    if seller.get("passed") is not True:
        approvals.append({"lab": "offerlab_seller_pilot", "reason": seller.get("reason", "seller_readiness_not_passed")})
    return approvals


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    output = []
    for value in values:
        total = round(total + value, 2)
        output.append(total)
    return output
