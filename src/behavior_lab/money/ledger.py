from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from behavior_lab.core import parse_time, stable_hash, to_jsonable, utc_now
from behavior_lab.ledger import DuplicateRecordError, ImmutableLedger
from behavior_lab.money.contracts import EVIDENCE_STATES


MONEY_LEDGER_RECORD_TYPE = "money_ledger_entry"
DESIGNATIONS = {"paper", "real"}
REAL_STATES = {"manually_approved_real", "resolved_real"}
RESOLVED_STATES = {"resolved_paper", "resolved_real"}
INITIAL_STATES = {
    "proposed",
    "historically_evaluated",
    "blind_passed",
    "prospectively_incubating",
    "prospectively_verified",
    "paper_decision",
    "rejected",
    "expired",
}
MATERIAL_ENTRY_COST_FIELDS = (
    "fees",
    "slippage",
    "shipping",
    "taxes_or_tax_assumption_reference",
    "holding_costs",
    "return_refund_allowance",
    "research_api_cost",
)
IMMUTABLE_DECISION_FIELDS = (
    "contract_hash",
    "decision_timestamp",
    "data_cutoff",
    "target",
    "action_alternatives",
    "selected_action",
    "no_action_alternative",
    "capital_required",
    "maximum_possible_loss",
    "expected_gross_value",
    "uncertainty_adjustment",
    "fees",
    "slippage",
    "shipping",
    "taxes_or_tax_assumption_reference",
    "holding_costs",
    "return_refund_allowance",
    "research_api_cost",
    "conservative_expected_net_value",
    "decision_deadline",
    "feature_program_hash",
    "designation",
    "assumption_versions",
    "material_costs_known",
    "ineligibility_reasons",
)


class MoneyLedgerError(ValueError):
    pass


@dataclass(frozen=True)
class MoneyLedgerEntry:
    decision_id: str
    contract_hash: str
    decision_timestamp: str
    data_cutoff: str
    target: dict[str, Any]
    action_alternatives: list[str]
    selected_action: str
    no_action_alternative: str
    capital_required: float
    maximum_possible_loss: float
    expected_gross_value: float
    uncertainty_adjustment: float
    fees: float | None
    slippage: float | None
    shipping: float | None
    taxes_or_tax_assumption_reference: float | str | None
    holding_costs: float | None
    return_refund_allowance: float | None
    research_api_cost: float | None
    conservative_expected_net_value: float | None
    decision_deadline: str
    feature_program_hash: str
    evidence_state: str
    designation: str
    resolution: dict[str, Any] | None = None
    realized_gross_value: float | None = None
    realized_net_value: float | None = None
    mechanically_defined_no_action_outcome: dict[str, Any] | None = None
    economic_event_key: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    assumption_versions: dict[str, str] = field(default_factory=dict)
    material_costs_known: bool = True
    ineligibility_reasons: list[str] = field(default_factory=list)
    supersedes_entry_hash: str | None = None
    correction_reason: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_nonempty(self.decision_id, "decision_id")
        _require_nonempty(self.contract_hash, "contract_hash")
        _require_nonempty(self.feature_program_hash, "feature_program_hash")
        parse_time(self.decision_timestamp)
        cutoff = parse_time(self.data_cutoff)
        decision = parse_time(self.decision_timestamp)
        deadline = parse_time(self.decision_deadline)
        parse_time(self.created_at)
        if cutoff > decision:
            raise MoneyLedgerError("data_cutoff may not be after decision_timestamp")
        if decision > deadline:
            raise MoneyLedgerError("decision_timestamp may not be after decision_deadline")
        if self.evidence_state not in EVIDENCE_STATES:
            raise MoneyLedgerError(f"unknown evidence_state: {self.evidence_state}")
        if self.evidence_state == "manually_approved_real":
            raise MoneyLedgerError("this wave cannot create manually_approved_real entries")
        if self.designation not in DESIGNATIONS:
            raise MoneyLedgerError("designation must be paper or real")
        if self.designation == "real":
            raise MoneyLedgerError("this wave cannot create real money ledger entries")
        if self.designation == "paper" and self.evidence_state in REAL_STATES:
            raise MoneyLedgerError("paper entries cannot use real evidence states")
        if self.designation == "real" and self.evidence_state == "paper_decision":
            raise MoneyLedgerError("real entries cannot use paper_decision evidence state")
        if not self.action_alternatives:
            raise MoneyLedgerError("action_alternatives may not be empty")
        if self.selected_action not in self.action_alternatives:
            raise MoneyLedgerError("selected_action must be in action_alternatives")
        if self.no_action_alternative not in self.action_alternatives:
            raise MoneyLedgerError("no_action_alternative must be in action_alternatives")
        for field_name in ("capital_required", "maximum_possible_loss", "uncertainty_adjustment"):
            if float(getattr(self, field_name)) < 0:
                raise MoneyLedgerError(f"{field_name} may not be negative")
        unknown_cost_fields = [field_name for field_name in MATERIAL_ENTRY_COST_FIELDS if getattr(self, field_name) is None]
        if self.material_costs_known and unknown_cost_fields:
            raise MoneyLedgerError(f"known-cost entries must explicitly set cost fields: {unknown_cost_fields}")
        if not self.material_costs_known and self.conservative_expected_net_value is not None:
            raise MoneyLedgerError("unknown material costs make conservative net value ineligible")
        if not self.material_costs_known and not self.ineligibility_reasons:
            raise MoneyLedgerError("ineligible entries must explain missing material costs")
        if self.evidence_state in RESOLVED_STATES:
            if self.resolution is None or self.realized_gross_value is None or self.realized_net_value is None:
                raise MoneyLedgerError("resolved entries require resolution and realized values")
            if not self.mechanically_defined_no_action_outcome:
                raise MoneyLedgerError("resolved entries require a mechanically defined no-action outcome")
            _validate_realized_resolution(self.resolution, self.realized_gross_value, self.realized_net_value)
        if self.evidence_state in INITIAL_STATES and self.resolution is not None:
            raise MoneyLedgerError("unresolved entries may not include resolution")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def entry_hash(self) -> str:
        return stable_hash(self.to_dict())


class MoneyLedger:
    """Append-only, hash-linked ledger for paper and future real decisions."""

    def __init__(self, path: str):
        self.path = path
        self._ledger = ImmutableLedger(path)

    def append_entry(self, entry: MoneyLedgerEntry) -> dict[str, Any]:
        record_id = f"money_entry_{entry.decision_id}_{entry.entry_hash()[:16]}"

        def guard(records: list[dict[str, Any]]) -> None:
            prior = [
                record
                for record in records
                if record.get("record_type") == MONEY_LEDGER_RECORD_TYPE
                and record.get("payload", {}).get("decision_id") == entry.decision_id
            ]
            if not prior:
                if entry.supersedes_entry_hash is not None:
                    raise MoneyLedgerError("first entry for a decision cannot supersede another entry")
                return
            latest = prior[-1]
            latest_payload = latest["payload"]
            if entry.supersedes_entry_hash != latest["record_hash"]:
                raise MoneyLedgerError("updates must supersede the latest hash-linked entry")
            if latest_payload.get("designation") != entry.designation:
                raise MoneyLedgerError("paper and real outcomes cannot be mixed for one decision")
            if latest_payload.get("contract_hash") != entry.contract_hash:
                raise MoneyLedgerError("contract_hash cannot change for a decision")
            _assert_immutable_decision_fields(latest_payload, entry.to_dict())
            if latest_payload.get("evidence_state") == entry.evidence_state and not entry.correction_reason:
                raise MoneyLedgerError("same-state superseding entries must be explicit corrections")
            if latest_payload.get("evidence_state") in RESOLVED_STATES and not entry.correction_reason:
                raise MoneyLedgerError("post-resolution changes must be explicit corrections")

        try:
            return self._ledger.append_guarded(
                MONEY_LEDGER_RECORD_TYPE,
                entry.to_dict(),
                record_id=record_id,
                unique_record_id=True,
                guard=guard,
            )
        except DuplicateRecordError as exc:
            raise MoneyLedgerError(f"duplicate money ledger entry: {record_id}") from exc

    def append_resolution(
        self,
        decision_id: str,
        *,
        resolution: dict[str, Any],
        realized_gross_value: float,
        realized_net_value: float,
        mechanically_defined_no_action_outcome: dict[str, Any],
    ) -> dict[str, Any]:
        latest = self.latest_record(decision_id)
        if latest is None:
            raise MoneyLedgerError(f"unknown decision_id: {decision_id}")
        payload = latest["payload"]
        if payload["designation"] != "paper":
            raise MoneyLedgerError("this wave can only resolve paper money ledger entries")
        entry = MoneyLedgerEntry(
            **{
                **payload,
                "evidence_state": "resolved_paper",
                "resolution": resolution,
                "realized_gross_value": realized_gross_value,
                "realized_net_value": realized_net_value,
                "mechanically_defined_no_action_outcome": mechanically_defined_no_action_outcome,
                "supersedes_entry_hash": latest["record_hash"],
                "created_at": utc_now(),
            }
        )
        return self.append_entry(entry)

    def append_correction(self, entry: MoneyLedgerEntry, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise MoneyLedgerError("correction reason is required")
        return self.append_entry(replace(entry, correction_reason=reason))

    def records(self) -> list[dict[str, Any]]:
        return self._ledger.scan(MONEY_LEDGER_RECORD_TYPE)

    def entries(self) -> list[MoneyLedgerEntry]:
        return [MoneyLedgerEntry(**record["payload"]) for record in self.records()]

    def latest_record(self, decision_id: str) -> dict[str, Any] | None:
        match = None
        for record in self.records():
            if record["payload"].get("decision_id") == decision_id:
                match = record
        return match

    def latest_entries(self) -> list[MoneyLedgerEntry]:
        by_decision: dict[str, dict[str, Any]] = {}
        for record in self.records():
            by_decision[str(record["payload"]["decision_id"])] = record["payload"]
        return [MoneyLedgerEntry(**payload) for payload in by_decision.values()]

    def verify(self) -> bool:
        return self._ledger.verify_hash_chain()


def _assert_immutable_decision_fields(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for field_name in IMMUTABLE_DECISION_FIELDS:
        if previous.get(field_name) != current.get(field_name):
            raise MoneyLedgerError(f"{field_name} cannot be rewritten for an existing decision")


def _validate_realized_resolution(
    resolution: dict[str, Any],
    realized_gross_value: float,
    realized_net_value: float,
) -> None:
    costs = resolution.get("realized_costs")
    if not isinstance(costs, dict) or not costs:
        raise MoneyLedgerError("resolved entries require non-empty resolution.realized_costs")
    missing = [field for field, value in costs.items() if value is None]
    if missing:
        raise MoneyLedgerError(f"realized costs may not be unknown: {sorted(missing)}")
    negative = [field for field, value in costs.items() if float(value) < 0]
    if negative:
        raise MoneyLedgerError(f"realized costs may not be negative: {sorted(negative)}")
    expected_net = round(float(realized_gross_value) - sum(float(value) for value in costs.values()), 2)
    if round(float(realized_net_value), 2) != expected_net:
        raise MoneyLedgerError("realized_net_value must equal realized_gross_value minus realized_costs")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MoneyLedgerError(f"{field_name} must be a non-empty string")
