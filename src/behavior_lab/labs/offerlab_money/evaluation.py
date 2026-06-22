from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.ledger import ImmutableLedger
from behavior_lab.money.accounting import compute_decision_accounting, summarize_money_entries
from behavior_lab.money.contracts import Action, FinancialDecisionContract
from behavior_lab.money.ledger import MoneyLedgerEntry
from behavior_lab.money.storage import MoneyStorage
from behavior_lab.offerlab_pilot import (
    OfferLabPilotError,
    PILOT_IMPORT_RECORD_TYPE,
    PILOT_ROW_RECORD_TYPE,
    audit_pilot,
    default_data_root,
)


EVALUATION_SCHEMA_VERSION = "offerlab_money_evaluation.v1"
REPORT_SCHEMA_VERSION = "offerlab_money_report.v1"
FEATURE_PROGRAM_VERSION = "offerlab_money_wave2a.v1"
CONTRACT_VERSION = "offerlab_money_financial_decision.v1"
BENCHMARK_V2_RESEARCH_ONLY = {
    "source": "OfferLab Benchmark v2",
    "role": "research_only_evidence",
    "financial_action_source": False,
    "production_export_allowed": False,
    "causal_profit_lift_claim_allowed": False,
}
MONEY_COST_FIELDS = (
    "fees",
    "slippage",
    "shipping",
    "taxes_or_tax_assumption_reference",
    "holding_costs",
    "return_refund_allowance",
    "research_api_cost",
)
VALUE_ELIGIBLE_STATUSES = {"completed", "returned"}


class OfferLabMoneyError(ValueError):
    pass


@dataclass(frozen=True)
class PilotRows:
    latest_import: dict[str, Any]
    by_dataset: dict[str, list[dict[str, Any]]]
    ledger_path: Path


@dataclass(frozen=True)
class DecisionArtifacts:
    contract: FinancialDecisionContract
    entry: MoneyLedgerEntry
    summary: dict[str, Any]


def evaluate(
    pilot_id: str,
    *,
    data_root: str | Path | None = None,
    money_root: str | Path | None = None,
    evaluation_timestamp: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Command-equivalent for ``behavior-lab money offerlab evaluate PILOT_ID``.

    The future shared CLI can wire this function directly. It deliberately
    performs no seller, marketplace, network, financial, or notification
    mutation. The only writes are paper-only financial decision contracts and
    append-only money ledger entries under ``money_root``.
    """

    return evaluate_pilot(
        pilot_id,
        data_root=data_root,
        money_root=money_root,
        evaluation_timestamp=evaluation_timestamp,
        output_path=output_path,
    )


def report(
    pilot_id: str,
    *,
    data_root: str | Path | None = None,
    money_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Command-equivalent for ``behavior-lab money offerlab report PILOT_ID``."""

    return report_pilot(pilot_id, data_root=data_root, money_root=money_root, output_path=output_path)


def evaluate_pilot(
    pilot_id: str,
    *,
    data_root: str | Path | None = None,
    money_root: str | Path | None = None,
    evaluation_timestamp: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _data_root(data_root)
    as_of = evaluation_timestamp or utc_now()
    parse_time(as_of)
    audit = audit_pilot(pilot_id, data_root=root)
    rows = _latest_pilot_rows(pilot_id, root)
    storage = MoneyStorage(_money_root(root, pilot_id, money_root))
    artifacts = _decision_artifacts(pilot_id=pilot_id, rows=rows, evaluation_timestamp=as_of)

    appended = 0
    skipped = 0
    contract_hashes: dict[str, str] = {}
    for artifact in artifacts:
        storage.write_contract(artifact.contract)
        contract_hashes[artifact.contract.contract_id] = artifact.contract.contract_hash()
        latest = storage.ledger.latest_record(artifact.entry.decision_id)
        if latest is not None:
            if latest.get("payload") != artifact.entry.to_dict():
                raise OfferLabMoneyError(
                    f"Existing money entry for {artifact.entry.decision_id!r} differs from this evaluation"
                )
            skipped += 1
            continue
        storage.ledger.append_entry(artifact.entry)
        appended += 1

    entries = [artifact.entry for artifact in artifacts]
    summaries = [artifact.summary for artifact in artifacts]
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "command_equivalent": "behavior-lab money offerlab evaluate PILOT_ID",
        "integration_hook": "behavior_lab.labs.offerlab_money.evaluate",
        "pilot_id": pilot_id,
        "pilot_id_hash": stable_hash(pilot_id),
        "generated_at": as_of,
        "latest_import_id": rows.latest_import["import_id"],
        "latest_import_hash": rows.latest_import["import_hash"],
        "seller_pilot_ledger": str(rows.ledger_path.resolve()),
        "money_root": str(storage.root.resolve()),
        "money_ledger": str(Path(storage.ledger_path).resolve()),
        "contracts_dir": str(storage.contracts_dir.resolve()),
        "read_only": True,
        "paper_only": True,
        "executes_seller_actions": False,
        "submits_seller_actions": False,
        "financial_action": False,
        "network_mutation": False,
        "notifications_allowed": False,
        "causal_profit_lift_claimed": False,
        "historical_policy_claim": "descriptive_shadow_value_against_documented_historical_action_not_causal_lift",
        "benchmark_v2_evidence": BENCHMARK_V2_RESEARCH_ONLY,
        "audit_snapshot": {
            "readiness_gate": audit.get("readiness_gate"),
            "offer_funnel": audit.get("offer_funnel"),
            "data_quality_gaps": audit.get("data_quality_gaps"),
        },
        "decisions_seen": len(entries),
        "contracts_written": len(contract_hashes),
        "ledger_entries_appended": appended,
        "ledger_entries_skipped_existing": skipped,
        "ledger_valid": storage.ledger.verify(),
        "status_counts": dict(Counter(summary["financial_status"] for summary in summaries)),
        "historical_action_counts": dict(Counter(summary["historical_action"] for summary in summaries)),
        "selected_shadow_action_counts": dict(Counter(entry.selected_action for entry in entries)),
        "explicit_silence_count": sum(1 for summary in summaries if summary["explicit_silence"]),
        "net_profit_claim_eligible_count": sum(1 for summary in summaries if summary["net_profit_claim_eligible"]),
        "net_profit_claim_ineligible_count": sum(1 for summary in summaries if not summary["net_profit_claim_eligible"]),
        "unknown_cost_basis_count": sum(
            1 for entry in entries if "unknown_cost_basis" in entry.ineligibility_reasons
        ),
        "decision_summaries": summaries,
    }
    result["evaluation_hash"] = stable_hash({key: value for key, value in result.items() if key != "generated_at"})
    _write_optional_json(output_path, result)
    return result


def report_pilot(
    pilot_id: str,
    *,
    data_root: str | Path | None = None,
    money_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _data_root(data_root)
    storage = MoneyStorage(_money_root(root, pilot_id, money_root))
    pilot_hash = stable_hash(pilot_id)
    entries = [
        entry
        for entry in storage.ledger.latest_entries()
        if (entry.provenance or {}).get("pilot_id_hash") == pilot_hash
    ]
    generated_at = utc_now()
    if not entries:
        result = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "command_equivalent": "behavior-lab money offerlab report PILOT_ID",
            "integration_hook": "behavior_lab.labs.offerlab_money.report",
            "pilot_id": pilot_id,
            "pilot_id_hash": pilot_hash,
            "generated_at": generated_at,
            "money_root": str(storage.root.resolve()),
            "money_ledger": str(Path(storage.ledger_path).resolve()),
            "read_only": True,
            "paper_only": True,
            "executes_seller_actions": False,
            "financial_action": False,
            "notifications_allowed": False,
            "causal_profit_lift_claimed": False,
            "explicit_silence": {
                "applies": True,
                "reason": "no_offerlab_money_evaluation_entries",
                "seller_action": "none",
            },
            "benchmark_v2_evidence": BENCHMARK_V2_RESEARCH_ONLY,
            "decision_count": 0,
        }
        result["report_hash"] = stable_hash({key: value for key, value in result.items() if key != "generated_at"})
        _write_optional_json(output_path, result)
        return result

    summary = summarize_money_entries(entries)
    eligible_values = [
        float(entry.conservative_expected_net_value)
        for entry in entries
        if entry.conservative_expected_net_value is not None
        and not (entry.provenance or {}).get("explicit_silence", False)
    ]
    ineligible_reasons: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for entry in entries:
        provenance = entry.provenance or {}
        status_counts[str(provenance.get("financial_status", "unknown"))] += 1
        action_counts[str(provenance.get("historical_action", "unknown"))] += 1
        for reason in entry.ineligibility_reasons:
            ineligible_reasons[str(reason)] += 1
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "command_equivalent": "behavior-lab money offerlab report PILOT_ID",
        "integration_hook": "behavior_lab.labs.offerlab_money.report",
        "pilot_id": pilot_id,
        "pilot_id_hash": pilot_hash,
        "generated_at": generated_at,
        "money_root": str(storage.root.resolve()),
        "money_ledger": str(Path(storage.ledger_path).resolve()),
        "read_only": True,
        "paper_only": True,
        "executes_seller_actions": False,
        "submits_seller_actions": False,
        "financial_action": False,
        "network_mutation": False,
        "notifications_allowed": False,
        "causal_profit_lift_claimed": False,
        "historical_policy_claim": "descriptive_shadow_value_against_documented_historical_action_not_causal_lift",
        "benchmark_v2_evidence": BENCHMARK_V2_RESEARCH_ONLY,
        "ledger_valid": storage.ledger.verify(),
        "summary": summary,
        "decision_count": len(entries),
        "financial_status_counts": dict(status_counts),
        "historical_action_counts": dict(action_counts),
        "explicit_silence_count": sum(1 for entry in entries if (entry.provenance or {}).get("explicit_silence")),
        "net_profit_claim_eligible_count": len(eligible_values),
        "net_profit_claim_ineligible_count": len(entries) - len(eligible_values),
        "ineligibility_reasons": dict(ineligible_reasons),
        "conservative_shadow_value": {
            "basis": "seller_documented_historical_action",
            "causal_lift_claimed": False,
            "eligible_decisions": len(eligible_values),
            "total": round(sum(eligible_values), 2),
        },
    }
    result["report_hash"] = stable_hash({key: value for key, value in result.items() if key != "generated_at"})
    _write_optional_json(output_path, result)
    return result


def _decision_artifacts(
    *,
    pilot_id: str,
    rows: PilotRows,
    evaluation_timestamp: str,
) -> list[DecisionArtifacts]:
    by_dataset = rows.by_dataset
    listings = {str(row["listing_id"]): row for row in by_dataset.get("listings", [])}
    offers = sorted(
        by_dataset.get("offers", []),
        key=lambda row: (str(row.get("event_time") or ""), str(row.get("offer_id") or "")),
    )
    orders = by_dataset.get("orders", [])
    returns = by_dataset.get("returns_refunds", [])
    cancellations = by_dataset.get("cancellations_unpaid", [])
    fees_by_order = _sum_by_order(by_dataset.get("fees", []), "fee_amount")
    shipping_by_order = _sum_by_order(by_dataset.get("shipping_costs", []), "shipping_cost_amount")
    refunds_by_order = _sum_by_order(returns, "refund_amount")
    cost_by_listing = _latest_cost_by_listing(by_dataset.get("cost_basis", []))
    orders_by_offer = {str(row.get("offer_id")): row for row in orders if row.get("offer_id")}
    orders_by_listing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for order in orders:
        orders_by_listing[str(order.get("listing_id"))].append(order)

    artifacts = []
    for offer in offers:
        listing_id = str(offer.get("listing_id") or "")
        offer_id = str(offer.get("offer_id") or "")
        listing = listings.get(listing_id, {})
        order = _linked_order(offer, orders_by_offer, orders_by_listing)
        order_id = str(order.get("order_id") or "") if order else ""
        return_rows = [row for row in returns if str(row.get("order_id") or "") == order_id]
        cancellation_rows = _matching_cancellations(cancellations, offer=offer, order=order)
        cost_row = cost_by_listing.get(listing_id)
        decision_key = {
            "pilot_id": pilot_id,
            "import_hash": rows.latest_import["import_hash"],
            "offer_id": offer_id,
            "listing_id": listing_id,
        }
        token = stable_hash(decision_key)[:20]
        contract_id = f"offerlab_money_{token}"
        historical_action = _historical_action(offer)
        selected_action = _selected_shadow_action(historical_action)
        financial_status = _financial_status(
            offer=offer,
            order=order,
            returns=return_rows,
            cancellations=cancellation_rows,
            evaluation_timestamp=evaluation_timestamp,
        )
        timestamps = _decision_timestamps(offer, order, return_rows, cancellation_rows)
        value = _shadow_value(
            offer=offer,
            listing=listing,
            order=order,
            returns=return_rows,
            cancellations=cancellation_rows,
            fee=fees_by_order.get(order_id),
            shipping=shipping_by_order.get(order_id),
            refund=refunds_by_order.get(order_id, 0.0) if order else 0.0,
            cost_row=cost_row,
            historical_action=historical_action,
            financial_status=financial_status,
            evaluation_timestamp=evaluation_timestamp,
        )
        explicit_silence = _explicit_silence(offer, historical_action, value.ineligibility_reasons)
        if explicit_silence and "insufficient_seller_data" not in value.ineligibility_reasons:
            value.ineligibility_reasons.append("insufficient_seller_data")
        contract = _contract(
            contract_id=contract_id,
            pilot_id=pilot_id,
            import_hash=str(rows.latest_import["import_hash"]),
            offer=offer,
            listing=listing,
            decision_key=decision_key,
            decision_deadline=timestamps["decision_deadline"],
        )
        provenance = _provenance(
            pilot_id=pilot_id,
            latest_import=rows.latest_import,
            offer=offer,
            listing=listing,
            order=order,
            returns=return_rows,
            cancellations=cancellation_rows,
            timestamps=timestamps,
            historical_action=historical_action,
            selected_action=selected_action,
            financial_status=financial_status,
            value=value,
            explicit_silence=explicit_silence,
        )
        entry = MoneyLedgerEntry(
            decision_id=f"offerlab_money_{token}",
            contract_hash=contract.contract_hash(),
            decision_timestamp=timestamps["decision_timestamp"],
            data_cutoff=timestamps["data_cutoff"],
            target=contract.target,
            action_alternatives=[action.action_id for action in contract.available_actions],
            selected_action=selected_action,
            no_action_alternative=contract.no_action_id,
            capital_required=0.0,
            maximum_possible_loss=0.0,
            expected_gross_value=value.expected_gross_value,
            uncertainty_adjustment=value.uncertainty_adjustment,
            fees=value.cost_fields["fees"],
            slippage=value.cost_fields["slippage"],
            shipping=value.cost_fields["shipping"],
            taxes_or_tax_assumption_reference=value.cost_fields["taxes_or_tax_assumption_reference"],
            holding_costs=value.cost_fields["holding_costs"],
            return_refund_allowance=value.cost_fields["return_refund_allowance"],
            research_api_cost=value.cost_fields["research_api_cost"],
            conservative_expected_net_value=value.conservative_expected_net_value,
            decision_deadline=timestamps["decision_deadline"],
            feature_program_hash=stable_hash({"program": FEATURE_PROGRAM_VERSION}),
            evidence_state="paper_decision",
            designation="paper",
            mechanically_defined_no_action_outcome={
                "seller_mutation": False,
                "seller_action_submitted": False,
                "no_action_id": "abstain",
                "comparison_basis": "seller_documented_historical_action",
                "causal_lift_claimed": False,
            },
            economic_event_key=f"offerlab_money:{token}",
            provenance=provenance,
            artifact_hashes={
                "seller_pilot_import_hash": str(rows.latest_import["import_hash"]),
                "decision_artifact_hash": stable_hash({"decision_key": decision_key, "value": value.to_dict()}),
            },
            assumption_versions={
                "offerlab_money": FEATURE_PROGRAM_VERSION,
                "money_ledger_contract": "v1",
                "benchmark_v2_evidence": "research_only",
            },
            material_costs_known=value.material_costs_known,
            ineligibility_reasons=sorted(set(value.ineligibility_reasons)),
            created_at=timestamps["decision_timestamp"],
        )
        summary = {
            "decision_id": entry.decision_id,
            "contract_id": contract.contract_id,
            "offer_id_hash": stable_hash(offer_id),
            "listing_id_hash": stable_hash(listing_id),
            "historical_action": historical_action,
            "selected_shadow_action": selected_action,
            "financial_status": financial_status,
            "offer_timestamp": timestamps["offer_timestamp"],
            "response_timestamp": timestamps["response_timestamp"],
            "payment_timestamp": timestamps["payment_timestamp"],
            "return_window_matured_at": timestamps["return_window_matured_at"],
            "cancelled_at": timestamps["cancelled_at"],
            "material_costs_known": value.material_costs_known,
            "net_profit_claim_eligible": value.conservative_expected_net_value is not None and not explicit_silence,
            "conservative_expected_net_value": value.conservative_expected_net_value,
            "ineligibility_reasons": sorted(set(value.ineligibility_reasons)),
            "explicit_silence": explicit_silence,
        }
        artifacts.append(DecisionArtifacts(contract=contract, entry=entry, summary=summary))
    return artifacts


@dataclass
class ShadowValue:
    expected_gross_value: float
    uncertainty_adjustment: float
    cost_fields: dict[str, float | str | None]
    conservative_expected_net_value: float | None
    material_costs_known: bool
    ineligibility_reasons: list[str]
    cost_basis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_gross_value": self.expected_gross_value,
            "uncertainty_adjustment": self.uncertainty_adjustment,
            "cost_fields": dict(self.cost_fields),
            "conservative_expected_net_value": self.conservative_expected_net_value,
            "material_costs_known": self.material_costs_known,
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "cost_basis": dict(self.cost_basis),
        }


def _shadow_value(
    *,
    offer: dict[str, Any],
    listing: dict[str, Any],
    order: dict[str, Any] | None,
    returns: list[dict[str, Any]],
    cancellations: list[dict[str, Any]],
    fee: float | None,
    shipping: float | None,
    refund: float,
    cost_row: dict[str, Any] | None,
    historical_action: str,
    financial_status: str,
    evaluation_timestamp: str,
) -> ShadowValue:
    reasons: list[str] = []
    unit_cost = _float_or_none(cost_row.get("unit_cost_amount")) if cost_row else None
    quantity = int(order.get("quantity") or 1) if order else 1
    cost_total = round(unit_cost * quantity, 2) if unit_cost is not None else None
    sale_price = _float_or_none(order.get("sale_price_amount")) if order else None
    material_costs_known = True
    if historical_action == "unknown":
        reasons.append("unknown_seller_historical_action")
    if not offer.get("seller_response_time"):
        reasons.append("missing_seller_response_timestamp")
    if cost_total is None and financial_status in {"accepted", "paid", "completed", "returned", "cancelled"}:
        material_costs_known = False
        reasons.append("unknown_cost_basis")
    if order and financial_status in {"paid", "completed", "returned"}:
        if fee is None:
            material_costs_known = False
            reasons.append("missing_actual_fees")
        if shipping is None:
            material_costs_known = False
            reasons.append("missing_shipping_costs")
    if financial_status in {"accepted", "paid"}:
        reasons.append("unresolved_financial_outcome")
    if financial_status in {"paid", "completed", "returned"} and not _return_maturity(order, returns, evaluation_timestamp)["matured"]:
        reasons.append("return_window_not_matured")
    if financial_status == "cancelled":
        reasons.append("cancelled_or_unpaid_not_net_profit_claim")
    if financial_status == "unresolved" and historical_action != "decline":
        reasons.append("unresolved_financial_outcome")

    cost_fields: dict[str, float | str | None] = {
        "fees": _money_or_none(fee),
        "slippage": 0.0,
        "shipping": _money_or_none(shipping),
        "taxes_or_tax_assumption_reference": "not_applicable_seller_pilot_shadow",
        "holding_costs": _money_or_none(cost_total),
        "return_refund_allowance": _money_or_none(refund),
        "research_api_cost": 0.0,
    }
    if historical_action == "decline" and financial_status == "unresolved":
        zero_fields = {
            "fees": 0.0,
            "slippage": 0.0,
            "shipping": 0.0,
            "taxes_or_tax_assumption_reference": "not_applicable_seller_pilot_shadow",
            "holding_costs": 0.0,
            "return_refund_allowance": 0.0,
            "research_api_cost": 0.0,
        }
        return ShadowValue(
            expected_gross_value=0.0,
            uncertainty_adjustment=0.0,
            cost_fields=zero_fields,
            conservative_expected_net_value=0.0,
            material_costs_known=True,
            ineligibility_reasons=[],
            cost_basis=_cost_basis_payload(cost_row, quantity, cost_total),
        )

    eligible_for_value = (
        historical_action == "accept"
        and financial_status in VALUE_ELIGIBLE_STATUSES
        and sale_price is not None
        and material_costs_known
        and _return_maturity(order, returns, evaluation_timestamp)["matured"]
    )
    if eligible_for_value:
        accounting = compute_decision_accounting(
            gross_value=sale_price,
            fees=fee,
            shipping=shipping,
            holding_costs=cost_total,
            return_refund_allowance=refund,
            slippage=0.0,
            research_api_cost=0.0,
            uncertainty_adjustment=0.0,
            material_cost_fields=["fees", "shipping", "holding_costs", "return_refund_allowance"],
        )
        cost_fields["fees"] = accounting.cost_breakdown["fees"]
        cost_fields["shipping"] = accounting.cost_breakdown["shipping"]
        cost_fields["holding_costs"] = accounting.cost_breakdown["holding_costs"]
        cost_fields["return_refund_allowance"] = accounting.cost_breakdown["return_refund_allowance"]
        return ShadowValue(
            expected_gross_value=accounting.gross_value,
            uncertainty_adjustment=0.0,
            cost_fields=cost_fields,
            conservative_expected_net_value=accounting.conservative_expected_net_value,
            material_costs_known=True,
            ineligibility_reasons=sorted(set(reasons)),
            cost_basis=_cost_basis_payload(cost_row, quantity, cost_total),
        )

    if material_costs_known:
        for field in ("fees", "shipping", "holding_costs", "return_refund_allowance"):
            if cost_fields[field] is None:
                cost_fields[field] = 0.0
    return ShadowValue(
        expected_gross_value=_money_or_none(sale_price) or 0.0,
        uncertainty_adjustment=0.0,
        cost_fields=cost_fields,
        conservative_expected_net_value=None,
        material_costs_known=material_costs_known,
        ineligibility_reasons=sorted(set(reasons)),
        cost_basis=_cost_basis_payload(cost_row, quantity, cost_total),
    )


def _contract(
    *,
    contract_id: str,
    pilot_id: str,
    import_hash: str,
    offer: dict[str, Any],
    listing: dict[str, Any],
    decision_key: dict[str, Any],
    decision_deadline: str,
) -> FinancialDecisionContract:
    currency = str(offer.get("currency") or listing.get("currency") or "USD")
    offer_amount = _float_or_none(offer.get("offer_amount"))
    asking_amount = _float_or_none(listing.get("asking_price_amount"))
    counter_min = offer_amount if offer_amount is not None else 0.01
    counter_max = asking_amount if asking_amount is not None else counter_min
    if counter_max < counter_min:
        counter_max = counter_min
    actions = [
        Action(
            action_id="abstain",
            action_type="no_action",
            parameters={"reason": "insufficient_data_or_shadow_only"},
        ),
        Action(
            action_id="decline",
            action_type="seller_offer_response",
            parameters={"response": "decline", "seller_action_submitted": False},
        ),
        Action(
            action_id="accept",
            action_type="seller_offer_response",
            parameters={"response": "accept", "seller_action_submitted": False},
        ),
        Action(
            action_id="counter",
            action_type="seller_offer_response",
            parameters={
                "response": "counter",
                "bounded_values_only": True,
                "minimum_counter_amount": round(counter_min, 2),
                "maximum_counter_amount": round(counter_max, 2),
                "currency": currency,
            },
        ),
    ]
    return FinancialDecisionContract(
        contract_id=contract_id,
        domain="seller",
        target={
            "type": "offerlab_seller_offer_decision",
            "pilot_id_hash": stable_hash(pilot_id),
            "latest_import_hash": import_hash,
            "offer_id_hash": stable_hash(str(offer.get("offer_id") or "")),
            "listing_id_hash": stable_hash(str(offer.get("listing_id") or "")),
            "economic_event_key": f"offerlab_money:{stable_hash(decision_key)[:20]}",
            "primary_metric": "mature_contribution_margin",
        },
        decision_horizon="seller_offer_response_to_return_maturity",
        decision_deadline=decision_deadline,
        available_actions=actions,
        no_action_id="abstain",
        payoff_specification={
            "value_basis": "seller_documented_historical_action",
            "gross_value": "actual sale price when the historical offer action produced an order",
            "net_value": "gross minus actual fees, shipping, refunds, and documented cost basis",
            "causal_profit_lift_claim_allowed": False,
        },
        cost_policy={
            "material_cost_fields": ["fees", "shipping", "cost_basis", "return_refund_allowance"],
            "unknown_cost_basis": "net_profit_claim_ineligible",
            "imputation_allowed": False,
        },
        risk_policy={
            "paper_only": True,
            "seller_action_submitted": False,
            "financial_action_submitted": False,
            "notifications_allowed": False,
        },
        liquidity_policy={"capital_required_for_shadow_evaluation": 0.0},
        resolution_source={
            "type": "seller_supplied_pilot_ledger",
            "latest_import_hash": import_hash,
            "benchmark_v2": BENCHMARK_V2_RESEARCH_ONLY,
        },
        data_cutoff_policy={
            "source": "imported immutable seller pilot rows",
            "no_network_refresh": True,
            "as_of_required": True,
        },
        prospective_requirement={
            "required_before_real_action": True,
            "mode": "paper_shadow_only",
            "real_action_requires_separate_authorization": True,
        },
        notification_threshold={"notifications_allowed": False},
        paper_only=True,
        contract_version=CONTRACT_VERSION,
    )


def _provenance(
    *,
    pilot_id: str,
    latest_import: dict[str, Any],
    offer: dict[str, Any],
    listing: dict[str, Any],
    order: dict[str, Any] | None,
    returns: list[dict[str, Any]],
    cancellations: list[dict[str, Any]],
    timestamps: dict[str, str | None],
    historical_action: str,
    selected_action: str,
    financial_status: str,
    value: ShadowValue,
    explicit_silence: bool,
) -> dict[str, Any]:
    order_id = str(order.get("order_id") or "") if order else ""
    return {
        "source_id": "offerlab_seller_pilot",
        "strategy_id": "seller_documented_historical_action_shadow",
        "pilot_id_hash": stable_hash(pilot_id),
        "latest_import_id": latest_import["import_id"],
        "latest_import_hash": latest_import["import_hash"],
        "read_only": True,
        "paper_only": True,
        "executes_seller_actions": False,
        "seller_action_submitted": False,
        "financial_action_submitted": False,
        "notifications_sent": False,
        "causal_lift_claimed": False,
        "historical_policy_claim": "descriptive_only_not_causal",
        "benchmark_v2_evidence": BENCHMARK_V2_RESEARCH_ONLY,
        "explicit_silence": explicit_silence,
        "historical_action": historical_action,
        "selected_shadow_action": selected_action,
        "financial_status": financial_status,
        "seller_row_keys": {
            "offer_id": offer.get("offer_id"),
            "listing_id": offer.get("listing_id"),
            "order_id": order.get("order_id") if order else None,
        },
        "timestamps": dict(timestamps),
        "offer": {
            "offer_timestamp": timestamps["offer_timestamp"],
            "offer_amount": _float_or_none(offer.get("offer_amount")),
            "currency": offer.get("currency") or listing.get("currency"),
            "offer_state": offer.get("offer_state"),
            "expires_at": offer.get("expires_at"),
        },
        "seller_response": {
            "response": offer.get("seller_response"),
            "response_timestamp": timestamps["response_timestamp"],
            "response_amount": _float_or_none(offer.get("seller_response_amount")),
            "decision_history_available_at": offer.get("decision_history_available_at"),
        },
        "payment": _payment_payload(order),
        "cancellation": {
            "cancelled": financial_status == "cancelled",
            "events": [_event_payload(row, amount_field="amount") for row in cancellations],
        },
        "return_maturity": {
            "return_window_matured": timestamps["return_window_matured_at"] is not None
            and financial_status in {"completed", "returned"},
            "return_window_matured_at": timestamps["return_window_matured_at"],
            "returns": [_event_payload(row, amount_field="refund_amount") for row in returns],
        },
        "cost_basis": value.cost_basis,
        "fees": {"amount": value.cost_fields["fees"], "currency": offer.get("currency") or listing.get("currency")},
        "shipping": {
            "amount": value.cost_fields["shipping"],
            "currency": offer.get("currency") or listing.get("currency"),
        },
        "return_refund_allowance": value.cost_fields["return_refund_allowance"],
        "net_profit_claim_eligible": value.conservative_expected_net_value is not None and not explicit_silence,
        "ineligibility_reasons": sorted(set(value.ineligibility_reasons)),
        "value_basis": {
            "against": "seller_documented_historical_action",
            "conservative_expected_net_value": value.conservative_expected_net_value,
            "causal_lift_claimed": False,
        },
        "economic_event_key": f"offerlab_money:{stable_hash({'offer': offer.get('offer_id'), 'order': order_id})[:20]}",
    }


def _latest_pilot_rows(pilot_id: str, root: Path) -> PilotRows:
    ledger = ImmutableLedger(root / pilot_id / "ledger.jsonl")
    imports = ledger.payloads(PILOT_IMPORT_RECORD_TYPE)
    if not imports:
        raise OfferLabPilotError(f"No imports found for pilot_id {pilot_id!r}")
    latest_import = imports[-1]
    rows = [
        record
        for record in ledger.payloads(PILOT_ROW_RECORD_TYPE)
        if record.get("pilot_id") == pilot_id and record.get("import_id") == latest_import["import_id"]
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row["canonical"])
    ledger.verify_hash_chain()
    return PilotRows(latest_import=latest_import, by_dataset=dict(by_dataset), ledger_path=Path(ledger.path))


def _linked_order(
    offer: dict[str, Any],
    orders_by_offer: dict[str, dict[str, Any]],
    orders_by_listing: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    offer_id = str(offer.get("offer_id") or "")
    if offer_id in orders_by_offer:
        return orders_by_offer[offer_id]
    listing_orders = orders_by_listing.get(str(offer.get("listing_id") or ""), [])
    if len(listing_orders) == 1:
        return listing_orders[0]
    return None


def _historical_action(offer: dict[str, Any]) -> str:
    response = str(offer.get("seller_response") or offer.get("offer_state") or "").strip().lower()
    if response in {"accept", "accepted", "seller_accepted"}:
        return "accept"
    if response in {"decline", "declined", "rejected"}:
        return "decline"
    if response in {"counter", "countered", "counter_offer"}:
        return "counter"
    return "unknown"


def _selected_shadow_action(historical_action: str) -> str:
    if historical_action in {"accept", "decline", "counter"}:
        return historical_action
    return "abstain"


def _financial_status(
    *,
    offer: dict[str, Any],
    order: dict[str, Any] | None,
    returns: list[dict[str, Any]],
    cancellations: list[dict[str, Any]],
    evaluation_timestamp: str,
) -> str:
    order_status = str(order.get("order_status") if order else "").strip().lower()
    if cancellations or order_status in {"cancelled", "canceled", "unpaid", "unpaid_order"}:
        return "cancelled"
    if returns or order_status in {"returned", "refunded"}:
        return "returned"
    if order and _buyer_paid(order) and _order_completed(order) and _return_maturity(order, returns, evaluation_timestamp)["matured"]:
        return "completed"
    if order and _buyer_paid(order):
        return "paid"
    if _historical_action(offer) == "accept":
        return "accepted"
    return "unresolved"


def _decision_timestamps(
    offer: dict[str, Any],
    order: dict[str, Any] | None,
    returns: list[dict[str, Any]],
    cancellations: list[dict[str, Any]],
) -> dict[str, str | None]:
    offer_timestamp = _time_or_none(offer.get("event_time"))
    response_timestamp = _time_or_none(offer.get("seller_response_time"))
    available_at = _time_or_none(offer.get("decision_history_available_at")) or _time_or_none(offer.get("available_at"))
    decision_timestamp = response_timestamp or available_at or offer_timestamp
    if decision_timestamp is None:
        raise OfferLabMoneyError(f"Offer {offer.get('offer_id')!r} has no usable decision timestamp")
    data_cutoff = _min_timestamp([available_at, decision_timestamp]) or decision_timestamp
    deadline = _max_timestamp([offer.get("expires_at"), response_timestamp, decision_timestamp]) or decision_timestamp
    payment_timestamp = _time_or_none(order.get("paid_at")) if order else None
    return_window_matured_at = _return_window_matured_at(order, returns)
    cancelled_at = _min_timestamp([row.get("event_time") for row in cancellations])
    return {
        "offer_timestamp": offer_timestamp,
        "response_timestamp": response_timestamp,
        "payment_timestamp": payment_timestamp,
        "completed_timestamp": _time_or_none(order.get("completed_at")) if order else None,
        "return_window_matured_at": return_window_matured_at,
        "cancelled_at": cancelled_at,
        "decision_timestamp": decision_timestamp,
        "data_cutoff": data_cutoff,
        "decision_deadline": deadline,
    }


def _return_maturity(
    order: dict[str, Any] | None,
    returns: list[dict[str, Any]],
    evaluation_timestamp: str,
) -> dict[str, Any]:
    matured_at = _return_window_matured_at(order, returns)
    if matured_at is None:
        return {"matured": False, "return_window_matured_at": None}
    return {
        "matured": parse_time(matured_at) <= parse_time(evaluation_timestamp),
        "return_window_matured_at": matured_at,
    }


def _return_window_matured_at(order: dict[str, Any] | None, returns: list[dict[str, Any]]) -> str | None:
    candidates = []
    if order:
        candidates.append(order.get("return_window_matured_at"))
    candidates.extend(row.get("return_window_matured_at") for row in returns)
    return _min_timestamp(candidates)


def _buyer_paid(order: dict[str, Any]) -> bool:
    status = str(order.get("order_status") or "").strip().lower()
    return bool(order.get("paid_at")) or status in {"paid", "completed", "shipped", "delivered", "returned", "refunded"}


def _order_completed(order: dict[str, Any]) -> bool:
    status = str(order.get("order_status") or "").strip().lower()
    return bool(order.get("completed_at")) or status in {"completed", "delivered", "returned", "refunded"}


def _matching_cancellations(
    rows: list[dict[str, Any]],
    *,
    offer: dict[str, Any],
    order: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    offer_id = str(offer.get("offer_id") or "")
    listing_id = str(offer.get("listing_id") or "")
    order_id = str(order.get("order_id") or "") if order else ""
    output = []
    for row in rows:
        if row.get("offer_id") and str(row.get("offer_id")) == offer_id:
            output.append(row)
        elif row.get("order_id") and str(row.get("order_id")) == order_id:
            output.append(row)
        elif row.get("listing_id") and str(row.get("listing_id")) == listing_id:
            output.append(row)
    return output


def _sum_by_order(rows: list[dict[str, Any]], amount_field: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        order_id = row.get("order_id")
        if not order_id:
            continue
        amount = _float_or_none(row.get(amount_field))
        if amount is None:
            continue
        key = str(order_id)
        totals[key] = round(totals.get(key, 0.0) + amount, 2)
    return totals


def _latest_cost_by_listing(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        listing_id = str(row.get("listing_id") or "")
        if not listing_id:
            continue
        prior = latest.get(listing_id)
        if prior is None or parse_time(str(row.get("available_at") or row.get("event_time"))) >= parse_time(
            str(prior.get("available_at") or prior.get("event_time"))
        ):
            latest[listing_id] = row
    return latest


def _explicit_silence(offer: dict[str, Any], historical_action: str, reasons: list[str]) -> bool:
    if historical_action == "unknown":
        return True
    if not offer.get("seller_response_time"):
        return True
    return "insufficient_seller_data" in reasons


def _payment_payload(order: dict[str, Any] | None) -> dict[str, Any]:
    if not order:
        return {
            "buyer_paid": False,
            "paid_at": None,
            "completed_at": None,
            "sale_price_amount": None,
            "currency": None,
            "order_status": None,
        }
    return {
        "buyer_paid": _buyer_paid(order),
        "paid_at": order.get("paid_at"),
        "completed_at": order.get("completed_at"),
        "sale_price_amount": _float_or_none(order.get("sale_price_amount")),
        "currency": order.get("currency"),
        "order_status": order.get("order_status"),
    }


def _event_payload(row: dict[str, Any], *, amount_field: str) -> dict[str, Any]:
    return {
        "event_time": row.get("event_time"),
        "available_at": row.get("available_at"),
        "event_type": row.get("event_type") or row.get("return_status"),
        "amount": _float_or_none(row.get(amount_field)),
        "currency": row.get("currency"),
        "return_window_matured_at": row.get("return_window_matured_at"),
    }


def _cost_basis_payload(row: dict[str, Any] | None, quantity: int, total: float | None) -> dict[str, Any]:
    return {
        "known": row is not None and total is not None,
        "unit_cost_amount": _float_or_none(row.get("unit_cost_amount")) if row else None,
        "quantity": quantity,
        "total_cost_basis": total,
        "currency": row.get("currency") if row else None,
        "cost_source": row.get("cost_source") if row else None,
        "available_at": row.get("available_at") if row else None,
    }


def _data_root(data_root: str | Path | None) -> Path:
    return Path(data_root) if data_root is not None else default_data_root()


def _money_root(data_root: Path, pilot_id: str, money_root: str | Path | None) -> Path:
    if money_root is not None:
        return Path(money_root)
    return data_root / pilot_id / "offerlab_money"


def _write_optional_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _money_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def _time_or_none(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value)
    parse_time(text)
    return text


def _min_timestamp(values: list[Any]) -> str | None:
    concrete = [_time_or_none(value) for value in values if value not in {None, ""}]
    if not concrete:
        return None
    return min(concrete, key=parse_time)


def _max_timestamp(values: list[Any]) -> str | None:
    concrete = [_time_or_none(value) for value in values if value not in {None, ""}]
    if not concrete:
        return None
    return max(concrete, key=parse_time)
