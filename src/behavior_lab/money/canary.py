from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.offerlab_research.api import AppendOnlyResearchStore


CANARY_SCHEMA_VERSION = "money_canary.v1"
DEFAULT_STATE_DIR = ".money_canaries"
CANARY_LABS = {"offerlab_seller_pilot", "weather_edge", "etf_risk"}
REAL_ACTION_FLAGS = {
    "seller_mutation": False,
    "exchange_authentication": False,
    "exchange_order_submission": False,
    "brokerage_connection": False,
    "brokerage_order_submission": False,
    "notifications": False,
    "real_financial_action": False,
}


class MoneyCanaryError(ValueError):
    pass


@dataclass(frozen=True)
class CanaryOptions:
    lab: str | None = None
    as_of: str | None = None
    strategy_version: str = "fixture_frozen_v1"
    source_version: str = "fixture_source_v1"
    seller_pilot_ready: bool = False


class MoneyCanaryManager:
    def __init__(self, state_dir: str | Path = DEFAULT_STATE_DIR) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = AppendOnlyResearchStore(self.state_dir / "canaries.jsonl")

    def start(self, contract_id: str, options: CanaryOptions | None = None) -> dict[str, Any]:
        opts = options or CanaryOptions()
        lab = _infer_lab(contract_id, opts.lab)
        timestamp = opts.as_of or utc_now()
        parse_time(timestamp)
        if lab == "offerlab_seller_pilot" and not opts.seller_pilot_ready:
            payload = self._base_payload(
                {
                    "contract_id": contract_id,
                    "lab": lab,
                    "status": "blocked",
                    "reason": "seller_pilot_readiness_gate_not_passed",
                    "paper_only": True,
                    "production_state": dict(REAL_ACTION_FLAGS),
                }
            )
            self.store.append("canary_start_blocked", payload)
            return payload

        protocol = _protocol(contract_id, lab, timestamp, opts)
        canary_id = str(protocol["canary_id"])
        existing = self._start_event(canary_id)
        if existing:
            return {**self.status(canary_id), "already_started": True}

        started = self.store.append(
            "canary_started",
            self._base_payload(
                {
                    "canary_id": canary_id,
                    "contract_id": contract_id,
                    "lab": lab,
                    "protocol": protocol,
                    "paper_only": True,
                    "production_state": dict(REAL_ACTION_FLAGS),
                }
            ),
        )
        snapshot = self._append_snapshot(protocol, timestamp, reason="start")
        return {
            "schema_version": CANARY_SCHEMA_VERSION,
            "canary_id": canary_id,
            "contract_id": contract_id,
            "lab": lab,
            "status": "started",
            "started_event_hash": started["event_hash"],
            "snapshot": snapshot,
            "protocol": protocol,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def resume(self, canary_id: str, *, as_of: str | None = None, strategy_version: str | None = None) -> dict[str, Any]:
        protocol = self._protocol_for(canary_id)
        if self._invalidated(canary_id):
            raise MoneyCanaryError("invalidated canary cannot be resumed")
        supplied_strategy = strategy_version or protocol["frozen_strategy"]["strategy_version"]
        if supplied_strategy != protocol["frozen_strategy"]["strategy_version"]:
            raise MoneyCanaryError("material canary component changed; start a new canary")
        current_protocol = _protocol(
            str(protocol["contract_id"]),
            str(protocol["lab"]),
            str(protocol["start_at"]),
            CanaryOptions(
                lab=str(protocol["lab"]),
                as_of=str(protocol["start_at"]),
                strategy_version=supplied_strategy,
                source_version=str(protocol["source_versions"].get("primary", "fixture_source_v1")),
                seller_pilot_ready=True,
            ),
        )
        if current_protocol["material_hash"] != protocol["material_hash"]:
            raise MoneyCanaryError("material canary component changed; start a new canary")
        timestamp = as_of or utc_now()
        parse_time(timestamp)
        snapshot = self._append_snapshot(protocol, timestamp, reason="resume")
        return {
            "schema_version": CANARY_SCHEMA_VERSION,
            "canary_id": canary_id,
            "status": "resumed",
            "snapshot": snapshot,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def status(self, canary_id: str) -> dict[str, Any]:
        protocol = self._protocol_for(canary_id)
        snapshots = self._snapshots(canary_id)
        invalidation = self._invalidated(canary_id)
        return {
            "schema_version": CANARY_SCHEMA_VERSION,
            "canary_id": canary_id,
            "contract_id": protocol["contract_id"],
            "lab": protocol["lab"],
            "status": "invalidated" if invalidation else "active",
            "started_at": protocol["start_at"],
            "scheduled_end_at": protocol["end_at"],
            "minimum_duration_days": protocol["minimum_duration_days"],
            "snapshot_count": len(snapshots),
            "last_snapshot_at": snapshots[-1]["observed_at"] if snapshots else None,
            "invalidated": invalidation,
            "immutability": protocol["immutability"],
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def report(self, canary_id: str) -> dict[str, Any]:
        protocol = self._protocol_for(canary_id)
        snapshots = self._snapshots(canary_id)
        invalidation = self._invalidated(canary_id)
        metrics = _aggregate_metrics(protocol, snapshots)
        return {
            "schema_version": CANARY_SCHEMA_VERSION,
            "canary_id": canary_id,
            "contract_id": protocol["contract_id"],
            "lab": protocol["lab"],
            "protocol": protocol,
            "source_health_history": [item["source_health"] for item in snapshots],
            "prediction_history": [prediction for item in snapshots for prediction in item["predictions"]],
            "decision_history": [decision for item in snapshots for decision in item["decisions"]],
            "resolution_history": [resolution for item in snapshots for resolution in item["resolutions"]],
            "pnl_savings_history": [item["pnl_or_savings"] for item in snapshots],
            "counterexamples": [counterexample for item in snapshots for counterexample in item["counterexamples"]],
            "metrics": metrics,
            "final_evidence_report": {
                "available": bool(invalidation) or metrics["minimum_duration_elapsed"] is True,
                "paper_only": True,
                "real_money_allowed": False,
                "reason": "capital_allocation_not_authorized_in_wave_4",
            },
            "invalidated": invalidation,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def invalidate(self, canary_id: str, *, reason: str, as_of: str | None = None) -> dict[str, Any]:
        if not reason.strip():
            raise MoneyCanaryError("invalidation reason is required")
        protocol = self._protocol_for(canary_id)
        if self._invalidated(canary_id):
            return self.status(canary_id)
        timestamp = as_of or utc_now()
        event = self.store.append(
            "canary_invalidated",
            self._base_payload(
                {
                    "canary_id": canary_id,
                    "contract_id": protocol["contract_id"],
                    "lab": protocol["lab"],
                    "reason": reason,
                    "invalidated_at": timestamp,
                    "paper_only": True,
                    "production_state": dict(REAL_ACTION_FLAGS),
                }
            ),
        )
        return {
            "schema_version": CANARY_SCHEMA_VERSION,
            "canary_id": canary_id,
            "status": "invalidated",
            "event_hash": event["event_hash"],
            "reason": reason,
            "paper_only": True,
            "production_state": dict(REAL_ACTION_FLAGS),
        }

    def verify(self) -> bool:
        return self.store.verify()

    def _append_snapshot(self, protocol: dict[str, Any], observed_at: str, *, reason: str) -> dict[str, Any]:
        snapshots = self._snapshots(str(protocol["canary_id"]))
        snapshot = _snapshot(protocol, observed_at, snapshot_index=len(snapshots) + 1, reason=reason)
        event = self.store.append("canary_snapshot", self._base_payload(snapshot))
        return {**snapshot, "event_hash": event["event_hash"]}

    def _base_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": CANARY_SCHEMA_VERSION, **payload}

    def _events(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.store.all_events() if event["event_type"] == event_type]

    def _start_event(self, canary_id: str) -> dict[str, Any] | None:
        for event in self._events("canary_started"):
            if event["payload"].get("canary_id") == canary_id:
                return event
        return None

    def _protocol_for(self, canary_id: str) -> dict[str, Any]:
        event = self._start_event(canary_id)
        if not event:
            raise MoneyCanaryError(f"unknown canary_id: {canary_id}")
        return dict(event["payload"]["protocol"])

    def _snapshots(self, canary_id: str) -> list[dict[str, Any]]:
        return [
            dict(event["payload"])
            for event in self._events("canary_snapshot")
            if event["payload"].get("canary_id") == canary_id
        ]

    def _invalidated(self, canary_id: str) -> dict[str, Any] | None:
        invalidation = None
        for event in self._events("canary_invalidated"):
            if event["payload"].get("canary_id") == canary_id:
                invalidation = dict(event["payload"])
        return invalidation


def start_fixture_canaries(state_dir: str | Path, *, as_of: str = "2026-07-01T12:00:00+00:00") -> dict[str, Any]:
    manager = MoneyCanaryManager(state_dir)
    started = [
        manager.start("weather-edge-fixture", CanaryOptions(lab="weather_edge", as_of=as_of)),
        manager.start("etf-risk-fixture", CanaryOptions(lab="etf_risk", as_of=as_of)),
        manager.start("offerlab-seller-fixture", CanaryOptions(lab="offerlab_seller_pilot", as_of=as_of, seller_pilot_ready=True)),
    ]
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "state_dir": str(Path(state_dir)),
        "canaries": started,
        "ledger_valid": manager.verify(),
        "paper_only": True,
        "production_state": dict(REAL_ACTION_FLAGS),
    }


def _infer_lab(contract_id: str, explicit: str | None) -> str:
    if explicit:
        if explicit not in CANARY_LABS:
            raise MoneyCanaryError(f"unsupported lab: {explicit}")
        return explicit
    lowered = contract_id.lower()
    if "weather" in lowered:
        return "weather_edge"
    if "etf" in lowered:
        return "etf_risk"
    if "seller" in lowered or "offer" in lowered:
        return "offerlab_seller_pilot"
    raise MoneyCanaryError("lab is required when contract_id does not identify weather, etf, seller, or offerlab")


def _protocol(contract_id: str, lab: str, start_at: str, options: CanaryOptions) -> dict[str, Any]:
    lab_protocol = _lab_protocol(lab)
    end_at = (parse_time(start_at) + timedelta(days=int(lab_protocol["minimum_duration_days"]))).isoformat()
    frozen_strategy = {
        "strategy_version": options.strategy_version,
        "program_hash": stable_hash({"lab": lab, "strategy_version": options.strategy_version, "canary": "wave4"}),
        "model_changes_allowed": False,
        "restart_required_on_material_change": True,
    }
    protocol = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "contract_id": contract_id,
        "lab": lab,
        "contract_hash": stable_hash({"contract_id": contract_id, "lab": lab, "wave": 4}),
        "start_at": start_at,
        "end_at": end_at,
        "minimum_duration_days": lab_protocol["minimum_duration_days"],
        "cadence": lab_protocol["cadence"],
        "data_cutoff_policy": lab_protocol["data_cutoff_policy"],
        "source_versions": {"primary": options.source_version, **lab_protocol["source_versions"]},
        "cost_assumptions": lab_protocol["cost_assumptions"],
        "prospective_gates": lab_protocol["prospective_gates"],
        "invalidation_conditions": lab_protocol["invalidation_conditions"],
        "frozen_strategy": frozen_strategy,
        "metrics_tracked": lab_protocol["metrics_tracked"],
        "paper_only": True,
        "capital_allocation_allowed": False,
        "notifications_allowed": False,
        "real_actions_allowed": False,
        "production_state": dict(REAL_ACTION_FLAGS),
        "immutability": {
            "material_fields": [
                "contract_hash",
                "frozen_strategy.program_hash",
                "minimum_duration_days",
                "cadence",
                "data_cutoff_policy",
                "source_versions",
                "cost_assumptions",
                "prospective_gates",
                "invalidation_conditions",
                "metrics_tracked",
            ],
            "material_change_creates_new_canary": True,
        },
    }
    material_hash = stable_hash(
        {
            key: protocol[key]
            for key in (
                "contract_id",
                "lab",
                "contract_hash",
                "minimum_duration_days",
                "cadence",
                "data_cutoff_policy",
                "source_versions",
                "cost_assumptions",
                "prospective_gates",
                "invalidation_conditions",
                "frozen_strategy",
                "metrics_tracked",
            )
        }
    )
    protocol["material_hash"] = material_hash
    protocol["canary_id"] = f"canary_{material_hash[:16]}"
    return protocol


def _lab_protocol(lab: str) -> dict[str, Any]:
    if lab == "weather_edge":
        return {
            "minimum_duration_days": 60,
            "cadence": "daily",
            "source_versions": {
                "market_depth": "fixture_executable_order_book_v1",
                "weather": "fixture_as_of_weather_v1",
                "settlement": "fixture_station_daily_high_v1",
            },
            "data_cutoff_policy": {
                "fixed_decision_horizons": True,
                "all_supported_city_events": True,
                "executable_prices_only": True,
                "midpoint_allowed": False,
                "candle_extremes_allowed": False,
            },
            "cost_assumptions": {
                "fees_included": True,
                "spread_included": True,
                "slippage_included": True,
                "liquidity_included": True,
                "order_book_quantity_preserved": True,
            },
            "prospective_gates": {
                "minimum_consecutive_days": 60,
                "minimum_prospective_days_before_real_money_review": 60,
                "calibration_required": True,
                "brier_score_required": True,
                "log_loss_required": True,
                "city_month_regime_concentration_required": True,
            },
            "invalidation_conditions": [
                "strategy_or_program_hash_changes",
                "data_cutoff_policy_changes",
                "source_version_changes",
                "cost_assumption_changes",
            ],
            "metrics_tracked": ["calibration", "brier_score", "log_loss", "paper_pnl", "drawdown", "city_month_regime_concentration"],
        }
    if lab == "etf_risk":
        return {
            "minimum_duration_days": 183,
            "cadence": "weekly",
            "source_versions": {"market_data": "fixture_authorized_adjusted_prices_v1"},
            "data_cutoff_policy": {
                "weekly_decisions": True,
                "as_of_prices_only": True,
                "availability_time_required": True,
            },
            "cost_assumptions": {
                "turnover_included": True,
                "transaction_costs_included": True,
                "long_only_cash_allowed": True,
                "leverage_allowed": False,
                "options_allowed": False,
                "shorts_allowed": False,
                "individual_stocks_allowed": False,
            },
            "prospective_gates": {
                "minimum_duration_days": 183,
                "realized_volatility_forecast_tracking": True,
                "drawdown_event_tracking": True,
                "no_action_comparison_required": True,
            },
            "invalidation_conditions": [
                "policy_hash_changes",
                "universe_changes",
                "cost_assumption_changes",
                "data_availability_policy_changes",
            ],
            "metrics_tracked": ["realized_volatility_forecasts", "drawdown_events", "exposure", "return", "drawdown", "no_action_comparison"],
        }
    if lab == "offerlab_seller_pilot":
        return {
            "minimum_duration_days": 30,
            "cadence": "daily_or_offer_event",
            "source_versions": {"seller_pilot": "fixture_readiness_passed_v1"},
            "data_cutoff_policy": {
                "actual_seller_action_preserved": True,
                "mature_outcomes_only": True,
                "shadow_decisions_only": True,
            },
            "cost_assumptions": {
                "fees_included": True,
                "shipping_included": True,
                "cost_basis_required": True,
                "cancellations_included": True,
                "returns_included": True,
            },
            "prospective_gates": {
                "seller_pilot_readiness_required": True,
                "minimum_duration_days": 30,
                "maximum_duration_days": 60,
                "causal_claim_allowed": False,
            },
            "invalidation_conditions": [
                "seller_data_schema_changes",
                "shadow_policy_hash_changes",
                "cost_assumption_changes",
                "readiness_gate_revoked",
            ],
            "metrics_tracked": ["seller_shadow_savings", "mature_outcome_count", "return_loss", "cancellation_loss", "no_action_comparison"],
        }
    raise MoneyCanaryError(f"unsupported lab: {lab}")


def _snapshot(protocol: dict[str, Any], observed_at: str, *, snapshot_index: int, reason: str) -> dict[str, Any]:
    lab = str(protocol["lab"])
    metrics = _snapshot_metrics(lab, snapshot_index)
    decision_id = stable_hash({"canary": protocol["canary_id"], "snapshot": snapshot_index, "lab": lab})[:16]
    return {
        "canary_id": protocol["canary_id"],
        "contract_id": protocol["contract_id"],
        "lab": lab,
        "snapshot_id": f"snapshot_{decision_id}",
        "snapshot_index": snapshot_index,
        "reason": reason,
        "observed_at": observed_at,
        "material_hash": protocol["material_hash"],
        "program_hash": protocol["frozen_strategy"]["program_hash"],
        "source_health": {
            "healthy": True,
            "checked_at": observed_at,
            "source_versions": protocol["source_versions"],
            "lag_hours": 0,
        },
        "predictions": [_prediction(lab, observed_at, snapshot_index)],
        "decisions": [_decision(lab, decision_id, observed_at, protocol)],
        "resolutions": [],
        "pnl_or_savings": metrics["pnl_or_savings"],
        "metrics": metrics,
        "counterexamples": [],
        "paper_only": True,
        "production_state": dict(REAL_ACTION_FLAGS),
    }


def _prediction(lab: str, observed_at: str, snapshot_index: int) -> dict[str, Any]:
    if lab == "weather_edge":
        return {
            "target": "daily_high_temperature_bracket",
            "as_of": observed_at,
            "probability": round(0.54 + min(snapshot_index, 10) * 0.001, 4),
            "calibration_bucket": "fixture_warm_regime",
        }
    if lab == "etf_risk":
        return {
            "target": "next_20_trading_day_risk",
            "as_of": observed_at,
            "probability_5pct_drawdown": 0.18,
            "expected_exposure": "normal_exposure",
        }
    return {
        "target": "seller_shadow_offer_decision",
        "as_of": observed_at,
        "expected_shadow_value": 2.5,
        "mature_outcome_required": True,
    }


def _decision(lab: str, decision_id: str, observed_at: str, protocol: dict[str, Any]) -> dict[str, Any]:
    if lab == "weather_edge":
        action = "paper_buy_yes_or_no_trade"
    elif lab == "etf_risk":
        action = "normal_exposure"
    else:
        action = "shadow_accept_or_decline"
    return {
        "decision_id": decision_id,
        "decision_timestamp": observed_at,
        "selected_action": action,
        "no_action_alternative": "cash" if lab == "etf_risk" else "no_action",
        "program_hash": protocol["frozen_strategy"]["program_hash"],
        "material_hash": protocol["material_hash"],
        "paper_only": True,
        "real_action_executed": False,
        "notifications_allowed": False,
    }


def _snapshot_metrics(lab: str, snapshot_index: int) -> dict[str, Any]:
    if lab == "weather_edge":
        pnl = round(snapshot_index * 0.25, 2)
        return {
            "calibration": {"available": snapshot_index >= 2, "sample_count": snapshot_index},
            "brier_score": round(0.24 - min(snapshot_index, 20) * 0.001, 4),
            "log_loss": round(0.68 - min(snapshot_index, 20) * 0.002, 4),
            "paper_pnl": pnl,
            "drawdown": 0.0,
            "city_month_regime_concentration": {"max_share": 1.0, "needs_more_regions": snapshot_index < 20},
            "pnl_or_savings": {"paper_pnl": pnl, "seller_shadow_savings": 0.0},
        }
    if lab == "etf_risk":
        pnl = round(snapshot_index * 1.0, 2)
        return {
            "realized_volatility_forecasts": {"tracked": True, "sample_count": snapshot_index},
            "drawdown_events": {"count": 0},
            "exposure": "normal_exposure",
            "return": pnl,
            "drawdown": 0.0,
            "no_action_comparison": {"cash_return": 0.0, "paper_excess_value": pnl},
            "pnl_or_savings": {"paper_pnl": pnl, "seller_shadow_savings": 0.0},
        }
    savings = round(snapshot_index * 2.5, 2)
    return {
        "seller_shadow_savings": savings,
        "mature_outcome_count": max(0, snapshot_index - 1),
        "return_loss": 0.0,
        "cancellation_loss": 0.0,
        "no_action_comparison": {"actual_seller_action_preserved": True},
        "pnl_or_savings": {"paper_pnl": 0.0, "seller_shadow_savings": savings},
    }


def _aggregate_metrics(protocol: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if not snapshots:
        return {
            "snapshot_count": 0,
            "minimum_duration_progress": 0.0,
            "minimum_duration_elapsed": False,
            "distinct_observation_periods": 0,
            "consecutive_observation_periods": 0,
            "required_observation_periods": 0,
            "elapsed_days": 0,
            "paper_pnl": 0.0,
            "seller_shadow_savings": 0.0,
            "maximum_drawdown": 0.0,
        }
    pnl = round(sum(float(item["pnl_or_savings"].get("paper_pnl", 0.0)) for item in snapshots), 2)
    savings = round(sum(float(item["pnl_or_savings"].get("seller_shadow_savings", 0.0)) for item in snapshots), 2)
    duration = int(protocol["minimum_duration_days"])
    observed_times = sorted(parse_time(str(item["observed_at"])) for item in snapshots)
    start = parse_time(str(protocol["start_at"]))
    elapsed_days = max(0, (observed_times[-1].date() - start.date()).days + 1)
    periods = _observation_periods(str(protocol["lab"]), observed_times)
    period_count = len(periods)
    required_periods = _required_observation_periods(protocol)
    consecutive_periods = _consecutive_observation_periods(str(protocol["lab"]), observed_times)
    elapsed_ok = elapsed_days >= duration and consecutive_periods >= required_periods
    return {
        "snapshot_count": len(snapshots),
        "distinct_observation_periods": period_count,
        "required_observation_periods": required_periods,
        "consecutive_observation_periods": consecutive_periods,
        "elapsed_days": elapsed_days,
        "minimum_duration_elapsed": elapsed_ok,
        "minimum_duration_progress": round(
            min(1.0, min(elapsed_days / max(duration, 1), consecutive_periods / max(required_periods, 1))),
            6,
        ),
        "paper_pnl": pnl,
        "seller_shadow_savings": savings,
        "maximum_drawdown": 0.0,
        "metrics_tracked": list(protocol["metrics_tracked"]),
    }


def _observation_periods(lab: str, observed_times: list[Any]) -> set[str]:
    if lab == "etf_risk":
        return {f"{item.isocalendar().year}-W{item.isocalendar().week:02d}" for item in observed_times}
    return {item.date().isoformat() for item in observed_times}


def _consecutive_observation_periods(lab: str, observed_times: list[Any]) -> int:
    if not observed_times:
        return 0
    if lab == "etf_risk":
        week_starts = sorted({item.date() - timedelta(days=item.weekday()) for item in observed_times})
        return _longest_period_streak(week_starts, step_days=7)
    dates = sorted({item.date() for item in observed_times})
    return _longest_period_streak(dates, step_days=1)


def _longest_period_streak(periods: list[Any], *, step_days: int) -> int:
    if not periods:
        return 0
    best = 1
    current = 1
    for previous, current_period in zip(periods, periods[1:]):
        if (current_period - previous).days == step_days:
            current += 1
        else:
            current = 1
        best = max(best, current)
    return best


def _required_observation_periods(protocol: dict[str, Any]) -> int:
    if protocol["lab"] == "etf_risk":
        return max(1, int(protocol["minimum_duration_days"]) // 7)
    return int(protocol["minimum_duration_days"])
