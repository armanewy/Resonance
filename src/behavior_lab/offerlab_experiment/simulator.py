from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentAssumptions:
    listings: int
    item_cost: float
    asking_price: float
    baseline_sale_probability: float
    baseline_sale_price: float
    fee_rate: float = 0.1325
    shipping_cost: float = 0.0
    holding_cost_per_listing_day: float = 0.0
    horizon_days: int = 30
    experiment_setup_cost: float = 0.0
    max_discount_fraction: float = 0.15

    def validate(self) -> None:
        if self.listings <= 0:
            raise ValueError("listings must be positive")
        for name in ["item_cost", "asking_price", "baseline_sale_price", "shipping_cost", "holding_cost_per_listing_day", "experiment_setup_cost"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} may not be negative")
        for name in ["baseline_sale_probability", "fee_rate", "max_discount_fraction"]:
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")


def simulate_self_funded_experiment(assumptions: ExperimentAssumptions | dict[str, Any]) -> dict[str, Any]:
    config = assumptions if isinstance(assumptions, ExperimentAssumptions) else ExperimentAssumptions(**assumptions)
    config.validate()
    strategies = [
        _strategy(config, name="no_experiment_current_policy", probability_delta=0.0, price_discount=0.0, cells=1, setup_multiplier=0.0),
        _strategy(config, name="shadow_decision_support", probability_delta=0.0, price_discount=0.0, cells=1, setup_multiplier=0.35),
        _strategy(config, name="two_policy_randomized_test", probability_delta=0.07, price_discount=0.05, cells=2),
        _strategy(config, name="multi_price_randomized_test", probability_delta=0.12, price_discount=0.10, cells=4, power_penalty=0.08),
    ]
    baseline = strategies[0]["expected_contribution_margin"]
    for row in strategies:
        row["expected_incremental_margin_vs_no_experiment"] = round(row["expected_contribution_margin"] - baseline, 2)
    return {
        "assumptions": asdict(config),
        "strategies": strategies,
        "recommended_next_step": _recommend(strategies),
        "primary_metric": "expected mature contribution margin over experiment horizon",
        "warnings": [
            "This is a planning simulator, not evidence of causal lift.",
            "All actions must remain within the seller-approved maximum discount.",
            "Replace assumptions with observed store data before spending real money.",
        ],
    }


def _strategy(
    config: ExperimentAssumptions,
    *,
    name: str,
    probability_delta: float,
    price_discount: float,
    cells: int,
    setup_multiplier: float = 1.0,
    power_penalty: float = 0.0,
) -> dict[str, Any]:
    discount = min(price_discount, config.max_discount_fraction)
    sale_probability = min(0.98, max(0.0, config.baseline_sale_probability + probability_delta - power_penalty))
    sale_price = config.baseline_sale_price if discount == 0 else config.asking_price * (1.0 - discount)
    sold = config.listings * sale_probability
    unsold = config.listings - sold
    margin_per_sale = sale_price * (1.0 - config.fee_rate) - config.shipping_cost - config.item_cost
    holding_cost = unsold * config.holding_cost_per_listing_day * config.horizon_days
    setup_cost = config.experiment_setup_cost * setup_multiplier
    total = sold * margin_per_sale - holding_cost - setup_cost
    return {
        "strategy": name,
        "cells": cells,
        "sale_probability": round(sale_probability, 4),
        "expected_units_sold": round(sold, 2),
        "expected_sale_price": round(sale_price, 2),
        "expected_margin_per_sale": round(margin_per_sale, 2),
        "expected_holding_cost": round(holding_cost, 2),
        "experiment_setup_cost": round(setup_cost, 2),
        "expected_contribution_margin": round(total, 2),
    }


def _recommend(strategies: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(strategies, key=lambda row: row["expected_contribution_margin"], reverse=True)
    winner = ranked[0]
    baseline = next(row for row in strategies if row["strategy"] == "no_experiment_current_policy")
    if winner["strategy"] == "no_experiment_current_policy" or winner["expected_contribution_margin"] <= 0 or winner["expected_contribution_margin"] <= baseline["expected_contribution_margin"]:
        return {
            "status": "do_not_spend_yet",
            "strategy": winner["strategy"],
            "reason": "no tested experiment strategy has positive expected margin advantage over current policy",
        }
    return {"status": "candidate_budget", "strategy": winner["strategy"], "expected_margin": winner["expected_contribution_margin"]}
