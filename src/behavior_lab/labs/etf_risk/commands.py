from __future__ import annotations

from typing import Any

from behavior_lab.money.accounting import summarize_money_entries
from behavior_lab.money.ledger import MoneyLedger

from behavior_lab.labs.etf_risk.engine import ETFRiskConfig, ETFRiskLab, real_money_eligibility
from behavior_lab.labs.etf_risk.market_data import AuthorizedMarketDataProvider


def backfill(
    provider: AuthorizedMarketDataProvider,
    *,
    ledger_path: str,
    config: ETFRiskConfig | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Run paper-only historical walk-forward decisions and write each to MoneyLedger."""

    lab = ETFRiskLab(provider, config)
    return lab.walk_forward(start=start, end=end, ledger_path=ledger_path, write_ledger=True)


def paper_cycle(
    provider: AuthorizedMarketDataProvider,
    *,
    ledger_path: str,
    config: ETFRiskConfig | None = None,
    decision_cutoff: str | None = None,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    """Run one prospective paper decision and append it to MoneyLedger."""

    lab = ETFRiskLab(provider, config)
    return lab.paper_cycle(ledger_path=ledger_path, decision_cutoff=decision_cutoff, strategy_id=strategy_id)


def report(
    provider: AuthorizedMarketDataProvider,
    *,
    ledger_path: str,
    config: ETFRiskConfig | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Build a paper-lab report without appending new ledger entries."""

    cfg = config or ETFRiskConfig()
    lab = ETFRiskLab(provider, cfg)
    ledger = MoneyLedger(ledger_path)
    entries = ledger.latest_entries()
    walk_forward = lab.walk_forward(start=start, end=end, write_ledger=False)
    return {
        "lab_version": walk_forward["lab_version"],
        "paper_only": True,
        "ledger_verified": ledger.verify(),
        "money_summary": summarize_money_entries(entries),
        "walk_forward_metrics": walk_forward["metrics"],
        "real_money_eligibility": real_money_eligibility(entries, cfg),
        "integration_hooks_needed": [
            "wire behavior-lab money etf-risk backfill to behavior_lab.labs.etf_risk.commands.backfill",
            "wire behavior-lab money etf-risk paper-cycle to behavior_lab.labs.etf_risk.commands.paper_cycle",
            "wire behavior-lab money etf-risk report to behavior_lab.labs.etf_risk.commands.report",
        ],
    }
