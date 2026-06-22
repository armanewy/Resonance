# ETF Risk Laboratory

Finance Pivot Wave 2C adds a paper-only broad-ETF risk/allocation laboratory.
It does not trade, connect to brokers, place orders, send notifications, or
cover individual stocks, options, leverage, shorts, intraday strategies, or HFT.

## Universe

The default universe is a configurable small set of broad proxies:

- US equities
- International equities
- Treasury bonds
- Investment-grade credit
- Gold
- Broad commodities
- Cash or cash-equivalent

Each asset is represented by an `AssetSpec` role, not by a broker instrument.
Universe validation rejects forbidden exposure terms such as options, leverage,
shorts, broker/order APIs, individual stocks, intraday, and HFT.

## Market Data Semantics

The lab consumes a provider-neutral `AuthorizedMarketDataProvider`. Providers
must expose:

- an explicit `DataAuthorization`
- a `MarketCalendar`
- adjusted price observations with market date, event time, availability time,
  source, revision ID, and adjustment policy

`AdjustedPrice.adjusted_close` preserves split/distribution semantics through an
explicit `adjustment_policy`. Corrections are selected only when their
`availability_time` is at or before the decision cutoff, so later revised values
cannot leak backward. Every MoneyLedger decision records the exact price
snapshot available at that cutoff.

## Targets

The lab forecasts and evaluates three 20-trading-day targets:

- next-20-trading-day realized volatility
- probability of a 5% drawdown
- probability equities underperform cash over 20 trading days

The target-only autoregression baseline uses prior 20-trading-day target windows
from data available at the decision cutoff.

## Actions

All actions are paper allocation states:

- `cash`
- `low_exposure`
- `normal_exposure`

Weights are long-only, unlevered, fully invested, and limited to the broad
universe. `cash` is the no-action comparator.

## Baselines

Walk-forward evaluation includes:

- buy and hold
- fixed allocation
- simple momentum
- volatility scaling
- target-only autoregression
- cash

## Evaluation

Evaluation is walk-forward only with a weekly decision cadence. Reports include
turnover, transaction-cost assumptions, maximum drawdown, calibration, risk-
adjusted return, no-action comparison, regime/period concentration, and
parameter-neighborhood sensitivity.

Every backfill and paper-cycle decision is appended to `MoneyLedger` as a
`paper_decision`. Ledger entries preserve the contract hash, feature program
hash, data cutoff, action alternatives, selected action, no-action alternative,
cost assumptions, provenance, and exact price snapshot.

Real-money eligibility is always blocked in this wave. A later review would
require at least six months of prospective paper decisions, plus separate
authorization and implementation work outside Wave 2C.

## Callable Hooks

Shared CLI wiring is intentionally not modified in this branch. Future command
wiring can call:

- `behavior_lab.labs.etf_risk.commands.backfill`
- `behavior_lab.labs.etf_risk.commands.paper_cycle`
- `behavior_lab.labs.etf_risk.commands.report`
