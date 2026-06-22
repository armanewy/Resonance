# Weather Edge Paper Lab

Wave 2B adds a paper-only, multicity daily high-temperature event-market lab.
It is isolated under `behavior_lab.labs.weather_edge` and does not wire shared
CLI commands in this branch.

## Scope

Weather Edge evaluates binary YES contracts for daily high-temperature brackets.
The event provider owns discovery and exact market/weather semantics. The engine
does not hard-code supported cities, stations, dates, or brackets.

Each `DailyHighTemperatureEvent` preserves:

- settlement series
- station ID and station name
- report source and report name
- local date, timezone, and DST status
- bracket bounds and inclusivity
- market open, close, and resolution timestamps

One `city_event_key` represents one independent city-day outcome. If a provider
exposes multiple bracket contracts for the same station/date/report, the lab
evaluates them but records at most one paper decision for that city-event.

## Provider Interface

Command-equivalent functions accept a provider implementing:

- `discover_events(as_of, include_resolved=False)`
- `market_depth(event_id, as_of)`
- `weather_snapshot(event_id, as_of)`
- `settlement(event_id)`
- `station_history(station_id, before_local_date=...)`

`FixtureWeatherEdgeProvider` implements this interface for local tests and JSON
fixtures. Market depth and weather snapshots are selected as latest records at
or before the decision time.

## Command-Equivalent Functions

The package exports:

- `backfill(provider, storage_root, as_of=...)`
- `paper_cycle(provider, storage_root, as_of=...)`
- `report(storage_root, provider=..., as_of=...)`

These are intentionally callable surfaces only. Later integration can wire
`behavior-lab money weather-edge ...` without changing this branch's shared CLI.

## Execution Semantics

Paper fills use the executable best YES ask from the current order book.
Midpoints, candle highs, candle lows, and other non-executable prices are never
treated as fills. The raw order-book quantity is copied into ledger provenance,
and paper quantity is bounded by the configured liquidity fraction and maximum
contract count.

The default fixed decision horizon is `close_minus_6h`. Backfill uses that
horizon for every event and evaluates walk-forward: station history is filtered
to dates before the target local date.

## Baselines

Each decision records:

- market-implied probability from executable YES ask
- station climatology from prior station high-temperature outcomes
- official forecast distribution probability
- station-bias-corrected forecast distribution probability

Station bias is the prior average of `observed_high_f - forecast_mean_f`; it is
then applied as a temperature shift to the official forecast distribution.

## Ledger and Costs

Every paper trade and explicit no-trade decision is represented as a
`MoneyLedgerEntry` with `designation="paper"` and `evidence_state="paper_decision"`.
Historical backfill resolves entries to `resolved_paper` when settlement is
available.

Cost accounting includes:

- per-contract fees
- per-contract slippage
- liquidity limits
- uncertainty buffer
- maximum possible loss bounded by cash outlay

No leverage is allowed. Unknown material costs are not imputed.

## Safety Boundary

This lab never authenticates for trading, never submits orders, never places
real trades, and never creates notifications. Contracts and ledger provenance
record those disabled states explicitly.

Strategy changes are not allowed during prospective incubation. Reports expose
the strategy versions observed across prospective paper-cycle entries.

## Evidence Gate

`report(...)` returns an evidence gate with:

- 150+ resolved city-days required when historical data permits
- comparison against the market-implied baseline
- pessimistic fee/slippage sensitivity
- city/month/regime concentration diagnostics
- at least 30 prospective paper days before any later real-money review
- prospective strategy-lock check

`real_money_enabled_in_this_wave` is always `False`.
