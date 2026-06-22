# MoneyLedger

The MoneyLedger is the shared financial-decision substrate for the private
financial decision laboratory. It records paper decisions and future
human-approved real outcomes as append-only, hash-linked records.

This wave defines contracts, storage, accounting, and an OfferLab mapping only.
It does not place trades, submit seller actions, purchase inventory, create
notifications, or approve real-money activity.

## Contracts

`FinancialDecisionContract` defines:

- domain: `seller`, `event_market`, `etf_risk`, or `procurement`
- target and horizon
- available actions and explicit no-action alternative
- payoff, cost, risk, liquidity, resolution, and data-cutoff policies
- prospective requirements
- notification thresholds
- paper-only status
- contract version

Only `reactive` actions are eligible for automatic deterministic evaluation.
`interventional` actions may be represented as alternatives but are not
recommendation-eligible in this wave.

## Ledger Entries

`MoneyLedgerEntry` records the immutable decision ID, contract hash, timestamp,
data cutoff, target, action alternatives, selected action, no-action
alternative, capital, maximum possible loss, expected value, uncertainty,
costs, conservative net value, deadline, feature/program hash, evidence state,
paper/real designation, resolution, realized value, no-action outcome,
provenance, artifact hashes, and assumption versions.

Evidence states are:

- `proposed`
- `historically_evaluated`
- `blind_passed`
- `prospectively_incubating`
- `prospectively_verified`
- `paper_decision`
- `resolved_paper`
- `manually_approved_real`
- `resolved_real`
- `rejected`
- `expired`

This wave rejects creation of `manually_approved_real` entries and rejects all
`designation="real"` MoneyLedger entries. Real action approval requires a later
explicit wave with a verifiable approval state machine.

## Accounting Rules

Accounting is deterministic and conservative:

- unknown material costs are never treated as zero
- unknown material costs make a decision ineligible
- known-cost ledger entries must explicitly set every cost field, even when
  net value is pending
- ordinary cost components may not be negative
- fees, slippage, shipping, holding costs, refunds, return losses,
  cancellations, and research/API costs are explicit
- paper and real outcomes cannot be summarized together
- drawdown is calculated from an ordered value curve
- resolved entries require a mechanically defined no-action outcome
- realized net value must reconcile to realized gross value minus realized
  costs, and realized costs may not be unknown
- value summaries preserve the no-action comparator and group by contract,
  strategy, and source
- multiple decisions from one economic event are counted as one opportunity

## Append-Only Corrections

The MoneyLedger wraps the existing local `ImmutableLedger`. Entries are
hash-linked. Resolution appends a new entry that supersedes the previous entry
hash. Corrections also append superseding records and must state a reason.
Original decision forecasts are not rewritten.

## OfferLab Mapping

The minimal OfferLab adapter maps seller-pilot shadow reports to paper
MoneyLedger entries. It preserves mature contribution margin, fee/shipping/cost
coverage gates, refunds, returns, cancellation/unpaid-order counts, redacted
data-quality gaps, and the non-causal historical-comparison boundary.

The adapter selects `abstain` in this wave. It records shadow-policy preparation
as provenance only and never mutates seller state.

## Storage Boundary

Contract IDs used by `MoneyStorage` must be simple filename tokens containing
only letters, numbers, underscores, and hyphens. Path separators and traversal
segments are rejected before contract files are read or written.
