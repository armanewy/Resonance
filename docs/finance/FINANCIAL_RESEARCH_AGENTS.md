# Financial Research Agents

Wave: `FINANCE_WAVE_2E`

This package defines bounded LLM-facing agents for financial research labor in
`behavior_lab.money_agents`. The agents are advisory only. They do not place
trades, submit offers, purchase inventory, promote data sources, change
statistical thresholds, inspect blind outcomes, select blind winners, rerun
consumed blind evaluations, or declare strategies valid.

The runtime is intentionally provider-agnostic. A caller supplies a provider
object with a `complete(request)` method and receives validated, append-only
audit records through the existing research-store pattern. Tests use
`StaticMoneyAgentProvider`; no test invokes a live LLM or network service.

## Roles

### Financial Source Scout

Allowed work:

- search or read official financial providers
- compare documented licenses, rate limits, and timestamp semantics
- propose metrics and connector candidates
- reject sources with unclear licensing, non-official provenance, or
  undocumented timestamp semantics

Forbidden work:

- activate a source
- accept unclear licensing
- infer undocumented timestamps
- treat source availability as predictive evidence

### Financial Hypothesis Scientist

Allowed work:

- propose structured executable hypotheses using lagged features,
  interactions, regimes, forecast revisions, liquidity effects, seller-policy
  hypotheses, or risk-state hypotheses
- include required data, executable feature plans, falsification tests, and
  lineage

Forbidden work:

- emit trading strategy source code
- emit trade, offer, purchase, or marketplace-action instructions
- declare any hypothesis commercially valid

### Financial Skeptic

Required audit checks:

- timing leakage
- survivorship bias
- corporate-action leakage
- selection bias
- target leakage
- stale pricing
- non-executable prices
- omitted costs
- correlated outcomes
- regime concentration
- prior failed equivalent hypotheses

The skeptic may recommend rejection or remediation. It may not access blind
outcomes, select winners, rerun consumed blind evaluations, or validate a
strategy.

### Connector Maintenance Diagnostician

Allowed work:

- diagnose connector issues using read-only checks, offline replay, or mock
  replay
- produce symptoms, repair plans, and maintenance tickets

Forbidden work:

- mutate provider state
- activate connectors
- promote sources into production
- change thresholds

### Weekly Research Allocator

Allowed work:

- allocate already-approved weekly hours, cost, and tool-call capacity across
  role work items
- defer work that exceeds explicit budgets or authority

Forbidden work:

- create new budgets
- exceed explicit budgets
- authorize blind access, source promotion, or real actions

## Audit Records

Each completed or rejected provider response is persisted as a hash-linked
event with:

- provider and model
- prompt version
- request and response hashes
- tool calls
- citations
- token and cost usage
- role ID and campaign ID
- proposal IDs, rejection IDs, and parent lineage
- authority boundaries active for the run

Invalid provider output is persisted as `money_agent_rejected` with the error
type, error text, metadata, and lineage available from the rejected response.

## Integration Hooks Needed

The package does not wire a live provider, scheduler, connector registry, or
MoneyLedger decision writer. Future integration should supply:

- a production LLM provider adapter that returns `ProviderResponse`
- a caller-owned state path for append-only agent audit events
- a read-only official-provider tool facade for source scouting
- connector registry metadata for the diagnostician
- an external scheduler that passes explicit weekly budgets
- a downstream human review step before any MoneyLedger paper decision or
  connector implementation work
