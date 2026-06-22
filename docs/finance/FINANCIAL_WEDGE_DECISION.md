# Financial Wedge Decision

Generated at: `2026-07-05T12:00:00+00:00`
Top-level result: `CONTINUE_MULTIPLE_CANARIES`
Selected wedge: `None`

## Rationale

- All current evidence is fixture or immature paper evidence.
- No contract has enough prospective canary evidence for commercial selection.
- No winner is selected from synthetic tests.

## Contract Classifications

- `etf_risk`: `CONTINUE_PAPER_RESEARCH`
- `offerlab_seller_pilot`: `CONTINUE_PAPER_RESEARCH`
- `weather_edge`: `DATA_STARVED`

## Capital Policy

No real capital is authorized. All entries are paper or shadow decisions, and the production-state flags are false.

## 90-Day Plan

- Continue the three prospective canaries without strategy mutation.
- Prioritize private seller-pilot data only if the seller readiness gate passes.
- Keep Weather Edge and ETF Risk as paper research until their duration and evidence gates mature.
- Re-run this tournament only from append-only ledger and canary records.

## Kill Criteria

- Stop a lab if source health fails, material costs are unknown, or paper value remains economically weak after prospective validation.
- Do not promote any fixture-only or synthetic-only result into a commercial wedge.

## Output Artifacts

- `reports/finance/FINANCIAL_TOURNAMENT.json`
- `reports/finance/FINANCIAL_TOURNAMENT.html`
