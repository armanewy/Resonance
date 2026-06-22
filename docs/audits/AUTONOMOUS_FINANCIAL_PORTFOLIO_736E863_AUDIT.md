# Autonomous Financial Opportunity Portfolio Audit

Verdict: PASS after audit fixes

Commit audited: `736e8636f7d12f1bc66a0b8821996b5117736f0a`

Feature: Wave 3 autonomous financial opportunity portfolio

Worktree: `C:\Users\aoztu\Downloads\BehaviorDiscoveryLab-audit-worktrees\AUTONOMOUS_FINANCIAL_PORTFOLIO-736e863`

## Scope Reviewed

- `src/behavior_lab/money/portfolio.py`
- `src/behavior_lab/cli.py` opportunity-portfolio and seek-value commands
- `tests/money/test_opportunity_portfolio.py`
- `docs/finance/AUTONOMOUS_FINANCIAL_OPPORTUNITY_PORTFOLIO.md`
- Relevant Contract Scout, Data Mesh, MoneyAutopilot, canary, and operations boundaries.

## Findings

### Fixed: Human approval budget was cycle-local for portfolio-level approvals

Severity: High

Original behavior: `AutonomousFinancialOpportunityPortfolio._notifications` counted approvals and paper-opportunity alerts only within the current notification batch. Data Mesh missing-source approvals and Contract Scout approval notifications could be emitted again in a later cycle even when the weekly approval budget had already been consumed.

Impact: The portfolio could exceed the human attention budget for approval notifications across repeated weekly runs, violating the attention-budget requirement even though per-cycle notifications were capped.

Fix: Notification gating now checks historical `opportunity_portfolio_notifications_evaluated` events in the weekly approval window and daily paper-opportunity alert window before emitting new notifications. Weekly reports now use the same persisted counters.

References:

- `src/behavior_lab/money/portfolio.py:493`
- `src/behavior_lab/money/portfolio.py:630`
- `src/behavior_lab/money/portfolio.py:648`
- `tests/money/test_opportunity_portfolio.py:86`

### Fixed: Local paths were not redacted from portfolio notifications and reports

Severity: Medium

Original behavior: Portfolio redaction covered secret-looking keys and values, but not local filesystem paths. A local source path could flow into missing-source approval notifications and into the weekly report notification payload.

Impact: Notifications or reports copied off-machine could disclose local usernames or private directory structure.

Fix: Portfolio redaction now masks Windows absolute paths, UNC paths, and common POSIX local absolute paths. Regression coverage verifies notification and weekly-report payloads do not expose `C:\Users\...` paths.

References:

- `src/behavior_lab/money/portfolio.py:730`
- `src/behavior_lab/money/portfolio.py:741`
- `src/behavior_lab/money/portfolio.py:748`
- `tests/money/test_opportunity_portfolio.py:107`

## Audit Question Results

1. Paper-only boundary: PASS. Portfolio, autopilot, Contract Scout, Data Mesh, canary, and operations paths keep real-action flags false and reject trade/broker/seller-mutation shapes.
2. Blocked contract isolation: PASS. Blocked seller/private-data contracts do not enter runnable autopilot configs and do not stop Weather Edge or ETF paper work.
3. Budget allocation signals: PASS. Allocation uses economic value, information gain, cadence, cost, failure rate, uncertainty, evidence need, and deadline relevance, with blocked/retired and dead-end contracts deprioritized.
4. Human attention budgets: PASS after fix. Portfolio-level approval and paper-opportunity notification gates now account for persisted weekly/daily usage.
5. Notification classes: PASS. Emitted notification kinds are limited to `approval_required`, `prospectively_verified_paper_opportunity`, and `operational_failure_requires_authority`; routine hypotheses/source health/model metrics stay out of notifications.
6. Contract Scout integration: PASS. Unsupported and seller/private-data families are not auto-activated; seller shadow contracts are blocked.
7. Data Mesh integration: PASS. Acquisition activates experimental catalog sources only and rejects production activation authority.
8. Scheduling and repeated evaluation: PASS. Continuous/nightly/weekly/monthly task maps are distinct, and MoneyAutopilot prevents repeated blind evaluation and unchanged failed connector retries.
9. Weekly report fields: PASS. Report leads with paper value, prospective value, hypothetical capital at risk, drawdown, no-action rate, costs, attention budget/time, allocations, source changes, and non-repeated failures.
10. Secrets/local paths: PASS after fix. Secrets and local paths are redacted from portfolio notifications and weekly reports.
11. Full suite: PASS.

## Tests Run

- `python -m pytest tests/money/test_opportunity_portfolio.py -q`
- `python -m pytest tests/money/test_contract_scout.py tests/finance_data/test_data_mesh.py tests/money/test_autopilot.py tests/money/test_canary.py tests/money/test_operations.py -q`
- `python -m pytest -q`

## Changed Files

- `src/behavior_lab/money/portfolio.py`
- `tests/money/test_opportunity_portfolio.py`
- `docs/audits/AUTONOMOUS_FINANCIAL_PORTFOLIO_736E863_AUDIT.md`
- `docs/audits/AUTONOMOUS_FINANCIAL_PORTFOLIO_736E863_AUDIT.json`
