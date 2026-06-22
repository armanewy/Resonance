# Autonomous Contract Scout Audit - 41b2d94

Verdict: PASS after audit fix

Commit audited: 41b2d9470650f24814c95e8125b35a24a0fe4c1a

Feature audited: Autonomous Financial Contract Scout, including remediations after prior audit failures.

## Scope

- `src/behavior_lab/money/contract_scout.py`
- `src/behavior_lab/money_agents/roles.py`
- `tests/money/test_contract_scout.py`

## Findings

### Fixed P1: Nested activation/allocation authority fields could bypass pre-eligibility rejection

The audited commit rejected top-level authority fields such as `production_source_activation=True` and `money_allocation={...}`, but `ContractScout.run()` only inspected those fields at the proposal root before converting otherwise valid proposals to `OpportunityContractProposal`. Extra authority metadata nested under valid schema objects, such as `resolution_source.source_activation`, `paper_mode_feasibility.contract_activation`, or `payoff_formula.capital_allocation`, could therefore be silently dropped by model coercion and the proposal could remain eligible.

References:

- `src/behavior_lab/money/contract_scout.py:98`
- `src/behavior_lab/money/contract_scout.py:685`
- `tests/money/test_contract_scout.py:209`

Fix:

- Added recursive raw proposal authority scanning in `ContractScout.run()` validation preflight.
- Included `activation_status` in the shared authority-field reason map.
- Added preservation coverage proving rejected malformed/nested-authority proposals keep their sanitized raw shape and reasons.

### Fixed P1: Money-agent Contract Scout role only checked top-level contract authority fields

`FinancialContractScout.validate_content()` had the same top-level-only blind spot. Nested `activation_status=activated` or `capital_allocation={...}` inside a structured proposal could pass the role-specific boundary check before normal proposal coercion discarded the extra fields.

References:

- `src/behavior_lab/money_agents/roles.py:151`
- `src/behavior_lab/money_agents/roles.py:258`
- `src/behavior_lab/money_agents/roles.py:562`
- `tests/money/test_contract_scout.py:493`

Fix:

- Added recursive Contract Scout authority scanning in the role validator.
- Added independent subtests for nested production activation and nested capital allocation.

## Audit Answers

1. Paper-only and no real actions: PASS after fix. Eligible, approval, approval action, run, and report payloads remain paper-only and explicitly set `production_source_activation=False` and `money_allocation=False`. Real action-shaped actions are rejected.
2. Malformed proposal rejection and preservation: PASS. Wrong-shaped capital/loss payloads are deterministically rejected, and raw sanitized malformed payload shape is preserved in rejected records.
3. Extra authority fields rejected before eligibility: PASS after fix. Top-level and nested activation/allocation fields are rejected pre-coercion.
4. Seller/account/listing mutation-shaped actions rejected: PASS. `seller.update_listing` / `revise_listing_price` action shapes are rejected with `proposed_real_action`.
5. First-audit blockers: PASS. Coverage confirms material-cost representation, unknown costs, bounded capital, bounded max loss, non-paper proposals, secret redaction, private seller data ambiguity, and operations context reading.
6. Rejected proposal preservation and secret redaction: PASS. Rejected records include reasons and sanitized proposals; secret-like values are redacted from run output, report/proposals output, and raw append-only state.
7. Append-only behavior and no production promotion: PASS. `AppendOnlyResearchStore` hash-chains appended JSONL events and verifies the chain on reads/writes; Contract Scout only appends proposal/report/approval/rejection records and does not invoke external mutation or production activation paths.

## Tests Run

- `python -m pytest tests\money\test_contract_scout.py`
  - `33 passed in 0.94s`
- `python -m pytest tests\money\test_contract_scout.py tests\money_agents`
  - `44 passed, 2 subtests passed in 0.83s`
- `python -m pytest`
  - first attempt timed out at 120s without a failure summary
  - rerun after final changes: `414 passed, 103 subtests passed in 118.29s`

## Changed Files

- `src/behavior_lab/money/contract_scout.py`
- `src/behavior_lab/money_agents/roles.py`
- `tests/money/test_contract_scout.py`
- `docs/audits/AUTONOMOUS_CONTRACT_SCOUT_41B2D94_AUDIT.md`
- `docs/audits/AUTONOMOUS_CONTRACT_SCOUT_41B2D94_AUDIT.json`
