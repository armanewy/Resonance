# Financial Operations Release

This release prepares the financial laboratories for unattended local evidence
collection. It does not add models, sources, instruments, decision types,
notifications, seller mutations, broker access, exchange authentication, or
real financial actions.

## Commands

```powershell
behavior-lab money operations start --state-dir C:\ResonanceOps
behavior-lab money operations status --state-dir C:\ResonanceOps
behavior-lab money operations doctor --state-dir C:\ResonanceOps
behavior-lab money operations weekly-report --state-dir C:\ResonanceOps
behavior-lab money operations recover --state-dir C:\ResonanceOps
behavior-lab money operations stop --state-dir C:\ResonanceOps
```

`start` writes an immutable `release_manifest.json` containing the audited
commit, contract hashes, canary hashes, program hashes, source versions,
thresholds, costs, fee/slippage assumptions, start dates, and scheduled end
dates. A material change requires a new state directory and new canaries.

`recover` is intended for scheduled-task style operation. It resumes missed
Weather Edge daily cycles and ETF weekly cycles only when source health is
fresh. It never repeats blind evaluation and it cannot mutate frozen canary
logic.

## Seller Onboarding

```powershell
behavior-lab offerlab-pilot onboard C:\SellerExports --output C:\SellerExports\readiness.json
```

Onboarding inspects exports, proposes mappings, validates them
deterministically, and emits a data-readiness report. Seller data remains local.
Material ambiguity, missing cost basis, or incomplete fee/shipping data blocks
canary start.

## Weekly Report

The weekly report leads with paper/shadow value, resolved and unresolved
decisions, no-action rate, hypothetical capital at risk, drawdown, calibration,
source health, research/API cost, canary comparability, seller readiness, and
approval requirements.

All outputs remain paper or shadow-only.
