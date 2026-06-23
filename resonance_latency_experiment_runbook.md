# Resonance Latency Experiment Runbook

Active line:

```text
branch: product/resonance-app
required ancestor: f7e3b34b2d54dcaf3cecbe0e59517bea36a090aa
```

Goal:

```text
Test whether transformed TCP latency is predicted by recent upload throughput,
CPU activity, charging state, or a simple interaction among those local signals.
```

## Dry Rehearsal

Run this while fresh data is still sparse:

```powershell
cd C:\Users\aoztu\Downloads\Resonance
.\resonance_latency_experiment_driver.ps1 -AllowEarly
```

This creates an operator log under:

```text
data\science\operator_logs\latency_YYYYMMDD_HHMMSS\
```

The dry rehearsal verifies the branch and required lineage, audits the
collector, verifies the ledger, runs a scanner dry-run, creates a snapshot when
enough local rows exist, generates deterministic file-provider hypotheses,
approves only review-accepted hypotheses, fits them, and runs tuning. It does
not preregister or spend the blind budget.

`-AllowEarly` uses an identity target transform and zero maximum lag so it can
exercise the operator path before several days of regular observations exist.
The normal pre-blind run uses robust-z-scored TCP latency and the normal lag
window.

## Normal Pre-Blind Run

After local collection has enough coverage:

```powershell
cd C:\Users\aoztu\Downloads\Resonance
.\resonance_latency_experiment_driver.ps1
```

Valid pre-blind outcomes:

```text
NO_TUNING_WINNER
TUNING_WINNER_SELECTED
SNAPSHOT_NOT_READY
NO_VALID_PROPOSALS
```

`NO_TUNING_WINNER` is a valid scientific outcome.

## Blind Spend

Only spend blind after an existing run reports:

```json
"status": "TUNING_WINNER_SELECTED"
```

Use the exact existing run directory:

```powershell
cd C:\Users\aoztu\Downloads\Resonance
.\resonance_blind_spend_existing_run.ps1 `
  -RunDir "data\science\operator_logs\latency_YYYYMMDD_HHMMSS" `
  -ConfirmBlindSpend
```

Do not rerun the full driver to spend blind. That could create a different
snapshot or selected candidate. The blind helper reads the existing
`operator_result.json`, preregisters the exact `selected_candidate_id`, runs
one blind evaluation, writes the report, verifies the ledger, and updates the
same operator result.

The blind helper requires the current Git commit to match the commit recorded
by the run that selected the tuning winner. This keeps tuning and blind
evaluation tied to the same code path.

## Do Not Do

```text
Do not merge main.
Do not port BehaviorDiscoveryLab finance code.
Do not add sources or models for this experiment.
Do not enable OpenAI.
Do not change thresholds to force a winner.
Do not spend blind before tuning selects a candidate.
Do not repeat blind evaluation.
Do not claim causality from this observational run.
```
