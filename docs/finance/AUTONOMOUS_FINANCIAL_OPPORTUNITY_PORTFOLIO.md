# Autonomous Financial Opportunity Portfolio

The opportunity portfolio coordinates the paper-only finance labs, Contract Scout,
and the experimental Data Mesh. It is not a trading system and cannot mutate
seller accounts, activate production sources, allocate real money, or send
ordinary research notifications.

## Behavior

- Continuous cycles collect active data, advance prospective candidates, resolve
  paper decisions, and keep blocked contracts isolated.
- Nightly cycles run deterministic search, source health/repair checks, and paper
  outcome accounting.
- Weekly cycles run bounded source research, Scientist/Skeptic bookkeeping,
  Contract Scout, source marginal-value review, and a research digest.
- Monthly cycles reallocate research budget, pause low-value contracts, extend
  promising canaries, and retire redundant sources.

## Notification Gate

Only these notification classes are emitted:

- `approval_required`
- `prospectively_verified_paper_opportunity`
- `operational_failure_requires_authority`

Routine hypotheses, rejected hypotheses, ordinary no-action decisions, source
health, and model metrics stay in the weekly report.

## CLI

```powershell
python -m behavior_lab.cli money seek-value --mode paper --monthly-budget 40
python -m behavior_lab.cli money opportunity-portfolio run --schedule weekly
python -m behavior_lab.cli money opportunity-portfolio weekly-report
```

All output remains paper-only and includes production-state flags showing that no
real action was executed.
