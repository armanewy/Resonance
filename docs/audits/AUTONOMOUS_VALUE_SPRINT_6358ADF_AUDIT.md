# Autonomous Value Sprint 6358adf Audit

Verdict: PASS after audit fixes.

Commit audited: 6358adf96c3b62f81c9e052409265a12afca8bdb

Feature: Wave 4 frozen 30-day Autonomous Value Sprint.

## Scope

- `src/behavior_lab/money/value_sprint.py`
- `src/behavior_lab/cli.py` `money value-sprint run`
- `tests/money/test_value_sprint.py`
- Interactions with money portfolio, data mesh, contract scout, and autopilot.

## Audit Result

The sprint now creates:

- `AUTONOMOUS_VALUE_SPRINT.html`
- `AUTONOMOUS_VALUE_SPRINT.json`
- `VALUE_SYSTEM_DECISION.md`

The generated sprint remains paper-only. Production-state flags stay false, data mesh source activation and repair are experimental-only, and the top-level decision is one of the six specified values. The default 30-day sprint decision is `AUTONOMY_WORKS_NO_EDGE_YET`, which is defensible from the evidence: prospective survivors exist, but paper opportunities and resolved paper value are zero.

## Findings

### Fixed P1: success criteria were not fully evidence-derived

Several criteria and required evidence fields were constants or weak proxies, including source repair, blind/prospective survivors, repeated blind evaluation, candidate progress, and 30-day completion. This could make the sprint pass even when evidence contradicted the report.

Fix: final evidence is now derived from daily portfolio/autopilot/data-mesh run summaries, short runs no longer satisfy the 30-day criterion, blind reuse is checked across runs, and the top-level decision degrades when criteria fail.

References:

- `src/behavior_lab/money/value_sprint.py:230`
- `src/behavior_lab/money/value_sprint.py:266`
- `src/behavior_lab/money/value_sprint.py:330`
- `tests/money/test_value_sprint.py:38`
- `tests/money/test_value_sprint.py:71`

### Fixed P1: source repair criterion had no supporting repair event

The report claimed a repaired source without performing or recording a repair. This made `at_least_one_source_failure_repaired_or_substituted` non-defensible.

Fix: the sprint now performs a deterministic experimental data mesh repair for the public billing source and records the repair as experimental-only with no production activation.

References:

- `src/behavior_lab/money/value_sprint.py:414`
- `src/behavior_lab/money/value_sprint.py:230`
- `tests/money/test_value_sprint.py:38`

### Fixed P2: public-only contract criterion used total active contracts

The public-only criterion previously used total active contract count. That could be inflated by non-public or private-readiness-gated contracts.

Fix: the criterion now counts only public sprint families and requires Weather Edge, ETF Risk, and compute cost avoidance.

References:

- `src/behavior_lab/money/value_sprint.py:407`
- `src/behavior_lab/money/value_sprint.py:266`

### Fixed P2: serializable sprint payload leaked local artifact paths

The returned payload exposed absolute artifact paths, which could leave the local machine through CLI output or copied JSON.

Fix: returned artifact identifiers are file names only. The files are still created in the configured output directory.

References:

- `src/behavior_lab/money/value_sprint.py:57`
- `tests/money/test_value_sprint.py:82`

## Required Evidence Check

The report includes and computes:

- user attention minutes
- approvals requested
- active contracts
- usable sources
- automatically added sources
- automatically repaired sources
- repeated failures avoided through memory
- candidate counts
- blind survivors
- prospective survivors
- paper opportunities
- no-action decisions
- resolved paper value
- research/API cost
- source maintenance cost

## Tests Run

- `python -m pytest tests/money/test_value_sprint.py -q`
- `python -m pytest tests/money -q`
- `python -m pytest -q`

All passed.
