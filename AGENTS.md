# Resonance Agent Guide

## Purpose

Resonance is a local-first personal observatory and experimental science workbench. It has three deliberate layers:

1. Collect personal computer/network signals and localized weather into SQLite and render them in Streamlit.
2. Explore and conservatively promote lagged associations with holdout, stability, permutation, and multiple-testing checks.
3. Run a sealed scientific loop: structured hypothesis imagination, restricted-program fitting/search, tuning, one-shot blind evaluation, manual controlled experiments, and tamper-evident scientific memory.

The system is allowed to remain silent or return `fail`/`inconclusive`. Persuasive prose never overrides numerical evaluation.

## Repository Layout

- `config.toml`: local location, collection, and optional notification settings.
- `run_local.py`: starts the collector and localhost-only Streamlit dashboard.
- `resonance/collector.py`, `personal.py`, `weather.py`: data collection.
- `resonance/public_collector.py`, `resonance/public_sources/`: optional bounded public data collection.
- `resonance/storage.py`: SQLite schema and storage helpers.
- `resonance/dashboard.py`, `resonance/ui/`: visual exploration and evidence cards.
- `resonance/analysis/`: alignment, transformations, lagged association, validation, scanning, and lifecycle.
- `resonance/science/`: snapshots, DSL contracts/interpreter, shared evaluation, fitting, selection, program search, preregistration, blind evaluation, providers, ledger, and scientific CLI.
- `resonance/science/experiments/`: low-risk, manually executed controlled experiments.
- `tests/`: deterministic unit, integration, adversarial, and synthetic-science tests.
- `data/resonance.db`: local runtime database; ignored by Git.
- `data/science/`: local content-addressed artifacts and ledger; ignored by Git.

## Common Commands

```bash
python -m pip install -r requirements.txt
pytest -q
python run_local.py
python -m resonance.public_collector
python -m resonance.public_sources.eia_grid status
python -m resonance.public_sources.eia_grid backfill --start 2026-06-19T00:00:00Z --end 2026-06-20T00:00:00Z
python -m resonance.public_sources.eia_grid poll
python -m resonance.audit --hours 24
python -m resonance.analyze_pair --x tcp_latency_ms --y cpu_percent --hours 24 --transform first_difference --max-lag-minutes 60
python -m resonance.scan --hours 168 --dry-run
python -m resonance.watch
python -m resonance.science.ledger_cli verify
```

Use `python -m resonance.science.cli --help`, `python -m resonance.science.search_cli --help`, and `python -m resonance.science.experiments.cli --help` for the scientific workflows.

## Coding Rules

- Support Python 3.11 or newer.
- Preserve local-only behavior unless a task explicitly changes it.
- Use UTC for storage and configured local time only for display/calendar semantics.
- Never forward-fill missing observations implicitly.
- Keep public-source credentials in environment variables only; never persist API keys in config, SQLite, raw archive metadata, logs, or dashboard output.
- Use parameterized SQL.
- Keep external requests explicit, bounded by timeouts, and recoverable.
- Avoid broad refactors, speculative abstractions, and infrastructure services.
- Add focused deterministic tests for every behavioral change.
- Run targeted tests plus the complete suite before committing.
- Do not fabricate unavailable measurements or statistical certainty.

## Statistical Integrity

- Association discovery uses the same statistic in discovery and validation.
- A lag selected on discovery data is frozen before holdout evaluation.
- A permutation null must repeat the full lag search.
- Automatic scans correct across the whole tested family and may return no findings.
- Public-involved automatic scanner pairs are dry-run only until prospective verification exists.
- Reject incompatible cadence, geography, and lineage combinations before running scanner statistics.
- Calendar residuals use the configured location timezone.
- Findings say `associated`, `precedes`, `follows`, or `predicts in this dataset`; never `causes` without intervention.

## Sealed Scientific-Loop Invariants

Follow `docs/scientific_loop.md`.

- The proposer receives exploration-only summaries, never tuning or blind observations.
- LLM output is validated structured data; arbitrary generated Python/shell code is never executed.
- Hypotheses compile to the restricted expression DSL.
- Fitting uses exploration only; candidate selection/program search may use tuning; neither may access blind data.
- Preregistration freezes the exact expression, fitted parameters, transform semantics, baseline strategy, metrics, controls, thresholds, split, evaluator identity, and seed.
- The blind budget is atomically consumed before blind data is loaded.
- One blind evaluation is allowed per snapshot+hypothesis scientific object. A retry needs a future snapshot.
- Fitting, tuning, and blind evaluation must use the shared program/target-transform semantics in `resonance/science/evaluation.py`.
- Baselines are recomputed on the evaluated partition; tuning baseline values are provenance, not blind scores.
- Negative, rejected, failed, inconclusive, aborted, and superseded results remain in the ledger.
- Every random process records a seed.
- Every result identifies snapshot, code/evaluator identity, hypothesis, and artifact hashes.
- The local ledger is tamper-evident, not immutable against the machine owner.

## Safety and Scope

Initial controlled experiments are human-executed, low-risk, and reversible. Do not automate medical/behavioral interventions, hazardous hardware actions, emergency-connectivity changes, or consequential machine control. Do not add cloud orchestration, accounts, multi-user collaboration, public marketplaces, or autonomous causal claims without a new explicit contract.
