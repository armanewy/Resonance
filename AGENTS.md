# Resonance Agent Guide

## Purpose

Resonance is a compact, local-only Python prototype that collects personal computer/network signals plus localized Open-Meteo weather data, stores them in SQLite, and renders clear Streamlit/Plotly time-series graphs.

The completed MVP intentionally does not include correlation discovery, alerts, accounts, cloud deployment, a separate API server, an ORM, a JavaScript frontend, or infrastructure services.

## Repository Layout

- `config.toml`: editable location and collection configuration.
- `run_local.py`: local launcher for the collector and Streamlit dashboard.
- `resonance/config.py`: config loading and validation.
- `resonance/storage.py`: SQLite connection, schema setup, inserts, queries, and demo cleanup.
- `resonance/personal.py`: CPU, memory, network delta, battery, TCP, and DNS sampling.
- `resonance/weather.py`: Open-Meteo request and parser.
- `resonance/collector.py`: long-running collector loop.
- `resonance/dashboard.py`: compact Streamlit dashboard.
- `resonance/seed_demo.py`: explicit synthetic demo-data seeder.
- `tests/`: focused pytest tests and the Open-Meteo fixture.
- `data/resonance.db`: local SQLite database created at runtime and ignored by Git.

## Commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full test suite:

```powershell
pytest
```

Run the collector:

```powershell
python -m resonance.collector
```

Run the dashboard only, bound to localhost:

```powershell
streamlit run resonance/dashboard.py --server.address=127.0.0.1
```

Run collector and dashboard together:

```powershell
python run_local.py
```

Seed demo data explicitly:

```powershell
python -m resonance.seed_demo
```

Remove demo data:

```powershell
python -m resonance.seed_demo --clear
```

Current passing baseline command:

```powershell
pytest
```

## Storage

The local SQLite database path is:

```text
data/resonance.db
```

Runtime data is local-only and ignored by Git. Keep database access in small storage helpers. Use UTC timestamps for storage and convert to local time only for display.

## Coding Conventions

- Use Python 3.11 or newer and the existing dependencies in `requirements.txt`.
- Prefer the standard library where practical.
- Keep modules small and easy to delete or rewrite.
- Use dataclasses for lightweight typed records when useful.
- Use parameterized SQL only.
- Keep tests focused and deterministic.
- Do not fabricate unavailable metrics.
- Record recoverable collector errors and continue running.
- Avoid logging every successful sample.

## Scope Rules

- Assigned tasks may modify only their declared files unless compilation or tests require a minimal adjacent change.
- Do not perform broad refactoring.
- Do not add correlation features unless the prompt explicitly assigns that work.
- Do not add a new server, cloud deployment, account system, ORM, frontend framework, Docker setup, queue, cache, scheduler service, or infrastructure service.
- Do not introduce AI or LLM functionality.
- Do not add speculative architecture or abstractions for hypothetical sources.
- Existing behavior must remain backward compatible unless the prompt explicitly says otherwise.
- Every task must add focused tests, run targeted tests where relevant, run the full suite, make one atomic commit, and stop.
- Every task must report files changed, commands run, test results, and commit hash.

