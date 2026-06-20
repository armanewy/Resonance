# Resonance

Resonance is a compact local-only prototype that collects personal computer/network signals plus local Open-Meteo weather observations, stores them in SQLite, and renders time-series graphs in Streamlit.

It does not do alerts, accounts, cloud deployment, or background OS service installation.

## Setup

Use Python 3.11 or newer.

```powershell
cd C:\Users\aoztu\Downloads\Resonance
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` and use `python -m pip install -r requirements.txt`.

## Configure Location

Edit `config.toml`:

```toml
[location]
name = "Framingham, Massachusetts"
latitude = 42.2793
longitude = -71.4162
timezone = "America/New_York"

[collection]
personal_interval_seconds = 30
weather_interval_seconds = 900
tcp_test_host = "1.1.1.1"
tcp_test_port = 443
dns_test_hostname = "example.com"
router_host = ""
```

The database is created at `data/resonance.db`. The dashboard binds to `127.0.0.1` only.

## Run

Start the collector and dashboard together:

```powershell
python run_local.py
```

Open:

```text
http://127.0.0.1:8501
```

Stop both processes with `Ctrl+C`.

Run the collector and dashboard separately:

```powershell
python -m resonance.collector
streamlit run resonance/dashboard.py --server.address=127.0.0.1
```

## Demo Data

Generate synthetic demo data explicitly:

```powershell
python -m resonance.seed_demo
```

Demo measurements use `source = "demo"` and are replaced each time you run the seeder.

Remove demo data:

```powershell
python -m resonance.seed_demo --clear
```

## Data Audit

Audit the configured SQLite database for recent coverage, gaps, stale metrics, duplicate timestamps, and collector errors:

```powershell
python -m resonance.audit --hours 24
```

Emit JSON for scripts:

```powershell
python -m resonance.audit --hours 24 --json
```

## Manual Pair Analysis

Analyze one requested metric pair from the local SQLite database without saving findings or scanning every pair:

```powershell
python -m resonance.analyze_pair --x tcp_latency_ms --y cpu_percent --hours 24 --transform first_difference --max-lag-minutes 60
```

The command reports association only; it does not establish causation. Use `--json` for machine-readable output.

## Conservative Correlation Scan

Scan eligible metric pairs locally with strict promotion thresholds. Dry runs do not write findings, and scans are silent when nothing passes:

```powershell
python -m resonance.scan --hours 168 --dry-run
```

Promoted findings, when any pass, are stored in SQLite as association evidence only.

Run the scanner continuously as a separate local process. It defaults to one scan every six hours, applies finding lifecycle classification, and uses `[notifications]` when enabled:

```powershell
python -m resonance.watch
```

## Scientific Snapshots

Freeze a reproducible, sealed scientific dataset snapshot from the local SQLite measurements:

```powershell
python -m resonance.science.snapshot_cli create --hours 720 --metrics tcp_latency_ms,dns_latency_ms,cpu_percent --max-lag-seconds 3600
```

Append snapshot creation to the scientific ledger:

```powershell
python -m resonance.science.snapshot_cli create --hours 720 --metrics tcp_latency_ms,dns_latency_ms,cpu_percent --max-lag-seconds 3600 --ledger data/science/ledger.jsonl
```

Inspect a snapshot without printing blind values:

```powershell
python -m resonance.science.snapshot_cli inspect SNAPSHOT_ID
```

Snapshots are written as content-addressed artifacts under `data/science/artifacts/sha256/`.
Rows are normalized to UTC, sorted deterministically, and split chronologically into exploration,
tuning, and blind partitions with an embargo around each split boundary. Missing metric
observations are left missing; no forward filling is applied.

## Scientific Ledger

Scientific-loop events are appended to a tamper-evident JSON Lines ledger at `data/science/ledger.jsonl`. Entries include chained hashes, payload hashes, artifact hashes, code commit, UTC timestamp, and sequence number so edits, inserted records, reordered records, and missing middle records are detected by verification. Existing entries are not updated through the API; corrections append a later event that refers to the prior entry. This is local tamper evidence, not immutability against the machine owner.

Record result identity in event payloads: dataset snapshot ID, hypothesis hash, evaluator version, random seed, parameters, metrics, and referenced graph or report artifact hashes.

Verify the ledger:

```powershell
python -m resonance.science.ledger_cli verify
```

Verify path-bearing artifact references recorded in the ledger:

```powershell
python -m resonance.science.ledger_cli verify --artifact-root data/science/artifacts
```

Show recent entries:

```powershell
python -m resonance.science.ledger_cli show --limit 20
```

## Manual Scientific Loop

Run the human-operated sealed loop without any LLM dependency:

```powershell
python -m resonance.science.cli snapshot synthetic --scenario strong_lag
python -m resonance.science.cli hypothesis validate examples\science\strong_lag_hypothesis.json --snapshot SNAPSHOT_ID
python -m resonance.science.cli fit examples\science\strong_lag_hypothesis.json --snapshot SNAPSHOT_ID
python -m resonance.science.cli tune --run RUN_ID
python -m resonance.science.cli preregister --candidate CANDIDATE_ID
python -m resonance.science.cli blind-evaluate PREREGISTRATION_ID
python -m resonance.science.cli report PREREGISTRATION_ID
python -m resonance.science.ledger_cli verify
```

The CLI records proposal, fit, tuning, preregistration, blind evaluation, and
report artifacts in the ledger. Blind evaluation remains one-shot per
preregistration, and reports include the exact snapshot and code versions.

### LLM Hypothesis Imagination Flow

The imagination flow proposes hypotheses from an exploration-only
`DiscoveryBrief`, runs deterministic skeptical review, and requires explicit
human approval before fitting:

```powershell
python -m resonance.science.cli imagine --snapshot SNAPSHOT_ID --provider mock --max-hypotheses 8
python -m resonance.science.cli review RUN_ID
python -m resonance.science.cli review RUN_ID --approve HYPOTHESIS_INDEX_OR_HASH
python -m resonance.science.cli fit-approved RUN_ID
```

`fit-approved` fits approved proposals on exploration and compares them on
tuning, selecting at most one default candidate. It does not preregister or run
blind evaluation; use the existing `preregister`, `blind-evaluate`, and
`report` commands for those sealed steps.

### Optional Hypothesis Providers

Science provider adapters are library-level helpers for proposing structured
hypotheses from a `DiscoveryBrief`. They do not run fitting, tuning, selection,
preregistration, blind evaluation, or CLI orchestration.

`OpenAIProvider` uses the OpenAI Responses API with Structured Outputs. The
`openai` package is optional and imported only when the provider is used; the
API key is read from the OpenAI SDK's normal environment configuration. Requests
send only prompt instructions and serialized `DiscoveryBrief` JSON, set
`store=False`, do not configure tools, and record response metadata in the
provider run request config.

`CommandProvider` runs a configured local command argument vector, sends the
serialized `DiscoveryBrief` JSON on stdin, reads hypothesis JSON from stdout,
and enforces timeout and stdout-size limits. It never uses `shell=True`.

## Synthetic Scenarios

Generate a deterministic synthetic time-series scenario:

```powershell
python -m resonance.synthetic --scenario strong_lag --output tmp\strong_lag.csv
```

Available scenarios are `strong_lag`, `shared_seasonality_only`, `single_shared_outlier`, `relationship_break`, `independent_autocorrelated`, and `missing_data`.

## Tests

```powershell
pytest
```

Tests do not require Internet access.

## Inspect Data

Count real personal samples by metric:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/resonance.db'); print(c.execute(\"select metric, count(*) from measurements where source='personal' group by metric order by metric\").fetchall()); c.close()"
```

Count real weather samples:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/resonance.db'); print(c.execute(\"select metric, count(*) from measurements where source='open-meteo' group by metric order by metric\").fetchall()); c.close()"
```

## Data Boundaries

All data stays on the local machine except:

- Open-Meteo forecast API requests for configured latitude and longitude.
- TCP connectivity test to `tcp_test_host:tcp_test_port`.
- DNS lookup for `dns_test_hostname`.

No telemetry is sent by this application. Streamlit usage stats are disabled in `.streamlit/config.toml`.
