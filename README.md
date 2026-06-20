# Resonance

Resonance is a local-first personal observatory and experimental science workbench. It continuously records a compact set of computer, network, battery, and localized weather signals; renders them as time-series evidence; searches conservatively for lagged associations; and can carry a structured hypothesis through fitting, tuning, preregistration, one-shot blind evaluation, and a tamper-evident scientific ledger.

It is intentionally capable of saying nothing passed.

## Quick start

Use Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_local.py
```

Open `http://127.0.0.1:8501`. The dashboard binds to localhost, and Streamlit telemetry is disabled in `.streamlit/config.toml`.

Run the collector or dashboard separately:

```bash
python -m resonance.collector
streamlit run resonance/dashboard.py --server.address=127.0.0.1
```

## Configuration and collected data

Edit `config.toml` to set location, timezone, collection intervals, connectivity targets, and optional notifications. The default location is Framingham, Massachusetts.

Personal samples include CPU, memory, network receive/send rates, battery state when available, TCP connection latency, and DNS resolution latency. Weather comes from Open-Meteo. Data is stored in `data/resonance.db` with UTC timestamps; display and calendar-seasonality logic use the configured timezone.

Audit recent coverage and collector health:

```bash
python -m resonance.audit --hours 24
python -m resonance.audit --hours 24 --json
```

Seed or remove explicitly marked demo data:

```bash
python -m resonance.seed_demo
python -m resonance.seed_demo --clear
```

## Pair exploration and conservative discovery

Analyze one selected pair without persisting a finding:

```bash
python -m resonance.analyze_pair \
  --x tcp_latency_ms \
  --y cpu_percent \
  --hours 24 \
  --transform first_difference \
  --max-lag-minutes 60
```

The Streamlit Pair Explorer renders the aligned transformed timeline, lag profile, lagged scatter, and stability evidence.

Run the automatic scanner in dry-run mode:

```bash
python -m resonance.scan --hours 168 --dry-run
```

The scanner uses lagged Spearman association, chronological holdout, max-over-lags block permutation, time-window stability, local-time calendar residuals when sufficient history exists, and conservative Benjamini–Yekutieli false-discovery correction across the tested family. It may legitimately return no findings.

Continuously re-evaluate existing/new findings in a separate process:

```bash
python -m resonance.watch
```

Notifications are disabled by default and are limited to lifecycle events such as a new verified relationship or a previously stable relationship breaking.

## Synthetic truth scenarios

```bash
python -m resonance.synthetic --scenario strong_lag --output tmp/strong_lag.csv
```

Available scenarios: `strong_lag`, `shared_seasonality_only`, `single_shared_outlier`, `relationship_break`, `independent_autocorrelated`, and `missing_data`.

## Sealed scientific loop

The scientific loop separates exploration, tuning, and blind partitions chronologically, with an embargo around split boundaries. The proposer cannot load blind observations. Preregistration freezes the executable claim and evaluator identity. Blind evaluation atomically consumes its one-shot budget before loading the holdout and never exposes raw blind rows.

Run a complete deterministic synthetic loop:

```bash
python -m resonance.science.cli snapshot synthetic --scenario strong_lag
python -m resonance.science.cli hypothesis validate examples/science/strong_lag_hypothesis.json --snapshot SNAPSHOT_ID
python -m resonance.science.cli fit examples/science/strong_lag_hypothesis.json --snapshot SNAPSHOT_ID
python -m resonance.science.cli tune --run RUN_ID
python -m resonance.science.cli preregister --candidate CANDIDATE_ID
python -m resonance.science.cli blind-evaluate PREREGISTRATION_ID
python -m resonance.science.cli report PREREGISTRATION_ID
python -m resonance.science.ledger_cli verify
```

Create a snapshot from your actual local measurements:

```bash
python -m resonance.science.cli snapshot create \
  --db data/resonance.db \
  --hours 720 \
  --metrics tcp_latency_ms,dns_latency_ms,cpu_percent \
  --max-lag-seconds 3600
```

Inspect a snapshot without printing blind values:

```bash
python -m resonance.science.cli snapshot inspect SNAPSHOT_ID
```

Artifacts are content-addressed below `data/science/artifacts/sha256/`. The append-only application ledger is `data/science/ledger.jsonl`:

```bash
python -m resonance.science.ledger_cli verify
python -m resonance.science.ledger_cli verify --artifact-root data/science/artifacts
python -m resonance.science.ledger_cli show --limit 20
```

The ledger is tamper-evident, not immutable against an owner deliberately rewriting both files and code.

## Hypothesis imagination and pluggable LLMs

The imagination flow constructs an exploration-only `DiscoveryBrief`, requests at most eight schema-valid hypotheses, runs deterministic skeptical validation, and requires explicit human approval before fitting. It never preregisters or blind-evaluates automatically.

Exercise the flow with the deterministic mock provider:

```bash
python -m resonance.science.cli imagine --snapshot SNAPSHOT_ID --provider mock --max-hypotheses 8
python -m resonance.science.cli review RUN_ID
python -m resonance.science.cli review RUN_ID --approve 0
python -m resonance.science.cli fit-approved RUN_ID
```

Use a checked-in/output JSON file:

```bash
python -m resonance.science.cli imagine --snapshot SNAPSHOT_ID --provider file --provider-file proposals.json
```

Use OpenAI Structured Outputs (optional dependency and API key required):

```bash
python -m pip install openai
python -m resonance.science.cli imagine \
  --snapshot SNAPSHOT_ID \
  --provider openai \
  --provider-model gpt-5.5 \
  --max-hypotheses 8
```

Use a local model or any other provider through a safe argument-vector command adapter. The command receives JSON on stdin and must emit schema-valid proposal JSON on stdout; no shell is used:

```bash
python -m resonance.science.cli imagine \
  --snapshot SNAPSHOT_ID \
  --provider command \
  --provider-command python local_hypothesis_provider.py \
  --provider-model my-local-model
```

Only the exploration brief is sent to a provider. OpenAI requests set `store=false` and enable no tools. Provider output is data, not executable source code.

Compare mock-generated hypotheses with deterministic baselines:

```bash
python -m resonance.science.cli ablate \
  --scenarios strong_lag,shared_seasonality_only \
  --provider mock \
  --seed 123
```

## Bounded program search

Search only exploration/tuning data over the restricted DSL:

```bash
python -m resonance.science.search_cli run \
  --snapshot SNAPSHOT_ID \
  --seed-hypothesis examples/science/strong_lag_hypothesis.json \
  --budget 100 \
  --beam-width 10 \
  --random-seed 123
```

The search maintains lineage and a performance/complexity tradeoff, can return no winner, and cannot query blind data.

## Manual controlled experiments

Controlled experiments are deliberately low-risk, reversible, and human-executed. A protocol is frozen before execution; each block requires confirmation; missed/noncompliant blocks remain recorded.

```bash
python -m resonance.science.experiments.cli preregister EXPERIMENT_SPEC.json
python -m resonance.science.experiments.cli start EXPERIMENT_ID
python -m resonance.science.experiments.cli begin-block EXPERIMENT_ID
python -m resonance.science.experiments.cli confirm-condition EXPERIMENT_ID
python -m resonance.science.experiments.cli end-block EXPERIMENT_ID
python -m resonance.science.experiments.cli status EXPERIMENT_ID
python -m resonance.science.experiments.cli evaluate EXPERIMENT_ID
```

No hardware or operating-system intervention is automated. Medical, behavioral, hazardous, and emergency-connectivity experiments are outside the current scope.

## Tests

```bash
pytest -q
```

Tests are deterministic and do not require Internet access. They include synthetic true relationships, seasonality-only/null cases, lag-search inflation, broken relationships, snapshot seals, provider leakage sentinels, restricted-program execution, one-shot blind-budget enforcement, ledger tamper detection, and controlled-experiment behavior.

## Network and trust boundaries

All persistent data remains local except explicit outbound operations:

- Open-Meteo requests for the configured latitude/longitude.
- TCP connectivity checks to the configured host and port.
- DNS lookup of the configured hostname.
- Optional `ntfy` HTTP notifications when enabled.
- Optional LLM provider calls explicitly initiated by the user.

The blind partition is architecturally sealed inside one local Python project; it is not protected from a machine owner who intentionally reads or edits the artifact files. Correlation thresholds are conservative prototype defaults, not proof of causality. See `docs/scientific_loop.md` and `AUDIT.md` for the exact integrity model and current limitations.
