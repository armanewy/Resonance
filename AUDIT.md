# Resonance Audit and Alignment Report

Audit date: 2026-06-20

## Scope

The uploaded repository was treated as the authoritative working state, including its uncommitted files. The audit covered:

- Local collection, SQLite storage, dashboard, correlation exploration, scanner, watcher, and notifications.
- Scientific snapshots, restricted hypothesis DSL, fitting, tuning/selection, program search, preregistration, blind evaluation, providers, ledger, and controlled experiments.
- Statistical consistency, holdout leakage, one-shot evaluation integrity, artifact path safety, provider isolation, operational shutdown, documentation, and tests.

The included `data/resonance.db` was retained. No test used the production database for writes.

## Executive assessment

The project was much further along than the original bounded MVP. Its main risk was not missing features; it was scientific inconsistency between stages that were supposed to evaluate the same frozen hypothesis. The highest-priority fixes therefore preserve meaning and prevent accidental holdout reuse rather than add presentation features.

The corrected architecture now matches the intended loop:

```text
LLM/manual imagination
        +
restricted numerical fitting and program search
        +
one-shot blind evaluation
        +
manual controlled experimentation
        +
tamper-evident scientific memory
```

## Critical findings and fixes

### 1. Fitting, tuning, and blind evaluation interpreted programs differently

**Finding:** The same hypothesis could have different lag, rolling-window, target-transform, metric, and missing-value semantics depending on the stage evaluating it.

**Fix:** Added `resonance/science/evaluation.py` as the shared implementation for target transformation, frozen-program evaluation, alignment, metrics, baselines, improvements, direction agreement, and window diagnostics. Fitting, selection, and blind evaluation now use the same interpreter and explicit transform configuration.

### 2. Blind baseline values came from the wrong partition

**Finding:** Blind results could be compared against numeric baseline values carried from tuning/preregistration instead of evaluating the frozen baseline strategy on the blind partition.

**Fix:** Preregistration now freezes a baseline **strategy**. Blind evaluation recomputes that strategy on blind data. Tuning baseline values remain labeled provenance and cannot affect the blind verdict.

### 3. The one-shot blind budget had a race window

**Finding:** Two concurrent evaluators could both pass a read-before-write completion check, or a crash could allow an apparent retry.

**Fix:** Added an atomic, file-locked `blind_evaluation_started` claim. The claim is written before blind data is loaded and consumes the budget even if evaluation crashes. The same preregistration or snapshot+hypothesis object cannot be queried again.

### 4. Evaluator identity was too weak or too broad

**Finding:** A static version could miss material evaluator changes, while requiring the exact repository commit would invalidate evaluation after unrelated edits.

**Fix:** The evaluator identity now fingerprints critical evaluator source files, the evaluator ruleset, and numerical dependency versions. The Git commit remains provenance but is not the sole semantic identity.

### 5. Hypothesis fields allowed unsupported semantics

**Finding:** The schema accepted fitting objectives and expected directions the implemented evaluator did not honestly support.

**Fix:** Observational prediction v1 now requires RMSE fitting, positive prediction-to-target direction, unique blind metrics, Spearman in blind metrics, and at least one error metric. Negative input effects must be encoded in the expression itself.

### 6. Discovery and validation used inconsistent association statistics

**Finding:** Candidate discovery used Spearman while some holdout, permutation, and stability paths used Pearson-like calculations.

**Fix:** Lag discovery, holdout, max-over-lags permutation, and stability now use consistent Spearman rank association.

### 7. Automatic scanning under-corrected dependent multiple testing

**Finding:** The scanner corrected only promoted/evaluated results with Benjamini-Hochberg defaults, despite dependent candidate families and skipped tests.

**Fix:** The default is now conservative Benjamini-Yekutieli correction over the whole eligible candidate family. The correction method and total test count are recorded in evidence.

### 8. Calendar seasonality used UTC rather than the user's local clock

**Finding:** Commute, sleep, and household patterns could be grouped by UTC hour and silently shift across daylight-saving changes.

**Fix:** Calendar residual slots are computed in the configured location timezone while stored/indexed timestamps remain UTC.

### 9. Snapshot/artifact loading needed stronger path validation

**Finding:** Snapshot IDs and artifact references required stricter identity and containment checks.

**Fix:** Snapshot IDs/digests are validated as lowercase SHA-256 values; index and manifest identities must match; compressed artifacts are hash-verified; relative paths must remain inside the configured artifact root.

### 10. Provider adapters were not reachable through the main imagination CLI

**Finding:** OpenAI and local-command adapters existed but the user-facing `imagine` command supported only mock/file providers.

**Fix:** The CLI now supports `mock`, `file`, `openai`, and `command`. The command adapter uses an argument vector without a shell. OpenAI requests receive only the exploration `DiscoveryBrief`, use schema-constrained output, set `store=false`, and enable no tools. The OpenAI SDK remains optional.

### 11. DNS timeout handling could accumulate blocked threads

**Finding:** Creating a new executor for every DNS sample could accumulate unresolved resolver threads after repeated timeouts.

**Fix:** DNS collection now uses one shared worker and permits only one outstanding lookup. Subsequent samples report a recoverable condition rather than starting additional blocked work.

### 12. Program search could randomly omit the central lag dimension

**Finding:** Seeded random mutation could fail to try plausible lag variants, making a known lagged synthetic relationship unrecoverable.

**Fix:** Program search now seeds a bounded structural frontier with low-cost lag additions/changes before open-ended mutation, retains the evaluation budget, and remains deterministic for a fixed seed.

### 13. Collector shutdown was abrupt under the launcher

**Finding:** `run_local.py` sends SIGTERM on Unix, but the collector only handled `KeyboardInterrupt`, so normal launcher shutdown could bypass a clean connection close.

**Fix:** The collector now handles SIGINT/SIGTERM through a stop event, exits its loop predictably, and closes SQLite in `finally`.

### 14. Documentation described an earlier MVP, not the uploaded system

**Finding:** `AGENTS.md` still prohibited correlation and LLM work, while the repository already implemented both. The scientific-loop document described future architecture even though most of it existed.

**Fix:** Rewrote `AGENTS.md`, `README.md`, and `docs/scientific_loop.md` around the actual implemented layers, commands, scientific invariants, provider boundaries, program search, controlled experiments, and limitations.

## Validation performed

- 285 tests collected and passed in a single full run.
- Synthetic true-lag, seasonality-only, random-walk, autocorrelated-null, outlier, relationship-break, and missing-data scenarios exercised.
- Added regression coverage for:
  - Shared fitting/frozen-evaluation semantics.
  - Blind baseline recomputation.
  - Terminal pre-evaluation budget claims.
  - Unsupported scientific contract fields.
  - Stable scientific hashes independent of proposal provenance.
  - Snapshot path traversal.
  - Local-time slots across daylight-saving offsets.
  - BY versus BH correction.
  - CLI exposure of OpenAI and local-command providers.
  - Graceful collector shutdown.
- `python -m compileall -q resonance` passed.
- `git diff --check` passed.
- Scientific ledger verification passed.
- The Streamlit dashboard started successfully on localhost in headless smoke testing.
- The uploaded database was read successfully: 7,444 measurements across 16 metrics, with no recorded collector errors. Its samples are currently stale, which is expected for an uploaded snapshot rather than a running collector.

A global `pip check` reported an unrelated pre-existing `moviepy`/`Pillow` conflict in the audit container. Resonance does not depend on MoviePy or Pillow, and all project tests/imports passed.

## Remaining limitations

- Blind isolation is architectural, not a security boundary against the machine owner. A separate account/container/remote evaluator would be required for stronger secrecy.
- The JSONL ledger is tamper-evident, not externally signed or immutable.
- Statistical promotion thresholds are conservative prototype defaults rather than universal scientific guarantees.
- The current LLM proposer receives compressed exploration summaries; prompt quality and usefulness still require ablation against deterministic/random baselines.
- The remote-provider seed is stored as provenance but cannot guarantee remote-model determinism.
- OpenAI is an optional dependency and is not pinned in the base requirements.
- Controlled experiments are manual and limited to low-risk reversible actions; no intervention is automated.
- Weather is the only localized public-world signal currently collected. Traffic, grid, and regional Internet connectors remain future experiments.
- The DNS worker prevents thread accumulation, but an operating-system resolver call that never returns can occupy that single worker until process exit.
- The system has strong synthetic validation, but it has not yet accumulated enough fresh personal data to establish that its discoveries are consistently useful in practice.

## Recommended next experiment

Do not add more data sources first. Restart the collector, accumulate several fresh days, and run one sealed question end to end:

> Is transformed external latency predicted by recent upload throughput, CPU activity, charging state, or a simple interaction among them?

Compare LLM proposals with the deterministic pairwise and bounded-program baselines under the same candidate budget. Preregister only one winner, spend the blind budget once, then design a small randomized manual upload-versus-idle experiment only if the observational result survives.
