# Sealed Scientific Loop

This document defines the implemented integrity contract for Resonance's experimental scientific loop.

## Intended process

```text
exploration-only summaries
        ↓
structured hypothesis imagination
        ↓
restricted executable expression
        ↓
numerical fitting on exploration
        ↓
candidate/program selection on tuning
        ↓
preregistration of exact semantics
        ↓
one-shot blind evaluation
        ↓
pass / fail / inconclusive
        ↓
optional manual controlled experiment
        ↓
tamper-evident scientific memory
```

The generator is replaceable. The deterministic evaluator and memory are authoritative.

## Dataset seal

Each snapshot is content-addressed and split chronologically:

- `exploration`: first 50% of eligible observations.
- `tuning`: next 25%.
- `blind`: final 25%.

An embargo at least as large as the maximum searched lag surrounds each split boundary. If the embargo leaves insufficient observations, snapshot creation fails closed. Missing observations remain missing; no forward filling occurs.

The exploration loader cannot return blind rows. The blind loader requires an internal evaluator capability. This is an architectural separation, not OS-level protection from the machine owner.

## Roles and permissions

### Explorer and proposer

They receive exploration observations or a `DiscoveryBrief` derived only from exploration. They may inspect summaries and prior selected memory, then produce structured `HypothesisSpec` objects. They never receive tuning/blind rows or blind outcomes for the current snapshot.

### Restricted program

A hypothesis compiles to a bounded expression AST. Allowed operations include metric references, constants, fitted parameters, arithmetic, safe division, clipping, differences, nonnegative lags, rolling means/standard deviations, and robust z-scores. The interpreter performs explicit dispatch; it never uses `eval`, `exec`, a shell, dynamic imports, filesystem access, or network access.

### Numerical fitter

Numeric parameters are fitted on exploration data only. The current observational-prediction contract optimizes RMSE. Random seeds, bounds, convergence details, evaluator version, target transform configuration, and artifacts are recorded.

### Program searcher and tuning selector

Candidate structure may be mutated/searched under a fixed budget. Fitting uses exploration; ranking uses tuning. Search cannot load blind data. It may return no winner and favors simpler programs when performance is effectively tied.

### Preregistration gate

Preregistration freezes:

- Snapshot and split identities.
- Exact expression and fitted parameters.
- Target/input metrics.
- Explicit target-transform semantics.
- Frozen baseline strategy.
- Requested blind metrics.
- Minimum effect and baseline-improvement thresholds.
- Negative controls and falsification conditions.
- Evaluator version and evaluator identity hash.
- Random seed and evaluation budget.

The evaluator identity fingerprints critical evaluator source files and key numerical-library versions. An unrelated Git commit is retained as provenance but does not invalidate an otherwise identical evaluator.

### Blind evaluator

The blind evaluator atomically appends `blind_evaluation_started` before it loads blind observations. That claim consumes the one-shot budget even if evaluation subsequently crashes. A repeated evaluation for the same preregistration or snapshot+hypothesis scientific object is refused.

The evaluator uses the same restricted-program and target-transform implementation as fitting/tuning. It recomputes the preregistered baseline strategy on the blind partition; tuning baseline values are provenance only. It returns aggregate preregistered metrics, controls, warnings, and `pass`, `fail`, or `inconclusive`. Raw blind rows are not included in result objects or artifacts.

A future retry requires a new future-data snapshot.

### Controlled experiment planner and runner

Only low-risk, reversible, human-executed N-of-1 protocols are currently permitted. Schedules and primary outcomes are frozen before execution. Every block requires confirmation. Missed, noncompliant, aborted, failed, and completed experiments remain recorded. No hardware or OS setting is changed automatically.

### Scientific ledger

The JSONL ledger is append-only through the application interface. Each canonical entry includes a sequence number, timestamp, event type, payload/artifact hashes, code commit, previous-entry hash, and entry hash. File locking and complete-line writes protect normal concurrent appends. Verification detects edits, line deletion, insertion/reordering, broken links, and truncation.

This is tamper-evident memory, not literal immutability against the machine owner. Corrections and supersessions append new records; old records are not rewritten.

## Shared evaluation semantics

`resonance/science/evaluation.py` is the common source for target transforms, frozen program evaluation, metrics, baseline construction, improvement calculations, movement-direction agreement, and window diagnostics. Fitting, tuning, and blind evaluation must not independently reinterpret the same hypothesis.

Time-based lag and rolling operations use timestamp semantics and present/past values only. Transform parameters such as difference periods and robust-z-score windows are made explicit before preregistration.

## Claim language

- Use `associated with` for observational relationships.
- Use `X precedes Y in this dataset` for a stable lagged relationship.
- Use `predicts in this dataset` only after preregistered blind evaluation.
- Use `inconclusive` when overlap, controls, stability, effect, or baseline improvement is insufficient.
- Never use `causes` without a controlled intervention supporting that claim.

Natural-language interpretation can criticize or explain a result but cannot change the numerical verdict.

## Required memory

The ledger/artifact graph retains:

- Accepted, rejected, invalid, duplicate, and superseded hypotheses.
- Exploration fits and baselines.
- Tuning selections and non-selections.
- Program-search lineage and budgets.
- Preregistrations and one-shot claims.
- Blind passes, failures, errors, and inconclusive outcomes.
- Negative controls and counterevidence.
- Planned, aborted, noncompliant, and completed experiments.
- Prospective replications and later contradictions.

Every reproducible result identifies its dataset snapshot, hypothesis, evaluator/code identity, random seed, parameters, metrics, and artifact hashes.

## Deliberate limits

The current system does not provide:

- General causal discovery.
- Arbitrary generated code execution.
- Autonomous intervention or machine control.
- Medical or behavioral experimentation.
- Thousands of automatic hypotheses.
- Cloud isolation of holdout data.
- Multi-user access control or external signing.
- Scientific paper generation or autonomous causal claims.

Silence, rejection, failure, and inconclusive outcomes are expected valid results.
