# Sealed Scientific Loop

This document defines the initial architecture contract for pivoting Resonance
toward a sealed scientific hypothesis-testing loop. It is documentation only:
it does not implement production functionality or change current analysis
behavior.

The current repository already contains local SQLite measurements, pairwise
analysis contracts, lagged association scans, synthetic scenarios, Pair Explorer
helpers, scanner lifecycle events, and deterministic tests. This contract sets
the boundary for future work so those pieces can evolve without leaking blind
observations, executing generated code, or overstating causal claims.

## Goals

- Separate exploration, tuning, preregistration, and blind evaluation.
- Preserve negative and inconclusive results alongside positive results.
- Make every result reproducible by recording data, code, evaluator, hypothesis,
  and artifact identities.
- Keep the first implementation local, conservative, and human-operated.

## Chronological Dataset Layout

Each data snapshot used for a scientific run is split by timestamp:

- `exploration`: first 50% of eligible observations.
- `tuning`: next 25% of eligible observations.
- `blind`: final 25% of eligible observations.

An embargo gap is required around each split boundary. The gap on each side of a
boundary must be at least as large as the maximum searched lag for that run. If
the requested maximum lag makes the embargo leave too little data for any
partition, the run must fail closed and record an inconclusive result instead of
shrinking the embargo silently.

The split identity is part of preregistration. It must name the data snapshot,
eligible timestamp range, partition boundaries, embargo duration, maximum
searched lag, and any inclusion or exclusion rules.

## Roles

### Explorer

The Explorer may inspect only the exploration partition. It produces summaries,
plots, descriptive diagnostics, and candidate questions. It may not inspect
tuning or blind observations, tune thresholds on tuning outcomes, or create
claims that require blind evaluation.

### Hypothesis Proposer

The Hypothesis proposer may receive exploration summaries and prior
scientific-memory records. It must not receive tuning observations, blind
observations, or tuning outcomes. It produces structured hypotheses, not
executable arbitrary code.

### Numerical Fitter

The Numerical fitter estimates numeric parameters using exploration data only.
It may fit coefficients, lags, windows, and thresholds that are explicitly part
of a candidate hypothesis. It must record the input data snapshot, seed for any
random process, fitted values, fitting method, warnings, and artifact hashes.

### Program Searcher

The Program searcher may compare candidate hypotheses or restricted expression
programs using exploration and tuning data. It must never access blind data and
must never query the blind evaluator. It may choose among candidates before
preregistration, but it cannot change a candidate after preregistration.

### Preregistration Gate

The Preregistration gate freezes the hypothesis, restricted expression program,
metrics, thresholds, negative controls, split identity, and evaluator version
before blind evaluation. It computes and stores the hypothesis hash and artifact
hashes. After this gate, any change requires a new preregistration record.

### Blind Evaluator

The Blind evaluator loads the blind partition internally. It evaluates a
preregistered hypothesis once, does not expose blind observations, and records
that the blind budget for that preregistered hypothesis was consumed. It returns
only preregistered metrics, pass/fail or inconclusive status, warnings, and
artifact hashes.

### Experiment Planner

The Experiment planner may initially propose only low-risk, reversible,
human-executed experiments. It may not control hardware, change machine
settings automatically, manipulate medical or behavioral conditions, or execute
interventions without explicit human action and review.

### Scientific Ledger

The Scientific ledger is append-only through the application interface. It
records proposals, fits, preregistrations, evaluations, failures, corrections,
experiments, and replications. Corrections create new records rather than
modifying old records.

The first local ledger is tamper-evident, not literally immutable against a
machine owner intentionally rewriting the repository, database, or artifacts.
Tamper evidence should come from chained hashes, artifact hashes, timestamps,
and explicit data/code identities, but local ownership still implies ultimate
write access.

## Restricted Hypothesis Form

Hypotheses must compile to a restricted expression DSL. The DSL may represent
approved metric references, transforms, lags, fitted numeric parameters,
comparisons, aggregations, and preregistered metrics. It must not allow
arbitrary Python, shell commands, filesystem access, network access, imports,
reflection, or dynamic execution.

Natural-language explanations may accompany a hypothesis or result, but they
never override numerical results. If the explanation and numerical result
conflict, the ledger must preserve the conflict and treat the numerical result
as authoritative.

## Required Invariants

- The LLM never sees blind observations.
- No arbitrary Python generated by an LLM is executed.
- Hypotheses compile to a restricted expression DSL.
- Program search may not query the blind evaluator.
- One blind evaluation is allowed per preregistered hypothesis.
- Repeating a blind evaluation requires a new future-data snapshot.
- Natural-language explanations never override numerical results.
- Negative and inconclusive results are retained.
- "Associated with" and "predicts in this dataset" are allowed.
- "Causes" is prohibited unless supported by an intervention.
- Any random process must have a stored seed.
- Every result must identify data snapshot, code commit, evaluator version,
  hypothesis hash, and artifact hashes.

## Claim Language

Allowed claim language is deliberately narrow:

- Use "associated with" for non-directional statistical relationships.
- Use "predicts in this dataset" only when a preregistered predictive metric was
  evaluated on the blind partition.
- Use "inconclusive" when thresholds, overlap, controls, or diagnostics are not
  sufficient.
- Do not use "causes" unless the result is supported by an intervention.

Current pairwise scanner and Pair Explorer language should remain
association-oriented unless and until an intervention-backed workflow exists.

## Result Identity

Every result record must include:

- Data snapshot identity and partition split identity.
- Code commit.
- Evaluator version.
- Hypothesis hash.
- Artifact hashes for input summaries, frozen programs, fit outputs, metrics,
  plots, and evaluator outputs.
- Stored seed for every random process, including permutations, search,
  synthetic data generation, sampling, and randomized controls.

## Negative Controls and Failures

Preregistration must include negative controls when applicable. Blind evaluation
must record failures, failed controls, missing data, low overlap, and
inconclusive outcomes. These records stay in the ledger and are not replaced by
later corrections or replications.

## Initial Scope

Not part of the initial scientific loop:

- Automatic hardware interventions.
- Medical or behavioral experimentation.
- Arbitrary generated code.
- General causal discovery.
- Scientific paper generation.
- Thousands of automatically proposed hypotheses.
- Cloud orchestration.
- Multi-user collaboration.
- Public hypothesis marketplaces.
- Autonomous claims of causality.

## Implementation Boundary

Future implementation work must preserve the current local-only posture unless a
later contract explicitly changes it. The first implementation should favor
small storage helpers, explicit dataclasses, deterministic tests, read-only
access for analysis partitions, and simple auditability over broad frameworks.
