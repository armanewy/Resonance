# Stress Test Notes

This repository is not just a behavior predictor. It is an infrastructure spike for a closed-loop discovery lab: hypotheses are generated, fitted, compared, falsified, and only trusted when they survive chronological and prospective evaluation.

## Issues found in the submitted version

1. **Test ergonomics gap**: the project used a `src/` layout but did not include pytest configuration, so `python -m pytest` failed unless the package was installed or `PYTHONPATH=src` was set manually.
2. **Discovery loop was too slow for iteration**: symbolic search repeatedly fitted logistic formulas with 650 gradient steps across too many candidate terms. That made the core tests feel hung and violated the “see things happen quickly” requirement.
3. **A/B comparison bookkeeping was too thin**: treatment-effect estimation returned a difference in means, but did not expose block-level effects, a standard error, or tiny-sample warnings.
4. **No executable self-audit**: the README claimed temporal-firewall and blind-evaluation properties, but there was no one-command stress test that exercised those claims.

## Changes made

- Added `pytest.ini` so tests run without manual path setup.
- Reduced logistic fitting iterations and bounded symbolic-search candidates to keep the MVP fast while preserving the model zoo idea.
- Added `behavior_lab.causal.TreatmentComparator` for transparent treatment-vs-comparator comparison with uncertainty intervals, block slices, and tiny-sample warnings.
- Updated `ExperimentScheduler.estimate_treatment_effect` to delegate to `TreatmentComparator`.
- Added assignment-probability validation so invalid randomized designs fail early.
- Added `behavior_lab.stress.LabStressTester`, a runnable self-audit that checks:
  - pre-decision temporal snapshots do not contain post-decision fields,
  - hidden evaluation payloads redact labels and failure rows,
  - a discovered model is compared against the base-rate baseline,
  - the best discovered formula is scored against the synthetic hidden mechanism,
  - a separately labeled known-driver formula probe checks whether the DSL can express parts of the mechanism,
  - prospective split gaps are surfaced as warnings.
- Added `python -m behavior_lab stress-test` and `--matrix` mode across the synthetic hidden worlds.
- Added tests for causal comparison, probability validation, and the stress tester.
- Added append-only split manifests so cases do not migrate between splits as new observations arrive.
- Added ledger-enforced hidden/prospective evaluation budgets through `ResearchAPI`.
- Persisted fitted formula artifacts so a later `ResearchAPI` session can rehydrate them.
- Added `ResearchAPI.run_offline_experiment` for preregistered synthetic trial ingestion.
- Added `batch-stress` with lock files, run start/end records, config hashes, and idempotent skip behavior.

## Remaining honest gaps

- The “LLM scientist” is represented by a typed `ResearchAPI` plus a provider-agnostic `LLMHypothesisGenerator` validation seam, but no hosted/local LLM client is included. That is intentional for the MVP; the evaluator should mature before an LLM is allowed to propose hypotheses automatically.
- The prospective split is synthetic and often small. Split manifests now prevent migration, but real credibility still requires freezing a model and collecting future events after the freeze.
- The DSL is safe and useful, but small. It cannot yet express richer temporal state machines beyond the included hand-built two-state baseline.
- Treatment-effect estimates are deliberately simple randomized difference-in-means summaries, not a full causal inference system.
- The personal-data adapters are still stubs/boundaries. That is correct for now: manual or synthetic data should prove the loop before importing sensitive personal data.

## Commands

```powershell
python -m pytest
python -m behavior_lab demo --data-dir .demo --episodes 120 --iterations 2 --offline-trials 4
python -m behavior_lab stress-test --data-dir runs/stress-habit --episodes 120
python -m behavior_lab stress-test --data-dir runs/stress-matrix --episodes 100 --matrix
python -m behavior_lab batch-stress --data-dir runs/batch --worlds habit,threshold --seeds 11,23 --episode-counts 100,300
```
