# Behavior Discovery Lab

Behavior Discovery Lab is a local, executable MVP for the four-wave research infrastructure described in the prompt:

1. **World Gym**: hidden synthetic behavioral worlds, append-only event ledger, temporal firewall, and blind evaluation.
2. **Formula Forge**: safe hypothesis DSL, logistic/rule/tree/state-style models, model comparison, Pareto frontier, residuals, counterexamples, and lineage.
3. **Personal N-of-1 Lab**: randomized intervention assignment, preregistration, crossover trials, treatment-effect estimation, and prospective model freezing.
4. **Autonomous Discovery Loop**: offline hypothesis generation, fitting, mutation, experiment proposal, observation consumption, and retire/promote decisions.

The core has no runtime dependencies beyond Python's standard library. Install the package in editable mode before using the CLI, or run with `PYTHONPATH=src`.

## Quick Start

```powershell
cd C:\Users\aoztu\Downloads\BehaviorDiscoveryLab
python -m pip install -e .
python -m behavior_lab demo --data-dir .demo --episodes 180 --iterations 3
python -m pytest
```

The demo seeds a hidden synthetic world, fits a heterogeneous model zoo, evaluates it through the blind judge, preregisters a randomized micro-experiment, estimates treatment effects from simulated trials, and runs an autonomous offline discovery loop.

## CLI

```powershell
python -m behavior_lab demo
python -m behavior_lab seed-world --data-dir .behavior_lab --world habit --episodes 200
python -m behavior_lab run-loop --data-dir .behavior_lab --iterations 5
python -m behavior_lab verify-ledger --data-dir .behavior_lab
python -m behavior_lab stress-test --data-dir runs/stress-habit --episodes 120
python -m behavior_lab stress-test --data-dir runs/stress-matrix --episodes 100 --matrix
python -m behavior_lab batch-stress --data-dir runs/batch --worlds habit,threshold --seeds 11,23 --episode-counts 100,300
python examples/first_research_session.py
```

## Design Notes

- The event ledger is append-only JSONL with a hash chain. Edits are represented as new facts, not rewrites.
- The temporal firewall builds prediction snapshots from pre-decision fields only.
- Hidden and prospective evaluation do not expose labels or failure rows.
- Split assignments are append-only ledger records, so existing cases do not migrate between training, development, hidden, and prospective splits when new observations arrive.
- `ResearchAPI` records hidden/prospective evaluation budget use in the ledger and defaults to one hidden and one prospective submission per campaign.
- Formula fits are persisted with terms, coefficients, feature schema, and a training snapshot hash, then rehydrated by new `ResearchAPI` sessions.
- Real intervention launch paths require explicit approval; offline synthetic experiments do not.
- Hypotheses are executable artifacts with stable IDs, parent lineage, assumptions, falsification conditions, and counted complexity.
- `ResearchAPI` is the LLM-facing facade for schema inspection, training-data queries, hypothesis submission, fitting, evaluation, residual inspection, model comparison, experiment proposal, simulation, and frozen-candidate submission.
- `LLMHypothesisGenerator` is a provider-agnostic adapter seam that validates proposed terms against the safe DSL and the variables exposed by `ResearchAPI`.
- `ResearchAPI.run_offline_experiment` preregisters, randomizes, appends synthetic trials, verifies the ledger, and returns an allowed summary.
- `batch-stress` runs fixed synthetic research matrices with per-run lock files and idempotent completion records.


## Stress testing

Run:

```powershell
python -m behavior_lab stress-test --data-dir runs/stress-habit --episodes 120
python -m behavior_lab stress-test --data-dir runs/stress-matrix --episodes 100 --matrix
```

The stress test is intentionally adversarial for an MVP: it checks temporal-firewall behavior, hidden-label redaction, baseline comparison, best-formula mechanism recall, and a separate formula-language known-driver probe. See `docs/STRESS_TEST.md` for the current audit, fixes, and remaining gaps.
