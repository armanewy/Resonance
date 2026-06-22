# Behavior Discovery Lab

Behavior Discovery Lab is a local research harness for discovering and falsifying executable behavioral hypotheses.

A hypothesis may be a compact formula, rule, threshold, state model, nearest-neighbor policy, or another reloadable predictor. The laboratory keeps creativity and judgment separate:

```text
Generate hypotheses
        ↓
Fit only on campaign training data
        ↓
Iterate on development data
        ↓
Freeze the selected model artifact and data cut
        ↓
Spend one hidden lockbox query
        ↓
Collect genuinely new observations
        ↓
Spend one prospective query
```

This repository is an infrastructure MVP, not a validated human-behavior oracle.

`v0.3.0` remains frozen as the laboratory version for the first real-use campaign. The current reviewed harness is `v0.4.0`. Campaign 001 is a scientific demonstration and fixture; Campaign 002 is the OfferLab commercial research direction.

## What is implemented

- Hidden synthetic behavior worlds with deterministic, restart-safe event generation.
- Append-only JSONL ledger with a hash chain and local-process write locking.
- Immutable, chronological split manifests scoped to research campaigns.
- Pre-decision snapshots that structurally exclude known post-decision fields.
- A bounded mathematical DSL for formulas.
- A heterogeneous model foundry: base rate, recent rate, nearest neighbor, threshold, decision stump, two-state model, linear formula, and symbolic search.
- Campaign-scoped, hashed, reloadable model artifacts.
- Development diagnostics, residuals, counterexamples, paired comparison, and complexity frontiers.
- Persistent hidden/prospective evaluation budgets, reserved before evaluation so crashes cannot create free retries.
- Exact artifact, training-snapshot, and hidden-case-set binding for OfferLab research lockboxes.
- Conservative NBER listing-purged temporal splits that keep all negotiation threads for a listing in one split region.
- Explicit censoring: incomplete negotiation threads are not converted into failed-agreement labels.
- Immutable authorization-evidence gates for production artifacts derived from current seller data.
- Cross-campaign protection against reusing previously queried hidden cases.
- Preregistered randomized experiments with assignment and outcome integrity checks.
- Deterministic, restart-safe randomization streams that remain independent across repeated registrations of the same design.
- Difference-in-means and inverse-probability-weighted treatment comparisons.
- A provider-neutral, validated LLM hypothesis-generator seam.
- Locked, idempotent synthetic batch stress runs.
- Campaign 002 OfferLab scaffolding: normalized eBay offer snapshots, append-only read-only ingest, mature-margin audit, five-section profit-audit report, and abstaining decision-support arithmetic.
- OfferLab evidence-wave scaffolding: external data-source registry, commercial-use firewall, reproducible cache, federated benchmark contracts, NBER Best Offer benchmark path, Open Bandit OPE checks, and Criteo uplift research-only checks.

The core runtime uses only Python's standard library.

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install -e .
python -m pytest
python -m behavior_lab stress-test \
  --data-dir runs/stress-habit \
  --world habit \
  --episodes 160 \
  --seed 17
```

Run the complete demonstration:

```bash
python -m behavior_lab demo \
  --data-dir runs/demo \
  --world habit \
  --episodes 180 \
  --iterations 3 \
  --offline-trials 12 \
  --prospective-episodes 40
```

The demo resets its output directory by default. Pass `--no-reset` only when you intentionally want to continue the same ledger.

## What you should see

The stress report should show:

```json
{
  "temporal_firewall_ok": true,
  "split_chronology_ok": true,
  "initial_prospective_empty": true,
  "hidden_payload_redacted": true
}
```

The initial campaign should contain training, development, and hidden cases, but **zero prospective cases**. Prospective means observations first recorded after a model freeze; it does not mean "the newest fraction of an existing file."

The discovery loop should:

1. Create a fresh campaign for each offline iteration.
2. Use only training and development results while mutating hypotheses.
3. Leave every intermediate hidden split unqueried.
4. Select one final candidate using development data.
5. Freeze the exact persisted artifact, split snapshot, and data cutoff.
6. Submit that frozen candidate once to the final hidden lockbox.
7. Generate new observations after the freeze.
8. Submit the same frozen artifact once to the prospective evaluator.

## Campaign semantics

A campaign is an immutable view of the observations available when it starts.

```text
Existing observations at campaign creation
  → chronological training/development/hidden assignment

New observations before freeze
  → staging

New observations after freeze
  → prospective, bound to that freeze ID
```

Staging data never moves backward into a campaign's training set. Start a new campaign to incorporate it.

Model artifacts are campaign-scoped. Reopen `ResearchAPI` using the same campaign ID to reload its models:

```python
api = ResearchAPI(gym, campaign_id="experiment-001")
# ... fit model ...

reloaded = ResearchAPI(gym, campaign_id="experiment-001")
```

A different campaign may deliberately refit or import a model, but it does not silently inherit another campaign's fitted registry.

## Lockbox limits

`ResearchAPI` defaults to:

- One hidden aggregate submission per hidden case set.
- One prospective aggregate submission per frozen candidate.

Renaming a campaign does not reset a hidden budget when the hidden cases overlap a previously queried set.

Hidden and prospective responses omit raw labels, failure rows, direct prevalence, and baseline lift. However, **any aggregate scoring metric carries some statistical information**. The one-query budget is therefore a scientific discipline, not perfect information-theoretic secrecy.

`ResearchAPI` is a logical boundary inside one Python process. Do not give untrusted generated code direct filesystem access to the ledger or evaluator. A production LLM researcher should run out of process and receive only typed RPC tools.

## Campaign 001: Task Initiation

The first real campaign is observational only:

```text
campaign_id: campaign_001_task_initiation
target: Did I begin the intended task within 10 minutes?
initial block: 50 natural episodes
interventions: none
```

Record pre-decision features before the task decision:

```text
task_type
time_of_day
fatigue: 0..3
ambiguity: 0..3
estimated_minutes
first_step_explicit
deadline_hours
recent_context_switches
public_commitment
```

Record outcomes afterward under `protected_outcome`:

```text
started_within_10_minutes
start_latency_seconds
worked_for_20_minutes
completed_that_day
```

Manual collection flow:

```bash
python -m behavior_lab campaign-001-template \
  --output campaigns/campaign_001_task_initiation/manual_entry_template.json

python -m behavior_lab bridge-hash \
  --input manual_raw.jsonl \
  --output export_hashed.jsonl

python -m behavior_lab bridge-validate \
  --input export_hashed.jsonl

python -m behavior_lab bridge-import \
  --input export_hashed.jsonl \
  --data-dir data/campaign_001_task_initiation
```

The bridge imports immutable Behavior Lab export snapshots into `data/campaign_001_task_initiation/ledger.jsonl`. It rejects missing fields, outcome leakage into pre-decision features, malformed source hashes, and duplicate episode IDs.

## Campaign 002: OfferLab

Campaign 002 is the commercial wedge:

```text
campaign_id: campaign_002_ebay_seller_offers
goal: optimize seller-side eBay offer and pricing decisions for contribution margin
stage: read-only profit audit
```

Current commands:

```bash
python -m behavior_lab offerlab-template
python -m behavior_lab offerlab-ingest --input campaigns/campaign_002_ebay_seller_offers/examples/historical_decisions.jsonl
python -m behavior_lab offerlab-audit
python -m behavior_lab offerlab-report --output reports/offerlab_profit_audit.md
python -m behavior_lab offerlab-recommend --input campaigns/campaign_002_ebay_seller_offers/examples/pending_offer_snapshot.json
python -m behavior_lab offerlab-pilot template
python -m behavior_lab offerlab-pilot inspect C:\OfferLabData\seller_pilot_drop
python -m behavior_lab offerlab-pilot import C:\OfferLabData\seller_pilot_drop
python -m behavior_lab offerlab-pilot audit PILOT_ID
```

OfferLab does not call eBay or execute seller actions yet. It records normalized snapshots, produces a read-only profit audit, and abstains from recommendation when seller cost basis, fee data, traffic freshness, or comparable mature outcomes are insufficient. See [`docs/OFFERLAB.md`](docs/OFFERLAB.md).

## OfferLab Evidence Waves

The public-data validation path is separate from the production OfferLab product:

```bash
python -m behavior_lab data-source list
python -m behavior_lab data-source verify nber_ebay_best_offer criteo_uplift --use production_export

python -m behavior_lab nber-best-offer build-sample --output-dir runs/nber_sample/raw
python -m behavior_lab nber-best-offer normalize \
  --input-dir runs/nber_sample/raw \
  --output-dir runs/nber_sample/normalized
python -m behavior_lab nber-best-offer benchmark --normalized-dir runs/nber_sample/normalized
python -m behavior_lab nber-best-offer audit --normalized-dir runs/nber_sample/normalized

python -m behavior_lab benchmark-suite run   # smoke fixtures only
python -m behavior_lab benchmark-suite permissions
```

The NBER sample commands and `benchmark-suite run` use tiny fixtures to verify plumbing. They are not evidence until run against real source files. The NBER, Open Bandit, Criteo, AuctionNet, and CraigslistBargain lanes are research benchmarks unless a source is explicitly cleared for commercial training and production export. See [`docs/DATASET_ROADMAP.md`](docs/DATASET_ROADMAP.md).

## CLI

```bash
python -m behavior_lab seed-world --data-dir runs/world --world habit --episodes 200 --seed 7
python -m behavior_lab run-loop --data-dir runs/world --world habit --iterations 4
python -m behavior_lab verify-ledger --data-dir runs/world
python -m behavior_lab bridge-import --input export_hashed.jsonl --data-dir data/campaign_001_task_initiation
python -m behavior_lab offerlab-report --output reports/offerlab_profit_audit.md
python -m behavior_lab offerlab-recommend --input campaigns/campaign_002_ebay_seller_offers/examples/pending_offer_snapshot.json
python -m behavior_lab nber-best-offer benchmark --normalized-dir runs/nber_sample/normalized
python -m behavior_lab stress-test --data-dir runs/matrix --episodes 120 --matrix
python -m behavior_lab batch-stress \
  --data-dir runs/batch \
  --worlds habit,two_mode,threshold,nonstationary,confounded \
  --seeds 11,23,47 \
  --episode-counts 100,300
python examples/first_research_session.py
```

## Automated background research

Start by automating synthetic falsification, not real-life interventions.

A safe researcher may repeatedly use:

```text
inspect_schema
list_variables
describe_target
query_training_data
submit_hypothesis
fit_hypothesis
evaluate on development
inspect residuals and counterexamples
propose synthetic experiment
run preregistered offline experiment
```

A gatekeeper should exclusively control:

```text
hidden evaluation
candidate freeze
prospective evaluation
real intervention launch
```

See [`docs/AUTOMATION.md`](docs/AUTOMATION.md) for the recommended worker lifecycle and budgets.

## Scientific interpretation

Do not celebrate a model because its prose sounds human or because it wins once on development data.

A credible progression is:

```text
beats base rate on development
→ survives multiple seeds
→ survives a hidden chronological block
→ is frozen
→ survives genuinely future observations
→ predicts intervention direction
→ remains competitive at lower complexity
```

The stress tester's mechanism score is only exact-variable recall against a synthetic hidden world. It is not proof that the recovered equation is causally or mathematically equivalent.

## Current limitations

- The LLM adapter validates proposals but does not include a hosted or local model client.
- The evaluator is a logical in-process boundary, not a hostile-code sandbox.
- The formula DSL is intentionally small.
- The causal layer supports randomized binary comparisons, not arbitrary observational causal identification.
- Personal data adapters are intentionally absent.
- Pre-decision structural filtering cannot detect semantic leakage hidden inside misleading field names or prose.
- Aggregate lockbox metrics leak limited information by their nature.
- Real credibility requires enough future observations collected after a model freeze.

See [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md) for the stress-test findings and fixes applied to this version.
