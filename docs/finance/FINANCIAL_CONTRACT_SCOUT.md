# Financial Contract Scout

The Contract Scout widens Resonance by finding paper-only economic decisions,
not by adding trading authority. It proposes structured contract families and
uses deterministic validation before anything can enter the experimental paper
portfolio.

Governing split:

```text
LLMs perform research labor.
Deterministic code controls belief.
Decision contracts control money.
```

## Commands

```powershell
behavior-lab money contract-scout run
behavior-lab money contract-scout proposals
behavior-lab money contract-scout approve PROPOSAL_ID
behavior-lab money contract-scout reject PROPOSAL_ID
behavior-lab money contract-scout report
```

For local source-tree execution:

```powershell
$env:PYTHONPATH='src'
python -m behavior_lab.cli money contract-scout run --state-dir C:\ResonanceOps\contract-scout
```

`run` can also accept a bounded research-agent proposal file:

```powershell
python -m behavior_lab.cli money contract-scout run `
  --state-dir C:\ResonanceOps\contract-scout `
  --proposals-json C:\ResonanceOps\contract-proposals.json
```

The existing money-agent runtime also exposes a `financial_contract_scout`
role. That role may produce structured proposals and rejected ideas, but it
cannot activate contracts, allocate money, change thresholds, or propose real
actions. The deterministic Contract Scout validator remains the authority gate.

## Proposal Gate

Every proposal must define:

- objective outcome and resolution source
- resolution cadence and decision deadline
- available actions and a no-action alternative
- executable payoff formula
- represented material costs
- bounded capital requirement and maximum loss
- source requirements and source coverage
- paper-mode feasibility
- platform, credential, licensing, maintenance, and information-value context

The scout stores rejected and duplicate proposals. It does not erase failed
ideas, activate production sources, allocate money, place trades, mutate seller
accounts, or send notifications.

## Automatic Eligibility

A proposal can become `eligible_experimental` only when:

- `paper_only` is true
- resolution is unambiguous
- the no-action alternative exists
- payoff is mechanically executable
- material costs are represented and not unknown
- maximum loss is bounded
- enough data exists or can be prospectively collected
- private-data dependencies have an acquisition path
- licensing does not require approval
- credentials are not required
- no real account mutation is required
- maintenance burden is not high relative to expected information value

Credential or unclear-license cases enter the approval inbox. Hard scientific
or accounting defects are rejected.

## Seed Families

The current release seeds the scout with:

- multicity Weather Edge paper contracts
- broad ETF Risk paper contracts
- seller shadow decisions, blocked until seller private data is acquired

Additional families can be proposed through structured JSON, but deterministic
validation decides whether they can enter the experimental paper portfolio.
