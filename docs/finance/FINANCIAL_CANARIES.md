# Financial Canary Protocol

Wave 4 adds immutable prospective paper canaries for the three financial decision contracts:

- Weather Edge daily temperature-event paper canary, minimum 60 consecutive days.
- ETF Risk weekly long-only/cash paper canary, minimum 183 days.
- OfferLab seller shadow canary, minimum 30 days and only after seller-pilot readiness passes.

Canaries are local, append-only, and paper-only. They record the contract hash, frozen program hash, data-cutoff policy, source versions, cost assumptions, start/end dates, prospective gates, invalidation conditions, snapshots, source health, predictions, decisions, resolutions, value history, and counterexamples.

Material changes do not mutate an active canary. A strategy/program/source/cost/data-policy change either creates a new canary at `start` or is rejected at `resume`.

Commands:

```powershell
behavior-lab money canary start CONTRACT_ID
behavior-lab money canary resume CANARY_ID
behavior-lab money canary status CANARY_ID
behavior-lab money canary report CANARY_ID
behavior-lab money canary invalidate CANARY_ID --reason "..."
```

No canary authenticates to exchanges, submits orders, mutates seller state, sends notifications, or authorizes capital allocation.
