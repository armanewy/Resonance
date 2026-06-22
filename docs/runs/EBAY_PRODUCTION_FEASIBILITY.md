# eBay Production Feasibility Probe

Status: blocked on 2026-06-21 because an authorized production test account and manually supplied listing IDs were not available in this environment.

## Intended Command

```powershell
$env:EBAY_ACCESS_TOKEN = "<production user token>"
python -m tools.ebay_api_probe.cli `
  --mode production `
  --seller-owned-listing-id "<owned item id>" `
  --buyer-participated-listing-id "<participated item id>" `
  --unrelated-listing-id "<manual unrelated ended item id>" `
  --output reports/ebay_production_feasibility.json
```

## Scope

Do not crawl. Use at most ten seller-owned or buyer-participated listings and ten manually supplied unrelated ended listing IDs. Do not request mutation scopes, respond to offers, send offers, or retain raw payloads.

## Current Conclusion

Result C: technically indeterminate. Current public-ended negotiation observation cannot be accepted or rejected until the authorized read-only probe runs against manually selected listing IDs.

## Gate

Do not build a public current-data observatory from this result. A future result A would only establish technical feasibility for the manually authorized read-only probe; any crawler, observatory, or broader collection would still require a separate explicit authorization and legal/compliance gate.
