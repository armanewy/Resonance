# eBay Sandbox Role Probe

Status: blocked on 2026-06-21 because Sandbox seller, bidder, and unrelated observer credentials/listings were not available in this environment.

## Intended Command

```powershell
$env:EBAY_SANDBOX_ACCESS_TOKEN = "<sandbox user token>"
python -m tools.ebay_api_probe.cli `
  --mode sandbox `
  --seller-owned-listing-id "<seller item id>" `
  --buyer-participated-listing-id "<bidder item id>" `
  --unrelated-listing-id "<unrelated item id>" `
  --output reports/ebay_sandbox_role_probe.json
```

## Manual Setup Required

- Create three eBay Sandbox users: seller, bidder, unrelated observer.
- Create one fixed-price listing with Best Offer enabled.
- Place at least two offers and one counteroffer.
- End or allow the listing to end.
- Run `GetBestOffers` as seller, bidder, and unrelated observer while active and after end.

## Data Retention Rule

Only retain the redacted field matrix, status code, API acknowledgment, warning/error codes, and hashed identifiers. Do not retain raw XML, raw JSON, message content, names, addresses, emails, access tokens, or refresh tokens.

## Gate

Sandbox role visibility is not evaluated. Current result: `blocked_missing_sandbox_credentials_and_manual_listing_setup`.

