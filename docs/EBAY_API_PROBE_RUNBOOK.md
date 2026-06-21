# eBay API Probe Runbook

This probe is read-only. It records field availability, status codes, acknowledgments, warning/error codes, and redacted visibility outcomes. It must not respond to offers, send offers, create listings, revise listings, send messages, or request mutation scopes.

Official behavior to test:

- `GetBestOffers` lets sellers see offer prices and currencies for their listings whether active or ended.
- Bidder visibility differs by role.
- A caller who is neither seller nor bidder sees offer prices and currencies only if the listing has ended; active unrelated listings return nothing.

## Tokens

Read tokens only from environment variables:

```powershell
$env:EBAY_SANDBOX_ACCESS_TOKEN = "<sandbox user token>"
$env:EBAY_ACCESS_TOKEN = "<production user token>"
```

Do not print, commit, or store tokens.

## Sandbox Probe

```powershell
python -m tools.ebay_api_probe.cli `
  --mode sandbox `
  --seller-owned-listing-id "<seller item id>" `
  --buyer-participated-listing-id "<bidder item id>" `
  --unrelated-listing-id "<unrelated ended or active item id>" `
  --output reports/ebay_sandbox_role_probe.json
```

Use separate seller, bidder, and unrelated observer users for the role experiment. Preserve only the JSON field matrix and redacted identifiers.

## Production Feasibility Probe

```powershell
python -m tools.ebay_api_probe.cli `
  --mode production `
  --seller-owned-listing-id "<owned item id>" `
  --buyer-participated-listing-id "<participated item id>" `
  --unrelated-listing-id "<manually supplied ended item id>" `
  --output reports/ebay_production_feasibility.json
```

Do not crawl. Use at most the manually selected listing IDs in the Wave 3 protocol. The unrelated-ended result must be interpreted as one of `accessible`, `denied`, `empty`, or `indeterminate`; unexpected permission results are observations, not exceptions.

## Retention

The committed repository may contain only redacted summaries:

- status code
- API acknowledgment
- warning/error code
- field-availability booleans
- hashed listing/user identifiers
- message-field detected/discarded flags

Raw XML/JSON responses, message content, names, addresses, emails, access tokens, and refresh tokens must not be retained.

