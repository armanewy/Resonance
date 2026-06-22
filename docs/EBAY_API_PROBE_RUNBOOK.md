# eBay API Probe Runbook

This probe is read-only. It records field availability, status codes, acknowledgments, warning/error codes, redacted visibility outcomes, and one aggregate feasibility conclusion. It must not respond to offers, send offers, create listings, revise listings, send messages, request mutation scopes, crawl listings, or discover arbitrary listing IDs.

Official behavior to test:

- `GetBestOffers` lets sellers see offer prices and currencies for their listings whether active or ended.
- Bidder visibility differs by role.
- A caller who is neither seller nor bidder sees offer prices and currencies only if the listing has ended; active unrelated listings return nothing.

## Token Policy

Tokens remain operator-owned and local. Do not share tokens with Codex, commit
them, print them, or store them in run artifacts. Live probing requires an
explicitly supplied production OAuth user token held in an operator-named
environment variable. The CLI requires the environment variable name; it never
accepts the token value as an argument.

The probe may be reviewed and tested with fixtures. Sandbox HTTP parsing tests
are allowed for redaction verification only; the live feasibility probe is
production-only.

## Production Feasibility Probe

```powershell
python -m tools.ebay_api_probe.cli `
  --mode production `
  --token-env EBAY_PRODUCTION_USER_TOKEN `
  --scope https://api.ebay.com/oauth/api_scope `
  --seller-owned-listing-id "<owned item id>" `
  --buyer-participated-listing-id "<participated item id>" `
  --unrelated-listing-id "<manually supplied ended item id>" `
  --output reports/ebay_production_feasibility.json
```

Repeat `--scope` for each authorized read-only scope on the token. Do not include
mutation scopes. Use exactly the manually selected seller-owned,
buyer-participated, and unrelated ended listing IDs. The probe makes only
role-scoped `GetBestOffers` read requests; it does not list inventory, orders,
transactions, traffic, search results, or other records that could discover
arbitrary listing IDs.

The unrelated-ended result must be interpreted as one of `accessible`, `denied`,
`empty`, or `indeterminate`; unexpected permission results are observations, not
exceptions. One failed request is not a platform-wide conclusion. The aggregate
probe conclusion is one of `technically feasible`, `partially feasible`,
`not feasible`, or `indeterminate`.

## Retention

The committed repository may contain only redacted summaries:

- status code
- API acknowledgment
- warning/error code
- field-availability booleans
- hashed listing identifiers
- user identifier presence flags, not raw or hashed user IDs
- message-field detected/discarded flags

Raw XML/JSON responses, message content, names, addresses, emails, access tokens, and refresh tokens must not be retained.

## Seller Export Pilot

This probe is an API feasibility check only. It must not block the seller-export
pilot, which remains a separate seller-authorized data path.
