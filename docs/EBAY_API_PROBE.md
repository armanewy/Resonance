# eBay API Probe

The probe is read-only. It should be run only against an explicitly authorized
production user token and three manually supplied listing IDs. It retains a
redacted field/permission matrix and aggregate feasibility conclusion, not raw
payloads.

Allowed intent:

- Verify whether seller-owned Best Offer data is accessible.
- Verify whether buyer-participated Best Offer data is accessible.
- Empirically record whether unrelated ended-listing requests are accessible or denied; do not hard-code either outcome.
- Verify whether offer history, redacted actor presence, timestamps, response
  state, and completed outcome linkage fields are available.
- Produce exactly one of `technically feasible`, `partially feasible`,
  `not feasible`, or `indeterminate`.

Forbidden intent:

- No `RespondToBestOffer`.
- No seller-initiated offer send.
- No inventory offer create/update/publish.
- No marketing discount create/pause/update.
- No message body collection.
- No crawler behavior.
- No inventory, order, finance, traffic, search, or listing-list requests that
  could discover arbitrary listing IDs during this probe.

Relevant eBay references checked on 2026-06-21:

- [Authorization and OAuth scopes](https://developer.ebay.com/develop/guides-v2/authorization)
- [Trading API Best Offer management](https://developer.ebay.com/develop/guides-v2/other-apis/other-apis-guide)
- [Negotiation API](https://developer.ebay.com/api-docs/sell/negotiation/resources/methods)
- [Sell Feed API](https://developer.ebay.com/api-docs/sell/static/feed/sell-feed.html)

The probe is not a production integration. Unrelated-listing visibility is an empirical result, not a permission assumption. A single failed role request is not a platform-wide conclusion. The probe exists to decide whether the official API surface can support a later read-only OfferLab pilot, and it must not block the seller-export pilot.
