# eBay API Probe

The probe is read-only. It should be run only against an authorized test account
and should retain a redacted field/permission matrix, not raw payloads.

Allowed intent:

- Verify whether seller-owned Best Offer data is accessible.
- Verify whether unrelated listing requests are denied.
- Verify whether active inventory, orders, finances, traffic, and completed
  transaction fields are available.
- Compare sandbox and production field availability.

Forbidden intent:

- No `RespondToBestOffer`.
- No seller-initiated offer send.
- No inventory offer create/update/publish.
- No marketing discount create/pause/update.
- No message body collection.
- No crawler behavior.

Relevant eBay references checked on 2026-06-21:

- [Authorization and OAuth scopes](https://developer.ebay.com/develop/guides-v2/authorization)
- [Trading API Best Offer management](https://developer.ebay.com/develop/guides-v2/other-apis/other-apis-guide)
- [Negotiation API](https://developer.ebay.com/api-docs/sell/negotiation/resources/methods)
- [Sell Feed API](https://developer.ebay.com/api-docs/sell/static/feed/sell-feed.html)

The probe is not a production integration. It exists to decide whether the
official API surface can support a later read-only OfferLab pilot.
