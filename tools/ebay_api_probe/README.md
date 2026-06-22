# eBay API Probe

This probe is a read-only feasibility check for an explicitly authorized
production user token and three manually selected listing IDs. It records field
availability and permission boundaries; it does not
respond to Best Offers, send seller offers, create/update listings, create
discounts, retrieve message content, or mutate eBay state.

The default live probe makes only role-scoped `GetBestOffers` read requests for
the operator-supplied seller-owned, buyer-participated, and unrelated ended
listing IDs. It does not crawl, search, list inventory, or discover arbitrary
listing IDs.
