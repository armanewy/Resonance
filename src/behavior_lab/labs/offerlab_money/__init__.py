"""OfferLab financial decision ledger integration.

Callable command-equivalent hooks for future CLI wiring:

- ``evaluate(pilot_id, ...)`` is the implementation surface for
  ``behavior-lab money offerlab evaluate PILOT_ID``.
- ``report(pilot_id, ...)`` is the implementation surface for
  ``behavior-lab money offerlab report PILOT_ID``.

This package is intentionally read-only with respect to seller systems: it
uses imported seller pilot ledgers, writes paper-only money contracts and
ledger entries, sends no notifications, and submits no marketplace actions.
"""

from behavior_lab.labs.offerlab_money.evaluation import (
    OfferLabMoneyError,
    evaluate,
    evaluate_pilot,
    report,
    report_pilot,
)

__all__ = [
    "OfferLabMoneyError",
    "evaluate",
    "evaluate_pilot",
    "report",
    "report_pilot",
]
