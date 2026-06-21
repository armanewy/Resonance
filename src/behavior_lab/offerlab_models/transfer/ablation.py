from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from behavior_lab.offerlab_models.common import PRODUCTION_EXPORT_ALLOWED, SOURCE_ID


@dataclass(frozen=True)
class TransferAblationResult:
    source_id: str
    ancillary_source_ids: list[str]
    raw_pooling_allowed: bool
    retained: bool
    status: str
    reason: str
    base_hidden_loss: float | None
    transfer_hidden_loss: float | None
    base_calibration_error: float | None
    transfer_calibration_error: float | None
    minimum_required_hidden_loss_delta: float
    minimum_required_calibration_delta: float
    production_export_allowed: bool = PRODUCTION_EXPORT_ALLOWED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_transfer_ablation(
    *,
    base_hidden_loss: float | None = None,
    transfer_hidden_loss: float | None = None,
    base_calibration_error: float | None = None,
    transfer_calibration_error: float | None = None,
    minimum_required_hidden_loss_delta: float = 0.005,
    minimum_required_calibration_delta: float = 0.005,
    ancillary_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate measured transfer results without fabricating defaults.

    Calling this function without actual measurements returns an explicit
    ``not_run`` result. This prevents sample reports from quietly presenting
    invented hidden losses as experimental evidence.
    """

    ancillary_source_ids = ancillary_source_ids or [
        "open_bandit_dataset",
        "criteo_uplift",
        "craigslist_bargain",
    ]
    measurements = [
        base_hidden_loss,
        transfer_hidden_loss,
        base_calibration_error,
        transfer_calibration_error,
    ]
    if all(value is None for value in measurements):
        return TransferAblationResult(
            source_id=SOURCE_ID,
            ancillary_source_ids=ancillary_source_ids,
            raw_pooling_allowed=False,
            retained=False,
            status="not_run",
            reason="No measured hidden/calibration results were supplied; transfer remains disabled.",
            base_hidden_loss=None,
            transfer_hidden_loss=None,
            base_calibration_error=None,
            transfer_calibration_error=None,
            minimum_required_hidden_loss_delta=minimum_required_hidden_loss_delta,
            minimum_required_calibration_delta=minimum_required_calibration_delta,
        ).to_dict()
    if any(value is None for value in measurements):
        raise ValueError("all four base/transfer hidden and calibration metrics are required")
    numeric = [float(value) for value in measurements if value is not None]
    if any(value < 0.0 for value in numeric):
        raise ValueError("loss and calibration metrics may not be negative")
    assert base_hidden_loss is not None
    assert transfer_hidden_loss is not None
    assert base_calibration_error is not None
    assert transfer_calibration_error is not None
    hidden_gain = base_hidden_loss - transfer_hidden_loss
    calibration_gain = base_calibration_error - transfer_calibration_error
    retained = (
        hidden_gain >= minimum_required_hidden_loss_delta
        and calibration_gain >= minimum_required_calibration_delta
    )
    if retained:
        reason = (
            "ancillary transfer retained because both measured NBER hidden loss "
            "and calibration improved beyond thresholds"
        )
    else:
        reason = (
            "ancillary transfer retired because measured results did not improve "
            "both NBER hidden loss and calibration"
        )
    return TransferAblationResult(
        source_id=SOURCE_ID,
        ancillary_source_ids=ancillary_source_ids,
        raw_pooling_allowed=False,
        retained=retained,
        status="completed",
        reason=reason,
        base_hidden_loss=base_hidden_loss,
        transfer_hidden_loss=transfer_hidden_loss,
        base_calibration_error=base_calibration_error,
        transfer_calibration_error=transfer_calibration_error,
        minimum_required_hidden_loss_delta=minimum_required_hidden_loss_delta,
        minimum_required_calibration_delta=minimum_required_calibration_delta,
    ).to_dict()
