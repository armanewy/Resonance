from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
import math
from typing import Any, ClassVar

from behavior_lab.core import parse_time, stable_hash, to_jsonable


class FinanceDataError(ValueError):
    pass


class AdjustmentBasis(StrEnum):
    RAW = "raw"
    AS_REPORTED = "as_reported"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN_ADJUSTED = "total_return_adjusted"
    POINT_IN_TIME_CORPORATE_ACTION_KNOWLEDGE = "point_in_time_corporate_action_knowledge"
    DERIVED_WITH_CURRENT_CORPORATE_ACTION_KNOWLEDGE = "derived_with_current_corporate_action_knowledge"


class MarketValueKind(StrEnum):
    INDICATIVE = "indicative"
    LAST_TRADED = "last_traded"
    EXECUTABLE_BID = "executable_bid"
    EXECUTABLE_ASK = "executable_ask"
    SETTLEMENT = "settlement"


class MarketSessionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    AUCTION_ONLY = "auction_only"
    HALTED = "halted"


SUPPORTED_OBSERVATION_KINDS = {
    "quote",
    "trade",
    "daily_bar",
    "adjusted_total_return_bar",
    "order_book_snapshot",
    "market_calendar",
    "corporate_action",
    "distribution",
    "settlement_event",
    "economic_release",
    "revision_record",
    "vintage_snapshot",
    "cash_risk_free_benchmark",
}


@dataclass(frozen=True)
class ObservationMetadata:
    instrument_id: str
    source_id: str
    event_time: str
    available_at: str
    ingested_at: str
    timezone: str
    currency: str
    unit: str
    adjustment_basis: str
    revision_id: str
    source_artifact_hash: str
    source_observation_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.instrument_id, "instrument_id")
        _require_nonempty(self.source_id, "source_id")
        _require_nonempty(self.timezone, "timezone")
        _require_nonempty(self.currency, "currency")
        _require_nonempty(self.unit, "unit")
        _require_nonempty(self.revision_id, "revision_id")
        _require_nonempty(self.source_artifact_hash, "source_artifact_hash")
        if self.adjustment_basis not in {item.value for item in AdjustmentBasis}:
            raise FinanceDataError("adjustment_basis is not a supported market-data basis")
        available_at = parse_time(self.available_at)
        ingested_at = parse_time(self.ingested_at)
        parse_time(self.event_time)
        if ingested_at < available_at:
            raise FinanceDataError("ingested_at may not be earlier than available_at")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def metadata_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class Quote:
    metadata: ObservationMetadata
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    indicative_price: float | None = None
    bid_is_executable: bool = False
    ask_is_executable: bool = False
    venue: str | None = None
    condition: str | None = None

    kind: ClassVar[str] = "quote"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_optional_positive(self.bid_price, "bid_price")
        _require_optional_positive(self.ask_price, "ask_price")
        _require_optional_nonnegative(self.bid_size, "bid_size")
        _require_optional_nonnegative(self.ask_size, "ask_size")
        _require_optional_positive(self.indicative_price, "indicative_price")
        if self.bid_price is None and self.ask_price is None and self.indicative_price is None:
            raise FinanceDataError("quote must preserve at least one bid, ask, or indicative price")
        if self.bid_price is not None and self.ask_price is not None and self.bid_price > self.ask_price:
            raise FinanceDataError("bid_price may not exceed ask_price")
        if self.bid_is_executable and self.bid_price is None:
            raise FinanceDataError("an executable bid requires bid_price")
        if self.ask_is_executable and self.ask_price is None:
            raise FinanceDataError("an executable ask requires ask_price")

    @property
    def value_kinds(self) -> list[str]:
        kinds: list[str] = []
        if self.indicative_price is not None:
            kinds.append(MarketValueKind.INDICATIVE.value)
        if self.bid_price is not None and self.bid_is_executable:
            kinds.append(MarketValueKind.EXECUTABLE_BID.value)
        if self.ask_price is not None and self.ask_is_executable:
            kinds.append(MarketValueKind.EXECUTABLE_ASK.value)
        return kinds

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class Trade:
    metadata: ObservationMetadata
    trade_price: float
    quantity: float
    trade_id: str | None = None
    venue: str | None = None
    condition: str | None = None

    kind: ClassVar[str] = "trade"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_positive(self.trade_price, "trade_price")
        _require_positive(self.quantity, "quantity")

    @property
    def value_kind(self) -> str:
        return MarketValueKind.LAST_TRADED.value

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class DailyBar:
    metadata: ObservationMetadata
    session_date: str
    open_price: float | None
    high_price: float | None
    low_price: float | None
    close_price: float | None
    volume: float | None
    market_status: str = MarketSessionStatus.OPEN.value

    kind: ClassVar[str] = "daily_bar"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _parse_date(self.session_date, "session_date")
        if self.market_status not in {item.value for item in MarketSessionStatus}:
            raise FinanceDataError("market_status is not supported")
        for field_name in ("open_price", "high_price", "low_price", "close_price"):
            _require_optional_positive(getattr(self, field_name), field_name)
        _require_optional_nonnegative(self.volume, "volume")
        values = [self.open_price, self.high_price, self.low_price, self.close_price]
        if self.market_status == MarketSessionStatus.OPEN.value and all(value is None for value in values):
            raise FinanceDataError("open daily bars need at least one observed price")
        if self.high_price is not None and self.low_price is not None and self.high_price < self.low_price:
            raise FinanceDataError("high_price may not be below low_price")
        if self.metadata.adjustment_basis == AdjustmentBasis.TOTAL_RETURN_ADJUSTED.value:
            raise FinanceDataError("use AdjustedTotalReturnBar for total-return adjusted bars")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class AdjustedTotalReturnBar:
    metadata: ObservationMetadata
    session_date: str
    open_value: float | None
    high_value: float | None
    low_value: float | None
    close_value: float | None
    total_return_factor: float
    corporate_action_knowledge_at: str
    adjustment_source_revision_ids: list[str] = field(default_factory=list)
    uses_current_corporate_action_knowledge: bool = False
    current_knowledge_disclosure: str | None = None

    kind: ClassVar[str] = "adjusted_total_return_bar"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _parse_date(self.session_date, "session_date")
        for field_name in ("open_value", "high_value", "low_value", "close_value"):
            _require_optional_positive(getattr(self, field_name), field_name)
        _require_positive(self.total_return_factor, "total_return_factor")
        parse_time(self.corporate_action_knowledge_at)
        if self.metadata.adjustment_basis != AdjustmentBasis.TOTAL_RETURN_ADJUSTED.value:
            raise FinanceDataError("adjusted total-return bars must use total_return_adjusted basis")
        if self.high_value is not None and self.low_value is not None and self.high_value < self.low_value:
            raise FinanceDataError("high_value may not be below low_value")
        if all(value is None for value in [self.open_value, self.high_value, self.low_value, self.close_value]):
            raise FinanceDataError("adjusted total-return bars need at least one adjusted value")
        if self.uses_current_corporate_action_knowledge:
            _require_nonempty(self.current_knowledge_disclosure, "current_knowledge_disclosure")
        if any(not isinstance(item, str) or not item.strip() for item in self.adjustment_source_revision_ids):
            raise FinanceDataError("adjustment_source_revision_ids must contain non-empty strings")

    @property
    def effective_available_at(self) -> str:
        return max(
            parse_time(self.metadata.available_at),
            parse_time(self.corporate_action_knowledge_at),
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float
    order_count: int | None = None

    def __post_init__(self) -> None:
        _require_positive(self.price, "price")
        _require_positive(self.size, "size")
        if self.order_count is not None and self.order_count < 0:
            raise FinanceDataError("order_count may not be negative")


@dataclass(frozen=True)
class OrderBookSnapshot:
    metadata: ObservationMetadata
    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)
    depth: int | None = None
    venue: str | None = None

    kind: ClassVar[str] = "order_book_snapshot"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        if not self.bids and not self.asks:
            raise FinanceDataError("order-book snapshots may be one-sided, but not empty")
        if self.depth is not None and self.depth < 0:
            raise FinanceDataError("depth may not be negative")
        if self.bids != sorted(self.bids, key=lambda level: level.price, reverse=True):
            raise FinanceDataError("bid levels must be sorted from highest to lowest price")
        if self.asks != sorted(self.asks, key=lambda level: level.price):
            raise FinanceDataError("ask levels must be sorted from lowest to highest price")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class MarketCalendar:
    metadata: ObservationMetadata
    calendar_id: str
    session_date: str
    status: str
    open_time: str | None = None
    close_time: str | None = None
    settlement_time: str | None = None
    reason: str | None = None

    kind: ClassVar[str] = "market_calendar"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.calendar_id, "calendar_id")
        _parse_date(self.session_date, "session_date")
        if self.status not in {item.value for item in MarketSessionStatus}:
            raise FinanceDataError("calendar status is not supported")
        for field_name in ("open_time", "close_time", "settlement_time"):
            value = getattr(self, field_name)
            if value is not None:
                parse_time(value)
        if self.status == MarketSessionStatus.CLOSED.value and (self.open_time or self.close_time):
            raise FinanceDataError("closed sessions may not carry open or close times")
        if self.status in {MarketSessionStatus.OPEN.value, MarketSessionStatus.PARTIAL.value}:
            _require_nonempty(self.open_time, "open_time")
            _require_nonempty(self.close_time, "close_time")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class CorporateAction:
    metadata: ObservationMetadata
    action_id: str
    action_type: str
    effective_date: str
    announcement_time: str
    terms: dict[str, Any]

    kind: ClassVar[str] = "corporate_action"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.action_id, "action_id")
        _require_nonempty(self.action_type, "action_type")
        _parse_date(self.effective_date, "effective_date")
        parse_time(self.announcement_time)
        if not isinstance(self.terms, dict) or not self.terms:
            raise FinanceDataError("corporate action terms must be a non-empty object")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class Distribution:
    metadata: ObservationMetadata
    distribution_id: str
    distribution_type: str
    ex_date: str
    record_date: str | None
    payable_date: str | None
    amount: float
    tax_treatment: str | None = None

    kind: ClassVar[str] = "distribution"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.distribution_id, "distribution_id")
        _require_nonempty(self.distribution_type, "distribution_type")
        _parse_date(self.ex_date, "ex_date")
        for field_name in ("record_date", "payable_date"):
            value = getattr(self, field_name)
            if value is not None:
                _parse_date(value, field_name)
        _require_nonnegative(self.amount, "amount")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class SettlementEvent:
    metadata: ObservationMetadata
    settlement_id: str
    settlement_date: str
    settlement_value: float
    contract_reference: str | None = None

    kind: ClassVar[str] = "settlement_event"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.settlement_id, "settlement_id")
        _parse_date(self.settlement_date, "settlement_date")
        _require_finite(self.settlement_value, "settlement_value")

    @property
    def value_kind(self) -> str:
        return MarketValueKind.SETTLEMENT.value

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class EconomicRelease:
    metadata: ObservationMetadata
    series_id: str
    period_start: str
    period_end: str
    value: float | None
    release_stage: str
    revision_group_id: str
    vintage_id: str

    kind: ClassVar[str] = "economic_release"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.series_id, "series_id")
        _parse_date(self.period_start, "period_start")
        _parse_date(self.period_end, "period_end")
        if date.fromisoformat(self.period_end) < date.fromisoformat(self.period_start):
            raise FinanceDataError("period_end may not be before period_start")
        _require_optional_finite(self.value, "value")
        _require_nonempty(self.release_stage, "release_stage")
        _require_nonempty(self.revision_group_id, "revision_group_id")
        _require_nonempty(self.vintage_id, "vintage_id")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class RevisionRecord:
    metadata: ObservationMetadata
    observed_entity_id: str
    revision_group_id: str
    supersedes_revision_id: str | None
    new_revision_id: str
    revision_reason: str
    changed_fields: list[str]

    kind: ClassVar[str] = "revision_record"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.observed_entity_id, "observed_entity_id")
        _require_nonempty(self.revision_group_id, "revision_group_id")
        _require_nonempty(self.new_revision_id, "new_revision_id")
        _require_nonempty(self.revision_reason, "revision_reason")
        if self.supersedes_revision_id is not None:
            _require_nonempty(self.supersedes_revision_id, "supersedes_revision_id")
        if not self.changed_fields or any(not isinstance(item, str) or not item.strip() for item in self.changed_fields):
            raise FinanceDataError("changed_fields must contain at least one non-empty field name")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class VintageSnapshot:
    metadata: ObservationMetadata
    vintage_id: str
    revision_group_id: str
    as_of_date: str
    observations: dict[str, float | None]
    source_release_ids: list[str] = field(default_factory=list)

    kind: ClassVar[str] = "vintage_snapshot"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.vintage_id, "vintage_id")
        _require_nonempty(self.revision_group_id, "revision_group_id")
        _parse_date(self.as_of_date, "as_of_date")
        if not isinstance(self.observations, dict) or not self.observations:
            raise FinanceDataError("vintage snapshots must contain observations")
        for key, value in self.observations.items():
            _require_nonempty(key, "vintage observation key")
            _require_optional_finite(value, f"observations[{key!r}]")
        if any(not isinstance(item, str) or not item.strip() for item in self.source_release_ids):
            raise FinanceDataError("source_release_ids must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class CashRiskFreeBenchmark:
    metadata: ObservationMetadata
    benchmark_id: str
    fixing_date: str
    rate: float | None
    tenor: str
    day_count_convention: str
    compounding: str
    collateral_basis: str | None = None

    kind: ClassVar[str] = "cash_risk_free_benchmark"

    def __post_init__(self) -> None:
        _validate_metadata(self.metadata)
        _require_nonempty(self.benchmark_id, "benchmark_id")
        _parse_date(self.fixing_date, "fixing_date")
        _require_optional_finite(self.rate, "rate")
        _require_nonempty(self.tenor, "tenor")
        _require_nonempty(self.day_count_convention, "day_count_convention")
        _require_nonempty(self.compounding, "compounding")

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


FinanceObservation = (
    Quote
    | Trade
    | DailyBar
    | AdjustedTotalReturnBar
    | OrderBookSnapshot
    | MarketCalendar
    | CorporateAction
    | Distribution
    | SettlementEvent
    | EconomicRelease
    | RevisionRecord
    | VintageSnapshot
    | CashRiskFreeBenchmark
)


def observation_kind(observation: FinanceObservation) -> str:
    kind = getattr(observation, "kind", None)
    if kind not in SUPPORTED_OBSERVATION_KINDS:
        raise FinanceDataError(f"unsupported finance observation type: {type(observation).__name__}")
    return str(kind)


def effective_available_at(observation: FinanceObservation) -> datetime:
    metadata = observation.metadata
    available_at = parse_time(metadata.available_at)
    if isinstance(observation, AdjustedTotalReturnBar):
        return max(available_at, parse_time(observation.corporate_action_knowledge_at))
    return available_at


def observation_hash(observation: FinanceObservation) -> str:
    return stable_hash(_observation_dict(observation))


def source_artifact_hash(content: bytes | str | dict[str, Any]) -> str:
    if isinstance(content, bytes):
        return stable_hash({"bytes_hex": content.hex()})
    return stable_hash(content)


def _observation_dict(observation: FinanceObservation) -> dict[str, Any]:
    return {"kind": observation_kind(observation), **to_jsonable(observation)}


def _validate_metadata(metadata: ObservationMetadata) -> None:
    if not isinstance(metadata, ObservationMetadata):
        raise FinanceDataError("metadata must be ObservationMetadata")


def _require_nonempty(value: str | None, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FinanceDataError(f"{field_name} must be a non-empty string")


def _require_finite(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FinanceDataError(f"{field_name} must be finite")


def _require_optional_finite(value: float | None, field_name: str) -> None:
    if value is not None:
        _require_finite(value, field_name)


def _require_positive(value: float, field_name: str) -> None:
    _require_finite(value, field_name)
    if float(value) <= 0:
        raise FinanceDataError(f"{field_name} must be positive")


def _require_nonnegative(value: float, field_name: str) -> None:
    _require_finite(value, field_name)
    if float(value) < 0:
        raise FinanceDataError(f"{field_name} may not be negative")


def _require_optional_positive(value: float | None, field_name: str) -> None:
    if value is not None:
        _require_positive(value, field_name)


def _require_optional_nonnegative(value: float | None, field_name: str) -> None:
    if value is not None:
        _require_nonnegative(value, field_name)


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise FinanceDataError(f"{field_name} must be an ISO-8601 date") from exc
