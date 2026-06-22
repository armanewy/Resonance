from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from behavior_lab.core import parse_time, to_jsonable


@dataclass(frozen=True)
class TemperatureBracket:
    label: str
    lower_f: float | None
    upper_f: float | None
    lower_inclusive: bool = True
    upper_inclusive: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("bracket label must be non-empty")
        if self.lower_f is None and self.upper_f is None:
            raise ValueError("bracket must have at least one bound")
        if self.lower_f is not None and self.upper_f is not None and self.lower_f >= self.upper_f:
            raise ValueError("bracket lower bound must be below upper bound")

    def contains(self, temperature_f: float) -> bool:
        value = float(temperature_f)
        if self.lower_f is not None:
            if self.lower_inclusive and value < self.lower_f:
                return False
            if not self.lower_inclusive and value <= self.lower_f:
                return False
        if self.upper_f is not None:
            if self.upper_inclusive and value > self.upper_f:
                return False
            if not self.upper_inclusive and value >= self.upper_f:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DailyHighTemperatureEvent:
    event_id: str
    city: str
    station_id: str
    station_name: str
    local_date: str
    timezone: str
    dst_status: str
    settlement_series: str
    report_source: str
    report_name: str
    bracket: TemperatureBracket
    open_time: str
    close_time: str
    resolution_time: str
    market_source: str = "fixture"

    def __post_init__(self) -> None:
        for field_name in (
            "event_id",
            "city",
            "station_id",
            "station_name",
            "local_date",
            "timezone",
            "dst_status",
            "settlement_series",
            "report_source",
            "report_name",
            "market_source",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        opened = parse_time(self.open_time)
        closed = parse_time(self.close_time)
        resolved = parse_time(self.resolution_time)
        if opened > closed:
            raise ValueError("event open_time may not be after close_time")
        if closed > resolved:
            raise ValueError("event close_time may not be after resolution_time")

    @property
    def city_event_key(self) -> str:
        return (
            f"weather_edge:{self.settlement_series}:{self.report_source}:"
            f"{self.station_id}:{self.local_date}"
        )

    def settlement_semantics(self) -> dict[str, Any]:
        return {
            "settlement_series": self.settlement_series,
            "station_id": self.station_id,
            "station_name": self.station_name,
            "report_source": self.report_source,
            "report_name": self.report_name,
            "timezone": self.timezone,
            "dst_status": self.dst_status,
            "local_date": self.local_date,
            "bracket": self.bracket.to_dict(),
            "open_time": self.open_time,
            "close_time": self.close_time,
            "resolution_time": self.resolution_time,
        }

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.price) <= 1.0:
            raise ValueError("order-book prices must be probabilities between 0 and 1")
        if int(self.quantity) <= 0:
            raise ValueError("order-book quantity must be positive")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class MarketDepth:
    event_id: str
    as_of: str
    yes_bids: list[OrderBookLevel] = field(default_factory=list)
    yes_asks: list[OrderBookLevel] = field(default_factory=list)
    no_bids: list[OrderBookLevel] = field(default_factory=list)
    no_asks: list[OrderBookLevel] = field(default_factory=list)
    source: str = "fixture"
    snapshot_id: str = "fixture-depth"

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("market depth event_id must be non-empty")
        if not self.source.strip():
            raise ValueError("market depth source must be non-empty")
        if not self.snapshot_id.strip():
            raise ValueError("market depth snapshot_id must be non-empty")
        parse_time(self.as_of)

    @property
    def best_yes_ask(self) -> OrderBookLevel | None:
        if not self.yes_asks:
            return None
        return min(self.yes_asks, key=lambda level: (float(level.price), -int(level.quantity)))

    @property
    def best_yes_bid(self) -> OrderBookLevel | None:
        if not self.yes_bids:
            return None
        return max(self.yes_bids, key=lambda level: (float(level.price), int(level.quantity)))

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ForecastPoint:
    temperature_f: float
    probability: float

    def __post_init__(self) -> None:
        if float(self.probability) < 0.0:
            raise ValueError("forecast probabilities may not be negative")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class WeatherSnapshot:
    event_id: str
    as_of: str
    station_id: str
    timezone: str
    forecast_issued_at: str
    official_forecast_source: str
    forecast_distribution: list[ForecastPoint]
    regime: str = "unspecified"
    snapshot_id: str = "fixture-weather"

    def __post_init__(self) -> None:
        for field_name in (
            "event_id",
            "station_id",
            "timezone",
            "official_forecast_source",
            "regime",
            "snapshot_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        as_of = parse_time(self.as_of)
        issued = parse_time(self.forecast_issued_at)
        if issued > as_of:
            raise ValueError("forecast_issued_at may not be after snapshot as_of")
        if not self.forecast_distribution:
            raise ValueError("weather snapshot forecast_distribution may not be empty")
        if sum(point.probability for point in self.forecast_distribution) <= 0.0:
            raise ValueError("forecast distribution must have positive probability mass")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class StationHistoricalDay:
    station_id: str
    local_date: str
    high_f: float
    settlement_series: str
    report_source: str
    forecast_mean_f: float | None = None
    regime: str = "unspecified"

    def __post_init__(self) -> None:
        for field_name in ("station_id", "local_date", "settlement_series", "report_source", "regime"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class Settlement:
    event_id: str
    observed_high_f: float
    finalized_at: str
    station_id: str
    settlement_series: str
    report_source: str
    report_name: str
    timezone: str
    dst_status: str

    def __post_init__(self) -> None:
        for field_name in (
            "event_id",
            "station_id",
            "settlement_series",
            "report_source",
            "report_name",
            "timezone",
            "dst_status",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        parse_time(self.finalized_at)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class CostPolicy:
    per_contract_fee: float = 0.01
    slippage_cents: float = 1.0
    liquidity_fraction: float = 0.5
    max_contracts: int = 5
    min_edge_probability: float = 0.03
    uncertainty_buffer_probability: float = 0.05
    pessimistic_cost_multiplier: float = 2.0
    version: str = "weather_edge_costs.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "per_contract_fee",
            "slippage_cents",
            "min_edge_probability",
            "uncertainty_buffer_probability",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} may not be negative")
        if not 0.0 <= float(self.liquidity_fraction) <= 1.0:
            raise ValueError("liquidity_fraction must be between 0 and 1")
        if int(self.max_contracts) < 0:
            raise ValueError("max_contracts may not be negative")
        if float(self.pessimistic_cost_multiplier) < 1.0:
            raise ValueError("pessimistic_cost_multiplier must be at least 1")
        if not self.version.strip():
            raise ValueError("cost policy version must be non-empty")

    @property
    def slippage_per_contract(self) -> float:
        return float(self.slippage_cents) / 100.0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str = "station_bias_corrected_edge"
    strategy_version: str = "weather_edge_strategy.v1"
    decision_horizon: str = "close_minus_6h"
    horizon_hours_before_close: int = 6
    model_probability_name: str = "station_bias_corrected"

    def __post_init__(self) -> None:
        for field_name in ("strategy_id", "strategy_version", "decision_horizon", "model_probability_name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if int(self.horizon_hours_before_close) < 0:
            raise ValueError("horizon_hours_before_close may not be negative")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
