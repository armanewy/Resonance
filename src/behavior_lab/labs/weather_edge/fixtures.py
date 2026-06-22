from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from behavior_lab.core import parse_time
from behavior_lab.labs.weather_edge.models import (
    DailyHighTemperatureEvent,
    ForecastPoint,
    MarketDepth,
    OrderBookLevel,
    Settlement,
    StationHistoricalDay,
    TemperatureBracket,
    WeatherSnapshot,
)


class WeatherEdgeProvider(Protocol):
    def discover_events(
        self,
        as_of: str,
        *,
        include_resolved: bool = False,
    ) -> list[DailyHighTemperatureEvent]:
        ...

    def market_depth(self, event_id: str, as_of: str) -> MarketDepth:
        ...

    def weather_snapshot(self, event_id: str, as_of: str) -> WeatherSnapshot:
        ...

    def settlement(self, event_id: str) -> Settlement | None:
        ...

    def station_history(self, station_id: str, *, before_local_date: str) -> list[StationHistoricalDay]:
        ...


class FixtureWeatherEdgeProvider:
    """Deterministic provider used by tests and local paper-lab fixtures."""

    def __init__(
        self,
        *,
        events: list[DailyHighTemperatureEvent],
        market_depths: list[MarketDepth],
        weather_snapshots: list[WeatherSnapshot],
        settlements: list[Settlement] | None = None,
        station_history: list[StationHistoricalDay] | None = None,
        historical_resolved_city_days: int | None = None,
    ):
        self._events = sorted(events, key=lambda event: (event.local_date, event.city, event.event_id))
        self._depths = sorted(market_depths, key=lambda depth: (depth.event_id, parse_time(depth.as_of)))
        self._snapshots = sorted(weather_snapshots, key=lambda snapshot: (snapshot.event_id, parse_time(snapshot.as_of)))
        self._settlements = {settlement.event_id: settlement for settlement in settlements or []}
        self._history = sorted(station_history or [], key=lambda day: (day.station_id, day.local_date))
        self._historical_resolved_city_days = historical_resolved_city_days

    @classmethod
    def from_json(cls, path: str | Path) -> "FixtureWeatherEdgeProvider":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixtureWeatherEdgeProvider":
        return cls(
            events=[_event(item) for item in payload.get("events", [])],
            market_depths=[_depth(item) for item in payload.get("market_depths", [])],
            weather_snapshots=[_snapshot(item) for item in payload.get("weather_snapshots", [])],
            settlements=[_settlement(item) for item in payload.get("settlements", [])],
            station_history=[_history_day(item) for item in payload.get("station_history", [])],
            historical_resolved_city_days=payload.get("historical_resolved_city_days"),
        )

    def discover_events(
        self,
        as_of: str,
        *,
        include_resolved: bool = False,
    ) -> list[DailyHighTemperatureEvent]:
        as_of_time = parse_time(as_of)
        output = []
        for event in self._events:
            opened = parse_time(event.open_time)
            closed = parse_time(event.close_time)
            if include_resolved:
                if opened <= as_of_time:
                    output.append(event)
            elif opened <= as_of_time <= closed:
                output.append(event)
        return output

    def market_depth(self, event_id: str, as_of: str) -> MarketDepth:
        as_of_time = parse_time(as_of)
        eligible = [
            depth
            for depth in self._depths
            if depth.event_id == event_id and parse_time(depth.as_of) <= as_of_time
        ]
        if not eligible:
            raise LookupError(f"no market depth for {event_id} at or before {as_of}")
        return max(eligible, key=lambda depth: parse_time(depth.as_of))

    def weather_snapshot(self, event_id: str, as_of: str) -> WeatherSnapshot:
        as_of_time = parse_time(as_of)
        eligible = [
            snapshot
            for snapshot in self._snapshots
            if snapshot.event_id == event_id and parse_time(snapshot.as_of) <= as_of_time
        ]
        if not eligible:
            raise LookupError(f"no weather snapshot for {event_id} at or before {as_of}")
        return max(eligible, key=lambda snapshot: parse_time(snapshot.as_of))

    def settlement(self, event_id: str) -> Settlement | None:
        return self._settlements.get(event_id)

    def station_history(self, station_id: str, *, before_local_date: str) -> list[StationHistoricalDay]:
        return [
            day
            for day in self._history
            if day.station_id == station_id and day.local_date < before_local_date
        ]

    def historical_resolved_count(self) -> int:
        if self._historical_resolved_city_days is not None:
            return int(self._historical_resolved_city_days)
        keys = {
            event.city_event_key
            for event in self._events
            if event.event_id in self._settlements
        }
        return len(keys)


def _event(payload: dict[str, Any]) -> DailyHighTemperatureEvent:
    bracket = payload.get("bracket")
    if not isinstance(bracket, TemperatureBracket):
        bracket = TemperatureBracket(**bracket)
    return DailyHighTemperatureEvent(**{**payload, "bracket": bracket})


def _level(payload: dict[str, Any]) -> OrderBookLevel:
    return payload if isinstance(payload, OrderBookLevel) else OrderBookLevel(**payload)


def _depth(payload: dict[str, Any]) -> MarketDepth:
    return MarketDepth(
        **{
            **payload,
            "yes_bids": [_level(item) for item in payload.get("yes_bids", [])],
            "yes_asks": [_level(item) for item in payload.get("yes_asks", [])],
            "no_bids": [_level(item) for item in payload.get("no_bids", [])],
            "no_asks": [_level(item) for item in payload.get("no_asks", [])],
        }
    )


def _point(payload: dict[str, Any]) -> ForecastPoint:
    return payload if isinstance(payload, ForecastPoint) else ForecastPoint(**payload)


def _snapshot(payload: dict[str, Any]) -> WeatherSnapshot:
    return WeatherSnapshot(
        **{
            **payload,
            "forecast_distribution": [_point(item) for item in payload.get("forecast_distribution", [])],
        }
    )


def _settlement(payload: dict[str, Any]) -> Settlement:
    return payload if isinstance(payload, Settlement) else Settlement(**payload)


def _history_day(payload: dict[str, Any]) -> StationHistoricalDay:
    return payload if isinstance(payload, StationHistoricalDay) else StationHistoricalDay(**payload)
