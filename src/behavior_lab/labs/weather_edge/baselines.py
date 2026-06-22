from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from behavior_lab.core import to_jsonable
from behavior_lab.labs.weather_edge.models import (
    MarketDepth,
    StationHistoricalDay,
    TemperatureBracket,
    WeatherSnapshot,
)


@dataclass(frozen=True)
class BaselineProbabilities:
    market_implied: float | None
    station_climatology: float | None
    official_forecast: float
    station_bias_corrected: float
    station_bias_f: float
    historical_sample_size: int
    station_bias_sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def compute_baselines(
    *,
    bracket: TemperatureBracket,
    market_depth: MarketDepth,
    weather_snapshot: WeatherSnapshot,
    station_history: list[StationHistoricalDay],
) -> BaselineProbabilities:
    market_probability = market_implied_probability(market_depth)
    climatology = station_climatology_probability(bracket, station_history)
    official = official_forecast_probability(bracket, weather_snapshot)
    bias, bias_n = station_bias(station_history)
    corrected = official_forecast_probability(bracket, weather_snapshot, shift_f=bias)
    return BaselineProbabilities(
        market_implied=market_probability,
        station_climatology=climatology,
        official_forecast=official,
        station_bias_corrected=corrected,
        station_bias_f=bias,
        historical_sample_size=len(station_history),
        station_bias_sample_size=bias_n,
    )


def market_implied_probability(market_depth: MarketDepth) -> float | None:
    """Use executable YES ask, never midpoint or candle extremes."""

    best_ask = market_depth.best_yes_ask
    if best_ask is None:
        return None
    return round(float(best_ask.price), 6)


def station_climatology_probability(
    bracket: TemperatureBracket,
    station_history: list[StationHistoricalDay],
) -> float | None:
    if not station_history:
        return None
    hits = sum(1 for day in station_history if bracket.contains(day.high_f))
    return round(hits / len(station_history), 6)


def official_forecast_probability(
    bracket: TemperatureBracket,
    weather_snapshot: WeatherSnapshot,
    *,
    shift_f: float = 0.0,
) -> float:
    total = sum(float(point.probability) for point in weather_snapshot.forecast_distribution)
    if total <= 0.0:
        raise ValueError("forecast distribution has no probability mass")
    matched = sum(
        float(point.probability)
        for point in weather_snapshot.forecast_distribution
        if bracket.contains(float(point.temperature_f) + float(shift_f))
    )
    return round(matched / total, 6)


def station_bias(station_history: list[StationHistoricalDay]) -> tuple[float, int]:
    deltas = [
        float(day.high_f) - float(day.forecast_mean_f)
        for day in station_history
        if day.forecast_mean_f is not None
    ]
    if not deltas:
        return 0.0, 0
    return round(sum(deltas) / len(deltas), 6), len(deltas)
