from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx

from resonance.config import AppConfig
from resonance.storage import Measurement
from resonance.time_utils import parse_open_meteo_time, to_utc_iso, utc_now


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_SOURCE = "open-meteo"

WEATHER_FIELDS = {
    "temperature_2m": ("weather_temperature_c", "C"),
    "relative_humidity_2m": ("weather_relative_humidity_percent", "percent"),
    "precipitation": ("weather_precipitation_mm", "mm"),
    "wind_speed_10m": ("weather_wind_speed_kmh", "km/h"),
    "surface_pressure": ("weather_surface_pressure_hpa", "hPa"),
    "weather_code": ("weather_code", "code"),
}


class WeatherError(RuntimeError):
    pass


def fetch_weather_measurements(config: AppConfig) -> list[Measurement]:
    params = {
        "latitude": config.location.latitude,
        "longitude": config.location.longitude,
        "current": ",".join(WEATHER_FIELDS.keys()),
        "timezone": "UTC",
    }
    timeout = httpx.Timeout(5.0, connect=2.0)
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = httpx.get(OPEN_METEO_URL, params=params, timeout=timeout)
            response.raise_for_status()
            return parse_weather_response(response.json(), utc_now(), config.location.name)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))

    raise WeatherError(f"Open-Meteo request failed: {last_error}")


def parse_weather_response(
    payload: dict[str, Any],
    collected_at_utc: datetime,
    location_name: str,
) -> list[Measurement]:
    current = payload.get("current")
    if not isinstance(current, dict):
        raise WeatherError("Open-Meteo response did not include a current weather object")

    api_time = current.get("time")
    weather_time = parse_open_meteo_time(api_time) or collected_at_utc
    metadata = {
        "api_time": api_time,
        "collected_at_utc": to_utc_iso(collected_at_utc),
        "location": location_name,
        "weather_time_utc": to_utc_iso(weather_time),
    }

    measurements: list[Measurement] = []
    for api_field, (metric, unit) in WEATHER_FIELDS.items():
        value = current.get(api_field)
        if value is None:
            continue
        measurements.append(
            Measurement(
                timestamp_utc=weather_time,
                metric=metric,
                value=float(value),
                unit=unit,
                source=WEATHER_SOURCE,
                metadata=metadata,
            )
        )
    return measurements

