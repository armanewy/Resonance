from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from resonance.weather import parse_weather_response


def test_weather_response_parsing_from_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "open_meteo_current.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    collected_at = datetime(2026, 6, 19, 18, 1, tzinfo=timezone.utc)

    measurements = parse_weather_response(payload, collected_at, "Framingham, Massachusetts")

    values = {measurement.metric: measurement.value for measurement in measurements}
    assert values["weather_temperature_c"] == 23.4
    assert values["weather_relative_humidity_percent"] == 64
    assert values["weather_precipitation_mm"] == 0.2
    assert values["weather_wind_speed_kmh"] == 12.5
    assert values["weather_surface_pressure_hpa"] == 1011.7
    assert values["weather_code"] == 3
    assert {measurement.source for measurement in measurements} == {"open-meteo"}
    assert measurements[0].timestamp_utc == datetime(2026, 6, 19, 18, 0, tzinfo=timezone.utc)

