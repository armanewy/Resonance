from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_CONFIG_PATH = Path("config.toml")


class ConfigError(ValueError):
    """Raised when config.toml is missing required values or has invalid values."""


@dataclass(frozen=True)
class LocationConfig:
    name: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True)
class CollectionConfig:
    personal_interval_seconds: int
    weather_interval_seconds: int
    tcp_test_host: str
    tcp_test_port: int
    dns_test_hostname: str
    router_host: str


@dataclass(frozen=True)
class AppConfig:
    location: LocationConfig
    collection: CollectionConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML: {exc}") from exc

    location = _require_table(raw, "location")
    collection = _require_table(raw, "collection")

    name = _required_str(location, "location.name")
    latitude = _required_float(location, "location.latitude")
    longitude = _required_float(location, "location.longitude")
    timezone_name = _required_str(location, "location.timezone")
    if not name.strip():
        raise ConfigError("location.name must not be blank")
    if not -90 <= latitude <= 90:
        raise ConfigError("location.latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ConfigError("location.longitude must be between -180 and 180")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(f"location.timezone is not recognized: {timezone_name}") from exc

    personal_interval = _required_int(collection, "collection.personal_interval_seconds")
    weather_interval = _required_int(collection, "collection.weather_interval_seconds")
    tcp_host = _required_str(collection, "collection.tcp_test_host")
    tcp_port = _required_int(collection, "collection.tcp_test_port")
    dns_hostname = _required_str(collection, "collection.dns_test_hostname")
    router_host = _optional_str(collection, "collection.router_host", "")

    if personal_interval <= 0:
        raise ConfigError("collection.personal_interval_seconds must be positive")
    if weather_interval <= 0:
        raise ConfigError("collection.weather_interval_seconds must be positive")
    if not tcp_host.strip():
        raise ConfigError("collection.tcp_test_host must not be blank")
    if not 1 <= tcp_port <= 65535:
        raise ConfigError("collection.tcp_test_port must be between 1 and 65535")
    if not dns_hostname.strip():
        raise ConfigError("collection.dns_test_hostname must not be blank")

    return AppConfig(
        location=LocationConfig(
            name=name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone_name,
        ),
        collection=CollectionConfig(
            personal_interval_seconds=personal_interval,
            weather_interval_seconds=weather_interval,
            tcp_test_host=tcp_host,
            tcp_test_port=tcp_port,
            dns_test_hostname=dns_hostname,
            router_host=router_host,
        ),
    )


def _require_table(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required [{key}] table")
    return value


def _required_str(table: dict, dotted_key: str) -> str:
    key = dotted_key.split(".")[-1]
    value = table.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"{dotted_key} must be a string")
    return value


def _optional_str(table: dict, dotted_key: str, default: str) -> str:
    key = dotted_key.split(".")[-1]
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{dotted_key} must be a string")
    return value


def _required_float(table: dict, dotted_key: str) -> float:
    key = dotted_key.split(".")[-1]
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted_key} must be a number")
    return float(value)


def _required_int(table: dict, dotted_key: str) -> int:
    key = dotted_key.split(".")[-1]
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{dotted_key} must be an integer")
    return value

