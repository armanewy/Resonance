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
class NotificationConfig:
    enabled: bool
    dry_run_stdout: bool
    ntfy_endpoint: str
    history_path: str
    dashboard_url: str
    discovery_cooldown_hours: int
    finding_cooldown_hours: int
    major_strengthening_threshold: float
    request_timeout_seconds: int


@dataclass(frozen=True)
class EiaGridPublicSourceConfig:
    enabled: bool
    poll_interval_seconds: int
    initial_backfill_hours: int
    normal_lookback_hours: int
    maximum_gap_repair_hours: int


@dataclass(frozen=True)
class PublicSourcesConfig:
    eia_grid: EiaGridPublicSourceConfig


@dataclass(frozen=True)
class AppConfig:
    location: LocationConfig
    collection: CollectionConfig
    notifications: NotificationConfig
    public_sources: PublicSourcesConfig


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
    notifications = _optional_table(raw, "notifications")
    public_sources = _optional_table(raw, "public_sources")
    eia_grid = _optional_table(public_sources, "eia_grid")

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

    notifications_enabled = _optional_bool(notifications, "notifications.enabled", False)
    dry_run_stdout = _optional_bool(notifications, "notifications.dry_run_stdout", True)
    ntfy_endpoint = _optional_str(notifications, "notifications.ntfy_endpoint", "")
    history_path = _optional_str(
        notifications,
        "notifications.history_path",
        "data/notification_history.json",
    )
    dashboard_url = _optional_str(
        notifications,
        "notifications.dashboard_url",
        "http://127.0.0.1:8501",
    )
    discovery_cooldown_hours = _optional_int(
        notifications,
        "notifications.discovery_cooldown_hours",
        24,
    )
    finding_cooldown_hours = _optional_int(
        notifications,
        "notifications.finding_cooldown_hours",
        24,
    )
    major_strengthening_threshold = _optional_float(
        notifications,
        "notifications.major_strengthening_threshold",
        0.20,
    )
    request_timeout_seconds = _optional_int(
        notifications,
        "notifications.request_timeout_seconds",
        5,
    )

    if ntfy_endpoint and not (
        ntfy_endpoint.startswith("http://") or ntfy_endpoint.startswith("https://")
    ):
        raise ConfigError("notifications.ntfy_endpoint must start with http:// or https://")
    if not history_path.strip():
        raise ConfigError("notifications.history_path must not be blank")
    if not dashboard_url.strip():
        raise ConfigError("notifications.dashboard_url must not be blank")
    if discovery_cooldown_hours <= 0:
        raise ConfigError("notifications.discovery_cooldown_hours must be positive")
    if finding_cooldown_hours <= 0:
        raise ConfigError("notifications.finding_cooldown_hours must be positive")
    if not 0 <= major_strengthening_threshold <= 2:
        raise ConfigError("notifications.major_strengthening_threshold must be between 0 and 2")
    if request_timeout_seconds <= 0:
        raise ConfigError("notifications.request_timeout_seconds must be positive")

    eia_enabled = _optional_bool(eia_grid, "public_sources.eia_grid.enabled", False)
    eia_poll_interval = _optional_int(eia_grid, "public_sources.eia_grid.poll_interval_seconds", 3600)
    eia_initial_backfill = _optional_int(eia_grid, "public_sources.eia_grid.initial_backfill_hours", 720)
    eia_normal_lookback = _optional_int(eia_grid, "public_sources.eia_grid.normal_lookback_hours", 72)
    eia_max_gap_repair = _optional_int(eia_grid, "public_sources.eia_grid.maximum_gap_repair_hours", 2160)
    if eia_poll_interval <= 0:
        raise ConfigError("public_sources.eia_grid.poll_interval_seconds must be positive")
    if eia_initial_backfill <= 0:
        raise ConfigError("public_sources.eia_grid.initial_backfill_hours must be positive")
    if eia_normal_lookback <= 0:
        raise ConfigError("public_sources.eia_grid.normal_lookback_hours must be positive")
    if eia_max_gap_repair <= 0:
        raise ConfigError("public_sources.eia_grid.maximum_gap_repair_hours must be positive")

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
        notifications=NotificationConfig(
            enabled=notifications_enabled,
            dry_run_stdout=dry_run_stdout,
            ntfy_endpoint=ntfy_endpoint,
            history_path=history_path,
            dashboard_url=dashboard_url,
            discovery_cooldown_hours=discovery_cooldown_hours,
            finding_cooldown_hours=finding_cooldown_hours,
            major_strengthening_threshold=major_strengthening_threshold,
            request_timeout_seconds=request_timeout_seconds,
        ),
        public_sources=PublicSourcesConfig(
            eia_grid=EiaGridPublicSourceConfig(
                enabled=eia_enabled,
                poll_interval_seconds=eia_poll_interval,
                initial_backfill_hours=eia_initial_backfill,
                normal_lookback_hours=eia_normal_lookback,
                maximum_gap_repair_hours=eia_max_gap_repair,
            )
        ),
    )


def _require_table(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required [{key}] table")
    return value


def _optional_table(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
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


def _optional_bool(table: dict, dotted_key: str, default: bool) -> bool:
    key = dotted_key.split(".")[-1]
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{dotted_key} must be a boolean")
    return value


def _required_float(table: dict, dotted_key: str) -> float:
    key = dotted_key.split(".")[-1]
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted_key} must be a number")
    return float(value)


def _optional_float(table: dict, dotted_key: str, default: float) -> float:
    key = dotted_key.split(".")[-1]
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted_key} must be a number")
    return float(value)


def _required_int(table: dict, dotted_key: str) -> int:
    key = dotted_key.split(".")[-1]
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{dotted_key} must be an integer")
    return value


def _optional_int(table: dict, dotted_key: str, default: int) -> int:
    key = dotted_key.split(".")[-1]
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{dotted_key} must be an integer")
    return value
