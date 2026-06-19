from __future__ import annotations

from datetime import datetime, timezone


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_utc_iso(value: datetime) -> str:
    return ensure_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return ensure_utc(parsed)


def parse_open_meteo_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return ensure_utc(parsed)

