from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from behavior_lab.core import parse_time, to_jsonable


BROAD_ASSET_ROLES = {
    "us_equity",
    "international_equity",
    "treasury_bond",
    "investment_grade_credit",
    "gold",
    "broad_commodities",
    "cash_equivalent",
}

FORBIDDEN_EXPOSURE_TERMS = {
    "broker",
    "hft",
    "individual_stock",
    "intraday",
    "inverse",
    "leveraged",
    "margin",
    "option",
    "options",
    "order_api",
    "short",
    "single_stock",
}


class MarketDataError(ValueError):
    pass


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    role: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.asset_id, "asset_id")
        _require_nonempty(self.description, "description")
        if self.role not in BROAD_ASSET_ROLES:
            raise MarketDataError(f"unsupported broad-ETF role: {self.role}")
        searchable = " ".join(
            [
                self.asset_id,
                self.role,
                self.description,
                " ".join(f"{key} {value}" for key, value in self.metadata.items()),
            ]
        ).lower()
        forbidden = sorted(term for term in FORBIDDEN_EXPOSURE_TERMS if term in searchable)
        if forbidden:
            raise MarketDataError(f"forbidden exposure terms are not allowed: {forbidden}")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class Universe:
    universe_id: str
    assets: tuple[AssetSpec, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.universe_id, "universe_id")
        if not self.assets:
            raise MarketDataError("universe must include at least one broad asset")
        ids = [asset.asset_id for asset in self.assets]
        if len(set(ids)) != len(ids):
            raise MarketDataError("asset IDs must be unique")
        roles = {asset.role for asset in self.assets}
        missing = BROAD_ASSET_ROLES - roles
        if missing:
            raise MarketDataError(f"universe is missing broad roles: {sorted(missing)}")

    @property
    def asset_ids(self) -> list[str]:
        return [asset.asset_id for asset in self.assets]

    def asset_for_role(self, role: str) -> AssetSpec:
        for asset in self.assets:
            if asset.role == role:
                return asset
        raise MarketDataError(f"universe has no asset for role {role!r}")

    def role_by_asset_id(self) -> dict[str, str]:
        return {asset.asset_id: asset.role for asset in self.assets}

    def to_dict(self) -> dict[str, Any]:
        return {"universe_id": self.universe_id, "assets": [asset.to_dict() for asset in self.assets]}


@dataclass(frozen=True)
class MarketCalendar:
    calendar_id: str
    sessions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.calendar_id, "calendar_id")
        if not self.sessions:
            raise MarketDataError("market calendar must include sessions")
        parsed = [_parse_date(session) for session in self.sessions]
        if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
            raise MarketDataError("market calendar sessions must be unique and sorted")

    def contains(self, market_date: str) -> bool:
        return market_date in set(self.sessions)

    def previous_sessions(self, through_session: str, count: int) -> list[str]:
        if count < 0:
            raise MarketDataError("count may not be negative")
        index = self.sessions.index(through_session)
        start = max(0, index - count + 1)
        return list(self.sessions[start : index + 1])

    def forward_sessions(self, after_session: str, count: int) -> list[str]:
        if count < 0:
            raise MarketDataError("count may not be negative")
        index = self.sessions.index(after_session)
        return list(self.sessions[index + 1 : index + 1 + count])

    def weekly_decision_sessions(
        self,
        *,
        min_history_sessions: int,
        horizon_sessions: int,
        start: str | None = None,
        end: str | None = None,
    ) -> list[str]:
        start_date = _parse_date(start) if start else None
        end_date = _parse_date(end) if end else None
        output = []
        for index, session in enumerate(self.sessions):
            session_date = _parse_date(session)
            if start_date and session_date < start_date:
                continue
            if end_date and session_date > end_date:
                continue
            if index < min_history_sessions:
                continue
            if index + horizon_sessions >= len(self.sessions):
                continue
            if (index - min_history_sessions) % 5 == 0:
                output.append(session)
        return output

    def to_dict(self) -> dict[str, Any]:
        return {"calendar_id": self.calendar_id, "sessions": list(self.sessions)}


@dataclass(frozen=True)
class DataAuthorization:
    provider_id: str
    authorized: bool
    permission_scope: str
    as_of: str
    restrictions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.provider_id, "provider_id")
        _require_nonempty(self.permission_scope, "permission_scope")
        parse_time(self.as_of)
        if not self.authorized:
            raise MarketDataError("market data provider is not authorized")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "authorized": self.authorized,
            "permission_scope": self.permission_scope,
            "as_of": self.as_of,
            "restrictions": list(self.restrictions),
        }


@dataclass(frozen=True)
class AdjustedPrice:
    asset_id: str
    market_date: str
    close: float
    adjusted_close: float
    event_time: str
    availability_time: str
    calendar_id: str
    source: str
    revision_id: str = "original"
    corrected_from: str | None = None
    adjustment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.asset_id, "asset_id")
        _parse_date(self.market_date)
        if float(self.close) <= 0.0 or float(self.adjusted_close) <= 0.0:
            raise MarketDataError("close and adjusted_close must be positive")
        event = parse_time(self.event_time)
        available = parse_time(self.availability_time)
        if available < event:
            raise MarketDataError("availability_time may not precede event_time")
        _require_nonempty(self.calendar_id, "calendar_id")
        _require_nonempty(self.source, "source")
        _require_nonempty(self.revision_id, "revision_id")
        if not isinstance(self.adjustment, dict) or not self.adjustment.get("adjustment_policy"):
            raise MarketDataError("adjusted prices must preserve an adjustment_policy")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class AuthorizedMarketDataProvider(Protocol):
    def data_authorization(self) -> DataAuthorization:
        ...

    def market_calendar(self) -> MarketCalendar:
        ...

    def history(self, asset_ids: list[str], decision_cutoff: str) -> list[AdjustedPrice]:
        ...

    def latest_cutoff(self) -> str:
        ...


class InMemoryMarketDataProvider:
    """Authorized provider-neutral fixture for tests and offline paper labs."""

    def __init__(
        self,
        *,
        prices: list[AdjustedPrice],
        calendar: MarketCalendar,
        authorization: DataAuthorization,
    ) -> None:
        self._prices = sorted(prices, key=lambda item: (item.market_date, item.asset_id, item.availability_time, item.revision_id))
        self._calendar = calendar
        self._authorization = authorization
        self._validate()

    def data_authorization(self) -> DataAuthorization:
        return self._authorization

    def market_calendar(self) -> MarketCalendar:
        return self._calendar

    def history(self, asset_ids: list[str], decision_cutoff: str) -> list[AdjustedPrice]:
        cutoff = parse_time(decision_cutoff)
        requested = set(asset_ids)
        latest_by_asset_date: dict[tuple[str, str], AdjustedPrice] = {}
        for price in self._prices:
            if price.asset_id not in requested:
                continue
            if parse_time(price.availability_time) > cutoff:
                continue
            key = (price.asset_id, price.market_date)
            previous = latest_by_asset_date.get(key)
            if previous is None or (
                parse_time(previous.availability_time),
                previous.revision_id,
            ) < (
                parse_time(price.availability_time),
                price.revision_id,
            ):
                latest_by_asset_date[key] = price
        return sorted(latest_by_asset_date.values(), key=lambda item: (item.market_date, item.asset_id))

    def latest_prices(self, asset_ids: list[str], decision_cutoff: str) -> dict[str, AdjustedPrice]:
        history = self.history(asset_ids, decision_cutoff)
        latest: dict[str, AdjustedPrice] = {}
        for price in history:
            latest[price.asset_id] = price
        missing = sorted(set(asset_ids) - set(latest))
        if missing:
            raise MarketDataError(f"missing price history at cutoff for assets: {missing}")
        return latest

    def latest_cutoff(self) -> str:
        if not self._prices:
            raise MarketDataError("provider has no prices")
        return max(self._prices, key=lambda item: parse_time(item.availability_time)).availability_time

    def _validate(self) -> None:
        self._authorization.to_dict()
        for price in self._prices:
            if price.calendar_id != self._calendar.calendar_id:
                raise MarketDataError("price calendar_id does not match provider calendar")
            if not self._calendar.contains(price.market_date):
                raise MarketDataError(f"price uses non-session market_date: {price.market_date}")


def default_universe() -> Universe:
    return Universe(
        universe_id="broad_etf_proxy_v1",
        assets=(
            AssetSpec("US_EQUITY", "us_equity", "Broad US equity market ETF proxy"),
            AssetSpec("INTL_EQUITY", "international_equity", "Broad developed and emerging ex-US equity ETF proxy"),
            AssetSpec("TREASURY_BOND", "treasury_bond", "Intermediate Treasury bond ETF proxy"),
            AssetSpec("IG_CREDIT", "investment_grade_credit", "Investment-grade corporate credit ETF proxy"),
            AssetSpec("GOLD", "gold", "Gold ETF or trust proxy"),
            AssetSpec("BROAD_COMMODITIES", "broad_commodities", "Broad commodities ETF proxy"),
            AssetSpec("CASH_EQUIVALENT", "cash_equivalent", "Cash or Treasury-bill ETF equivalent proxy"),
        ),
    )


def _parse_date(value: str | None) -> date:
    if not isinstance(value, str) or not value.strip():
        raise MarketDataError("dates must be non-empty ISO dates")
    return date.fromisoformat(value)


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MarketDataError(f"{field_name} must be a non-empty string")
