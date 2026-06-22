from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from behavior_lab.core import parse_time, to_jsonable
from behavior_lab.finance_data.contracts import (
    FinanceDataError,
    FinanceObservation,
    MarketCalendar,
    MarketSessionStatus,
    effective_available_at,
    observation_hash,
    observation_kind,
)


REVISION_POLICIES = {"latest", "all_available"}


@dataclass(frozen=True)
class AsOfQuery:
    kind: str
    instrument_id: str
    as_of: str
    event_time: str | None = None
    source_id: str | None = None
    revision_policy: str = "latest"
    require_ingested: bool = True
    calendar_id: str | None = None

    def __post_init__(self) -> None:
        if self.revision_policy not in REVISION_POLICIES:
            raise FinanceDataError(f"revision_policy must be one of {sorted(REVISION_POLICIES)}")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise FinanceDataError("kind must be non-empty")
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise FinanceDataError("instrument_id must be non-empty")
        parse_time(self.as_of)
        if self.event_time is not None:
            parse_time(self.event_time)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class AsOfResult:
    query: AsOfQuery
    observations: list[FinanceObservation] = field(default_factory=list)
    missing_reason: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "found": self.found,
            "missing_reason": self.missing_reason,
            "observation_hashes": [observation_hash(observation) for observation in self.observations],
            "observations": [observation.to_dict() for observation in self.observations],
        }


class FinanceDataStore:
    """Provider-neutral point-in-time store for already-approved observations.

    The store does not fetch, scrape, subscribe, or activate sources. It only
    indexes typed observations that callers have already supplied.
    """

    def __init__(self, observations: list[FinanceObservation] | None = None) -> None:
        self._observations: list[FinanceObservation] = []
        if observations:
            self.add_many(observations)

    def add(self, observation: FinanceObservation) -> None:
        observation_kind(observation)
        observation_hash(observation)
        self._observations.append(observation)

    def add_many(self, observations: list[FinanceObservation]) -> None:
        for observation in observations:
            self.add(observation)

    def list(self) -> list[FinanceObservation]:
        return list(self._observations)

    def query(
        self,
        *,
        kind: str,
        instrument_id: str,
        as_of: str,
        event_time: str | None = None,
        source_id: str | None = None,
        revision_policy: str = "latest",
        require_ingested: bool = True,
    ) -> list[FinanceObservation]:
        query = AsOfQuery(
            kind=kind,
            instrument_id=instrument_id,
            source_id=source_id,
            event_time=event_time,
            as_of=as_of,
            revision_policy=revision_policy,
            require_ingested=require_ingested,
        )
        return self.sample(query).observations

    def sample(self, query: AsOfQuery) -> AsOfResult:
        observations = self._eligible_observations(query)
        if observations:
            return AsOfResult(query=query, observations=observations)
        if query.event_time is not None:
            closed_reason = self._closed_market_reason(query)
            if closed_reason is not None:
                return AsOfResult(query=query, missing_reason=closed_reason)
            return AsOfResult(query=query, missing_reason="no_observation_at_event_time_no_forward_fill")
        return AsOfResult(query=query, missing_reason="no_observation_available_as_of")

    def _eligible_observations(self, query: AsOfQuery) -> list[FinanceObservation]:
        as_of_time = parse_time(query.as_of)
        candidates: list[FinanceObservation] = []
        for observation in self._observations:
            metadata = observation.metadata
            if observation_kind(observation) != query.kind:
                continue
            if metadata.instrument_id != query.instrument_id:
                continue
            if query.source_id is not None and metadata.source_id != query.source_id:
                continue
            if query.event_time is not None and parse_time(metadata.event_time) != parse_time(query.event_time):
                continue
            if effective_available_at(observation) > as_of_time:
                continue
            if query.require_ingested and parse_time(metadata.ingested_at) > as_of_time:
                continue
            candidates.append(observation)
        candidates.sort(key=_availability_sort_key)
        if query.revision_policy == "all_available":
            return candidates
        return _latest_by_revision_group(candidates)

    def _closed_market_reason(self, query: AsOfQuery) -> str | None:
        if query.event_time is None:
            return None
        target_date = parse_time(query.event_time).date().isoformat()
        calendar_id = query.calendar_id or query.instrument_id
        calendar_query = AsOfQuery(
            kind=MarketCalendar.kind,
            instrument_id=calendar_id,
            source_id=None,
            event_time=None,
            as_of=query.as_of,
            revision_policy="latest",
            require_ingested=query.require_ingested,
        )
        calendars = [
            observation
            for observation in self._eligible_observations(calendar_query)
            if isinstance(observation, MarketCalendar) and observation.session_date == target_date
        ]
        if not calendars:
            return None
        calendar = sorted(calendars, key=_availability_sort_key)[-1]
        if calendar.status != MarketSessionStatus.OPEN.value:
            return f"market_{calendar.status}_no_forward_fill"
        return None


def _availability_sort_key(observation: FinanceObservation) -> tuple[str, str, str]:
    metadata = observation.metadata
    return (
        effective_available_at(observation).isoformat(),
        metadata.ingested_at,
        metadata.revision_id,
    )


def _latest_by_revision_group(observations: list[FinanceObservation]) -> list[FinanceObservation]:
    latest: dict[tuple[str, str, str, str, str], FinanceObservation] = {}
    for observation in observations:
        key = _revision_group_key(observation)
        current = latest.get(key)
        if current is None or _availability_sort_key(current) <= _availability_sort_key(observation):
            latest[key] = observation
    return [latest[key] for key in sorted(latest)]


def _revision_group_key(observation: FinanceObservation) -> tuple[str, str, str, str, str]:
    metadata = observation.metadata
    group = getattr(observation, "revision_group_id", None) or metadata.source_observation_id or metadata.event_time
    event_bucket = metadata.event_time
    if getattr(observation, "revision_group_id", None) is not None:
        event_bucket = str(group)
    if isinstance(observation, MarketCalendar):
        group = f"{observation.calendar_id}:{observation.session_date}"
        event_bucket = observation.session_date
    fixing_date = getattr(observation, "fixing_date", None)
    if fixing_date is not None:
        group = f"{group}:{fixing_date}"
        event_bucket = str(fixing_date)
    session_date = getattr(observation, "session_date", None)
    if session_date is not None:
        group = f"{group}:{session_date}"
        event_bucket = str(session_date)
    return (
        observation_kind(observation),
        metadata.instrument_id,
        metadata.source_id,
        event_bucket,
        str(group),
    )


def session_event_time(session_date: str, timezone_offset: str = "+00:00") -> str:
    date.fromisoformat(session_date)
    return f"{session_date}T00:00:00{timezone_offset}"
