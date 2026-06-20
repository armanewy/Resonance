from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from resonance.analysis.scanner import _select_scanner_candidate_pairs
from resonance.analysis.service import ValidationOptions, analyze_metric_pair, list_analyzable_metrics
from resonance.config import LocationConfig, RipeAtlasPublicSourceConfig
from resonance.dashboard import _ripe_chart_rows
from resonance.public_health import ripe_source_health_rows
from resonance.public_sources.ripe_atlas import (
    SOURCE_ID,
    RipeAtlasClient,
    RipeAtlasError,
    RipeHttpFetchResult,
    aggregate_ping_results,
    backfill_regional_ipv4_health,
    cohort_rows,
    ensure_ripe_registry,
    parse_ping_results,
    poll_regional_ipv4_health,
    select_probe_cohort,
    status_payload,
    _planned_result_requests,
)
from resonance.storage import Measurement, ensure_database, insert_measurements, insert_public_observations


START = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc)
LOCATION = LocationConfig("Framingham, Massachusetts", 42.2793, -71.4162, "America/New_York")


def test_probe_cohort_selection_expands_radius_and_preserves_diversity(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    calls: list[int] = []

    def fetcher(_lat: float, _lon: float, radius: int):
        calls.append(radius)
        rows = [_probe(index, asn=64500 + index, distance=10 + index) for index in range(1, 4)]
        if radius >= 125:
            rows.extend(
                [
                    _probe(4, asn=64504, distance=20, is_anchor=True),
                    _probe(5, asn=64504, distance=21, is_anchor=True),
                    _probe(6, asn=64506, distance=22, status=2),
                    _probe(7, asn=64507, distance=23, is_public=False),
                    _probe(8, asn=64508, distance=24, address_v4=""),
                    _probe(9, asn=None, distance=25),
                    _probe(10, asn=None, distance=26),
                ]
            )
        return (_http("probes", {"results": rows, "next": None}, f"https://atlas.ripe.net/api/v2/probes/?radius={radius}"),)

    try:
        cohort, members, pages = select_probe_cohort(
            conn,
            config=_config(initial_radius_km=100, maximum_radius_km=150, desired_probe_count=6, minimum_probe_count=6, maximum_probes_per_asn=1, maximum_anchor_count=1),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=fetcher,
        )
    finally:
        conn.close()

    assert calls == [100, 125]
    assert cohort.selected_radius_km == 125
    assert len(members) == 6
    assert sum(1 for member in members if member.is_anchor) == 1
    assert sum(1 for member in members if member.asn_v4 is None) == 1
    assert len(pages) == 2


def test_probe_cohort_reuses_recent_valid_cohort_without_network(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    try:
        first, members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=_probe_fetcher(count=6),
        )

        def should_not_fetch(*_args):
            raise AssertionError("cohort refresh should not fetch probes")

        second, second_members, pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6, cohort_refresh_hours=24),
            location=LOCATION,
            effective_start_utc=START + timedelta(hours=1),
            now=START + timedelta(hours=1),
            probe_fetcher=should_not_fetch,
        )
    finally:
        conn.close()

    assert second.cohort_id == first.cohort_id
    assert [member.probe_id for member in second_members] == [member.probe_id for member in members]
    assert pages == ()


def test_probe_discovery_accepts_geojson_geometry_coordinates(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")

    def fetcher(_lat: float, _lon: float, _radius: int):
        row = _probe(1, asn=64501, distance=1.0)
        row["latitude"] = None
        row["longitude"] = None
        row["geometry"] = {"type": "Point", "coordinates": [LOCATION.longitude, LOCATION.latitude]}
        return (_http("probes", {"results": [row], "next": None}, "https://atlas.ripe.net/api/v2/probes/?radius=fixture"),)

    try:
        _cohort, members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=1, minimum_probe_count=1),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=fetcher,
        )
    finally:
        conn.close()

    assert len(members) == 1
    assert members[0].latitude == LOCATION.latitude
    assert members[0].longitude == LOCATION.longitude


def test_probe_cohort_refresh_replaces_disconnected_probe_and_preserves_effective_period(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    refresh_time = START + timedelta(hours=25)

    def refreshed(_lat: float, _lon: float, _radius: int):
        rows = [_probe(index, asn=64500 + index, distance=float(index)) for index in range(1, 6)]
        rows.append(_probe(6, asn=64506, distance=6.0, status=2))
        rows.append(_probe(7, asn=64507, distance=7.0))
        return (_http("probes", {"results": rows, "next": None}, "https://atlas.ripe.net/api/v2/probes/?radius=fixture"),)

    try:
        old, _members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=_probe_fetcher(count=6),
        )
        new, new_members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            effective_start_utc=refresh_time,
            now=refresh_time,
            force_refresh=True,
            probe_fetcher=refreshed,
        )
        old_row = conn.execute("SELECT effective_end_utc FROM ripe_probe_cohorts WHERE cohort_id = ?", (old.cohort_id,)).fetchone()
        old_member = conn.execute(
            "SELECT effective_end_utc FROM ripe_probe_cohort_members WHERE cohort_id = ? AND probe_id = ?",
            (old.cohort_id, 6),
        ).fetchone()
    finally:
        conn.close()

    assert new.cohort_id != old.cohort_id
    assert [member.probe_id for member in new_members] == [1, 2, 3, 4, 5, 7]
    assert old_row["effective_end_utc"] == "2026-06-20T01:00:00Z"
    assert old_member["effective_end_utc"] == "2026-06-20T01:00:00Z"
    assert all(member.effective_start_utc == refresh_time for member in new_members)


def test_result_request_planning_chunks_batches_and_measurements() -> None:
    requests = _planned_result_requests(
        config=_config(result_chunk_hours=2, maximum_probe_batch_size=3, measurement_ids=(1001, 1004)),
        start_utc=START,
        end_utc=START + timedelta(hours=5),
        probe_ids=(1, 2, 3, 4),
    )

    assert len(requests) == 12
    assert requests[0] == (1001, START, START + timedelta(hours=2), (1, 2, 3))
    assert requests[1] == (1001, START, START + timedelta(hours=2), (4,))
    assert requests[-1] == (1004, START + timedelta(hours=4), START + timedelta(hours=5), (4,))


def test_http_client_honors_retry_after_and_bounds_retries(monkeypatch) -> None:
    sleeps: list[float] = []

    class Response:
        def __init__(self, status_code: int, payload, *, retry_after: str | None = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers = {"Retry-After": retry_after} if retry_after else {}
            self.content = json.dumps(payload).encode("utf-8")
            self.url = "https://atlas.ripe.net/api/v2/measurements/1001/results/?redacted"

        def json(self):
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status {self.status_code}")

    responses = [Response(429, {"detail": "rate limited"}, retry_after="2"), Response(200, [{"ok": True}])]

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            self.headers = kwargs.get("headers", {})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url, params=None):
            return responses.pop(0)

    monkeypatch.setattr("resonance.public_sources.ripe_atlas.httpx.Client", Client)
    monkeypatch.setattr("resonance.public_sources.ripe_atlas.time.sleep", sleeps.append)

    result = RipeAtlasClient(retries=1).fetch_results(1001, START, START + timedelta(hours=1), [1, 2])

    assert sleeps == [2.0]
    assert result.status_code == 200
    assert result.request_metadata["attempt"] == 2
    assert result.request_metadata["authenticated_read"] is False


def test_http_client_stops_after_bounded_retries(monkeypatch) -> None:
    calls = 0

    class Response:
        status_code = 500
        headers = {}
        content = b"{}"
        url = "https://atlas.ripe.net/api/v2/measurements/1001/results/"

        def json(self):
            return {}

        def raise_for_status(self) -> None:
            request = httpx.Request("GET", self.url)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("server failed", request=request, response=response)

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url, params=None):
            nonlocal calls
            calls += 1
            return Response()

    monkeypatch.setattr("resonance.public_sources.ripe_atlas.httpx.Client", Client)
    monkeypatch.setattr("resonance.public_sources.ripe_atlas.time.sleep", lambda _seconds: None)

    try:
        RipeAtlasClient(retries=2).fetch_results(1001, START, START + timedelta(hours=1), [1])
    except RipeAtlasError as exc:
        assert "RIPE Atlas request failed" in str(exc)
    else:
        raise AssertionError("expected bounded retry failure")

    assert calls == 3


def test_ping_parser_handles_top_level_packet_fallback_loss_and_malformed_rows() -> None:
    payload = [
        _result_row(1001, 1, START, avg=10.0, sent=3, rcvd=3),
        _result_row(1001, 2, START, result=[{"rtt": 20.0}, {"x": "*"}, {"rtt": 30.0}]),
        _result_row(1001, 3, START, result=[{"x": "*"}, {"error": "timeout"}]),
        _result_row(1001, 4, START, avg=-1.0),
        {**_result_row(1001, 5, START), "timestamp": "not-a-time"},
    ]

    parsed = parse_ping_results(payload)

    assert len(parsed) == 3
    assert parsed[0].avg_rtt_ms == 10.0
    assert parsed[1].avg_rtt_ms == 25.0
    assert parsed[1].packets_sent == 3
    assert parsed[1].packets_received == 2
    assert parsed[2].avg_rtt_ms is None
    assert parsed[2].packet_loss_fraction == 1.0


def test_aggregation_gates_quality_and_avoids_overweighting_repeated_probe_samples(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    try:
        ensure_ripe_registry(conn, config=_config(desired_probe_count=6, minimum_probe_count=6), location=LOCATION, enabled=True)
        cohort, members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
        )
        results = parse_ping_results(
            [
                _result_row(measurement_id, probe_id, START + timedelta(minutes=1), avg=10.0 + probe_id + measurement_id % 10)
                for measurement_id in (1001, 1004, 1009)
                for probe_id in range(1, 7)
            ]
            + [
                _result_row(1001, 1, START + timedelta(minutes=2), avg=500.0),
            ]
        )
        observations = aggregate_ping_results(
            results,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            cohort=cohort,
            members=members,
            ingested_at_utc=START + timedelta(hours=1),
        )
        insert_public_observations(conn, observations)
        rows = conn.execute(
            "SELECT series_id, value, metadata_json FROM public_observations WHERE valid_start_utc = ?",
            ("2026-06-19T00:00:00Z",),
        ).fetchall()
    finally:
        conn.close()

    by_series = {row["series_id"]: row for row in rows}
    assert "ripe_atlas_ipv4_ping:regional:median_rtt_ms" in by_series
    assert by_series["ripe_atlas_ipv4_ping:regional:responding_probe_count"]["value"] == 6
    assert by_series["ripe_atlas_ipv4_ping:regional:unique_responding_asn_count"]["value"] == 3
    assert json.loads(by_series["ripe_atlas_ipv4_ping:regional:median_rtt_ms"]["metadata_json"])["quality_score"] == 1.0


def test_low_quality_bins_emit_diagnostics_but_no_composite_rtt(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    try:
        ensure_ripe_registry(conn, config=_config(desired_probe_count=6, minimum_probe_count=6), location=LOCATION, enabled=True)
        cohort, members, _pages = select_probe_cohort(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            effective_start_utc=START,
            now=START,
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
        )
        results = parse_ping_results(
            [_result_row(1001, probe_id, START + timedelta(minutes=1), avg=10.0 + probe_id) for probe_id in range(1, 4)]
        )
        observations = aggregate_ping_results(
            results,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            cohort=cohort,
            members=members,
            ingested_at_utc=START + timedelta(hours=1),
        )
    finally:
        conn.close()

    series_ids = {observation.series_id for observation in observations}
    assert "ripe_atlas_ipv4_ping:regional:median_rtt_ms" not in series_ids
    assert "ripe_atlas_ipv4_ping:regional:responding_probe_count" in series_ids


def test_backfill_archives_fetch_events_is_idempotent_and_late_data_revises(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    result_payload = _full_result_payload(START, avg_offset=0.0)
    late_payload = _full_result_payload(START, avg_offset=5.0)

    def fetcher_factory(payload):
        def fetch(measurement_id: int, start: datetime, stop: datetime, probe_ids):
            rows = [row for row in payload if row["msm_id"] == measurement_id and row["prb_id"] in probe_ids]
            return _http(f"results:{measurement_id}", rows, f"https://atlas.ripe.net/api/v2/measurements/{measurement_id}/results/?probe_ids={','.join(map(str, probe_ids))}")

        return fetch

    try:
        first = backfill_regional_ipv4_health(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6, result_chunk_hours=1),
            location=LOCATION,
            start_utc=START,
            end_utc=START + timedelta(hours=1),
            raw_root=tmp_path / "raw",
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
            result_fetcher=fetcher_factory(result_payload),
            now=START + timedelta(hours=2),
        )
        second = backfill_regional_ipv4_health(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6, result_chunk_hours=1),
            location=LOCATION,
            start_utc=START,
            end_utc=START + timedelta(hours=1),
            raw_root=tmp_path / "raw",
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
            result_fetcher=fetcher_factory(result_payload),
            now=START + timedelta(hours=3),
        )
        late = backfill_regional_ipv4_health(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6, result_chunk_hours=1),
            location=LOCATION,
            start_utc=START,
            end_utc=START + timedelta(hours=1),
            raw_root=tmp_path / "raw",
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
            result_fetcher=fetcher_factory(late_payload),
            now=START + timedelta(hours=4),
        )
        archive_count = conn.execute("SELECT COUNT(*) AS count FROM public_raw_archives WHERE source_id = ?", (SOURCE_ID,)).fetchone()["count"]
        fetch_count = conn.execute("SELECT COUNT(*) AS count FROM public_fetch_events WHERE source_id = ?", (SOURCE_ID,)).fetchone()["count"]
        revisions = conn.execute(
            "SELECT COUNT(*) AS count FROM public_observations WHERE series_id = ?",
            ("ripe_atlas_ipv4_ping:regional:median_rtt_ms",),
        ).fetchone()["count"]
    finally:
        conn.close()

    assert first.inserted_observations > 0
    assert second.inserted_observations == 0
    assert late.inserted_observations > 0
    assert archive_count >= 4
    assert fetch_count >= 7
    assert revisions == 2


def test_poll_refresh_collects_lookback_with_old_cohort_before_selecting_new_one(tmp_path) -> None:
    conn = ensure_database(tmp_path / "resonance.db")
    config = _config(
        desired_probe_count=6,
        minimum_probe_count=6,
        initial_backfill_hours=1,
        normal_lookback_hours=1,
        result_chunk_hours=1,
        cohort_refresh_hours=1,
    )
    seen_fetch_probe_sets: list[tuple[int, ...]] = []

    def refreshed(_lat: float, _lon: float, _radius: int):
        rows = [_probe(index, asn=64500 + index, distance=float(index)) for index in range(1, 6)]
        rows.append(_probe(6, asn=64506, distance=6.0, status=2))
        rows.append(_probe(7, asn=64507, distance=7.0))
        return (_http("probes", {"results": rows, "next": None}, "https://atlas.ripe.net/api/v2/probes/?radius=fixture"),)

    def fetch_results(measurement_id: int, start: datetime, stop: datetime, probe_ids):
        seen_fetch_probe_sets.append(tuple(probe_ids))
        rows = [_result_row(measurement_id, probe_id, start + timedelta(minutes=1), avg=15.0 + probe_id) for probe_id in probe_ids]
        return _http(f"results:{measurement_id}", rows, f"https://atlas.ripe.net/api/v2/measurements/{measurement_id}/results/")

    try:
        backfill_regional_ipv4_health(
            conn,
            config=config,
            location=LOCATION,
            start_utc=START,
            end_utc=START + timedelta(hours=1),
            raw_root=tmp_path / "raw",
            probe_fetcher=_probe_fetcher(count=6),
            result_fetcher=fetch_results,
            now=START + timedelta(hours=1, minutes=30),
        )
        seen_fetch_probe_sets.clear()
        result = poll_regional_ipv4_health(
            conn,
            config=config,
            location=LOCATION,
            raw_root=tmp_path / "raw",
            probe_fetcher=refreshed,
            result_fetcher=fetch_results,
            now=START + timedelta(hours=3),
        )
        active = conn.execute("SELECT cohort_id, effective_start_utc FROM ripe_probe_cohorts WHERE effective_end_utc IS NULL").fetchone()
        active_members = conn.execute(
            "SELECT probe_id FROM ripe_probe_cohort_members WHERE cohort_id = ? ORDER BY probe_id",
            (active["cohort_id"],),
        ).fetchall()
    finally:
        conn.close()

    assert result.cohort_id == 1
    assert seen_fetch_probe_sets
    assert all(7 not in probe_ids for probe_ids in seen_fetch_probe_sets)
    assert active["effective_start_utc"] == "2026-06-19T03:00:00Z"
    assert [row["probe_id"] for row in active_members] == [1, 2, 3, 4, 5, 7]


def test_status_probe_rows_pair_explorer_and_scanner_lineage(tmp_path) -> None:
    db_path = tmp_path / "resonance.db"
    conn = ensure_database(db_path)
    try:
        result = backfill_regional_ipv4_health(
            conn,
            config=_config(desired_probe_count=6, minimum_probe_count=6),
            location=LOCATION,
            start_utc=START,
            end_utc=START + timedelta(hours=4),
            raw_root=tmp_path / "raw",
            probe_fetcher=_probe_fetcher(count=6, asn_cycle=3),
            result_fetcher=_hourly_result_fetcher(),
            now=START + timedelta(hours=5),
        )
        insert_measurements(
            conn,
            [
                Measurement(START + timedelta(minutes=15 * index), "tcp_latency_ms", 10.0 + index, "ms", "personal")
                for index in range(16)
            ],
        )
        status = status_payload(conn, config=_config(), location=LOCATION, now=START + timedelta(hours=6))
        probes = cohort_rows(conn)
        health_rows = ripe_source_health_rows(conn, config=_config(), location=LOCATION, now_utc=START + timedelta(hours=6))
        chart_rows = _ripe_chart_rows(conn, START, START + timedelta(hours=4), ZoneInfo(LOCATION.timezone))
    finally:
        conn.close()

    assert result.newest_finalized_bin_utc is not None
    assert status["active_probe_count"] == 6
    assert len(probes) == 6
    assert health_rows[0]["source"] == "RIPE Atlas regional IPv4 Internet health"
    assert {row["series_id"] for row in chart_rows} >= {
        "ripe_atlas_ipv4_ping:regional:median_rtt_ms",
        "ripe_atlas_ipv4_ping:regional:p90_rtt_ms",
        "ripe_atlas_ipv4_ping:regional:packet_loss_fraction",
    }

    metrics = list_analyzable_metrics(db_path, START, START + timedelta(hours=4))
    labels = {metric.display_name for metric in metrics}
    assert "Regional IPv4 median RTT [42.2793,-71.4162:regional]" in labels

    analysis = analyze_metric_pair(
        db_path,
        "tcp_latency_ms",
        "ripe_atlas_ipv4_ping:regional:median_rtt_ms",
        START,
        START + timedelta(hours=4),
        "raw",
        max_lag_steps=1,
        validation_options=ValidationOptions(min_aligned_points=4, min_overlap=3, permutations=9),
    )
    assert analysis.y_metric_summary.series_id == "ripe_atlas_ipv4_ping:regional:median_rtt_ms"

    selection = _select_scanner_candidate_pairs(
        db_path,
        START,
        START + timedelta(hours=4),
        include_public=True,
        options={"min_observations": 4, "min_coverage": 0.1, "min_aligned_bins": 2},
    )
    reasons = {frozenset(rejection.metrics): rejection.reason for rejection in selection.rejections}
    pairs = {frozenset((pair.x_metric, pair.y_metric)) for pair in selection.pairs}
    assert frozenset(("tcp_latency_ms", "ripe_atlas_ipv4_ping:regional:median_rtt_ms")) in pairs
    assert reasons[frozenset(("ripe_atlas_ipv4_ping:regional:median_rtt_ms", "ripe_atlas_ipv4_ping:regional:p90_rtt_ms"))] == "shared_lineage"
    assert reasons[frozenset(("ripe_atlas_ipv4_ping:regional:responding_probe_fraction",))] == "diagnostic_series"


def _config(**overrides) -> RipeAtlasPublicSourceConfig:
    values = {
        "enabled": True,
        "poll_interval_seconds": 900,
        "initial_backfill_hours": 168,
        "normal_lookback_hours": 6,
        "aggregation_seconds": 900,
        "finalization_delay_seconds": 600,
        "initial_radius_km": 150,
        "maximum_radius_km": 500,
        "desired_probe_count": 8,
        "minimum_probe_count": 5,
        "maximum_probes_per_asn": 2,
        "maximum_anchor_count": 4,
        "cohort_refresh_hours": 24,
        "result_chunk_hours": 6,
        "maximum_probe_batch_size": 50,
        "maximum_requests_per_poll": 200,
        "measurement_ids": (1001, 1004, 1009),
    }
    values.update(overrides)
    return RipeAtlasPublicSourceConfig(**values)


def _http(route: str, payload, url: str) -> RipeHttpFetchResult:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return RipeHttpFetchResult(payload, raw, url, 200, START + timedelta(hours=1), route, {"route": route})


def _probe(probe_id: int, *, asn: int | None, distance: float, status: int = 1, is_public: bool = True, address_v4: str = "192.0.2.1", is_anchor: bool = False) -> dict:
    return {
        "id": probe_id,
        "asn_v4": asn,
        "latitude": LOCATION.latitude + distance / 111.0,
        "longitude": LOCATION.longitude,
        "status": {"id": status, "name": "Connected" if status == 1 else "Disconnected"},
        "is_public": is_public,
        "address_v4": address_v4,
        "is_anchor": is_anchor,
    }


def _probe_fetcher(*, count: int, asn_cycle: int | None = None):
    def fetch(_lat: float, _lon: float, _radius: int):
        rows = [
            _probe(index, asn=64500 + ((index - 1) % (asn_cycle or count)), distance=float(index))
            for index in range(1, count + 1)
        ]
        return (_http("probes", {"results": rows, "next": None}, "https://atlas.ripe.net/api/v2/probes/?radius=fixture"),)

    return fetch


def _result_row(measurement_id: int, probe_id: int, timestamp: datetime, *, avg: float | None = None, sent: int | None = None, rcvd: int | None = None, result: list[dict] | None = None) -> dict:
    row = {
        "msm_id": measurement_id,
        "prb_id": probe_id,
        "timestamp": int(timestamp.timestamp()),
        "af": 4,
        "dst_addr": "193.0.14.129",
        "fw": 5080,
    }
    if avg is not None:
        row.update({"avg": avg, "min": avg - 1 if avg >= 1 else avg, "max": avg + 1})
    if sent is not None:
        row["sent"] = sent
    if rcvd is not None:
        row["rcvd"] = rcvd
    if result is not None:
        row["result"] = result
    elif avg is None:
        row["result"] = [{"rtt": 10.0}, {"rtt": 11.0}, {"rtt": 12.0}]
    return row


def _full_result_payload(timestamp: datetime, *, avg_offset: float) -> list[dict]:
    return [
        _result_row(measurement_id, probe_id, timestamp + timedelta(minutes=1), avg=20.0 + avg_offset + probe_id + measurement_id % 10, sent=3, rcvd=3)
        for measurement_id in (1001, 1004, 1009)
        for probe_id in range(1, 7)
    ]


def _hourly_result_fetcher():
    def fetch(measurement_id: int, start: datetime, stop: datetime, probe_ids):
        rows = []
        cursor = start
        while cursor < stop:
            rows.extend(
                _result_row(measurement_id, probe_id, cursor + timedelta(minutes=1), avg=20.0 + probe_id + cursor.hour)
                for probe_id in probe_ids
            )
            cursor += timedelta(minutes=15)
        return _http(f"results:{measurement_id}", rows, f"https://atlas.ripe.net/api/v2/measurements/{measurement_id}/results/")

    return fetch
