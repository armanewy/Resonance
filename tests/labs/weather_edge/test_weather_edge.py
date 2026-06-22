from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests"
for path in [str(ROOT), str(TESTS)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import _bootstrap  # noqa: E402,F401

from behavior_lab.labs.weather_edge import (  # noqa: E402
    DailyHighTemperatureEvent,
    FixtureWeatherEdgeProvider,
    ForecastPoint,
    MarketDepth,
    OrderBookLevel,
    Settlement,
    StationHistoricalDay,
    TemperatureBracket,
    WeatherSnapshot,
    backfill,
    paper_cycle,
    report,
)
from behavior_lab.money.storage import MoneyStorage  # noqa: E402


AS_OF = "2026-07-01T03:00:00-04:00"


class WeatherEdgeTests(unittest.TestCase):
    def test_backfill_walks_forward_groups_city_event_and_resolves_paper_trade(self) -> None:
        provider = _provider(include_settlements=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = backfill(provider, tmp, as_of="2026-07-03T00:00:00-04:00")

            self.assertEqual(result["discovered_event_count"], 2)
            self.assertEqual(result["city_event_count"], 1)
            self.assertEqual(result["decisions_appended"], 1)
            self.assertEqual(result["decisions_resolved"], 1)
            self.assertFalse(result["authenticates_for_trading"])
            self.assertFalse(result["submits_orders"])

            storage = MoneyStorage(tmp)
            self.assertTrue(storage.ledger.verify())
            entries = storage.ledger.latest_entries()
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.evidence_state, "resolved_paper")
            self.assertEqual(entry.selected_action, "buy_yes")
            self.assertEqual(entry.economic_event_key, "weather_edge:NOAA_DAILY_HIGH:CLI:KNYC:2026-07-01")
            self.assertTrue(entry.resolution["outcome_yes"])
            self.assertEqual(entry.resolution["observed_high_f"], 88.0)
            self.assertEqual(entry.realized_net_value, 2.15)
            self.assertEqual(entry.provenance["baselines"]["historical_sample_size"], 4)
            self.assertEqual(entry.provenance["baselines"]["station_bias_f"], 1.0)
            self.assertEqual(entry.provenance["execution"]["quantity"], 5)
            self.assertEqual(entry.provenance["execution"]["raw_order_book_quantity"], 10)
            self.assertEqual(entry.provenance["market_depth"]["yes_asks"][0]["quantity"], 10)

    def test_paper_cycle_uses_executable_ask_not_midpoint_and_preserves_quantity(self) -> None:
        provider = _provider(include_settlements=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = paper_cycle(provider, tmp, as_of=AS_OF)

            self.assertEqual(result["decisions_appended"], 1)
            entry = MoneyStorage(tmp).ledger.latest_entries()[0]
            execution = entry.provenance["execution"]
            self.assertEqual(entry.evidence_state, "paper_decision")
            self.assertEqual(entry.selected_action, "buy_yes")
            self.assertEqual(execution["executable_price"], 0.55)
            self.assertFalse(execution["midpoint_used"])
            self.assertFalse(execution["candle_extreme_used"])
            self.assertEqual(execution["raw_order_book_quantity"], 10)
            self.assertEqual(execution["quantity"], 5)
            self.assertIn("midpoint", entry.provenance["prohibited_fill_sources"])
            self.assertFalse(entry.provenance["authenticates_for_trading"])
            self.assertFalse(entry.provenance["submits_orders"])
            self.assertFalse(entry.provenance["notifications_allowed"])

    def test_paper_cycle_records_explicit_no_trade_decision(self) -> None:
        provider = _provider(include_settlements=False, first_yes_ask=0.95)
        with tempfile.TemporaryDirectory() as tmp:
            paper_cycle(provider, tmp, as_of=AS_OF)
            entry = MoneyStorage(tmp).ledger.latest_entries()[0]

            self.assertEqual(entry.selected_action, "no_trade")
            self.assertEqual(entry.capital_required, 0.0)
            self.assertEqual(entry.maximum_possible_loss, 0.0)
            self.assertEqual(entry.conservative_expected_net_value, 0.0)
            self.assertIn("edge_not_positive_after_uncertainty_buffer", entry.provenance["no_trade_reasons"])
            self.assertEqual(entry.mechanically_defined_no_action_outcome["realized_net_value"], 0.0)

    def test_report_exposes_evidence_gates_without_enabling_real_money(self) -> None:
        provider = _provider(include_settlements=True, historical_resolved_city_days=200)
        with tempfile.TemporaryDirectory() as tmp:
            backfill(provider, tmp, as_of="2026-07-03T00:00:00-04:00")
            output = report(tmp, provider=provider, as_of="2026-07-04T00:00:00-04:00")

            gate = output["evidence_gate"]
            self.assertEqual(gate["minimum_resolved_city_days"]["historical_available"], 200)
            self.assertFalse(gate["minimum_resolved_city_days"]["passes"])
            self.assertTrue(gate["market_baseline_comparison"]["available"])
            self.assertTrue(gate["market_baseline_comparison"]["passes"])
            self.assertTrue(gate["pessimistic_cost_sensitivity"]["passes"])
            self.assertFalse(gate["prospective_incubation"]["passes"])
            self.assertFalse(gate["future_real_money_review_allowed"])
            self.assertFalse(gate["real_money_enabled_in_this_wave"])
            self.assertFalse(output["authenticates_for_trading"])
            self.assertFalse(output["submits_orders"])


def _provider(
    *,
    include_settlements: bool,
    first_yes_ask: float = 0.55,
    historical_resolved_city_days: int | None = None,
) -> FixtureWeatherEdgeProvider:
    event_a = _event("nyc-20260701-85-90", TemperatureBracket("85-90", 85.0, 90.0))
    event_b = _event("nyc-20260701-90-95", TemperatureBracket("90-95", 90.0, 95.0))
    settlements = []
    if include_settlements:
        settlements = [_settlement(event_a), _settlement(event_b)]
    return FixtureWeatherEdgeProvider(
        events=[event_a, event_b],
        market_depths=[
            _depth(event_a.event_id, first_yes_ask),
            _depth(event_b.event_id, 0.65),
        ],
        weather_snapshots=[
            _snapshot(event_a.event_id),
            _snapshot(event_b.event_id),
        ],
        settlements=settlements,
        station_history=[
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-26",
                high_f=86.0,
                forecast_mean_f=85.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-27",
                high_f=87.0,
                forecast_mean_f=86.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-28",
                high_f=88.0,
                forecast_mean_f=87.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-06-29",
                high_f=91.0,
                forecast_mean_f=90.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="heat",
            ),
            StationHistoricalDay(
                station_id="KNYC",
                local_date="2026-07-02",
                high_f=70.0,
                forecast_mean_f=50.0,
                settlement_series="NOAA_DAILY_HIGH",
                report_source="CLI",
                regime="post_event_future_record",
            ),
        ],
        historical_resolved_city_days=historical_resolved_city_days,
    )


def _event(event_id: str, bracket: TemperatureBracket) -> DailyHighTemperatureEvent:
    return DailyHighTemperatureEvent(
        event_id=event_id,
        city="New York",
        station_id="KNYC",
        station_name="Central Park",
        local_date="2026-07-01",
        timezone="America/New_York",
        dst_status="EDT",
        settlement_series="NOAA_DAILY_HIGH",
        report_source="CLI",
        report_name="Daily Climate Report",
        bracket=bracket,
        open_time="2026-06-30T09:00:00-04:00",
        close_time="2026-07-01T09:00:00-04:00",
        resolution_time="2026-07-02T12:00:00-04:00",
        market_source="fixture_event_market",
    )


def _depth(event_id: str, yes_ask: float) -> MarketDepth:
    return MarketDepth(
        event_id=event_id,
        as_of=AS_OF,
        yes_bids=[OrderBookLevel(price=0.25, quantity=7)],
        yes_asks=[OrderBookLevel(price=yes_ask, quantity=10)],
        no_bids=[OrderBookLevel(price=0.25, quantity=9)],
        no_asks=[OrderBookLevel(price=0.50, quantity=8)],
        source="fixture_order_book",
        snapshot_id=f"depth-{event_id}",
    )


def _snapshot(event_id: str) -> WeatherSnapshot:
    return WeatherSnapshot(
        event_id=event_id,
        as_of=AS_OF,
        station_id="KNYC",
        timezone="America/New_York",
        forecast_issued_at="2026-07-01T02:45:00-04:00",
        official_forecast_source="NWS_NBM_FIXTURE",
        forecast_distribution=[
            ForecastPoint(temperature_f=84.0, probability=0.2),
            ForecastPoint(temperature_f=86.0, probability=0.3),
            ForecastPoint(temperature_f=88.0, probability=0.3),
            ForecastPoint(temperature_f=91.0, probability=0.2),
        ],
        regime="heat",
        snapshot_id=f"weather-{event_id}",
    )


def _settlement(event: DailyHighTemperatureEvent) -> Settlement:
    return Settlement(
        event_id=event.event_id,
        observed_high_f=88.0,
        finalized_at="2026-07-02T12:00:00-04:00",
        station_id=event.station_id,
        settlement_series=event.settlement_series,
        report_source=event.report_source,
        report_name=event.report_name,
        timezone=event.timezone,
        dst_status=event.dst_status,
    )


if __name__ == "__main__":
    unittest.main()
