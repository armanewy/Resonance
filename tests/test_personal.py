from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from resonance.personal import NetCountersSnapshot, calculate_network_rates, collect_battery_measurements


def test_network_throughput_delta_calculation() -> None:
    start = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    previous = NetCountersSnapshot(1_000, 2_000, start)
    current = NetCountersSnapshot(4_000, 2_900, start + timedelta(seconds=30))

    rates = calculate_network_rates(previous, current, 30)

    assert rates.bytes_recv_per_second == 100
    assert rates.bytes_sent_per_second == 30


def test_network_throughput_first_sample_has_no_bogus_rate() -> None:
    start = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    current = NetCountersSnapshot(4_000, 2_900, start)

    rates = calculate_network_rates(None, current, 0)

    assert rates.bytes_recv_per_second is None
    assert rates.bytes_sent_per_second is None


def test_unavailable_battery_does_not_crash() -> None:
    timestamp = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)

    measurements, error = collect_battery_measurements(timestamp, battery_provider=lambda: None)

    assert measurements == []
    assert error is None


def test_available_battery_is_recorded() -> None:
    timestamp = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    battery = SimpleNamespace(percent=81, power_plugged=True)

    measurements, error = collect_battery_measurements(timestamp, battery_provider=lambda: battery)

    assert error is None
    assert {measurement.metric for measurement in measurements} == {"battery_percent", "battery_plugged"}

