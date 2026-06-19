from __future__ import annotations

import argparse
import math
import random
from datetime import timedelta

from resonance.storage import Measurement, delete_measurements_by_source, ensure_database, insert_measurements
from resonance.time_utils import utc_now


DEMO_SOURCE = "demo"


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed or clear synthetic Resonance demo data.")
    parser.add_argument("--hours", type=float, default=6.0, help="Hours of demo data to generate.")
    parser.add_argument("--clear", action="store_true", help="Remove demo measurements and exit.")
    args = parser.parse_args()

    conn = ensure_database()
    try:
        removed = delete_measurements_by_source(conn, DEMO_SOURCE)
        if args.clear:
            print(f"Removed {removed} demo measurements.")
            return 0

        measurements = generate_demo_measurements(args.hours)
        inserted = insert_measurements(conn, measurements)
        print(f"Replaced {removed} old demo measurements with {inserted} new demo measurements.")
        return 0
    finally:
        conn.close()


def generate_demo_measurements(hours: float) -> list[Measurement]:
    random.seed(42)
    end = utc_now()
    start = end - timedelta(hours=hours)
    measurements: list[Measurement] = []
    sample_count = max(1, int(hours * 60 * 2))

    for index in range(sample_count + 1):
        timestamp = start + timedelta(seconds=30 * index)
        phase = index / max(sample_count, 1)
        cpu = 28 + 20 * math.sin(phase * math.tau * 4) + random.uniform(-4, 4)
        memory = 58 + 5 * math.sin(phase * math.tau * 2 + 0.8) + random.uniform(-1.5, 1.5)
        recv = 95_000 + 75_000 * max(0, math.sin(phase * math.tau * 7)) + random.uniform(0, 18_000)
        sent = 28_000 + 22_000 * max(0, math.sin(phase * math.tau * 5 + 1.2)) + random.uniform(0, 8_000)
        tcp = 19 + 6 * math.sin(phase * math.tau * 8) + random.uniform(-1, 1)
        dns = 14 + 4 * math.sin(phase * math.tau * 6 + 0.4) + random.uniform(-1, 1)
        plugged = 1.0 if index % 120 < 80 else 0.0
        battery = min(100, max(20, 84 - phase * 18 + plugged * 4))

        metadata = {"demo": True}
        measurements.extend(
            [
                Measurement(timestamp, "cpu_percent", max(0, min(100, cpu)), "percent", DEMO_SOURCE, metadata),
                Measurement(timestamp, "memory_percent", max(0, min(100, memory)), "percent", DEMO_SOURCE, metadata),
                Measurement(timestamp, "network_recv_bytes_per_second", max(0, recv), "bytes/second", DEMO_SOURCE, metadata),
                Measurement(timestamp, "network_sent_bytes_per_second", max(0, sent), "bytes/second", DEMO_SOURCE, metadata),
                Measurement(timestamp, "tcp_success", 1.0, "boolean", DEMO_SOURCE, metadata),
                Measurement(timestamp, "tcp_latency_ms", max(1, tcp), "ms", DEMO_SOURCE, metadata),
                Measurement(timestamp, "dns_success", 1.0, "boolean", DEMO_SOURCE, metadata),
                Measurement(timestamp, "dns_latency_ms", max(1, dns), "ms", DEMO_SOURCE, metadata),
                Measurement(timestamp, "battery_percent", battery, "percent", DEMO_SOURCE, metadata),
                Measurement(timestamp, "battery_plugged", plugged, "boolean", DEMO_SOURCE, metadata),
            ]
        )

        if index % 30 == 0:
            weather_phase = index / max(sample_count, 1)
            temp = 20 + 5 * math.sin(weather_phase * math.tau)
            precipitation = max(0, 0.9 * math.sin(weather_phase * math.tau * 3 - 1.2))
            wind = 10 + 5 * math.sin(weather_phase * math.tau * 2 + 0.5)
            measurements.extend(
                [
                    Measurement(timestamp, "weather_temperature_c", temp, "C", DEMO_SOURCE, metadata),
                    Measurement(timestamp, "weather_relative_humidity_percent", 62 + random.uniform(-8, 8), "percent", DEMO_SOURCE, metadata),
                    Measurement(timestamp, "weather_precipitation_mm", precipitation, "mm", DEMO_SOURCE, metadata),
                    Measurement(timestamp, "weather_wind_speed_kmh", max(0, wind), "km/h", DEMO_SOURCE, metadata),
                    Measurement(timestamp, "weather_surface_pressure_hpa", 1012 + random.uniform(-4, 4), "hPa", DEMO_SOURCE, metadata),
                    Measurement(timestamp, "weather_code", 2, "code", DEMO_SOURCE, metadata),
                ]
            )

    return measurements


if __name__ == "__main__":
    raise SystemExit(main())

