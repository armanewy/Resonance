from __future__ import annotations

import logging
import signal
import threading
import time

from resonance.config import ConfigError, load_config
from resonance.personal import collect_personal_measurements
from resonance.storage import CollectorError, ensure_database, insert_collector_error, insert_collector_errors, insert_measurements
from resonance.time_utils import utc_now
from resonance.weather import WeatherError, fetch_weather_measurements


LOG = logging.getLogger("resonance.collector")


def main(stop_requested: threading.Event | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop_event = stop_requested or threading.Event()
    if stop_requested is None:
        _install_stop_handlers(stop_event)
    try:
        config = load_config()
    except ConfigError as exc:
        LOG.error("Configuration error: %s", exc)
        return 2

    conn = ensure_database()
    LOG.info("Resonance collector started")
    previous_network_snapshot = None
    next_personal_at = 0.0
    next_weather_at = 0.0

    try:
        while not stop_event.is_set():
            monotonic_now = time.monotonic()
            if monotonic_now >= next_personal_at:
                result = collect_personal_measurements(config.collection, previous_network_snapshot)
                insert_measurements(conn, result.measurements)
                if result.errors:
                    insert_collector_errors(conn, result.errors)
                    for error in result.errors:
                        LOG.warning("%s: %s", error.error_type, error.message)
                previous_network_snapshot = result.network_snapshot
                next_personal_at = monotonic_now + config.collection.personal_interval_seconds

            if monotonic_now >= next_weather_at:
                try:
                    weather_measurements = fetch_weather_measurements(config)
                    inserted = insert_measurements(conn, weather_measurements)
                    if inserted:
                        LOG.info("Stored %s weather measurements", inserted)
                except WeatherError as exc:
                    insert_collector_error(
                        conn,
                        CollectorError(utc_now(), "weather", "weather_request_failed", str(exc)),
                    )
                    LOG.warning("weather_request_failed: %s", exc)
                next_weather_at = monotonic_now + config.collection.weather_interval_seconds

            stop_event.wait(0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        LOG.info("Resonance collector stopping")
        conn.close()
    return 0


def _install_stop_handlers(stop_event: threading.Event) -> None:
    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)


if __name__ == "__main__":
    raise SystemExit(main())

