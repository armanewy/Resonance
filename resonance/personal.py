from __future__ import annotations

import concurrent.futures
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import psutil

from resonance.config import CollectionConfig
from resonance.storage import CollectorError, Measurement
from resonance.time_utils import utc_now


PERSONAL_SOURCE = "personal"


@dataclass(frozen=True)
class NetCountersSnapshot:
    bytes_recv: int
    bytes_sent: int
    timestamp_utc: datetime


@dataclass(frozen=True)
class NetworkRates:
    bytes_recv_per_second: float | None
    bytes_sent_per_second: float | None


@dataclass(frozen=True)
class ProbeResult:
    succeeded: bool
    latency_ms: float | None
    error_message: str | None = None


@dataclass(frozen=True)
class PersonalSampleResult:
    measurements: list[Measurement]
    errors: list[CollectorError]
    network_snapshot: NetCountersSnapshot | None


def calculate_network_rates(
    previous: NetCountersSnapshot | None,
    current: NetCountersSnapshot,
    elapsed_seconds: float,
) -> NetworkRates:
    if previous is None or elapsed_seconds <= 0:
        return NetworkRates(None, None)

    recv_delta = current.bytes_recv - previous.bytes_recv
    sent_delta = current.bytes_sent - previous.bytes_sent
    recv_rate = recv_delta / elapsed_seconds if recv_delta >= 0 else None
    sent_rate = sent_delta / elapsed_seconds if sent_delta >= 0 else None
    return NetworkRates(recv_rate, sent_rate)


def collect_personal_measurements(
    collection_config: CollectionConfig,
    previous_network_snapshot: NetCountersSnapshot | None,
) -> PersonalSampleResult:
    timestamp = utc_now()
    measurements: list[Measurement] = []
    errors: list[CollectorError] = []

    try:
        measurements.append(
            Measurement(timestamp, "cpu_percent", psutil.cpu_percent(interval=0.1), "percent", PERSONAL_SOURCE)
        )
    except Exception as exc:  # pragma: no cover - defensive around platform psutil behavior
        errors.append(_error(timestamp, "cpu", exc))

    try:
        measurements.append(
            Measurement(timestamp, "memory_percent", psutil.virtual_memory().percent, "percent", PERSONAL_SOURCE)
        )
    except Exception as exc:  # pragma: no cover - defensive around platform psutil behavior
        errors.append(_error(timestamp, "memory", exc))

    current_network_snapshot = _network_snapshot(timestamp)
    if current_network_snapshot is None:
        errors.append(
            CollectorError(timestamp, "personal", "network_counters_unavailable", "psutil.net_io_counters() returned no data")
        )
    else:
        previous_time = previous_network_snapshot.timestamp_utc if previous_network_snapshot else None
        elapsed = (timestamp - previous_time).total_seconds() if previous_time else 0
        rates = calculate_network_rates(previous_network_snapshot, current_network_snapshot, elapsed)
        if rates.bytes_recv_per_second is not None:
            measurements.append(
                Measurement(
                    timestamp,
                    "network_recv_bytes_per_second",
                    rates.bytes_recv_per_second,
                    "bytes/second",
                    PERSONAL_SOURCE,
                )
            )
        if rates.bytes_sent_per_second is not None:
            measurements.append(
                Measurement(
                    timestamp,
                    "network_sent_bytes_per_second",
                    rates.bytes_sent_per_second,
                    "bytes/second",
                    PERSONAL_SOURCE,
                )
            )

    battery_measurements, battery_error = collect_battery_measurements(timestamp)
    measurements.extend(battery_measurements)
    if battery_error:
        errors.append(battery_error)

    tcp_result = measure_tcp_latency(collection_config.tcp_test_host, collection_config.tcp_test_port)
    measurements.append(
        Measurement(timestamp, "tcp_success", 1.0 if tcp_result.succeeded else 0.0, "boolean", PERSONAL_SOURCE)
    )
    if tcp_result.latency_ms is not None:
        measurements.append(Measurement(timestamp, "tcp_latency_ms", tcp_result.latency_ms, "ms", PERSONAL_SOURCE))
    elif tcp_result.error_message:
        errors.append(CollectorError(timestamp, "personal", "tcp_connection_failed", tcp_result.error_message))

    dns_result = measure_dns_latency(collection_config.dns_test_hostname)
    measurements.append(
        Measurement(timestamp, "dns_success", 1.0 if dns_result.succeeded else 0.0, "boolean", PERSONAL_SOURCE)
    )
    if dns_result.latency_ms is not None:
        measurements.append(Measurement(timestamp, "dns_latency_ms", dns_result.latency_ms, "ms", PERSONAL_SOURCE))
    elif dns_result.error_message:
        errors.append(CollectorError(timestamp, "personal", "dns_lookup_failed", dns_result.error_message))

    return PersonalSampleResult(measurements, errors, current_network_snapshot)


def collect_battery_measurements(
    timestamp_utc: datetime,
    battery_provider: Callable[[], object | None] = psutil.sensors_battery,
) -> tuple[list[Measurement], CollectorError | None]:
    try:
        battery = battery_provider()
    except Exception as exc:  # pragma: no cover - defensive around platform psutil behavior
        return [], _error(timestamp_utc, "battery", exc)

    if battery is None:
        return [], None

    measurements = [
        Measurement(timestamp_utc, "battery_percent", float(battery.percent), "percent", PERSONAL_SOURCE)
    ]
    plugged = getattr(battery, "power_plugged", None)
    if plugged is not None:
        measurements.append(
            Measurement(timestamp_utc, "battery_plugged", 1.0 if plugged else 0.0, "boolean", PERSONAL_SOURCE)
        )
    return measurements, None


def measure_tcp_latency(host: str, port: int, timeout_seconds: float = 1.5) -> ProbeResult:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            latency_ms = (time.perf_counter() - started) * 1000
            return ProbeResult(True, latency_ms)
    except OSError as exc:
        return ProbeResult(False, None, f"{type(exc).__name__}: {exc}")


def measure_dns_latency(hostname: str, timeout_seconds: float = 1.5) -> ProbeResult:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    started = time.perf_counter()
    future = executor.submit(socket.getaddrinfo, hostname, None)
    try:
        future.result(timeout=timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000
        return ProbeResult(True, latency_ms)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return ProbeResult(False, None, f"DNS lookup timed out after {timeout_seconds:.1f}s")
    except OSError as exc:
        return ProbeResult(False, None, f"{type(exc).__name__}: {exc}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _network_snapshot(timestamp_utc: datetime) -> NetCountersSnapshot | None:
    counters = psutil.net_io_counters()
    if counters is None:
        return None
    return NetCountersSnapshot(
        bytes_recv=int(counters.bytes_recv),
        bytes_sent=int(counters.bytes_sent),
        timestamp_utc=timestamp_utc,
    )


def _error(timestamp_utc: datetime, metric: str, exc: Exception) -> CollectorError:
    return CollectorError(timestamp_utc, "personal", f"{metric}_unavailable", f"{type(exc).__name__}: {exc}")

