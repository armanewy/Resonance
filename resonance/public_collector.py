from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable

from resonance.config import ConfigError, EiaGridPublicSourceConfig, load_config
from resonance.public_sources.eia_grid import (
    DEFAULT_RAW_ROOT,
    ROUTES,
    SOURCE_ID,
    EiaGridError,
    ensure_eia_registry,
    poll_new_england_grid,
)
from resonance.storage import (
    CollectorError,
    PublicCollectionState,
    ensure_database,
    fetch_public_collection_state,
    insert_collector_error,
    upsert_public_collection_state,
)
from resonance.time_utils import utc_now


LOG = logging.getLogger("resonance.public_collector")
LOCK_PATH = Path("data/public/eia_grid.lock")


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

    eia_config = config.public_sources.eia_grid
    if not eia_config.enabled:
        LOG.info("No public source enabled; public collector exiting.")
        conn = ensure_database()
        try:
            ensure_eia_registry(conn, enabled=False)
        finally:
            conn.close()
        return 0

    lock = _acquire_lock(LOCK_PATH)
    if lock is None:
        LOG.warning("EIA public collector lock is already held; exiting.")
        return 0
    try:
        return run_loop(eia_config, stop_event=stop_event)
    finally:
        _release_lock(lock, LOCK_PATH)


def run_loop(
    eia_config: EiaGridPublicSourceConfig,
    *,
    stop_event: threading.Event,
    poller: Callable[..., object] = poll_new_england_grid,
    database_path: str | Path = "data/resonance.db",
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> int:
    conn = ensure_database(database_path)
    try:
        next_poll = 0.0
        while not stop_event.is_set():
            now_monotonic = time.monotonic()
            if now_monotonic >= next_poll:
                run_once(eia_config, database_path=database_path, raw_root=raw_root, poller=poller)
                next_poll = now_monotonic + eia_config.poll_interval_seconds
            stop_event.wait(0.5)
    finally:
        conn.close()
    return 0


def run_once(
    eia_config: EiaGridPublicSourceConfig,
    *,
    database_path: str | Path = "data/resonance.db",
    raw_root: Path = DEFAULT_RAW_ROOT,
    poller: Callable[..., object] = poll_new_england_grid,
) -> bool:
    conn = ensure_database(database_path)
    try:
        api_key = os.environ.get("EIA_API_KEY", "")
        ensure_eia_registry(conn, enabled=eia_config.enabled and bool(api_key))
        if not api_key:
            error = EiaGridError("EIA_API_KEY is required when public_sources.eia_grid.enabled is true")
            _record_failure(conn, error)
            insert_collector_error(
                conn,
                CollectorError(utc_now(), SOURCE_ID, "missing_credentials", str(error)),
            )
            LOG.warning("%s", error)
            return False
        try:
            result = poller(conn, config=eia_config, api_key=api_key, raw_root=raw_root)
        except Exception as exc:
            _record_failure(conn, exc)
            insert_collector_error(
                conn,
                CollectorError(utc_now(), SOURCE_ID, exc.__class__.__name__, str(exc).replace(api_key, "REDACTED")),
            )
            LOG.warning("EIA public collection failed: %s", str(exc).replace(api_key, "REDACTED"))
            return False
        LOG.info("EIA public collection inserted %s observations", getattr(result, "inserted_observations", "unknown"))
        return True
    finally:
        conn.close()


def _record_failure(conn, exc: BaseException) -> None:
    now = utc_now()
    api_key = os.environ.get("EIA_API_KEY", "")
    message = str(exc).replace(api_key, "REDACTED") if api_key else str(exc)
    for route in ROUTES:
        previous = fetch_public_collection_state(conn, source_id=SOURCE_ID, route=route)
        upsert_public_collection_state(
            conn,
            PublicCollectionState(
                source_id=SOURCE_ID,
                route=route,
                last_successful_poll_utc=previous.last_successful_poll_utc if previous else None,
                newest_complete_valid_period_utc=previous.newest_complete_valid_period_utc if previous else None,
                earliest_unresolved_gap_utc=previous.earliest_unresolved_gap_utc if previous else None,
                latest_error_utc=now,
                latest_error=message,
                consecutive_failure_count=(previous.consecutive_failure_count if previous else 0) + 1,
                metadata=previous.metadata if previous else {},
            ),
        )
    conn.commit()


def _acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(handle, str(os.getpid()).encode("ascii"))
    return handle


def _release_lock(handle, path: Path) -> None:
    os.close(handle)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _install_stop_handlers(stop_event: threading.Event) -> None:
    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)


if __name__ == "__main__":
    raise SystemExit(main())
