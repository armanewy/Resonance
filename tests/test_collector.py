from __future__ import annotations

import threading
from types import SimpleNamespace

from resonance import collector


def test_collector_closes_database_when_stop_is_already_requested(monkeypatch) -> None:
    stop_event = threading.Event()
    stop_event.set()
    closed = []

    class FakeConnection:
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(collector, "load_config", lambda: SimpleNamespace(collection=object()))
    monkeypatch.setattr(collector, "ensure_database", FakeConnection)

    assert collector.main(stop_event) == 0
    assert closed == [True]
