from __future__ import annotations

import sqlite3

import pytest

from resonance.storage import init_db


@pytest.fixture()
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()

