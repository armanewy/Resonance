from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


DASHBOARD_URL = "http://127.0.0.1:8501"


def main() -> int:
    stop_requested = threading.Event()

    def request_stop(signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    project_root = Path(__file__).resolve().parent
    collector = subprocess.Popen([sys.executable, "-m", "resonance.collector"], cwd=project_root)
    dashboard = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "resonance/dashboard.py",
            "--server.address=127.0.0.1",
            "--server.port=8501",
            "--server.headless=true",
        ],
        cwd=project_root,
    )

    processes = {"collector": collector, "dashboard": dashboard}
    print(f"Resonance dashboard: {DASHBOARD_URL}", flush=True)
    print("Press Ctrl+C to stop collector and dashboard.", flush=True)

    try:
        while not stop_requested.is_set():
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    print(f"{name} exited with code {code}; stopping remaining processes.", flush=True)
                    return code
            stop_requested.wait(1)
    except KeyboardInterrupt:
        stop_requested.set()
    finally:
        print("Stopping Resonance...", flush=True)
        for process in processes.values():
            _terminate(process)
    return 0


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        process.terminate()
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
