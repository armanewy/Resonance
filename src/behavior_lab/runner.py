from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from behavior_lab.core import stable_hash, utc_now
from behavior_lab.ledger import ImmutableLedger
from behavior_lab.stress import LabStressTester


@dataclass(frozen=True)
class BatchConfig:
    worlds: list[str]
    seeds: list[int]
    episode_counts: list[int]

    def hash(self) -> str:
        return stable_hash(asdict(self))


class RunAlreadyLocked(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: str | Path, payload: dict[str, Any] | None = None):
        self.path = Path(path)
        self.payload = payload or {}
        self._fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunAlreadyLocked(f"Run lock already exists: {self.path}") from exc
        body = dict(self.payload, locked_at=utc_now(), pid=os.getpid())
        os.write(self._fd, json.dumps(body, sort_keys=True).encode("utf-8"))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.path.exists():
            self.path.unlink()


class SyntheticBatchRunner:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def run(self, config: BatchConfig) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        config_hash = config.hash()
        for world in config.worlds:
            for seed in config.seeds:
                for episodes in config.episode_counts:
                    run_id = f"{world}-seed{seed}-n{episodes}"
                    run_dir = self.base_dir / run_id
                    ledger = ImmutableLedger(run_dir / "ledger.jsonl")
                    if self._completed(ledger, run_id, config_hash):
                        reports.append({"run_id": run_id, "status": "skipped", "reason": "already_complete"})
                        continue
                    with RunLock(run_dir / ".run.lock", {"run_id": run_id, "config_hash": config_hash}):
                        ledger.append(
                            "research_run_start",
                            {
                                "run_id": run_id,
                                "world": world,
                                "seed": seed,
                                "episodes": episodes,
                                "config_hash": config_hash,
                                "code_commit": _git_commit(),
                                "started_at": utc_now(),
                            },
                        )
                        try:
                            report = LabStressTester().run(run_dir, episodes=episodes, seed=seed, world=world)
                            ledger.append(
                                "research_run_end",
                                {
                                    "run_id": run_id,
                                    "status": "complete",
                                    "config_hash": config_hash,
                                    "ended_at": utc_now(),
                                    "summary": report,
                                },
                            )
                            reports.append({"run_id": run_id, "status": "complete", "summary": report})
                        except Exception as exc:
                            ledger.append(
                                "research_run_end",
                                {
                                    "run_id": run_id,
                                    "status": "failed",
                                    "config_hash": config_hash,
                                    "ended_at": utc_now(),
                                    "error": repr(exc),
                                },
                            )
                            raise
        return reports

    def _completed(self, ledger: ImmutableLedger, run_id: str, config_hash: str) -> bool:
        return any(
            payload.get("run_id") == run_id
            and payload.get("config_hash") == config_hash
            and payload.get("status") == "complete"
            for payload in ledger.payloads("research_run_end")
        )


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()
