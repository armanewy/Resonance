from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator

from behavior_lab.core import new_id, stable_hash, to_jsonable, utc_now


class LedgerIntegrityError(RuntimeError):
    pass


class LedgerLockTimeout(RuntimeError):
    pass


class DuplicateRecordError(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _ExclusiveFileLock:
    def __init__(self, path: Path, *, timeout: float = 10.0, stale_after: float = 120.0):
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: int | None = None

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {"pid": os.getpid(), "locked_at": utc_now(), "created_epoch": time.time()}
                os.write(self._fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
                os.fsync(self._fd)
                return
            except (FileExistsError, PermissionError):
                if self._is_stale():
                    try:
                        self.path.unlink()
                    except (FileNotFoundError, PermissionError):
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(f"Timed out acquiring ledger lock: {self.path}")
                time.sleep(0.02)

    def _is_stale(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        age = time.time() - stat.st_mtime
        if age < self.stale_after:
            return False
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw else {}
            pid = int(payload.get("pid", -1))
        except (OSError, ValueError, json.JSONDecodeError):
            # A truncated lock from a crashed process is recoverable only after
            # the full stale interval; a newly-created empty lock remains safe.
            return True
        return not _pid_alive(pid)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        deadline = time.monotonic() + 1.0
        while True:
            try:
                self.path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.02)

    def __enter__(self) -> "_ExclusiveFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


Guard = Callable[[list[dict[str, Any]]], None]


class ImmutableLedger:
    """Append-only JSONL ledger with hash chaining and guarded writes.

    The hash chain detects mutation; the sidecar write lock prevents two local
    processes from calculating the same previous hash and corrupting the chain.
    It is an integrity mechanism, not an adversarial remote database.
    """

    genesis_hash = "GENESIS"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".write.lock")

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with _ExclusiveFileLock(self.lock_path):
            yield

    def append(
        self,
        record_type: str,
        payload: Any,
        record_id: str | None = None,
        *,
        unique_record_id: bool = False,
    ) -> dict[str, Any]:
        return self.append_guarded(
            record_type,
            payload,
            record_id=record_id,
            unique_record_id=unique_record_id,
        )

    def append_guarded(
        self,
        record_type: str,
        payload: Any,
        *,
        record_id: str | None = None,
        unique_record_id: bool = False,
        guard: Guard | None = None,
    ) -> dict[str, Any]:
        if not record_type or not str(record_type).strip():
            raise ValueError("record_type must be non-empty")
        actual_record_id = record_id or new_id("r")
        with self.exclusive():
            records = self._scan_unlocked()
            self._verify_records(records)
            if unique_record_id and any(record.get("record_id") == actual_record_id for record in records):
                raise DuplicateRecordError(f"Record ID already exists: {actual_record_id}")
            if guard is not None:
                guard(records)
            previous = str(records[-1]["record_hash"]) if records else self.genesis_hash
            body = {
                "record_id": actual_record_id,
                "record_type": record_type,
                "written_at": utc_now(),
                "previous_hash": previous,
                "payload": to_jsonable(payload),
            }
            body["record_hash"] = stable_hash(body)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(body, sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return body

    def append_many_guarded(
        self,
        entries: list[tuple[str, Any, str | None]],
        *,
        unique_record_ids: bool = False,
        guard: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically append several records with one scan and one fsync.

        This is the preferred path for seeding datasets and materializing split
        manifests.  It avoids the quadratic cost of rescanning the complete
        ledger for each item while preserving the same hash-chain guarantees.
        """

        if not entries:
            return []
        normalized: list[tuple[str, Any, str]] = []
        for record_type, payload, record_id in entries:
            if not isinstance(record_type, str) or not record_type.strip():
                raise ValueError("record_type must be a non-empty string")
            normalized.append((record_type, payload, record_id or new_id("r")))
        with self.exclusive():
            records = self._scan_unlocked()
            self._verify_records(records)
            if guard is not None:
                guard(records)
            if unique_record_ids:
                existing_ids = {str(record.get("record_id")) for record in records}
                new_ids = [record_id for _, _, record_id in normalized]
                counts = Counter(new_ids)
                duplicates = {record_id for record_id in new_ids if record_id in existing_ids}
                duplicates.update({record_id for record_id, count in counts.items() if count > 1})
                if duplicates:
                    raise DuplicateRecordError(
                        f"Record IDs already exist or repeat in batch: {sorted(duplicates)}"
                    )
            previous = str(records[-1]["record_hash"]) if records else self.genesis_hash
            bodies: list[dict[str, Any]] = []
            written_at = utc_now()
            for record_type, payload, record_id in normalized:
                body = {
                    "record_id": record_id,
                    "record_type": record_type,
                    "written_at": written_at,
                    "previous_hash": previous,
                    "payload": to_jsonable(payload),
                }
                body["record_hash"] = stable_hash(body)
                previous = str(body["record_hash"])
                bodies.append(body)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                for body in bodies:
                    handle.write(json.dumps(body, sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return bodies

    def scan(self, record_type: str | None = None) -> list[dict[str, Any]]:
        with self.exclusive():
            records = self._scan_unlocked()
        if record_type is None:
            return records
        return [record for record in records if record.get("record_type") == record_type]

    def _scan_unlocked(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise LedgerIntegrityError(f"Invalid ledger JSON at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise LedgerIntegrityError(f"Ledger record at line {line_number} is not an object")
                records.append(record)
        return records

    def payloads(self, record_type: str | None = None) -> list[dict[str, Any]]:
        return [record["payload"] for record in self.scan(record_type)]

    def iter_payloads(self, record_type: str | None = None) -> Iterable[dict[str, Any]]:
        for record in self.scan(record_type):
            yield record["payload"]

    def last_hash(self) -> str:
        records = self.scan()
        return str(records[-1]["record_hash"]) if records else self.genesis_hash

    def verify_hash_chain(self) -> bool:
        with self.exclusive():
            self._verify_records(self._scan_unlocked())
        return True

    def _verify_records(self, records: list[dict[str, Any]]) -> None:
        previous = self.genesis_hash
        seen_record_ids: set[str] = set()
        for index, record in enumerate(records, start=1):
            required = {"record_id", "record_type", "written_at", "previous_hash", "payload", "record_hash"}
            missing = required - set(record)
            if missing:
                raise LedgerIntegrityError(f"Ledger record {index} is missing fields: {sorted(missing)}")
            record_id = str(record["record_id"])
            # Historic ledgers may contain duplicate IDs, so duplicates are not an
            # integrity failure. New critical records opt into uniqueness on append.
            seen_record_ids.add(record_id)
            observed_hash = record.get("record_hash")
            if record.get("previous_hash") != previous:
                raise LedgerIntegrityError(
                    f"Broken hash chain at {record_id}: expected previous {previous}, "
                    f"found {record.get('previous_hash')}"
                )
            body = dict(record)
            body.pop("record_hash", None)
            expected_hash = stable_hash(body)
            if observed_hash != expected_hash:
                raise LedgerIntegrityError(
                    f"Record hash mismatch at {record_id}: expected {expected_hash}, found {observed_hash}"
                )
            previous = str(observed_hash)

    def latest_by_payload_key(self, record_type: str, key: str, value: str) -> dict[str, Any] | None:
        match = None
        for record in self.scan(record_type):
            payload = record["payload"]
            if payload.get(key) == value:
                match = payload
        return match

    def find_record(self, record_id: str, record_type: str | None = None) -> dict[str, Any] | None:
        match = None
        for record in self.scan(record_type):
            if record.get("record_id") == record_id:
                match = record
        return match
