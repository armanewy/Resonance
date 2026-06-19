from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


DEFAULT_LEDGER_PATH = Path("data/science/ledger.jsonl")
GENESIS_PREVIOUS_HASH = "0" * 64
ENTRY_FIELDS = (
    "sequence_number",
    "timestamp_utc",
    "event_type",
    "payload",
    "artifact_hashes",
    "code_commit",
    "previous_entry_hash",
    "payload_hash",
    "entry_hash",
)
ENTRY_FIELD_SET = set(ENTRY_FIELDS)
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "snapshot_created",
        "hypothesis_proposed",
        "fit_completed",
        "hypothesis_preregistered",
        "blind_evaluation_completed",
        "result_interpreted",
        "hypothesis_superseded",
        "experiment_planned",
        "experiment_started",
        "experiment_observation",
        "experiment_completed",
        "prospective_replication_completed",
    }
)


class LedgerError(Exception):
    """Raised when a ledger operation cannot be completed safely."""


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    entry_count: int
    head_hash: str | None
    errors: tuple[str, ...]


def append_event(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    artifact_hashes: Mapping[str, str] | None = None,
    code_commit: str | None = None,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    timestamp_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """Append one canonical event to the tamper-evident JSONL ledger."""

    _validate_event_type(event_type)
    normalized_payload = _normalize_mapping(payload, "payload")
    normalized_artifacts = _normalize_artifact_hashes(artifact_hashes or {})
    _canonical_json(normalized_payload)
    _canonical_json(normalized_artifacts)

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FileLock(path):
        verification = verify_ledger(path)
        if not verification.valid:
            joined = "; ".join(verification.errors)
            raise LedgerError(f"Cannot append to invalid ledger: {joined}")

        entries = _load_entries(path)
        previous_hash = entries[-1]["entry_hash"] if entries else GENESIS_PREVIOUS_HASH
        entry = _build_entry(
            sequence_number=len(entries) + 1,
            timestamp_utc=_normalize_timestamp(timestamp_utc),
            event_type=event_type,
            payload=normalized_payload,
            artifact_hashes=normalized_artifacts,
            code_commit=code_commit or current_code_commit(),
            previous_entry_hash=previous_hash,
        )
        _append_canonical_line(path, entry)
        return entry


def verify_ledger(ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> LedgerVerification:
    """Verify hashes, sequencing, links, canonical JSON, and complete lines."""

    path = Path(ledger_path)
    if not path.exists():
        return LedgerVerification(valid=True, entry_count=0, head_hash=None, errors=())

    errors: list[str] = []
    expected_sequence = 1
    expected_previous_hash = GENESIS_PREVIOUS_HASH
    head_hash: str | None = None
    valid_entry_count = 0

    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        return LedgerVerification(False, 0, None, (f"Could not read ledger: {exc}",))

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line_errors: list[str] = []
        if not raw_line.endswith(b"\n"):
            line_errors.append("line is truncated or missing final newline")

        try:
            line_text = raw_line.rstrip(b"\r\n").decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"line {line_number}: invalid UTF-8: {exc}")
            break

        try:
            entry = json.loads(line_text)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            break

        if not isinstance(entry, dict):
            line_errors.append("entry must be a JSON object")
            errors.extend(_prefix_errors(line_number, line_errors))
            break

        line_errors.extend(_validate_entry(entry, expected_sequence, expected_previous_hash))
        canonical_line = _canonical_json(_ordered_entry(entry)) + "\n" if not line_errors else None
        if canonical_line is not None and raw_line.decode("utf-8") != canonical_line:
            line_errors.append("entry is not canonical JSON")

        if line_errors:
            errors.extend(_prefix_errors(line_number, line_errors))
            break

        valid_entry_count += 1
        head_hash = str(entry["entry_hash"])
        expected_previous_hash = head_hash
        expected_sequence += 1

    return LedgerVerification(
        valid=not errors,
        entry_count=valid_entry_count,
        head_hash=head_hash,
        errors=tuple(errors),
    )


def read_entries(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read verified ledger entries, optionally returning only the newest entries."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than 0")
    verification = verify_ledger(ledger_path)
    if not verification.valid:
        raise LedgerError("; ".join(verification.errors))
    entries = _load_entries(Path(ledger_path))
    if limit is None:
        return entries
    return entries[-limit:]


def verify_ledger_artifacts(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    *,
    artifact_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Verify path-bearing artifacts referenced by verified ledger payloads."""

    entries = read_entries(ledger_path)
    errors: list[str] = []
    for entry in entries:
        sequence_number = entry["sequence_number"]
        payload = entry["payload"]
        base_root = _ledger_entry_artifact_root(payload, artifact_root)
        for label, reference in _iter_artifact_references(payload):
            path_value = reference.get("path")
            digest = reference.get("sha256")
            if not isinstance(path_value, str) or not isinstance(digest, str):
                errors.append(f"entry {sequence_number} {label}: artifact reference must include path and sha256")
                continue
            path = Path(path_value)
            if not path.is_absolute():
                if base_root is None:
                    errors.append(f"entry {sequence_number} {label}: relative path has no artifact root")
                    continue
                path = base_root / path
            if not path.exists():
                errors.append(f"entry {sequence_number} {label}: missing artifact {path}")
                continue
            actual = _file_sha256(path)
            if actual != digest:
                errors.append(f"entry {sequence_number} {label}: hash mismatch for {path}")
    return tuple(errors)


def current_code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _build_entry(
    *,
    sequence_number: int,
    timestamp_utc: str,
    event_type: str,
    payload: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    code_commit: str,
    previous_entry_hash: str,
) -> dict[str, Any]:
    payload_hash = hash_json(payload)
    entry_without_hash = {
        "sequence_number": sequence_number,
        "timestamp_utc": timestamp_utc,
        "event_type": event_type,
        "payload": dict(payload),
        "artifact_hashes": dict(artifact_hashes),
        "code_commit": code_commit,
        "previous_entry_hash": previous_entry_hash,
        "payload_hash": payload_hash,
    }
    return _ordered_entry({**entry_without_hash, "entry_hash": hash_json(entry_without_hash)})


def _validate_entry(
    entry: Mapping[str, Any],
    expected_sequence: int,
    expected_previous_hash: str,
) -> list[str]:
    errors: list[str] = []
    field_names = set(entry)
    missing = ENTRY_FIELD_SET - field_names
    extra = field_names - ENTRY_FIELD_SET
    if missing:
        errors.append(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"unexpected fields: {', '.join(sorted(extra))}")
    if missing or extra:
        return errors

    if entry["sequence_number"] != expected_sequence:
        errors.append(
            f"sequence_number {entry['sequence_number']!r} does not match expected {expected_sequence}"
        )
    if entry["previous_entry_hash"] != expected_previous_hash:
        errors.append("previous_entry_hash does not match prior entry")
    if entry["event_type"] not in SUPPORTED_EVENT_TYPES:
        errors.append(f"unsupported event_type {entry['event_type']!r}")
    if not isinstance(entry["payload"], dict):
        errors.append("payload must be a JSON object")
    if not isinstance(entry["artifact_hashes"], dict):
        errors.append("artifact_hashes must be a JSON object")
    else:
        for key, value in entry["artifact_hashes"].items():
            if not isinstance(key, str) or not isinstance(value, str):
                errors.append("artifact_hashes keys and values must be strings")
                break
    if not isinstance(entry["code_commit"], str) or not entry["code_commit"]:
        errors.append("code_commit must be a non-empty string")
    if not _is_hash_string(entry["previous_entry_hash"]):
        errors.append("previous_entry_hash must be a 64-character hex string")
    if not _is_hash_string(entry["payload_hash"]):
        errors.append("payload_hash must be a 64-character hex string")
    if not _is_hash_string(entry["entry_hash"]):
        errors.append("entry_hash must be a 64-character hex string")
    try:
        parsed_timestamp = parse_utc(str(entry["timestamp_utc"]))
    except ValueError:
        errors.append("timestamp_utc must be an ISO UTC timestamp")
    else:
        if to_utc_iso(parsed_timestamp) != entry["timestamp_utc"]:
            errors.append("timestamp_utc must be canonical UTC format")

    if errors:
        return errors

    payload_hash = hash_json(entry["payload"])
    if entry["payload_hash"] != payload_hash:
        errors.append("payload_hash does not match payload")

    entry_without_hash = {key: entry[key] for key in ENTRY_FIELDS if key != "entry_hash"}
    entry_hash = hash_json(entry_without_hash)
    if entry["entry_hash"] != entry_hash:
        errors.append("entry_hash does not match entry contents")
    return errors


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line:
            entries.append(json.loads(raw_line))
    return entries


def _append_canonical_line(path: Path, entry: Mapping[str, Any]) -> None:
    line = (_canonical_json(entry) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise LedgerError("Append wrote only part of the ledger line")
        os.fsync(fd)
    finally:
        os.close(fd)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in ENTRY_FIELDS if key in entry}


def _normalize_timestamp(value: datetime | str | None) -> str:
    if value is None:
        return to_utc_iso(utc_now())
    if isinstance(value, datetime):
        return to_utc_iso(ensure_utc(value))
    return to_utc_iso(parse_utc(value))


def _normalize_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = dict(value)
    for key in normalized:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    return normalized


def _normalize_artifact_hashes(value: Mapping[str, str]) -> dict[str, str]:
    normalized = _normalize_mapping(value, "artifact_hashes")
    for key, artifact_hash in normalized.items():
        if not isinstance(artifact_hash, str) or not artifact_hash:
            raise TypeError(f"artifact hash for {key!r} must be a non-empty string")
    return normalized


def _ledger_entry_artifact_root(
    payload: Mapping[str, Any],
    override_root: str | Path | None,
) -> Path | None:
    if override_root is not None:
        return Path(override_root)
    root = payload.get("artifact_root")
    if isinstance(root, str) and root:
        return Path(root)
    return None


def _iter_artifact_references(value: Any, prefix: str = "payload") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield prefix, value
            return
        for key, child in value.items():
            yield from _iter_artifact_references(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_artifact_references(child, f"{prefix}[{index}]")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_event_type(event_type: str) -> None:
    if event_type not in SUPPORTED_EVENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_EVENT_TYPES))
        raise ValueError(f"unsupported event_type {event_type!r}; supported values: {supported}")


def _is_hash_string(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _prefix_errors(line_number: int, errors: Sequence[str]) -> list[str]:
    return [f"line {line_number}: {error}" for error in errors]


class _FileLock:
    def __init__(self, ledger_path: Path) -> None:
        self._lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
        self._file: Any = None

    def __enter__(self) -> "_FileLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._lock_path.open("a+b")
        if self._file.tell() == 0:
            self._file.write(b"\0")
            self._file.flush()
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._file is None:
            return
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()

