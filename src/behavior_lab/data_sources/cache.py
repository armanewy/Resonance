from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import BinaryIO
from uuid import uuid4

from behavior_lab.ledger import _ExclusiveFileLock


@dataclass(frozen=True)
class CachedFile:
    sha256: str
    bytes: int
    path: str
    original_name: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContentAddressedCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self._metadata_lock = self.root / "manifest.jsonl.write.lock"

    def add_file(self, path: str | Path) -> CachedFile:
        source = Path(path)
        digest = _sha256_file(source)
        destination = self.objects / digest[:2] / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        object_lock = destination.with_suffix(".write.lock")
        with _ExclusiveFileLock(object_lock):
            if not destination.exists():
                tmp = destination.parent / f".{digest}.{os.getpid()}.{uuid4().hex}.tmp"
                try:
                    with source.open("rb") as input_handle, tmp.open("xb") as output_handle:
                        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                        output_handle.flush()
                        os.fsync(output_handle.fileno())
                    os.replace(tmp, destination)
                finally:
                    try:
                        tmp.unlink()
                    except FileNotFoundError:
                        pass
            elif _sha256_file(destination) != digest:
                raise RuntimeError(f"cache object hash mismatch at {destination}")
        cached = CachedFile(digest, source.stat().st_size, str(destination), source.name)
        self._write_metadata(cached)
        return cached

    def add_stream(self, stream: BinaryIO, original_name: str) -> CachedFile:
        digest = hashlib.sha256()
        safe_name = Path(original_name).name or "stream"
        tmp = self.root / f".{safe_name}.{os.getpid()}.{uuid4().hex}.tmp"
        size = 0
        try:
            with tmp.open("xb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha = digest.hexdigest()
            destination = self.objects / sha[:2] / sha
            destination.parent.mkdir(parents=True, exist_ok=True)
            object_lock = destination.with_suffix(".write.lock")
            with _ExclusiveFileLock(object_lock):
                if destination.exists():
                    if _sha256_file(destination) != sha:
                        raise RuntimeError(f"cache object hash mismatch at {destination}")
                else:
                    os.replace(tmp, destination)
            cached = CachedFile(sha, size, str(destination), safe_name)
            self._write_metadata(cached)
            return cached
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def inspect(self, sha256: str) -> dict[str, object]:
        path = self.objects / sha256[:2] / sha256
        return {
            "sha256": sha256,
            "exists": path.exists(),
            "path": str(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }

    def _write_metadata(self, cached: CachedFile) -> None:
        metadata = self.root / "manifest.jsonl"
        serialized = json.dumps(cached.to_dict(), sort_keys=True)
        with _ExclusiveFileLock(self._metadata_lock):
            existing = set()
            if metadata.exists():
                existing = {
                    line.strip()
                    for line in metadata.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            if serialized in existing:
                return
            with metadata.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
