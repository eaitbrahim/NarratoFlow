"""Content-addressed blob storage for sketch files."""

import hashlib
import os
from pathlib import Path

from narratoflow.errors import LibraryError


class BlobStore:
    """Stores bytes under ``root/<hash[:2]>/<hash><suffix>``, deduplicated by sha256."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _location(self, blob_hash: str, suffix: str) -> Path:
        return self._root / blob_hash[:2] / f"{blob_hash}{suffix}"

    def put(self, data: bytes, suffix: str) -> str:
        blob_hash = hashlib.sha256(data).hexdigest()
        target = self._location(blob_hash, suffix)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        return blob_hash

    def path(self, blob_hash: str, suffix: str) -> Path:
        target = self._location(blob_hash, suffix)
        if not target.is_file():
            raise LibraryError(f"blob not found: {blob_hash}{suffix}")
        return target
