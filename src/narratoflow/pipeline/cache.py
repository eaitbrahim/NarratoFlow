"""Content-addressed stage cache: same inputs + config => reuse prior output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(stage: str, version: str, payload: Any) -> str:
    """Stable hash of a stage's name, implementation version and inputs."""
    blob = json.dumps(
        {"stage": stage, "version": version, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class StageCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, stage: str, key: str, suffix: str = "") -> Path:
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}{suffix}"

    def get(self, stage: str, key: str, suffix: str = "") -> Path | None:
        path = self.root / stage / f"{key}{suffix}"
        return path if path.exists() else None
