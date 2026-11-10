"""Free bundled icons, usable only for generic idea keys."""

import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from narratoflow.errors import InvalidIdeaKeyError
from narratoflow.library.keys import IdeaKey, parse_idea_key


class BundledIcon(BaseModel):
    key: str
    file: Path
    license: str
    source: str


class BundledIcons:
    def __init__(self, icons: Mapping[str, BundledIcon]) -> None:
        self._icons = dict(icons)

    @classmethod
    def from_index(cls, index_path: Path) -> "BundledIcons":
        data = tomllib.loads(index_path.read_text(encoding="utf-8"))
        icons: dict[str, BundledIcon] = {}
        for raw_key, entry in data.get("icons", {}).items():
            key = parse_idea_key(raw_key)
            if not key.is_generic:
                raise InvalidIdeaKeyError(f"bundled icon key must be generic: {raw_key!r}")
            icons[str(key)] = BundledIcon(
                key=str(key),
                file=index_path.parent / entry["file"],
                license=entry["license"],
                source=entry["source"],
            )
        return cls(icons)

    def get(self, key: IdeaKey) -> BundledIcon | None:
        return self._icons.get(str(key))
