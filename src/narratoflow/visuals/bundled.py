"""Free bundled icons, usable only for generic idea keys."""

import logging
import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from narratoflow.errors import ChannelConfigError, InvalidIdeaKeyError
from narratoflow.library.keys import IdeaKey, parse_idea_key

logger = logging.getLogger(__name__)


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
        try:
            return cls(cls._parse(index_path))
        except (tomllib.TOMLDecodeError, KeyError, TypeError, OSError, UnicodeDecodeError) as exc:
            raise ChannelConfigError(f"bundled icon index {index_path} is broken: {exc!r}") from exc

    @staticmethod
    def _parse(index_path: Path) -> dict[str, BundledIcon]:
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
        return icons

    def get(self, key: IdeaKey) -> BundledIcon | None:
        return self._icons.get(str(key))


def _dev_index() -> Path:
    return Path(__file__).parents[3] / "assets" / "icons" / "index.toml"


def default_bundled_icons() -> BundledIcons:
    """The repository's bundled icon index, or no icons when it is not present."""
    index = _dev_index()
    if not index.is_file():
        logger.warning("bundled icon index not found; bundled icons disabled")
        return BundledIcons({})
    return BundledIcons.from_index(index)
