"""Free bundled icons, usable only for generic idea keys."""

import logging
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

from pydantic import BaseModel

from narratoflow.errors import ChannelConfigError, InvalidIdeaKeyError
from narratoflow.library.keys import IdeaKey, parse_idea_key

logger = logging.getLogger(__name__)
_FIELDS = frozenset({"file", "license", "source"})


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
        root = index_path.parent.resolve()
        icons: dict[str, BundledIcon] = {}
        for raw_key, entry in data.get("icons", {}).items():
            try:
                key = parse_idea_key(raw_key)
            except InvalidIdeaKeyError as exc:
                raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: {exc}") from exc
            if not key.is_generic:
                raise ChannelConfigError(
                    f"bundled icon index entry {raw_key!r}: key must be generic"
                )
            if str(key) in icons:
                raise ChannelConfigError(
                    f"bundled icon index entry {raw_key!r}: duplicate key {str(key)!r}"
                )
            icons[str(key)] = _build(raw_key, key, entry, index_path, root)
        return icons

    def get(self, key: IdeaKey) -> BundledIcon | None:
        return self._icons.get(str(key))


def _text(raw_key: str, entry: object, field: str) -> str:
    value = entry.get(field) if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: {field} is required")
    return value


def _build(raw_key: str, key: IdeaKey, entry: object, index_path: Path, root: Path) -> BundledIcon:
    if not isinstance(entry, dict):
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: entry must be a table")
    unknown = sorted(set(entry) - _FIELDS)
    if unknown:
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: unknown fields {unknown}")
    license_ = _text(raw_key, entry, "license")
    source = _text(raw_key, entry, "source")
    rel = Path(_text(raw_key, entry, "file"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ChannelConfigError(
            f"bundled icon index entry {raw_key!r}: file must be a relative path inside"
        )
    path = index_path.parent / rel
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ChannelConfigError(
            f"bundled icon index entry {raw_key!r}: file resolves outside the index"
        )
    if not resolved.is_file():
        raise ChannelConfigError(
            f"bundled icon index entry {raw_key!r}: file {str(rel)!r} does not exist"
        )
    return BundledIcon(key=str(key), file=path, license=license_, source=source)


def _packaged_index() -> Path:
    return Path(str(resources.files("narratoflow") / "assets" / "icons" / "index.toml"))


def _dev_index() -> Path:
    return Path(__file__).parents[3] / "assets" / "icons" / "index.toml"


def default_bundled_icons() -> BundledIcons:
    """The repository's bundled icon index, or no icons when it is not present."""
    for index in (_packaged_index(), _dev_index()):
        if index.is_file():
            return BundledIcons.from_index(index)
    logger.warning("bundled icon index not found; bundled icons disabled")
    return BundledIcons({})
