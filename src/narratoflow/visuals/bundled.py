"""Free bundled icons, usable only for generic idea keys."""

import hashlib
import logging
import tomllib
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from pydantic import BaseModel

from narratoflow.errors import ChannelConfigError, InvalidIdeaKeyError
from narratoflow.library.keys import IdeaKey, parse_idea_key
from narratoflow.planning.ideas import MAX_DESCRIPTION_WORDS

logger = logging.getLogger(__name__)
_FIELDS = frozenset({"file", "description", "license", "source", "sha256"})


class BundledIcon(BaseModel):
    key: str
    file: Path
    license: str
    source: str
    description: str
    sha256: str


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
            icons[str(key)] = _build(raw_key, key, entry, index_path)
        return icons

    def __len__(self) -> int:
        return len(self._icons)

    def get(self, key: IdeaKey) -> BundledIcon | None:
        return self._icons.get(str(key))

    def catalog(self) -> list[tuple[str, str]]:
        """(key, description) pairs sorted by key, like `SketchLibrary.idea_catalog`."""
        return [(key, self._icons[key].description) for key in sorted(self._icons)]

    def search(self, words: Sequence[str]) -> list[BundledIcon]:
        """Icons, sorted by key, whose key and description contain every word."""
        needles = [word.strip().lower() for word in words if word.strip()]
        found: list[BundledIcon] = []
        for key in sorted(self._icons):
            icon = self._icons[key]
            haystack = f"{key} {icon.description}".lower()
            if all(needle in haystack for needle in needles):
                found.append(icon)
        return found


def _text(raw_key: str, entry: object, field: str) -> str:
    value = entry.get(field) if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: {field} is required")
    return value


def confined_file(where: str, rel: Path, index_path: Path, suffix: str) -> Path:
    """The existing `suffix` file `rel` names, which must stay inside the index directory.

    `where` prefixes every error ("<where>: file must be ..."). Shared with the SFX pack.
    """
    if rel.is_absolute() or ".." in rel.parts:
        raise ChannelConfigError(f"{where}: file must be a relative path inside")
    if rel.suffix.lower() != suffix:
        raise ChannelConfigError(f"{where}: file must be a {suffix}")
    path = index_path.parent / rel
    resolved = path.resolve()
    if not resolved.is_relative_to(index_path.parent.resolve()):
        raise ChannelConfigError(f"{where}: file resolves outside the index")
    if not resolved.is_file():
        raise ChannelConfigError(f"{where}: file {str(rel)!r} does not exist")
    return path


def _build(raw_key: str, key: IdeaKey, entry: object, index_path: Path) -> BundledIcon:
    if not isinstance(entry, dict):
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: entry must be a table")
    unknown = sorted(set(entry) - _FIELDS)
    if unknown:
        raise ChannelConfigError(f"bundled icon index entry {raw_key!r}: unknown fields {unknown}")
    license_ = _text(raw_key, entry, "license")
    source = _text(raw_key, entry, "source")
    description = _text(raw_key, entry, "description")
    if len(description.split()) > MAX_DESCRIPTION_WORDS:
        raise ChannelConfigError(
            f"bundled icon index entry {raw_key!r}: description has more than "
            f"{MAX_DESCRIPTION_WORDS} words"
        )
    sha256 = _text(raw_key, entry, "sha256").strip().lower()
    path = confined_file(
        f"bundled icon index entry {raw_key!r}",
        Path(_text(raw_key, entry, "file")),
        index_path,
        ".png",
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ChannelConfigError(
            f"bundled icon index entry '{key}': sha256 mismatch for {path.name}"
        )
    return BundledIcon(
        key=str(key),
        file=path,
        license=license_,
        source=source,
        description=description,
        sha256=sha256,
    )


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
