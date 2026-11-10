"""Canonical idea keys of the form ``object|context``."""

import re
from dataclasses import dataclass

from narratoflow.errors import InvalidIdeaKeyError

GENERIC = "generic"
_PART = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SEPARATORS = re.compile(r"[ _]+")


@dataclass(frozen=True)
class IdeaKey:
    object: str
    context: str

    def __str__(self) -> str:
        return f"{self.object}|{self.context}"

    @property
    def is_generic(self) -> bool:
        return self.context == GENERIC


def parse_idea_key(raw: str) -> IdeaKey:
    """Normalize and validate a raw idea key."""
    normalized = _SEPARATORS.sub("-", raw.strip().lower())
    parts = normalized.split("|")
    if len(parts) != 2 or not all(_PART.fullmatch(p) for p in parts):
        raise InvalidIdeaKeyError(f"invalid idea key: {raw!r}")
    return IdeaKey(parts[0], parts[1])
