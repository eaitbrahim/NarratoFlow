"""Decide whether to reuse a stored sketch, a bundled icon, or generate a new one."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from narratoflow.library.keys import IdeaKey
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchRecord
from narratoflow.visuals.bundled import BundledIcon, BundledIcons


class MatchKind(StrEnum):
    REUSED = "reused"
    BUNDLED = "bundled"
    NEW = "new"


@dataclass(frozen=True)
class MatchResult:
    kind: MatchKind
    key: IdeaKey
    candidates: tuple[SketchRecord, ...] = ()
    icon: BundledIcon | None = None


def match_idea(
    library: SketchLibrary, style_version_id: int, key: IdeaKey, bundled: BundledIcons
) -> MatchResult:
    found = library.find_reusable(style_version_id, str(key))
    if found:
        return MatchResult(MatchKind.REUSED, key, candidates=tuple(found))
    if key.is_generic and (icon := bundled.get(key)) is not None:
        return MatchResult(MatchKind.BUNDLED, key, icon=icon)
    return MatchResult(MatchKind.NEW, key)


def choose_variant(
    candidates: Sequence[SketchRecord], video_id: str, scene_id: str
) -> SketchRecord:
    """Pick a variant deterministically per (video, scene)."""
    if not candidates:
        raise ValueError("no candidates to choose from")
    digest = hashlib.sha256(f"{video_id}:{scene_id}".encode()).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]
