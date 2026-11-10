"""Guard: a diacritizer may add harakat but must never change the words."""

from __future__ import annotations

import unicodedata

# Arabic harakat, tanwin, shadda, sukun, superscript alef, Qur'anic marks, tatweel.
_DIACRITICS = frozenset(
    chr(c) for c in [*range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE), 0x0640]
)


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics and tatweel, NFC-normalizing first."""
    normalized = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in normalized if ch not in _DIACRITICS)


class SkeletonMismatchError(ValueError):
    """Raised when diacritized output alters the consonantal skeleton."""


def assert_skeleton_preserved(original: str, diacritized: str) -> None:
    a, b = strip_diacritics(original), strip_diacritics(diacritized)
    if a != b:
        diffs = (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
        idx = next(diffs, min(len(a), len(b)))
        raise SkeletonMismatchError(
            f"Diacritizer changed the text at char {idx}: "
            f"{a[max(0, idx - 10) : idx + 10]!r} -> {b[max(0, idx - 10) : idx + 10]!r}"
        )
