"""Spelling-insensitive normalization used to match quoted text against the corpus."""

import re
import unicodedata

# Deleted outright:
# harakat and related marks (U+064B-U+065F), superscript alef (U+0670), tatweel (U+0640),
# Qur'anic annotation marks (U+06D6-U+06ED), honorific/extended marks (U+0610-U+061A,
# U+08D3-U+08FF), and invisible format marks (U+200C-U+200F, U+061C, U+FEFF).
_STRIP = re.compile(
    "[\u064b-\u065f\u0670\u0640\u06d6-\u06ed\u0610-\u061a\u08d3-\u08ff\u200c-\u200f\u061c\ufeff]"
)
_TRANSLATE = str.maketrans(
    {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
        "\u0624": "\u0621",
        "\u0626": "\u0621",
    }
)
_SPACES = re.compile(r"\s+")


def is_punctuation(ch: str) -> bool:
    """True for characters `match_normalize` treats as word separators (Unicode P*)."""
    return unicodedata.category(ch).startswith("P")


def match_normalize(text: str) -> str:
    """Return a skeleton form of `text` for comparison; pure and idempotent.

    Punctuation (including the ornate brackets U+FD3E/U+FD3F and Arabic comma,
    semicolon, question mark and full stop) becomes a space, so `a:b` stays two words.
    """
    out = unicodedata.normalize("NFC", text)
    out = _STRIP.sub("", out).translate(_TRANSLATE)
    out = "".join(" " if is_punctuation(ch) else ch for ch in out)
    return _SPACES.sub(" ", out).strip()
