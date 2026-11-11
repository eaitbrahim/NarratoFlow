"""Spelling-insensitive normalization used to match quoted text against the corpus."""

import re
import unicodedata

# Harakat and related marks (U+064B-U+065F), superscript alef (U+0670), tatweel (U+0640),
# Qur'anic annotation marks (U+06D6-U+06ED).
_STRIP = re.compile("[\u064b-\u065f\u0670\u0640\u06d6-\u06ed]")
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
        "\ufd3e": None,
        "\ufd3f": None,
    }
)
_SPACES = re.compile(r"\s+")


def match_normalize(text: str) -> str:
    """Return a skeleton form of `text` for comparison; pure and idempotent."""
    out = unicodedata.normalize("NFC", text)
    out = _STRIP.sub("", out).translate(_TRANSLATE)
    out = "".join(ch for ch in out if not unicodedata.category(ch).startswith("P"))
    return _SPACES.sub(" ", out).strip()
