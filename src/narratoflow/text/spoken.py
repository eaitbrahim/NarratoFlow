"""The spoken form of narration text: numbers spelled out, pronunciations applied.

Captions keep the author's tokens; the voice is given the spoken words.
`SpokenText.groups` maps each author token to how many spoken words it became,
so word timings can be merged back onto the caption tokens.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from narratoflow.audio.words import norm_for_timing, tokens
from narratoflow.models import Language
from narratoflow.text.numbers import spell_number

if TYPE_CHECKING:
    from narratoflow.channels.config import ChannelConfig


@dataclass(frozen=True)
class Lexicon:
    """A frozen pronunciation table: `norm_for_timing(key)` to its replacement phrase."""

    entries: Mapping[str, str]
    EMPTY: ClassVar[Lexicon]

    @classmethod
    def from_entries(cls, entries: Mapping[str, str]) -> Lexicon:
        """Validate and freeze `entries`; errors name the key, never the replacement."""
        table: dict[str, str] = {}
        seen: dict[str, str] = {}
        for key, replacement in entries.items():
            if not key.strip():
                raise ValueError("pronunciation key is empty")
            if key != key.strip() or len(tokens(key)) != 1:
                raise ValueError(f"pronunciation key {key!r} must be one word without spaces")
            norm = norm_for_timing(key)
            if not norm:
                raise ValueError(f"pronunciation key {key!r} is empty once punctuation is ignored")
            if not replacement.strip():
                raise ValueError(f"pronunciation replacement for key {key!r} is empty")
            if norm in seen:
                raise ValueError(
                    f"pronunciation keys {seen[norm]!r} and {key!r} are duplicates"
                    " (they compare equal)"
                )
            seen[norm] = key
            table[norm] = " ".join(replacement.split())
        return cls(MappingProxyType(table))

    def lookup(self, token: str) -> str | None:
        """The replacement for `token`, comparing `norm_for_timing` with the keys."""
        norm = norm_for_timing(token)
        return self.entries.get(norm) if norm else None


Lexicon.EMPTY = Lexicon(MappingProxyType({}))


@dataclass(frozen=True)
class SpokenText:
    """Text for the voice, and how many spoken words each author token became."""

    text: str
    groups: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(g < 1 for g in self.groups):
            raise ValueError("groups must each be at least 1")
        if sum(self.groups) != len(self.text.split()):
            raise ValueError("sum of groups must equal the number of words in text")


def _is_edge(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _split_edges(token: str) -> tuple[str, str, str]:
    start = 0
    end = len(token)
    while start < end and _is_edge(token[start]):
        start += 1
    while end > start and _is_edge(token[end - 1]):
        end -= 1
    return token[:start], token[start:end], token[end:]


def _attach(words: list[str], prefix: str, suffix: str) -> list[str]:
    return (
        [prefix + words[0], *words[1:-1], words[-1] + suffix]
        if len(words) > 1
        else [prefix + words[0] + suffix]
    )


def spoken_text(text: str, language: Language, lexicon: Lexicon) -> SpokenText:
    """`text` as the voice should say it.

    Per whitespace token: a lexicon entry wins (its phrase replaces the token's
    core, edge punctuation and symbols stay attached to the first and last
    word); otherwise a number the speller handles is spelled out; otherwise the
    token is unchanged.
    """
    words: list[str] = []
    groups: list[int] = []
    for token in tokens(text):
        prefix, core, suffix = _split_edges(token)
        replacement = lexicon.lookup(token)
        if replacement is None and core:
            replacement = spell_number(core, language)
        if replacement is None:
            words.append(token)
            groups.append(1)
            continue
        spoken = _attach(replacement.split(), prefix, suffix)
        words.extend(spoken)
        groups.append(len(spoken))
    return SpokenText(" ".join(words), tuple(groups))


def channel_lexicon(channel: ChannelConfig) -> Lexicon:
    """The channel's pronunciation table as a Lexicon."""
    return Lexicon.from_entries(channel.pronunciation)
