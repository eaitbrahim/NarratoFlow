"""The spoken form of narration text: numbers spelled out, pronunciations applied.

Captions keep the author's tokens; the voice is given the spoken words.
`SpokenText.groups` maps each author token to how many spoken words it became,
so word timings can be merged back onto the caption tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from narratoflow.audio.words import norm_for_timing, tokens
from narratoflow.models import Language
from narratoflow.text.amounts import spell_amount
from narratoflow.text.numbers import spell_number
from narratoflow.text.tashkeel.verify import is_diacritic

if TYPE_CHECKING:
    from narratoflow.channels.config import ChannelConfig


def _is_edge(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in "PS" or cat == "Cf"


def _split_edges(token: str) -> tuple[str, str, str]:
    start = 0
    end = len(token)
    while start < end and _is_edge(token[start]):
        start += 1
    while end > start and _is_edge(token[end - 1]):
        end -= 1
    return token[:start], token[start:end], token[end:]


def _exact_core(token: str) -> str:
    """The token without edge punctuation or diacritics; case is kept."""
    core = unicodedata.normalize("NFD", _split_edges(token)[1])
    return "".join(ch for ch in core if not is_diacritic(ch) and unicodedata.category(ch)[0] != "M")


def _is_exact_key(core: str) -> bool:
    """Keys with an uppercase letter or internal punctuation match case-sensitively."""
    return any(
        unicodedata.category(ch) in ("Lu", "Lt") or unicodedata.category(ch)[0] == "P"
        for ch in core
    )


@dataclass(frozen=True)
class Lexicon:
    """A frozen pronunciation table.

    `entries` maps `norm_for_timing(key)` to its phrase for loose keys (no
    uppercase letter, no internal punctuation). `exact` maps the case-sensitive
    core of the other keys (for example "U.S." or "WHO") to its phrase.
    """

    entries: Mapping[str, tuple[str, str]]
    exact: Mapping[str, tuple[str, str]] = MappingProxyType({})
    EMPTY: ClassVar[Lexicon]

    @classmethod
    def from_entries(cls, entries: Mapping[str, str]) -> Lexicon:
        """Validate and freeze `entries`; errors name the key, never the replacement."""
        loose: dict[str, tuple[str, str]] = {}
        exact: dict[str, tuple[str, str]] = {}
        seen: dict[tuple[bool, str], str] = {}
        for key, replacement in entries.items():
            if not key.strip():
                raise ValueError("pronunciation key is empty")
            if key != key.strip() or len(tokens(key)) != 1:
                raise ValueError(f"pronunciation key {key!r} must be one word without spaces")
            if any(unicodedata.category(ch)[0] == "S" for ch in key):
                raise ValueError(f"pronunciation key {key!r} contains a symbol character")
            norm = norm_for_timing(key)
            if not norm:
                raise ValueError(f"pronunciation key {key!r} is empty once punctuation is ignored")
            if not replacement.strip():
                raise ValueError(f"pronunciation replacement for key {key!r} is empty")
            core = _exact_core(key)
            is_exact = _is_exact_key(core)
            ident = (is_exact, core if is_exact else norm)
            if ident in seen:
                raise ValueError(
                    f"pronunciation keys {seen[ident]!r} and {key!r} are duplicates"
                    " (they compare equal)"
                )
            seen[ident] = key
            phrase = " ".join(replacement.split())
            (exact if is_exact else loose)[ident[1]] = (phrase, _split_edges(key)[2])
        return cls(MappingProxyType(loose), MappingProxyType(exact))

    def __hash__(self) -> int:
        return hash((frozenset(self.entries.items()), frozenset(self.exact.items())))

    def match(self, token: str) -> tuple[str, str] | None:
        """`(phrase, key_tail)` for `token`: an exact key first, then a loose one.

        `key_tail` is the key's own trailing punctuation (the "." of "U.S."), which
        belongs to the key, not to the sentence.
        """
        core = _exact_core(token)
        if core in self.exact:
            return self.exact[core]
        norm = norm_for_timing(token)
        return self.entries.get(norm) if norm else None

    def lookup(self, token: str) -> str | None:
        """The replacement phrase for `token`, or None."""
        hit = self.match(token)
        return None if hit is None else hit[0]


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


def _attach(words: list[str], prefix: str, suffix: str) -> list[str]:
    return (
        [prefix + words[0], *words[1:-1], words[-1] + suffix]
        if len(words) > 1
        else [prefix + words[0] + suffix]
    )


def _continues(match: re.Match[str]) -> bool:
    """The sentence goes on after this token: the next word starts lowercase."""
    rest = match.string[match.end() :].lstrip()
    return bool(rest) and rest[0].islower()


def spoken_text(text: str, language: Language, lexicon: Lexicon) -> SpokenText:
    """`text` as the voice should say it.

    Per whitespace token: a lexicon entry wins (its phrase replaces the token's
    core, edge punctuation and symbols stay attached to the first and last
    word, except the key's own trailing punctuation when a lowercase word
    follows, as in "the U.S. economy"); otherwise a number the speller
    handles is spelled out; otherwise the token is unchanged.
    """
    groups: list[int] = []

    def rewrite(match: re.Match[str]) -> str:
        token = match.group()
        prefix, core, suffix = _split_edges(token)
        hit = lexicon.match(token)
        replacement = None if hit is None else hit[0]
        if hit is not None and hit[1] and suffix.startswith(hit[1]) and _continues(match):
            suffix = suffix[len(hit[1]) :]  # "the U.S. economy": the key's "." ends no sentence
        if replacement is None and core:
            amount = spell_amount(prefix, core, suffix, language)
            if amount is not None:
                prefix, replacement, suffix = amount
            else:
                replacement = spell_number(core, language)
        if replacement is None:
            groups.append(1)
            return token
        spoken = _attach(replacement.split(), prefix, suffix)
        groups.append(len(spoken))
        return " ".join(spoken)

    # Only the tokens that change are rewritten: every separator survives as written.
    return SpokenText(re.sub(r"\S+", rewrite, text), tuple(groups))


def channel_lexicon(channel: ChannelConfig) -> Lexicon:
    """The channel's pronunciation table as a Lexicon."""
    return Lexicon.from_entries(channel.pronunciation)
