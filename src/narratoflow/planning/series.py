"""Series splitting: pack scenes into short-video parts with on-screen text.

Hook, recap, teaser and end card are on-screen text only; they are never spoken and never
alter the author's narration. A scene is never split across parts.
"""

from __future__ import annotations

import string

from narratoflow.models import Language, Part, Scene, SceneKind

END_CARD_DEFAULTS = {
    Language.ENGLISH: "Follow for more",
    Language.ARABIC: (
        "\u062a\u0627\u0628\u0639\u0648\u0646\u0627 \u0644\u0644\u0645\u0632\u064a\u062f"
    ),
}
NARRATION_WPM = {Language.ARABIC: 120, Language.ENGLISH: 150}
NARRATION_PAUSE_S = 0.4
VERSE_WPM = 80
VERSE_BUFFER_S = 0.8
TAIL_MERGE_SLACK = 1.15
HOOK_FALLBACK_WORDS = 8
EPSILON = 1e-9
# U+200B-U+200F, U+061C, U+2066-U+2069, U+FEFF: zero-width and bidi marks.
_INVISIBLE = "\u200b\u200c\u200d\u200e\u200f\u061c\u2066\u2067\u2068\u2069\ufeff"

# U+060C, U+061B: Arabic comma and semicolon.
_OPEN_ENDINGS = (":", "\u060c", ",", "\u061b", ";")
# . ! ? U+061F U+2026 U+00BB " U+201D ) U+FD3E
_CLOSED_ENDINGS = (".", "!", "?", "\u061f", "\u2026", "\u00bb", '"', "\u201d", ")", "\ufd3e")
_TRIM = "".join(_OPEN_ENDINGS + _CLOSED_ENDINGS) + _INVISIBLE + string.whitespace


def _count_words(text: str) -> int:
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def estimate_scene_seconds(scene: Scene, language: Language) -> float:
    n = _count_words(scene.narration.source_text)
    if scene.kind is SceneKind.VERSE:
        return n / VERSE_WPM * 60 + VERSE_BUFFER_S
    return n / NARRATION_WPM[language] * 60 + NARRATION_PAUSE_S


def _is_unfinished(text: str) -> bool:
    stripped = text.rstrip(_INVISIBLE + string.whitespace)
    if not stripped:
        return True
    if stripped.endswith(_OPEN_ENDINGS):
        return True
    return not stripped.endswith(_CLOSED_ENDINGS)


def end_card_text(channel_end_card: str | None, language: Language) -> str:
    """The channel's end card; None means the language default, "" turns it off."""
    if channel_end_card is None:
        return END_CARD_DEFAULTS[language]
    return channel_end_card.strip()


def _hook(scenes: list[Scene]) -> str:
    for scene in scenes:
        if scene.title:
            return scene.title
    for scene in scenes:
        if scene.kind is SceneKind.NARRATION:
            tokens = scene.narration.source_text.split()[:HOOK_FALLBACK_WORDS]
            snippet = " ".join(tokens).rstrip(
                "".join(_OPEN_ENDINGS + _CLOSED_ENDINGS) + _INVISIBLE + string.whitespace
            )
            return snippet + "\u2026"
    return ""


def _group_scenes(timed: list[Scene], max_s: int) -> tuple[list[list[Scene]], list[bool]]:
    """Greedy groups of at most `max_s` seconds; the flag marks a lone scene over the max."""
    groups: list[list[Scene]] = []
    current: list[Scene] = []
    total = 0.0
    for scene in timed:
        d = scene.est_duration_s
        if current and total + d > max_s + EPSILON:
            groups.append(current)
            current, total = [], 0.0
        current.append(scene)
        total += d
    groups.append(current)
    over = [len(g) == 1 and g[0].est_duration_s > max_s for g in groups]
    return groups, over


def _merge_short_tail(groups: list[list[Scene]], over: list[bool], min_s: int, max_s: int) -> None:
    """Fold a very short last group into the one before it when that still fits."""
    if len(groups) < 2:
        return
    tail = sum(s.est_duration_s for s in groups[-1])
    prev = sum(s.est_duration_s for s in groups[-2])
    if tail < min_s / 2 and prev + tail <= max_s * TAIL_MERGE_SLACK:
        last = groups.pop()
        groups[-1] = groups[-1] + last
        over.pop()


def _part_flags(group: list[Scene], is_over: bool, is_last: bool) -> list[str]:
    flags = ["over_max"] if is_over else []
    if is_last:
        narrations = [s for s in group if s.kind is SceneKind.NARRATION]
        if narrations and _is_unfinished(narrations[-1].narration.source_text):
            flags.append("unfinished_ending")
    return flags


def split_into_parts(
    scenes: list[Scene], language: Language, *, min_s: int, max_s: int, end_card: str = ""
) -> list[Part]:
    if not scenes:
        return []
    timed = [
        s.model_copy(update={"est_duration_s": estimate_scene_seconds(s, language)}) for s in scenes
    ]

    groups, over = _group_scenes(timed, max_s)
    _merge_short_tail(groups, over, min_s, max_s)

    hooks = [_hook(g) for g in groups]
    count = len(groups)
    parts = [
        Part(
            index=i + 1,
            count=count,
            hook_text=hooks[i],
            recap_text=hooks[i - 1] if i > 0 else "",
            teaser_text=hooks[i + 1] if i < count - 1 else "",
            end_card_text=end_card if i == count - 1 else "",
            scenes=group,
            flags=_part_flags(group, over[i], i == count - 1),
        )
        for i, group in enumerate(groups)
    ]
    return parts
