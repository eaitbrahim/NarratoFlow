"""On-screen overlays (part label, topic badge, hook/recap/teaser/end-card headline) and text cards.

Every author string is cut for display by `truncate` and passed through `escape_text`,
so it can never open an override block. Windows are on the part clock.
"""

from __future__ import annotations

import unicodedata

from narratoflow.audio.timeline import PartTimeline, SceneTiming
from narratoflow.audio.words import tokens
from narratoflow.captions.ass import EMOJI_MODIFIERS, AssEvent, display_units, escape_text
from narratoflow.captions.verse import VERSE_POS, localize_digits
from narratoflow.errors import RenderError
from narratoflow.models import Language, Part, Scene, SceneKind, Series

HOOK_MS = 2500
RECAP_MS = 2500
TEASER_MS = 3000
END_CARD_MS = 3000
MIN_OVERLAY_MS = 1000
OVERLAY_MAX_CHARS = 60
# An Arabic headline stays on one line: at the Headline em two Amiri lines (line height =
# the win-metric Fontsize) overflow TOP_BAR. Measured with libass, the widest Arabic
# letters (sad, dad) are 23.0 px each, so 40 of them (39 and the ellipsis) fit the 960 px
# bar. Latin letters in Amiri are wider (W 28.5 px): `truncate` counts each as 5/4.
OVERLAY_MAX_CHARS_AR = 40
# The widest token one headline line holds (English wraps to two lines; W is 29.2 px).
HEADLINE_LINE_CHARS = 30
BADGE_MAX_CHARS = 24  # TopRight shares the TopLeft row; 60 chars would run into it
CARD_MAX_CHARS = OVERLAY_MAX_CHARS
# The widest token one card line holds in the 780 px wrap width (W 56 px in Noto Sans
# Bold, dad 43.7 px in Amiri Bold).
CARD_LINE_CHARS = {Language.ENGLISH: 13, Language.ARABIC: 16}
CARD_FALLBACK_WORDS = 6
CARD_FADE = (150, 200)
ELLIPSIS = "\u2026"

LABELS = {
    Language.ENGLISH: ("Part {n} of {m}", "Previously: ", "Next: "),
    Language.ARABIC: (
        "\u0627\u0644\u062c\u0632\u0621 {n} \u0645\u0646 {m}",
        "\u0633\u0627\u0628\u0642\u064b\u0627: ",
        "\u0627\u0644\u062a\u0627\u0644\u064a: ",
    ),
}

_ZWJ, _ZWNJ, _TATWEEL = "\u200d", "\u200c", "\u0640"


def _boundary(text: str, i: int) -> bool:
    """True when cutting `text` before index `i` keeps every grapheme whole."""
    if i <= 0 or i >= len(text):
        return True
    nxt, prev = text[i], text[i - 1]
    if nxt in (_ZWJ, _ZWNJ, _TATWEEL) or prev == _ZWJ:
        return False  # joiners and a tatweel stay with the letter before; ZWJ glues both
    return unicodedata.category(nxt) not in ("Mn", "Me") and ord(nxt) not in EMOJI_MODIFIERS


def _cut(word: str, room: int, language: Language) -> str:
    """The longest grapheme-whole prefix of `word` within `room` quarter units."""
    used = i = 0
    while i < len(word) and used + display_units(word[i], language) <= room:
        used += display_units(word[i], language)
        i += 1
    while not _boundary(word, i):
        i -= 1
    return word[:i]


def truncate(
    text: str,
    limit: int = OVERLAY_MAX_CHARS,
    *,
    line: int | None = None,
    language: Language = Language.ENGLISH,
) -> str:
    """Display cut: whitespace collapsed, cut at a word boundary, ``\u2026`` appended.

    Widths are counted by `display_units`. A token wider than `line` (one display line) is cut
    with the ellipsis and ends the text, so an unbreakable string cannot run sideways out
    of its zone. A cut never splits a grapheme.
    """
    words = text.split()
    budget = limit * 4
    line_room = min(limit if line is None else line, limit) * 4
    ell = display_units(ELLIPSIS, language)
    kept: list[str] = []
    used = 0
    for word in words:
        sep = 4 if kept else 0
        width = sum(display_units(ch, language) for ch in word)
        if width > line_room:
            cut = _cut(word, min(line_room, budget - used - sep) - ell, language)
            if cut:
                return " ".join([*kept, cut]) + ELLIPSIS
            break
        if used + sep + width > budget:
            break
        kept.append(word)
        used += sep + width
    else:
        return " ".join(kept)
    while kept and used + ell > budget:
        last = kept.pop()
        used -= sum(display_units(ch, language) for ch in last) + (4 if kept else 0)
    if not kept:
        kept = [_cut(words[0], line_room - ell, language)]
    return " ".join(kept) + ELLIPSIS


def headline_max_chars(language: Language) -> int:
    """The headline character limit: one line for Arabic, up to two lines for English."""
    return OVERLAY_MAX_CHARS_AR if language is Language.ARABIC else OVERLAY_MAX_CHARS


def headline_line_chars(language: Language) -> int:
    """The widest token one headline line holds."""
    return OVERLAY_MAX_CHARS_AR if language is Language.ARABIC else HEADLINE_LINE_CHARS


def part_label(index: int, count: int, language: Language) -> str | None:
    """``Part n of m`` in the channel language and digits; None for a single part."""
    if count == 1:
        return None
    return localize_digits(LABELS[language][0].format(n=index, m=count), language)


def _tail_window(part: Part, d: int, h: int, teaser_prefix: str) -> tuple[int, int, str] | None:
    """The closing window: the teaser, or on the final part the end card; None if no room."""
    if d - h < MIN_OVERLAY_MS:
        return None
    if part.index < part.count:
        text, ms, prefix = part.teaser_text, TEASER_MS, teaser_prefix
    else:
        text, ms, prefix = part.end_card_text, END_CARD_MS, ""
    if not text.strip():
        return None
    return max(h, d - ms), d, prefix + text


def _gaps(windows: list[tuple[int, int, str]], duration_ms: int) -> list[tuple[int, int]]:
    """The spans of `[0, duration_ms)` that no sorted window covers."""
    gaps: list[tuple[int, int]] = []
    t = 0
    for start, end, _ in windows:
        if start > t:
            gaps.append((t, start))
        t = max(t, end)
    if duration_ms > t:
        gaps.append((t, duration_ms))
    return gaps


def _headline(series: Series, part: Part, duration_ms: int) -> list[AssEvent]:
    _, recap_prefix, teaser_prefix = LABELS[series.language]
    d = duration_ms
    language = series.language
    limit, line = headline_max_chars(language), headline_line_chars(language)
    h = min(HOOK_MS, d)
    windows: list[tuple[int, int, str]] = []
    if part.hook_text.strip():
        windows.append((0, h, part.hook_text))
    teaser_start = d
    tail = _tail_window(part, d, h, teaser_prefix)
    if tail is not None:
        teaser_start = tail[0]
        windows.append(tail)
    if part.index > 1 and part.recap_text.strip():
        end = min(h + RECAP_MS, teaser_start)
        if end - h >= MIN_OVERLAY_MS:
            windows.append((h, end, recap_prefix + part.recap_text))
    windows = sorted(w for w in windows if w[1] > w[0])
    gaps = _gaps(windows, d)
    if series.title.strip():
        windows += [(a, b, series.title) for a, b in gaps if b - a >= MIN_OVERLAY_MS]
    return [
        AssEvent(a, b, "Headline", escape_text(truncate(text, limit, line=line, language=language)))
        for a, b, text in sorted(windows)
    ]


def timeline_scenes(part: Part, timeline: PartTimeline) -> list[tuple[SceneTiming, Scene]]:
    """Each timeline slot with its part scene; a slot naming another scene raises."""
    scenes = {s.id: s for s in part.scenes}
    pairs: list[tuple[SceneTiming, Scene]] = []
    for slot in timeline.scenes:
        scene = scenes.get(slot.scene_id)
        if scene is None:
            raise RenderError(f"timeline scene {slot.scene_id} is not in part {part.index}")
        pairs.append((slot, scene))
    return pairs


def overlay_events(series: Series, part: Part, timeline: PartTimeline) -> list[AssEvent]:
    """The part label, each non-verse scene's topic badge and the headline row."""
    d = timeline.duration_ms
    events: list[AssEvent] = []
    label = part_label(part.index, part.count, series.language)
    if label is not None and d > 0:
        events.append(AssEvent(0, d, "TopLeft", escape_text(label)))
    for slot, scene in timeline_scenes(part, timeline):
        # A verse scene's reference line (localized digits) already labels it.
        if scene.kind is SceneKind.VERSE:
            continue
        title = scene.title
        if title.strip() and slot.end_ms > slot.start_ms:
            text = escape_text(truncate(title, BADGE_MAX_CHARS, language=series.language))
            events.append(AssEvent(slot.start_ms, slot.end_ms, "TopRight", text))
    return events + _headline(series, part, d)


def card_text(scene: Scene) -> str:
    """The card's text: `card_text`, else the title, else the first caption words."""
    for text in (scene.card_text, scene.title):
        if text.strip():
            return " ".join(text.split())
    words = tokens(scene.narration.caption_text or scene.narration.source_text)
    shown = " ".join(words[:CARD_FALLBACK_WORDS])
    return shown + ELLIPSIS if len(words) > CARD_FALLBACK_WORDS else shown


def card_events(scene: Scene, timing: SceneTiming, language: Language) -> list[AssEvent]:
    """One ``Card`` event over the scene slot, centred on the stage."""
    text = card_text(scene)
    if not text or timing.end_ms <= timing.start_ms:
        return []
    x, y = VERSE_POS
    fade_in, fade_out = CARD_FADE
    head = f"{{\\an5\\pos({x},{y})\\fad({fade_in},{fade_out})}}"
    line = CARD_LINE_CHARS[language]
    body = escape_text(truncate(text, CARD_MAX_CHARS, line=line, language=language))
    return [AssEvent(timing.start_ms, timing.end_ms, "Card", head + body)]
