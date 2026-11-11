"""On-screen overlays (part label, topic badge, hook/recap/teaser headline) and text cards.

Every author string is cut for display by `truncate` and passed through `escape_text`,
so it can never open an override block. Windows are on the part clock.
"""

from __future__ import annotations

from narratoflow.audio.timeline import PartTimeline, SceneTiming
from narratoflow.audio.words import tokens
from narratoflow.captions.ass import AssEvent, escape_text
from narratoflow.captions.verse import VERSE_POS, localize_digits
from narratoflow.errors import RenderError
from narratoflow.models import Language, Part, Scene, Series

HOOK_MS = 2500
RECAP_MS = 2500
TEASER_MS = 3000
MIN_OVERLAY_MS = 1000
OVERLAY_MAX_CHARS = 60
# An Arabic headline stays on one line: at the Headline em two Amiri lines (line height =
# the win-metric Fontsize) overflow TOP_BAR. The widest letters measure ~20.1 px each, so
# 45 characters fit the 960 px bar (906 px measured with libass).
OVERLAY_MAX_CHARS_AR = 45
BADGE_MAX_CHARS = 24  # TopRight shares the TopLeft row; 60 chars would run into it
CARD_MAX_CHARS = OVERLAY_MAX_CHARS
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


def truncate(text: str, limit: int = OVERLAY_MAX_CHARS) -> str:
    """Display cut: whitespace collapsed, cut at a word boundary, ``\u2026`` appended."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    if text[limit - 1] != " " and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip() + ELLIPSIS


def headline_max_chars(language: Language) -> int:
    """The headline character limit: one line for Arabic, up to two lines for English."""
    return OVERLAY_MAX_CHARS_AR if language is Language.ARABIC else OVERLAY_MAX_CHARS


def part_label(index: int, count: int, language: Language) -> str | None:
    """``Part n of m`` in the channel language and digits; None for a single part."""
    if count == 1:
        return None
    return localize_digits(LABELS[language][0].format(n=index, m=count), language)


def _headline(series: Series, part: Part, duration_ms: int) -> list[AssEvent]:
    _, recap_prefix, teaser_prefix = LABELS[series.language]
    d = duration_ms
    limit = headline_max_chars(series.language)
    h = min(HOOK_MS, d)
    windows: list[tuple[int, int, str]] = []
    if part.hook_text.strip():
        windows.append((0, h, part.hook_text))
    teaser_start = d
    if part.index < part.count and part.teaser_text.strip():
        teaser_start = max(h, d - TEASER_MS)
        windows.append((teaser_start, d, teaser_prefix + part.teaser_text))
    if part.index > 1 and part.recap_text.strip():
        end = min(h + RECAP_MS, teaser_start)
        if end - h >= MIN_OVERLAY_MS:
            windows.append((h, end, recap_prefix + part.recap_text))
    windows = sorted(w for w in windows if w[1] > w[0])
    gaps: list[tuple[int, int]] = []
    t = 0
    for start, end, _ in windows:
        if start > t:
            gaps.append((t, start))
        t = max(t, end)
    if d > t:
        gaps.append((t, d))
    if series.title.strip():
        windows += [(a, b, series.title) for a, b in gaps if b - a >= MIN_OVERLAY_MS]
    return [
        AssEvent(a, b, "Headline", escape_text(truncate(text, limit)))
        for a, b, text in sorted(windows)
    ]


def overlay_events(series: Series, part: Part, timeline: PartTimeline) -> list[AssEvent]:
    """The part label, each scene's topic badge and the headline row."""
    d = timeline.duration_ms
    events: list[AssEvent] = []
    label = part_label(part.index, part.count, series.language)
    if label is not None and d > 0:
        events.append(AssEvent(0, d, "TopLeft", escape_text(label)))
    titles = {s.id: s.title for s in part.scenes}
    for slot in timeline.scenes:
        if slot.scene_id not in titles:
            raise RenderError(f"timeline scene {slot.scene_id} is not in part {part.index}")
        title = titles[slot.scene_id]
        if title.strip() and slot.end_ms > slot.start_ms:
            text = escape_text(truncate(title, BADGE_MAX_CHARS))
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


def card_events(scene: Scene, timing: SceneTiming) -> list[AssEvent]:
    """One ``Card`` event over the scene slot, centred on the stage."""
    text = card_text(scene)
    if not text or timing.end_ms <= timing.start_ms:
        return []
    x, y = VERSE_POS
    fade_in, fade_out = CARD_FADE
    head = f"{{\\an5\\pos({x},{y})\\fad({fade_in},{fade_out})}}"
    body = escape_text(truncate(text, CARD_MAX_CHARS))
    return [AssEvent(timing.start_ms, timing.end_ms, "Card", head + body)]
