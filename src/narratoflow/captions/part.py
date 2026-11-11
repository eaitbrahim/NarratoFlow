"""The part's subtitle document: captions, verses, text cards and overlays together."""

from __future__ import annotations

from narratoflow.audio.timeline import PartTimeline
from narratoflow.captions.ass import (
    AssDocument,
    AssEvent,
    build_styles,
    caption_events,
    strip_tags,
    style_font_id,
)
from narratoflow.captions.overlays import card_events, overlay_events, timeline_scenes
from narratoflow.captions.verse import check_verse_text, verse_events
from narratoflow.errors import RenderError
from narratoflow.models import Language, Part, SceneKind, Series, VisualSource
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.render.assets import FontPack


def _glyph_warnings(
    owner: str, events: list[AssEvent], language: Language, fonts: FontPack
) -> list[str]:
    """One warning per font naming the codepoints its events' text lacks."""
    texts: dict[str, list[str]] = {}
    for e in events:
        texts.setdefault(style_font_id(language, e.style), []).append(strip_tags(e.text))
    warnings: list[str] = []
    for font_id, parts in texts.items():
        missing = fonts.missing_glyphs(font_id, " ".join(parts))
        if missing:
            warnings.append(f"{owner}: caption glyphs missing from {font_id}: {' '.join(missing)}")
    return warnings


def build_part_ass(
    series: Series,
    part: Part,
    timeline: PartTimeline,
    *,
    corpus: QuranCorpus | None,
    fonts: FontPack,
    ink: str,
) -> tuple[AssDocument, list[str]]:
    """The part's `AssDocument` and its glyph warnings; a bad verse raises ``FAIL``."""
    language = series.language
    events: list[AssEvent] = []
    warnings: list[str] = []
    for timing, scene in timeline_scenes(part, timeline):
        if timing.kind is SceneKind.VERSE:
            if corpus is None:
                raise RenderError(f"FAIL: verse scene {scene.id} needs the Quran corpus")
            check_verse_text(scene, timing, corpus, fonts)
            events += verse_events(scene, timing, corpus, language)
            continue
        own = caption_events(timing, language)
        if scene.visual.source is VisualSource.TEXT_CARD:
            own += card_events(scene, timing, language)
        warnings += _glyph_warnings(scene.id, own, language, fonts)
        events += own
    overlays = overlay_events(series, part, timeline)
    warnings += _glyph_warnings("overlays", overlays, language, fonts)
    events += overlays
    return AssDocument(styles=build_styles(language, ink, fonts), events=events), warnings
