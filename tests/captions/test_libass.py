"""libass checks on full-size 1080x1920 stills; ink is gray < 128 on the white paper."""

import dataclasses
from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

from narratoflow.audio.timeline import SceneTiming, TimelineWord, TimingSource
from narratoflow.captions.ass import (
    CAPTION_MAX_CHARS_AR,
    AssDocument,
    AssEvent,
    AssStyle,
    build_styles,
    caption_events,
    caption_pages,
    karaoke,
)
from narratoflow.models import Language, SceneKind
from narratoflow.render.assets import default_fonts
from narratoflow.render.ffmpeg import RenderTools, render_ass_still
from narratoflow.render.layout import CAPTIONS, FRAME_H, FRAME_W, Rect

INK = "#111111"
SEEN3 = "\u0633\u0633\u0633"
MEEM3 = "\u0645\u0645\u0645"
BEH3 = "\u0628\u0628\u0628"
ALEF = "\u0627"
# Six words of the widest letters (seen, sad, sheen, dad): 28 characters with the spaces.
ARABIC_WIDE = [
    "\u0633\u0633\u0633\u0633",
    "\u0635\u0635\u0635\u0635",
    "\u0634\u0634\u0634",
    "\u0636\u0636\u0636\u0636",
    "\u0633\u0633\u0633\u0633",
    "\u0635\u0635\u0635\u0635",
]
# "dhahaba al-rajul ila al-suq fi al-layl": six words, 28 characters with the spaces.
ARABIC_SIX = [
    "\u0630\u0647\u0628",
    "\u0627\u0644\u0631\u062c\u0644",
    "\u0625\u0644\u0649",
    "\u0627\u0644\u0633\u0648\u0642",
    "\u0641\u064a",
    "\u0627\u0644\u0644\u064a\u0644",
]


def _caption_style(language: Language) -> AssStyle:
    styles = build_styles(language, INK, default_fonts())
    return next(s for s in styles if s.name == "Caption")


def _still(
    tools: RenderTools, work: Path, style: AssStyle, events: list[AssEvent], t_ms: int
) -> np.ndarray:
    work.mkdir(parents=True, exist_ok=True)
    default_fonts().copy_to(work / "fonts")
    AssDocument(styles=[style], events=events).write(work / "captions.ass")
    frame = render_ass_still(tools, work, "captions.ass", (FRAME_W, FRAME_H), t_ms)
    assert frame.shape == (FRAME_H, FRAME_W, 3)
    return frame


def _ink(frame: np.ndarray) -> np.ndarray:
    mask: np.ndarray = frame.mean(axis=2) < 128
    return mask


def _bbox(mask: np.ndarray) -> Rect:
    ys, xs = np.nonzero(mask)
    assert xs.size > 0, "no ink rendered"
    x0, y0 = int(xs.min()), int(ys.min())
    return Rect(x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)


def _components(mask: np.ndarray) -> list[Rect]:
    """Ink components as bounding boxes, largest area first."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    return [
        Rect(
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        for i in found
    ]


def _line_runs(mask: np.ndarray) -> int:
    rows = mask.any(axis=1)
    edges = np.diff(np.concatenate(([0], rows.astype(np.int8), [0])))
    return int((edges == 1).sum())


def _event(text: str) -> AssEvent:
    return AssEvent(0, 3_000, "Caption", text)


def _scene(texts: list[str]) -> SceneTiming:
    words = [
        TimelineWord(text=t, start_ms=i * 100, end_ms=i * 100 + 80) for i, t in enumerate(texts)
    ]
    return SceneTiming(
        scene_id="s1",
        kind=SceneKind.NARRATION,
        start_ms=0,
        end_ms=3_000,
        speech_start_ms=0,
        speech_end_ms=words[-1].end_ms,
        words=words,
        timing_source=TimingSource.PROVIDER,
    )


def test_arabic_letters_join(render_tools: RenderTools, tmp_path: Path) -> None:
    style = _caption_style(Language.ARABIC)
    frame = _still(render_tools, tmp_path, style, [_event(SEEN3)], 500)
    assert len(_components(_ink(frame))) == 1


def test_arabic_trailing_punctuation_on_left(render_tools: RenderTools, tmp_path: Path) -> None:
    style = _caption_style(Language.ARABIC)
    frame = _still(render_tools, tmp_path, style, [_event(SEEN3 + ".")], 500)
    parts = _components(_ink(frame))
    assert len(parts) == 2
    seen, period = parts
    assert period.w * period.h < seen.w * seen.h
    assert period.right <= seen.x


def _column_runs(mask: np.ndarray, gap: int = 6) -> list[tuple[int, int]]:
    """Inclusive x ranges of ink columns, split where more than `gap` columns are empty."""
    xs = np.nonzero(mask.any(axis=0))[0]
    runs: list[tuple[int, int]] = []
    for x in xs.tolist():
        if runs and x - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], x)
        else:
            runs.append((x, x))
    return runs


def test_karaoke_reveals_rtl_first_word_rightmost(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    # Three words of distinct shapes, spoken at 0, 1000 and 2000 ms. Unrevealed words are
    # fully transparent, so only lit words leave ink. RTL: the first spoken word is the
    # rightmost, and the reveal moves right to left one word at a time.
    style = dataclasses.replace(_caption_style(Language.ARABIC), secondary_alpha=0xFF)
    events = [_event(karaoke([(SEEN3, 0), (MEEM3, 1_000), (BEH3, 2_000)], 0))]
    shots = {
        t: _ink(_still(render_tools, tmp_path / str(t), style, events, t))
        for t in (500, 1_500, 2_500)
    }
    words = _column_runs(shots[2_500])
    assert len(words) == 3, words
    third, second, first = words  # left to right on screen

    def lit(t: int, run: tuple[int, int]) -> bool:
        return bool(shots[t][:, run[0] : run[1] + 1].any())

    assert [lit(500, w) for w in (first, second, third)] == [True, False, False]
    assert [lit(1_500, w) for w in (first, second, third)] == [True, True, False]
    assert [lit(2_500, w) for w in (first, second, third)] == [True, True, True]
    # A lit word is drawn whole: its ink matches the final frame's ink for that word.
    x0, x1 = first
    assert np.array_equal(shots[500][:, x0 : x1 + 1], shots[2_500][:, x0 : x1 + 1])


def test_karaoke_leading_offset_delays_first_word(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    # The leading empty {\k25} syllable keeps the first word dark until 250 ms.
    style = dataclasses.replace(_caption_style(Language.ENGLISH), secondary_alpha=0xFF)
    text = karaoke([("one", 250), ("two", 1_000)], 0)
    assert text.startswith("{\\k25}")
    events = [_event(text)]
    before = _ink(_still(render_tools, tmp_path / "a", style, events, 150))
    after = _ink(_still(render_tools, tmp_path / "b", style, events, 500))
    both = _ink(_still(render_tools, tmp_path / "c", style, events, 1_200))
    assert not before.any()
    assert len(_column_runs(after, gap=10)) == 1
    assert len(_column_runs(both, gap=10)) == 2


def _glyph_height_units(font_id: str, ch: str) -> tuple[float, int]:
    """(ink height of `ch`'s glyph in font units, unitsPerEm)."""
    font = TTFont(default_fonts().get(font_id).file)
    try:
        glyphs = font.getGlyphSet()
        pen = BoundsPen(glyphs)
        glyphs[font.getBestCmap()[ord(ch)]].draw(pen)
        assert pen.bounds is not None
        return pen.bounds[3] - pen.bounds[1], int(font["head"].unitsPerEm)
    finally:
        font.close()


def _rendered_em(
    tools: RenderTools, work: Path, language: Language, font_id: str, ch: str
) -> float:
    """The em in pixels libass used: a glyph's ink height over its height in em units."""
    frame = _still(tools, work, _caption_style(language), [_event(ch)], 500)
    height_px = _bbox(_ink(frame)).h
    units, upem = _glyph_height_units(font_id, ch)
    return height_px * upem / units


def test_arabic_em_matches_english_em(render_tools: RenderTools, tmp_path: Path) -> None:
    # Metric: the em libass actually rendered, recovered from one tall, plain glyph per
    # script (Latin "H", Arabic alef: both about 0.71 em, no marks, no joining). Comparing
    # the em, not raw letter heights, separates the font scale from the type design.
    en = _rendered_em(render_tools, tmp_path / "en", Language.ENGLISH, "notosans-bold", "H")
    ar = _rendered_em(render_tools, tmp_path / "ar", Language.ARABIC, "amiri-bold", ALEF)
    assert abs(en - 39.5) <= 0.1 * 39.5, en
    assert abs(ar / en - 1) <= 0.10, (ar, en)


def test_english_caption_two_lines_inside_caption_band(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    texts = ["WWWW"] * 5 + ["WWW"]
    assert len(" ".join(texts)) == 28
    scene = _scene(texts)
    assert len(caption_pages(scene.words)) == 1
    events = caption_events(scene, Language.ENGLISH)
    assert len(events) == 1
    frame = _still(render_tools, tmp_path, _caption_style(Language.ENGLISH), events, 800)
    ink = _ink(frame)
    assert _line_runs(ink) == 2
    box = _bbox(ink)
    assert CAPTIONS.contains(box), f"caption ink {box} leaves {CAPTIONS}"


def test_arabic_caption_inside_caption_band(render_tools: RenderTools, tmp_path: Path) -> None:
    style = _caption_style(Language.ARABIC)
    for name, texts in (("prose", ARABIC_SIX), ("wide", ARABIC_WIDE)):
        assert len(" ".join(texts)) == CAPTION_MAX_CHARS_AR
        scene = _scene(texts)
        assert len(caption_pages(scene.words, CAPTION_MAX_CHARS_AR)) == 1
        events = caption_events(scene, Language.ARABIC)
        frame = _still(render_tools, tmp_path / name, style, events, 800)
        box = _bbox(_ink(frame))
        assert CAPTIONS.contains(box), f"{name}: caption ink {box} leaves {CAPTIONS}"
        # One line: a second line would start a whole line height (= Fontsize) lower.
        assert box.h < style.size, f"{name}: ink {box.h} px tall, wrapped to two lines?"
