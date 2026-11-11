"""libass checks on full-size 1080x1920 stills; ink is gray < 128 on the white paper."""

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from narratoflow.audio.timeline import SceneTiming, TimelineWord, TimingSource
from narratoflow.captions.ass import (
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


def test_karaoke_reveals_rtl_first_word_rightmost(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    # Unrevealed words are fully transparent, so only lit words leave ink.
    style = dataclasses.replace(_caption_style(Language.ARABIC), secondary_alpha=0xFF)
    # Two equal-width words, so each half of the line holds exactly one of them.
    events = [_event(karaoke([(SEEN3, 0), (SEEN3, 1_000)], 0))]
    early = _ink(_still(render_tools, tmp_path / "early", style, events, 500))
    final = _ink(_still(render_tools, tmp_path / "final", style, events, 1_500))
    line = _bbox(final)
    mid = line.x + line.w // 2
    lit = _bbox(early)
    assert lit.x >= mid, f"first word ink starts at x={lit.x}, left of the line middle {mid}"
    assert not early[:, :mid].any()
    assert final[:, line.x : mid].any()
    assert final[:, mid : line.right].any()


def test_english_caption_two_lines_inside_caption_band(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    texts = ["WWWW"] * 5 + ["WWW"]
    assert len(" ".join(texts)) == 28
    scene = _scene(texts)
    assert len(caption_pages(scene.words)) == 1
    events = caption_events(scene)
    assert len(events) == 1
    frame = _still(render_tools, tmp_path, _caption_style(Language.ENGLISH), events, 800)
    ink = _ink(frame)
    assert _line_runs(ink) == 2
    box = _bbox(ink)
    assert CAPTIONS.contains(box), f"caption ink {box} leaves {CAPTIONS}"


def test_arabic_caption_inside_caption_band(render_tools: RenderTools, tmp_path: Path) -> None:
    assert len(" ".join(ARABIC_SIX)) == 28
    scene = _scene(ARABIC_SIX)
    assert len(caption_pages(scene.words)) == 1
    events = caption_events(scene)
    frame = _still(render_tools, tmp_path, _caption_style(Language.ARABIC), events, 800)
    box = _bbox(_ink(frame))
    assert CAPTIONS.contains(box), f"caption ink {box} leaves {CAPTIONS}"
