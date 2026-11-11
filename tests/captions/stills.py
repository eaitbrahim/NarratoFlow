"""libass stills for zone checks: full 1080x1920 frames, ink is gray < 128 on white."""

import dataclasses
from pathlib import Path

import numpy as np

from narratoflow.captions.ass import AssDocument, AssEvent, build_styles
from narratoflow.models import Language
from narratoflow.render.assets import default_fonts
from narratoflow.render.ffmpeg import RenderTools, render_ass_still
from narratoflow.render.layout import (
    FRAME_H,
    FRAME_W,
    LEFT_MARGIN,
    PLATFORM_BOTTOM,
    PLATFORM_TOP,
    RIGHT_RAIL,
    Rect,
)

INK = "#111111"
UNSAFE = (PLATFORM_TOP, RIGHT_RAIL, PLATFORM_BOTTOM, LEFT_MARGIN)


def still(
    tools: RenderTools,
    work: Path,
    language: Language,
    events: list[AssEvent],
    t_ms: int,
    *,
    reveal_all: bool = False,
) -> np.ndarray:
    """Render `events` with the real style set at `t_ms` and return the RGB frame."""
    work.mkdir(parents=True, exist_ok=True)
    fonts = default_fonts()
    fonts.copy_to(work / "fonts")
    styles = build_styles(language, INK, fonts)
    if reveal_all:  # karaoke secondaries opaque: every word drawn, revealed or not
        styles = [dataclasses.replace(s, secondary_alpha=0) for s in styles]
    AssDocument(styles=styles, events=events).write(work / "captions.ass")
    frame = render_ass_still(tools, work, "captions.ass", (FRAME_W, FRAME_H), t_ms)
    assert frame.shape == (FRAME_H, FRAME_W, 3)
    return frame


def ink(frame: np.ndarray) -> np.ndarray:
    mask: np.ndarray = frame.mean(axis=2) < 128
    return mask


def bbox(mask: np.ndarray) -> Rect:
    ys, xs = np.nonzero(mask)
    assert xs.size > 0, "no ink rendered"
    x0, y0 = int(xs.min()), int(ys.min())
    return Rect(x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)


def assert_in_zone(box: Rect, zone: Rect, what: str) -> None:
    assert zone.contains(box), f"{what}: ink {box} leaves {zone}"
    for area in UNSAFE:
        assert not area.intersects(box), f"{what}: ink {box} enters unsafe {area}"
