import itertools
import random
from fractions import Fraction

import pytest

from narratoflow.render import layout as L
from narratoflow.render.layout import Canvas, Rect, frame_of, frame_range


def test_zones_pinned_values() -> None:
    assert Rect(0, 0, 1080, 160) == L.PLATFORM_TOP
    assert Rect(888, 768, 192, 1152) == L.RIGHT_RAIL
    assert Rect(0, 1440, 1080, 480) == L.PLATFORM_BOTTOM
    assert Rect(0, 0, 60, 1920) == L.LEFT_MARGIN
    assert Rect(60, 170, 960, 200) == L.TOP_BAR
    assert Rect(60, 390, 828, 850) == L.STAGE
    assert Rect(60, 1260, 828, 180) == L.CAPTIONS
    assert (L.FPS, L.FRAME_W, L.FRAME_H, L.STAGE_PADDING_PX) == (30, 1080, 1920, 24)


def test_content_zones_disjoint_and_clear_of_unsafe() -> None:
    zones = [L.TOP_BAR, L.STAGE, L.CAPTIONS]
    unsafe = [L.PLATFORM_TOP, L.RIGHT_RAIL, L.PLATFORM_BOTTOM, L.LEFT_MARGIN]
    for i, a in enumerate(zones):
        assert a.x >= 60
        for b in zones[i + 1 :]:
            assert not a.intersects(b)
        for u in unsafe:
            assert not a.intersects(u)


def test_rect_helpers() -> None:
    r = Rect(10, 10, 20, 20)
    assert (r.right, r.bottom) == (30, 30)
    assert r.contains(Rect(12, 12, 5, 5))
    assert not r.contains(Rect(0, 0, 50, 50))
    assert not r.intersects(Rect(30, 10, 5, 5))


def test_canvas_scales_even_and_adjacent(tiny_canvas: Canvas) -> None:
    assert (tiny_canvas.width, tiny_canvas.height) == (180, 320)
    assert tiny_canvas.stage.bottom <= tiny_canvas.captions.y
    for rect in (tiny_canvas.stage, tiny_canvas.captions, tiny_canvas.top_bar):
        assert all(isinstance(v, int) for v in (rect.x, rect.y, rect.w, rect.h))
    for s in (1, Fraction(1, 2), Fraction(1, 3)):
        c = Canvas(Fraction(s))
        assert c.width % 2 == 0 and c.height % 2 == 0


def test_bad_scale_rejected() -> None:
    with pytest.raises(ValueError):
        Canvas(Fraction(1, 4))


def test_frame_of_round_half_up() -> None:
    assert frame_of(0) == 0
    assert frame_of(16) == 0
    assert frame_of(17) == 1
    assert frame_of(1000) == 30
    assert frame_of(90_000) == 2700


def test_frame_time_ms() -> None:
    assert L.frame_time_ms(30) == 1000.0


def test_frame_ranges_contiguous_sum() -> None:
    rng = random.Random(5)  # noqa: S311
    bounds = [0]
    for _ in range(200):
        bounds.append(bounds[-1] + rng.randint(0, 400))
    frames: list[int] = []
    for a, b in itertools.pairwise(bounds):
        frames.extend(frame_range(a, b))
    assert frames == list(range(frame_of(bounds[-1])))
