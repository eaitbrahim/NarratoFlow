from __future__ import annotations

import math
from fractions import Fraction

import pytest

from narratoflow.models import Language, RightsBasis, ScriptSource, Series
from narratoflow.render import variation as var
from narratoflow.render.layout import STAGE_PADDING_PX, Canvas
from narratoflow.render.variation import (
    MIRROR_P,
    OFFSET_FRAC,
    ROTATION_DEG,
    SCALE_RANGE,
    WEIGHT_RANGE,
    art_size,
    card_frame_strokes,
    choose_variation,
    sketch_to_art,
)
from narratoflow.visuals import marker
from narratoflow.visuals.marker import prepare_marker
from narratoflow.visuals.strokes import Stroke, StrokeSet


def _strokes(w: int = 400, h: int = 400, n: int = 6) -> StrokeSet:
    return StrokeSet(
        width=w,
        height=h,
        strokes=[Stroke(points=[(0.0, i / n), (0.5, (i + 1) / n), (1.0, i / n)]) for i in range(n)],
    )


def test_variation_deterministic_per_video_and_scene() -> None:
    a = choose_variation("h/S", "s1", mirror_safe=True, stroke_count=5)
    b = choose_variation("h/S", "s1", mirror_safe=True, stroke_count=5)
    assert a == b
    assert a != choose_variation("h/S", "s2", mirror_safe=True, stroke_count=5)


def test_different_videos_differ() -> None:
    assert choose_variation("a/x", "s1", mirror_safe=True, stroke_count=5) != choose_variation(
        "b/x", "s1", mirror_safe=True, stroke_count=5
    )


def test_mirror_unsafe_never_mirrored() -> None:
    assert not any(
        choose_variation("h/S", f"s{i}", mirror_safe=False, stroke_count=3).mirrored
        for i in range(500)
    )


def test_mirror_safe_mirrors_about_half() -> None:
    n = sum(
        choose_variation("h/S", f"s{i}", mirror_safe=True, stroke_count=3).mirrored
        for i in range(500)
    )
    assert 0.4 <= n / 500 <= 0.6
    assert MIRROR_P == 0.5


def test_ranges_respected() -> None:
    reversed_seen = 0
    for i in range(300):
        v = choose_variation("h/S", f"s{i}", mirror_safe=True, stroke_count=8)
        assert abs(v.rotation_deg) <= ROTATION_DEG
        assert SCALE_RANGE[0] <= v.scale <= SCALE_RANGE[1]
        assert abs(v.dx) <= OFFSET_FRAC and abs(v.dy) <= OFFSET_FRAC
        assert WEIGHT_RANGE[0] <= v.line_weight_factor <= WEIGHT_RANGE[1]
        assert all(0 <= j < 8 for j in v.reversed_strokes)
        reversed_seen += len(v.reversed_strokes)
    assert 0.4 <= reversed_seen / 2400 <= 0.6


def test_reversal_never_reorders() -> None:
    strokes = _strokes()
    v = choose_variation("h/S", "s1", mirror_safe=True, stroke_count=len(strokes.strokes))
    r = [i in v.reversed_strokes for i in range(len(strokes.strokes))]
    assert any(r) and not all(r)
    plain = prepare_marker(strokes, (100, 100), line_weight_px=3.0, seed=v.wobble_seed)
    flipped = prepare_marker(strokes, (100, 100), line_weight_px=3.0, seed=v.wobble_seed, reverse=r)
    assert len(plain) == len(flipped)
    for i, (p, f) in enumerate(zip(plain, flipped, strict=True)):
        assert f.points == (p.points[::-1] if r[i] else p.points)


def test_sketch_to_art_returns_combined_scale() -> None:
    strokes = _strokes(300, 200)
    art = (1000, 1000)
    base = choose_variation("h/S", "s1", mirror_safe=True, stroke_count=6)
    one = base.model_copy(update={"scale": 1.0})
    _, fit = sketch_to_art(strokes, one, art, 24.0)
    _, got = sketch_to_art(strokes, base, art, 24.0)
    assert got == pytest.approx(fit * base.scale)


@pytest.mark.parametrize("size", [(100, 400), (400, 400), (400, 100)])
def test_transformed_points_inside_padded_art(size: tuple[int, int]) -> None:
    strokes = _strokes(*size)
    canvas = Canvas(Fraction(1, 2))
    art = art_size(canvas)
    pad = STAGE_PADDING_PX * float(canvas.scale)
    lo = pad * var.ZOOM_MAX
    for seed in range(200):
        v = choose_variation("h/S", f"s{seed}", mirror_safe=True, stroke_count=6)
        to_art, _ = sketch_to_art(strokes, v, art, pad)
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            ax, ay = to_art(x, y)
            assert lo - 1e-6 <= ax <= art[0] - lo + 1e-6
            assert lo - 1e-6 <= ay <= art[1] - lo + 1e-6


def test_art_size_is_zoomed_stage() -> None:
    c = Canvas()
    assert art_size(c) == (round(c.stage.w * 1.3), round(c.stage.h * 1.3))


def test_card_frame_is_four_edges_inset() -> None:
    s = card_frame_strokes()
    assert len(s.strokes) == 4
    xs = [x for st in s.strokes for x, _ in st.points]
    assert math.isclose(min(xs), 0.08) and math.isclose(max(xs), 0.92)


def test_video_id_is_single_definition() -> None:
    series = Series(
        title="Scenes",
        channel_slug="history",
        language=Language.ARABIC,
        source=ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN),
        parts=[],
    )
    assert var.video_id is marker.video_id
    assert marker.video_id(series) == f"{series.channel_slug}/{series.title}"
