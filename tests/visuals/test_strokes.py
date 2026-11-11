import io
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fixtures.images import (
    brick,
    calligraphy,
    label_map,
    line_drawing_scroll,
    noisy_line_drawing,
    road_map,
    scroll,
)
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

from narratoflow.errors import NarratoFlowError, StrokeDataError
from narratoflow.visuals.lineart import binarize
from narratoflow.visuals.strokes import (
    PACING_STROKES,
    Stroke,
    StrokeSet,
    extract_strokes,
)

SIZE = 256


def _ink(draw_fn: object, size: int = SIZE) -> np.ndarray:
    image = Image.new("L", (size, size), 255)
    draw_fn(ImageDraw.Draw(image))  # type: ignore[operator]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return binarize(buffer.getvalue())


def _px_length(stroke: Stroke, width: int, height: int) -> float:
    pts = [(x * width, y * height) for x, y in stroke.points]
    return sum(math.dist(a, b) for a, b in itertools.pairwise(pts))


def _is_dot(stroke: Stroke, width: int) -> bool:
    return len(stroke.points) == 2 and _px_length(stroke, width, width) <= 0.51


def _render(strokes: StrokeSet, thickness: int) -> np.ndarray:
    image = np.zeros((strokes.height, strokes.width), np.uint8)
    for stroke in strokes.strokes:
        pts = np.array(
            [[round(x * strokes.width), round(y * strokes.height)] for x, y in stroke.points],
            np.int32,
        )
        cv2.polylines(image, [pts], False, 255, thickness=max(1, thickness), lineType=cv2.LINE_8)
        if stroke.width is not None:  # Task 5 draws a dot as a filled disc
            centre = (int(pts[0][0]), int(pts[0][1]))
            cv2.circle(image, centre, max(1, round(stroke.width * strokes.width / 2)), 255, -1)
    return image > 0


def _line_width(ink: np.ndarray) -> int:
    depth = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    skeleton = skeletonize(ink)  # type: ignore[no-untyped-call]
    return max(1, round(2 * float(depth[skeleton].mean()) - 1))


def _ink_coverage(ink: np.ndarray, strokes: StrokeSet) -> float:
    """Share of ink pixels within 2 px (at 512, scaled) of a stroke drawn at the line width."""
    width = _line_width(ink)
    tolerance = 2 * min(ink.shape) / 512
    centre = _render(strokes, 1)
    to_stroke = cv2.distanceTransform((~centre).astype(np.uint8), cv2.DIST_L2, 3)
    return float((to_stroke[ink] <= tolerance + width / 2).mean())


def test_square_outline_is_one_closed_stroke_or_at_most_four() -> None:
    ink = _ink(lambda d: d.rectangle((60, 60, 190, 190), outline=0, width=3))
    result = extract_strokes(ink)
    assert 1 <= len(result.strokes) <= 4
    total = sum(_px_length(s, SIZE, SIZE) for s in result.strokes)
    assert total == pytest.approx(4 * 130, rel=0.1)
    if len(result.strokes) == 1:
        first, last = result.strokes[0].points[0], result.strokes[0].points[-1]
        assert math.dist(first, last) * SIZE < 3


def test_l_shape_is_one_stroke() -> None:
    ink = _ink(lambda d: d.line((60, 40, 60, 200, 200, 200), fill=0, width=3, joint="curve"))
    result = extract_strokes(ink)
    assert len(result.strokes) == 1
    assert _px_length(result.strokes[0], SIZE, SIZE) == pytest.approx(300, rel=0.08)


def test_t_shape_splits_at_the_junction() -> None:
    def draw(d: ImageDraw.ImageDraw) -> None:
        d.line((40, 60, 216, 60), fill=0, width=3)
        d.line((128, 60, 128, 220), fill=0, width=3)

    result = extract_strokes(_ink(draw))
    assert 2 <= len(result.strokes) <= 3


def test_plus_gives_four_strokes() -> None:
    # Documented behaviour: the graph is never merged through a junction of
    # degree >= 3, so a "+" traces as 4 arms. (At 256 px CHAIN_PX is 1 px and
    # the arm ends sit on different cluster pixels, so they are not chained.)
    def draw(d: ImageDraw.ImageDraw) -> None:
        d.line((40, 128, 216, 128), fill=0, width=3)
        d.line((128, 40, 128, 216), fill=0, width=3)

    ink = _ink(draw)
    result = extract_strokes(ink)
    assert len(result.strokes) == 4
    assert _ink_coverage(ink, result) == 1.0


def test_circle_is_one_closed_stroke() -> None:
    ink = _ink(lambda d: d.ellipse((50, 50, 200, 200), outline=0, width=3))
    result = extract_strokes(ink)
    assert len(result.strokes) == 1
    points = result.strokes[0].points
    assert math.dist(points[0], points[-1]) * SIZE < 2
    assert _px_length(result.strokes[0], SIZE, SIZE) == pytest.approx(math.pi * 148, rel=0.08)


def test_three_pixel_dot_survives_as_a_tap() -> None:
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[100:103, 50] = True  # a 3 px dot
    ink[30, 200:230] = True  # and a line
    result = extract_strokes(ink)
    dots = [s for s in result.strokes if _is_dot(s, SIZE)]
    assert len(dots) == 1
    (x0, y0), (x1, y1) = dots[0].points
    assert x0 * SIZE == pytest.approx(50, abs=1)
    assert y0 * SIZE == pytest.approx(101, abs=1)
    assert (x1 - x0) * SIZE == pytest.approx(0.5)
    assert y1 == y0


def test_dots_are_never_pruned() -> None:
    # nuqta-sized blob next to a junction-free line
    ink = _ink(
        lambda d: (
            d.line((40, 128, 200, 128), fill=0, width=3),
            d.ellipse((120, 140, 124, 144), fill=0),
        )
    )
    result = extract_strokes(ink)
    assert len(result.strokes) == 2


def test_noisy_drawing_matches_the_clean_one() -> None:
    # binarize's speck filter removes the single-pixel noise before tracing
    noisy = extract_strokes(binarize(noisy_line_drawing()))
    assert noisy == extract_strokes(binarize(line_drawing_scroll()))


def test_short_spur_is_pruned_at_512() -> None:
    ink = np.zeros((512, 512), dtype=bool)
    ink[100, 40:200] = True
    ink[101:104, 120] = True  # 3 px spur: shorter than 8 and within the reach
    result = extract_strokes(ink)
    assert len(result.strokes) == 1
    assert _px_length(result.strokes[0], 512, 512) == pytest.approx(159, abs=2)


def test_spur_threshold_scales_with_resolution() -> None:
    # the same 3 px spur is pruned at 512 (limit 8 px) but kept at 128 (2 px)
    def build(size: int) -> np.ndarray:
        ink = np.zeros((size, size), dtype=bool)
        ink[60, 10:110] = True
        ink[61:64, 50] = True
        return ink

    assert len(extract_strokes(build(512)).strokes) == 1
    assert len(extract_strokes(build(128)).strokes) == 3


def test_short_real_stroke_is_not_pruned() -> None:
    # a 6 px tick reaches well past the 1 px line: real ink, kept
    ink = np.zeros((512, 512), dtype=bool)
    ink[100, 40:200] = True
    ink[101:107, 120] = True
    result = extract_strokes(ink)
    assert _ink_coverage(ink, result) == 1.0
    assert len(result.strokes) >= 2


def _ring(size: int, tails: tuple[tuple[int, int], ...]) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    cv2.circle(canvas, (64, 64), 30, 255, 1)
    for top, bottom in tails:
        canvas[top:bottom, 64] = 255
    return canvas > 0


@pytest.mark.parametrize(
    "tails",
    [((94, 96),), ((94, 96), (32, 34))],
    ids=["one-bump", "two-bumps"],
)
def test_ring_with_bumps_is_one_closed_stroke(tails: tuple[tuple[int, int], ...]) -> None:
    # a 1 px ring traced through junction clusters: the 2 px bumps are pruned,
    # the clusters drop to degree 2 and the arcs join into one closed loop
    ink = _ring(512, tails)
    before = ink.copy()
    result = extract_strokes(ink)
    assert np.array_equal(ink, before)
    assert len(result.strokes) == 1
    points = result.strokes[0].points
    assert math.dist(points[0], points[-1]) * 512 <= 2
    assert _px_length(result.strokes[0], 512, 512) == pytest.approx(2 * math.pi * 30, rel=0.08)


@pytest.mark.parametrize(
    ("tails", "count"),
    [(((94, 99),), 1), (((94, 99), (29, 34)), 2)],
    ids=["one-tail", "two-tails"],
)
def test_ring_with_real_tails_keeps_all_ink(tails: tuple[tuple[int, int], ...], count: int) -> None:
    # 5 px tails are real ink and are kept; chaining keeps the pen down
    ink = _ring(512, tails)
    result = extract_strokes(ink)
    assert len(result.strokes) == count
    assert _ink_coverage(ink, result) == 1.0


def test_touching_strokes_are_chained() -> None:
    ink = np.zeros((512, 512), dtype=bool)
    ink[10, 10:51] = True
    ink[10, 52:91] = True  # separate component, 2 px from the pen: chained
    ink[10, 94:130] = True  # 4 px gap: pen lift
    result = extract_strokes(ink)
    assert len(result.strokes) == 2
    assert _px_length(result.strokes[0], 512, 512) == pytest.approx(80, abs=1)


def test_dots_are_not_chained() -> None:
    ink = np.zeros((512, 512), dtype=bool)
    ink[10, 10:51] = True
    ink[10, 52] = True  # a 1 px dot touching the pen's end
    result = extract_strokes(ink)
    assert len(result.strokes) == 2
    assert result.strokes[1].width is not None


def test_large_dot_keeps_its_diameter() -> None:
    ink = _ink(lambda d: (d.ellipse((100, 100, 118, 118), fill=0),), size=512)
    assert ink.sum() > 250  # a 19 x 19 disc
    result = extract_strokes(ink)
    (dot,) = result.strokes
    assert dot.width is not None
    assert dot.width * 512 == pytest.approx(19, abs=1)
    (x, y), _ = dot.points
    assert x * 512 == pytest.approx(109, abs=1)
    assert y * 512 == pytest.approx(109, abs=1)


def test_line_strokes_have_no_width() -> None:
    result = extract_strokes(binarize(line_drawing_scroll()))
    assert all(s.width is None for s in result.strokes)
    assert b'"width":null' not in result.to_json()


def test_pinholes_are_filled_before_skeletonizing() -> None:
    ink = np.zeros((512, 512), dtype=bool)
    ink[100:108, 40:200] = True
    ink[103:105, 80] = False  # a 2 px pinhole inside the thick line
    result = extract_strokes(ink)
    assert len(result.strokes) == 1


def test_calligraphy_does_not_fragment_at_higher_resolution() -> None:
    # guards only against fragmentation blowups; ink fidelity wins over a low
    # count (Task 5 paces the reveal from the count)
    low = len(extract_strokes(binarize(calligraphy(size=512))).strokes)
    high = len(extract_strokes(binarize(calligraphy(size=1024))).strokes)
    assert high <= 3 * low


def _busy(name: str, size: int) -> bytes:
    width = max(1, round(3 * size / 512))
    if name == "road_map":
        return road_map(size=size, width=width)
    if name == "brick 20x10":
        return brick(brick_w=20, brick_h=10, size=size)
    return label_map(size=size)


@pytest.mark.parametrize("size", [512, 1024, 2048])
@pytest.mark.parametrize("name", ["road_map", "brick 20x10", "label_map"])
def test_no_ink_is_dropped(name: str, size: int) -> None:
    ink = binarize(_busy(name, size))
    assert _ink_coverage(ink, extract_strokes(ink)) >= 0.98, (name, size)


@pytest.mark.parametrize("size", [512, 1024, 2048])
def test_calligraphy_keeps_its_ink(size: int) -> None:
    ink = binarize(calligraphy(size=size))
    assert _ink_coverage(ink, extract_strokes(ink)) >= 0.97, size


def test_pacing_hint_is_never_enforced() -> None:
    assert PACING_STROKES == 400
    assert len(extract_strokes(binarize(brick(brick_w=20, brick_h=10))).strokes) > PACING_STROKES


def test_scroll_fixture_has_few_strokes() -> None:
    result = extract_strokes(binarize(line_drawing_scroll()))
    assert 5 <= len(result.strokes) < 60


@pytest.mark.parametrize(
    ("png", "max_strokes"),
    [
        (scroll(size=1024), 60),
        (scroll(size=2048, width=6), 60),
        (road_map(size=2048, width=12), None),
    ],
    ids=["scroll-1024", "scroll-2048", "road-map-2048"],
)
def test_extraction_is_fast(png: bytes, max_strokes: int | None) -> None:
    ink = binarize(png)
    start = time.perf_counter()
    result = extract_strokes(ink)
    elapsed = time.perf_counter() - start
    if max_strokes is not None:
        assert len(result.strokes) < max_strokes
    assert elapsed < 2.0


def test_rtl_changes_the_first_stroke() -> None:
    def draw(d: ImageDraw.ImageDraw) -> None:
        d.line((20, 30, 90, 30), fill=0, width=3)
        d.line((160, 30, 236, 30), fill=0, width=3)

    ink = _ink(draw)
    ltr = extract_strokes(ink)
    rtl = extract_strokes(ink, rtl=True)
    assert ltr.strokes[0].points[0][0] < 0.2
    assert rtl.strokes[0].points[0][0] > 0.8
    # the pen then crosses to the other line, entering at its nearer end
    assert ltr.strokes[1].points[0][0] * SIZE == pytest.approx(160, abs=1)
    assert rtl.strokes[1].points[0][0] * SIZE == pytest.approx(90, abs=1)


def _starts(result: StrokeSet) -> list[tuple[int, int]]:
    return [
        (round(s.points[0][0] * result.width), round(s.points[0][1] * result.height))
        for s in result.strokes
    ]


def test_greedy_order_reverses_far_strokes() -> None:
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[10, 10:60] = True
    ink[20, 70:120] = True
    ink[200, 10:60] = True
    # pen ends at (119, 20); the right end (59, 200) of the last line is nearer
    assert _starts(extract_strokes(ink)) == [(10, 10), (70, 20), (59, 200)]


def test_order_ties_break_on_lowest_y_then_x() -> None:
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[0, 0:21] = True  # first stroke, pen ends at (20, 0)
    ink[50:91, 20] = True  # near end (20, 50): distance 50
    ink[30, 60:91] = True  # near end (60, 30): distance 50, lower y wins
    assert _starts(extract_strokes(ink)) == [(0, 0), (60, 30), (20, 50)]


def test_order_ties_break_on_x_when_y_is_equal() -> None:
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[10, 0:101] = True  # first stroke from (0, 10), pen ends at (100, 10)
    ink[50, 130:141] = True  # near end (130, 50): distance 50
    ink[50, 60:71] = True  # near end (70, 50): distance 50, lower x wins
    assert _starts(extract_strokes(ink))[:2] == [(0, 10), (70, 50)]


def test_points_are_normalized_x_by_width_y_by_height() -> None:
    ink = np.zeros((100, 200), dtype=bool)
    ink[10, 20:180] = True
    result = extract_strokes(ink)
    assert (result.width, result.height) == (200, 100)
    xs = sorted(p[0] for p in result.strokes[0].points)
    ys = {p[1] for p in result.strokes[0].points}
    assert xs[0] == pytest.approx(20 / 200)
    assert xs[-1] == pytest.approx(179 / 200)
    assert ys == {10 / 100}


def test_empty_ink_gives_no_strokes() -> None:
    result = extract_strokes(np.zeros((32, 48), dtype=bool))
    assert result.strokes == []
    assert (result.width, result.height) == (48, 32)


def test_rejects_non_2d_ink() -> None:
    with pytest.raises(StrokeDataError):
        extract_strokes(np.zeros((4, 4, 3), dtype=bool))


def test_deterministic() -> None:
    ink = binarize(noisy_line_drawing())
    assert extract_strokes(ink).to_json() == extract_strokes(ink.copy()).to_json()


def test_json_roundtrip_is_compact_and_sorted() -> None:
    result = extract_strokes(binarize(line_drawing_scroll()))
    raw = result.to_json()
    assert isinstance(raw, bytes)
    assert b" " not in raw
    decoded = json.loads(raw)
    assert list(decoded) == sorted(decoded)
    assert StrokeSet.from_json(raw) == result


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'{"version":1,"width":10,"height":10,"strokes":[{"points":[[0.1,0.1]]}]}',
        b'{"version":1,"width":10,"height":10,"strokes":[{"points":[[0.1,0.1],[1.5,0.2]]}]}',
        b'{"version":1,"width":0,"height":10,"strokes":[]}',
        b'{"version":2,"width":10,"height":10,"strokes":[]}',
        b'{"version":1,"width":10,"height":10,"strokes":[],"extra":1}',
        b'{"version":1,"width":10,"height":10,"strokes":[{"points":[[0,0],[0,0]],"width":0}]}',
    ],
)
def test_from_json_rejects_bad_data(raw: bytes) -> None:
    with pytest.raises(StrokeDataError):
        StrokeSet.from_json(raw)


def test_stroke_data_error_is_a_narratoflow_error() -> None:
    assert issubclass(StrokeDataError, NarratoFlowError)


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow"
    for path in (root / "visuals" / "strokes.py", root / "errors.py", Path(__file__)):
        assert path.read_bytes().isascii()
