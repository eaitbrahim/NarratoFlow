import io
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from fixtures.images import line_drawing_scroll, noisy_line_drawing, scroll
from PIL import Image, ImageDraw

from narratoflow.errors import NarratoFlowError, StrokeDataError
from narratoflow.visuals import strokes as strokes_module
from narratoflow.visuals.lineart import binarize
from narratoflow.visuals.strokes import (
    MAX_STROKES,
    MIN_SPUR_PX,
    Stroke,
    StrokeSet,
    _Edge,
    _merge,
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
    # Documented behaviour: no merging through a junction of degree >= 3.
    def draw(d: ImageDraw.ImageDraw) -> None:
        d.line((40, 128, 216, 128), fill=0, width=3)
        d.line((128, 40, 128, 216), fill=0, width=3)

    result = extract_strokes(_ink(draw))
    assert len(result.strokes) == 4


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


def test_spurs_pruned_on_noisy_drawing() -> None:
    result = extract_strokes(binarize(noisy_line_drawing()))
    assert 0 < len(result.strokes) <= MAX_STROKES
    for stroke in result.strokes:
        if not _is_dot(stroke, result.width):
            assert _px_length(stroke, result.width, result.height) >= MIN_SPUR_PX


def test_short_spur_is_pruned() -> None:
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[100, 40:200] = True
    ink[101:105, 120] = True  # 4 px spur off the line
    result = extract_strokes(ink)
    assert len(result.strokes) == 1
    assert _px_length(result.strokes[0], SIZE, SIZE) == pytest.approx(159, abs=2)


def test_scroll_fixture_has_few_strokes() -> None:
    result = extract_strokes(binarize(line_drawing_scroll()))
    assert 5 <= len(result.strokes) < 60


def test_scroll_1024_is_fast() -> None:
    ink = binarize(scroll(size=1024))
    start = time.perf_counter()
    result = extract_strokes(ink)
    elapsed = time.perf_counter() - start
    assert len(result.strokes) < 60
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


def test_cap_drops_the_shortest_strokes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strokes_module, "MAX_STROKES", 2)
    ink = np.zeros((SIZE, SIZE), dtype=bool)
    ink[10, 10:30] = True  # 19 px, dropped
    ink[60, 10:100] = True
    ink[120, 10:200] = True
    result = extract_strokes(ink)
    lengths = sorted(round(_px_length(s, SIZE, SIZE)) for s in result.strokes)
    assert lengths == [89, 189]


def test_loop_through_a_degree_two_cluster_becomes_closed() -> None:
    # two arcs between clusters 0 and 1: merged at 0, then closed at 1
    arcs = [
        _Edge(0, 1, [(0, 0), (0, 1), (0, 2)]),
        _Edge(1, 0, [(1, 2), (1, 1), (1, 0)]),
    ]
    (loop,) = _merge(arcs)
    assert (loop.a, loop.b) == (-1, -1)
    assert loop.pixels[0] == loop.pixels[-1] == (0, 2)
    assert len(loop.pixels) == 7


def test_already_closed_loop_at_a_cluster_is_not_extended() -> None:
    (loop,) = _merge([_Edge(3, 3, [(0, 0), (0, 1), (1, 1), (0, 0)])])
    assert (loop.a, loop.b) == (-1, -1)
    assert loop.pixels == [(0, 0), (0, 1), (1, 1), (0, 0)]


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
