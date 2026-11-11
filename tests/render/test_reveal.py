import itertools
import math

import numpy as np
import pytest
from PIL import Image

from narratoflow.render.reveal import RevealCanvas, pen_at, stroke_progress
from narratoflow.visuals.marker import PreparedStroke, prepare_marker, render_marker
from narratoflow.visuals.schedule import StrokeEvent, draw_schedule
from narratoflow.visuals.strokes import Stroke, StrokeSet

SIZE = (200, 120)
INK = "#112233"


def _set() -> StrokeSet:
    return StrokeSet(
        width=200,
        height=120,
        strokes=[
            Stroke(points=[(0.1, 0.2), (0.5, 0.2), (0.9, 0.5)]),
            Stroke(points=[(0.1, 0.8), (0.9, 0.8)]),
            Stroke(points=[(0.5, 0.5), (0.5, 0.5)], width=0.05),
        ],
    )


def _ink(img: Image.Image) -> int:
    return int((np.asarray(img).sum(axis=2) < 3 * 255).sum())


def _prepared(seed: int = 1) -> list[PreparedStroke]:
    return prepare_marker(_set(), SIZE, line_weight_px=4.0, seed=seed)


def test_progress_zero_length_events_complete_at_start() -> None:
    events = [StrokeEvent(0, 0, 0), StrokeEvent(1, 0, 500), StrokeEvent(2, 500, 500)]
    assert stroke_progress(events, 0) == (1, 0.0)
    assert stroke_progress(events, 500) == (3, 0.0)


def test_progress_mid_stroke_fraction() -> None:
    events = [StrokeEvent(0, 100, 300), StrokeEvent(1, 300, 700)]
    assert stroke_progress(events, 0) == (0, 0.0)
    assert stroke_progress(events, 150) == (0, 0.25)
    assert stroke_progress(events, 300) == (1, 0.0)
    assert stroke_progress(events, 400) == (1, 0.25)
    assert stroke_progress(events, 700) == (2, 0.0)
    assert stroke_progress([], 5) == (0, 0.0)


@pytest.mark.parametrize("seed", range(1, 21))
def test_final_reveal_equals_render_marker(seed: int) -> None:
    events = draw_schedule(_set(), 1000, 4000)
    canvas = RevealCanvas(_prepared(seed), events, SIZE, INK)
    final = canvas.at(events[-1].end_ms)
    expected = render_marker(_set(), SIZE, line_weight_px=4.0, ink=INK, seed=seed, upto=1.0)
    assert final.tobytes() == expected.tobytes()


def test_reveal_monotonic_ink() -> None:
    events = draw_schedule(_set(), 0, 3000)
    canvas = RevealCanvas(_prepared(), events, SIZE, INK)
    counts = [_ink(canvas.at(t)) for t in np.linspace(0, 3100, 30)]
    assert counts[0] <= 2 * 4 * 4  # at most the pen-down disc
    assert all(a <= b for a, b in itertools.pairwise(counts))
    assert counts[-1] == _ink(
        render_marker(_set(), SIZE, line_weight_px=4.0, ink=INK, seed=1, upto=1.0)
    )


def test_reveal_rejects_time_going_back() -> None:
    events = draw_schedule(_set(), 0, 3000)
    canvas = RevealCanvas(_prepared(), events, SIZE, INK)
    canvas.at(1000)
    canvas.at(1000)
    with pytest.raises(ValueError):
        canvas.at(999)


def test_reveal_rejects_length_mismatch() -> None:
    events = draw_schedule(_set(), 0, 3000)
    with pytest.raises(ValueError):
        RevealCanvas(_prepared()[:2], events, SIZE, INK)


def test_reveal_partial_frame_is_a_copy() -> None:
    events = draw_schedule(_set(), 0, 3000)
    canvas = RevealCanvas(_prepared(), events, SIZE, INK)
    mid = canvas.at(events[1].start_ms + 1)
    before = mid.tobytes()
    later = canvas.at(events[1].end_ms - 1)
    assert mid.tobytes() == before
    assert _ink(later) > _ink(mid)


def test_pen_at_none_outside_window_and_on_stroke_inside() -> None:
    prepared = _prepared()
    events = draw_schedule(_set(), 1000, 4000)
    assert pen_at(prepared, events, 999) is None
    assert pen_at(prepared, events, events[-1].end_ms) is None
    assert pen_at(prepared, events, 5000) is None
    assert pen_at(prepared, [], 0) is None
    assert pen_at(prepared, events, 1000) == prepared[0].points[0]
    first = events[0]
    mid = pen_at(prepared, events, (first.start_ms + first.end_ms) / 2)
    assert mid is not None
    # On the first stroke's polyline: distance to some segment is ~0.
    pts = prepared[0].points
    dist = min(_seg_dist(mid, a, b) for a, b in itertools.pairwise(pts))
    assert dist < 1e-6
    assert pen_at(prepared, events, events[2].start_ms) == prepared[2].points[0]


def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / (ax * ax + ay * ay)))
    return math.dist(p, (a[0] + ax * t, a[1] + ay * t))
