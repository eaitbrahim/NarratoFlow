from __future__ import annotations

import itertools

from narratoflow.visuals.schedule import StrokeEvent, draw_schedule, stroke_lengths
from narratoflow.visuals.strokes import Stroke, StrokeSet


def _set(strokes: list[Stroke], width: int = 200, height: int = 100) -> StrokeSet:
    return StrokeSet(width=width, height=height, strokes=strokes)


def test_schedule_proportional_contiguous_and_exact_end() -> None:
    s = _set(
        [
            Stroke(points=[(0.0, 0.0), (0.5, 0.0)]),  # 100 px
            Stroke(points=[(0.0, 0.0), (0.0, 1.0)]),  # 100 px
            Stroke(points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),  # 300 px
        ]
    )
    assert stroke_lengths(s) == [100.0, 100.0, 300.0]
    events = draw_schedule(s, 1000, 6000)
    assert events == [
        StrokeEvent(0, 1000, 2000),
        StrokeEvent(1, 2000, 3000),
        StrokeEvent(2, 3000, 6000),
    ]
    assert events[-1].end_ms == 6000
    assert all(a.end_ms == b.start_ms for a, b in itertools.pairwise(events))


def test_schedule_dot_uses_dot_width() -> None:
    s = _set(
        [
            Stroke(points=[(0.5, 0.5), (0.5, 0.5)], width=0.1),
            Stroke(points=[(0.0, 0.0), (0.1, 0.0)]),
        ]
    )
    assert stroke_lengths(s) == [20.0, 20.0]


def test_schedule_empty() -> None:
    assert draw_schedule(_set([]), 0, 1000) == []
