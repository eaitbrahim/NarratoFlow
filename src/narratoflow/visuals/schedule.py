"""Per-stroke draw timing. `draw_schedule` is the authority for when each stroke is drawn.

Plan 5 must reveal strokes one by one from these events (or give `render_marker` a
per-stroke progress input) so the marker SFX land on the strokes they were placed on.
"""

import itertools
import math
from dataclasses import dataclass

from narratoflow.audio.words import split_ms
from narratoflow.visuals.strokes import StrokeSet


@dataclass(frozen=True)
class StrokeEvent:
    index: int
    start_ms: int
    end_ms: int


def stroke_lengths(strokes: StrokeSet) -> list[float]:
    """Schedule weights in pixels of width x height; not `render_marker`'s own lengths."""
    out: list[float] = []
    for stroke in strokes.strokes:
        if stroke.width is not None:
            out.append(stroke.width * strokes.width)
            continue
        pts = [(x * strokes.width, y * strokes.height) for x, y in stroke.points]
        out.append(math.fsum(math.dist(a, b) for a, b in itertools.pairwise(pts)))
    return out


def draw_schedule(strokes: StrokeSet, start_ms: int, end_ms: int) -> list[StrokeEvent]:
    """Contiguous events, durations proportional to stroke length, ending at `end_ms`."""
    spans = split_ms(stroke_lengths(strokes), start_ms, end_ms)
    return [StrokeEvent(i, a, b) for i, (a, b) in enumerate(spans)]
