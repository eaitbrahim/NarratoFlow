"""Per-stroke reveal on the audio draw schedule, and the pen position for the hand."""

from collections.abc import Sequence

from PIL import Image, ImageColor, ImageDraw

from narratoflow.visuals.marker import PreparedStroke, draw_stroke, stroke_point
from narratoflow.visuals.schedule import StrokeEvent


def stroke_progress(events: Sequence[StrokeEvent], t_ms: float) -> tuple[int, float]:
    """(strokes fully drawn at `t_ms`, fraction of the next one)."""
    done = 0
    for event in events:
        if t_ms < event.end_ms:
            break
        done += 1
    if done == len(events):
        return done, 0.0
    event = events[done]
    if t_ms <= event.start_ms:
        return done, 0.0
    return done, (t_ms - event.start_ms) / (event.end_ms - event.start_ms)


def _in_progress(events: Sequence[StrokeEvent], t_ms: float) -> tuple[int, float] | None:
    done, fraction = stroke_progress(events, t_ms)
    if done == len(events) or t_ms < events[done].start_ms:
        return None
    return done, fraction


def pen_at(
    prepared: Sequence[PreparedStroke], events: Sequence[StrokeEvent], t_ms: float
) -> tuple[float, float] | None:
    """Pen tip on the stroke in progress; None before the first start or after the last end."""
    current = _in_progress(events, t_ms)
    if current is None:
        return None
    return stroke_point(prepared[current[0]], current[1])


class RevealCanvas:
    """Incremental reveal: finished strokes are committed once, in order."""

    def __init__(
        self,
        prepared: list[PreparedStroke],
        events: list[StrokeEvent],
        size: tuple[int, int],
        ink: str,
    ) -> None:
        if len(prepared) != len(events):
            raise ValueError(f"{len(prepared)} strokes but {len(events)} draw events")
        colour = ImageColor.getrgb(ink)
        self._rgb = (colour[0], colour[1], colour[2])
        self._prepared = prepared
        self._events = events
        self._done = Image.new("RGB", size, (255, 255, 255))
        self._draw = ImageDraw.Draw(self._done)
        self._committed = 0
        self._last_t: float | None = None

    def at(self, t_ms: float) -> Image.Image:
        """Frame at `t_ms`. Without a stroke in progress this is the shared "done" image."""
        if self._last_t is not None and t_ms < self._last_t:
            raise ValueError(f"time went back from {self._last_t} to {t_ms}")
        self._last_t = t_ms
        done, _ = stroke_progress(self._events, t_ms)
        while self._committed < done:
            draw_stroke(self._draw, self._prepared[self._committed], 1.0, self._rgb)
            self._committed += 1
        current = _in_progress(self._events, t_ms)
        if current is None:
            return self._done
        frame = self._done.copy()
        draw_stroke(ImageDraw.Draw(frame), self._prepared[current[0]], current[1], self._rgb)
        return frame
