"""Virtual camera: a pure function of time that follows the pen inside the stage."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from narratoflow.render.reveal import pen_at
from narratoflow.render.variation import ZOOM_MAX
from narratoflow.visuals.marker import PreparedStroke
from narratoflow.visuals.schedule import StrokeEvent

ZOOM_DRAW = ZOOM_MAX
ZOOM_IN_MS = 500
ZOOM_OUT_MS = 700
CAMERA_SMOOTH_MS = 1200
CAMERA_SAMPLE_MS = 40
CAMERA_MIN_DRAW_MS = 2500
_HALF_SAMPLES = CAMERA_SMOOTH_MS // CAMERA_SAMPLE_MS // 2  # 15


@dataclass(frozen=True)
class View:
    cx: float
    cy: float
    zoom: float


def _smoothstep(x: float) -> float:
    return 3 * x * x - 2 * x * x * x


class CameraTrack:
    def __init__(
        self,
        prepared: Sequence[PreparedStroke],
        events: Sequence[StrokeEvent],
        art: tuple[int, int],
        draw: tuple[int, int] | None,
    ) -> None:
        self._prepared = prepared
        self._events = events
        self._art = art
        self._overview = View(art[0] / 2, art[1] / 2, 1.0)
        self._window: tuple[float, float] | None = None
        if draw is not None and events and draw[1] - draw[0] >= CAMERA_MIN_DRAW_MS:
            self._window = (float(draw[0]), float(draw[1]))

    def _smoothed_pen(self, t_ms: float, start: float, end: float) -> tuple[float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for k in range(-_HALF_SAMPLES, _HALF_SAMPLES + 1):
            s = min(max(t_ms + k * CAMERA_SAMPLE_MS, start), end - 1e-6)
            p = pen_at(self._prepared, self._events, s)
            if p is not None:
                xs.append(p[0])
                ys.append(p[1])
        if not xs:
            return self._overview.cx, self._overview.cy
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def at(self, t_ms: float) -> View:
        if self._window is None:
            return self._overview
        start, end = self._window
        if t_ms < start or t_ms > end + ZOOM_OUT_MS:
            return self._overview
        if t_ms <= end:
            e = _smoothstep(min(1.0, (t_ms - start) / ZOOM_IN_MS))
        else:
            e = 1.0 - _smoothstep((t_ms - end) / ZOOM_OUT_MS)
        zoom = 1 + (ZOOM_DRAW - 1) * e
        px, py = self._smoothed_pen(t_ms, start, end)
        cx = self._overview.cx + (px - self._overview.cx) * e
        cy = self._overview.cy + (py - self._overview.cy) * e
        half_w = self._art[0] / zoom / 2
        half_h = self._art[1] / zoom / 2
        cx = min(max(cx, half_w), self._art[0] - half_w)
        cy = min(max(cy, half_h), self._art[1] - half_h)
        return View(cx, cy, zoom)

    def affine(self, view: View, stage_size: tuple[int, int]) -> np.ndarray:
        """2x3 matrix mapping art pixels to stage pixels; the view window fills the stage."""
        view_w = self._art[0] / view.zoom
        view_h = self._art[1] / view.zoom
        sx = stage_size[0] / view_w
        sy = stage_size[1] / view_h
        return np.array(
            [
                [sx, 0.0, -(view.cx - view_w / 2) * sx],
                [0.0, sy, -(view.cy - view_h / 2) * sy],
            ],
            dtype=np.float64,
        )
