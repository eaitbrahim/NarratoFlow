"""Deterministic free variations of a sketch and its fit onto the art canvas.

Stroke order is never changed: the audio draw schedule and `SfxHit.stroke_index`
index the stored order. The spec's "different stroke order" is realized as
per-stroke direction reversal, recorded in `Variation.reversed_strokes`.
"""

import math
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from narratoflow.render.layout import Canvas
from narratoflow.visuals.marker import marker_seed, video_id
from narratoflow.visuals.strokes import Stroke, StrokeSet

__all__ = [
    "MIRROR_P",
    "OFFSET_FRAC",
    "REVERSE_P",
    "ROTATION_DEG",
    "SCALE_RANGE",
    "WEIGHT_RANGE",
    "ZOOM_MAX",
    "Variation",
    "art_size",
    "card_frame_strokes",
    "choose_variation",
    "sketch_to_art",
    "video_id",
]

ROTATION_DEG = 3.0
SCALE_RANGE = (0.90, 1.00)
OFFSET_FRAC = 0.03
WEIGHT_RANGE = (0.90, 1.10)
MIRROR_P = 0.5
REVERSE_P = 0.5
ZOOM_MAX = 1.30

_CARD_INSET = 0.08


class Variation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mirrored: bool
    rotation_deg: float
    scale: float
    dx: float
    dy: float
    line_weight_factor: float
    reversed_strokes: list[int]
    wobble_seed: int


def choose_variation(
    video_id: str, scene_id: str, *, mirror_safe: bool, stroke_count: int
) -> Variation:
    seed = marker_seed(video_id, scene_id)
    rng = np.random.default_rng([seed, 1])
    mirror_u = float(rng.random())
    rotation = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
    scale = float(rng.uniform(*SCALE_RANGE))
    dx = float(rng.uniform(-OFFSET_FRAC, OFFSET_FRAC))
    dy = float(rng.uniform(-OFFSET_FRAC, OFFSET_FRAC))
    weight = float(rng.uniform(*WEIGHT_RANGE))
    reversal = rng.random(stroke_count)
    return Variation(
        mirrored=mirror_safe and mirror_u < MIRROR_P,
        rotation_deg=rotation,
        scale=scale,
        dx=dx,
        dy=dy,
        line_weight_factor=weight,
        reversed_strokes=[i for i in range(stroke_count) if float(reversal[i]) < REVERSE_P],
        wobble_seed=seed,
    )


def art_size(canvas: Canvas) -> tuple[int, int]:
    stage = canvas.stage
    return round(stage.w * ZOOM_MAX), round(stage.h * ZOOM_MAX)


def _half_extent(w: float, h: float, deg: float) -> tuple[float, float]:
    """Half width and height of the w x h box after a rotation of `deg`."""
    c, s = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    return (w * c + h * s) / 2.0, (w * s + h * c) / 2.0


def sketch_to_art(
    strokes: StrokeSet, variation: Variation, art: tuple[int, int], pad_px: float
) -> tuple[Callable[[float, float], tuple[float, float]], float]:
    """Map normalized stroke points to art pixels; also return sketch px -> art px scale."""
    sw, sh = float(strokes.width), float(strokes.height)
    art_w, art_h = float(art[0]), float(art[1])
    avail_x = art_w / 2.0 - pad_px * ZOOM_MAX
    avail_y = art_h / 2.0 - pad_px * ZOOM_MAX
    fx, fy = _half_extent(sw, sh, ROTATION_DEG)
    fit = min(avail_x / fx, avail_y / fy)
    scale = fit * variation.scale
    hx, hy = _half_extent(sw, sh, variation.rotation_deg)
    slack_x = max(avail_x - scale * hx, 0.0)
    slack_y = max(avail_y - scale * hy, 0.0)
    tx = min(max(variation.dx * art_w, -slack_x), slack_x)
    ty = min(max(variation.dy * art_h, -slack_y), slack_y)
    rad = math.radians(variation.rotation_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    mirrored = variation.mirrored

    def to_art(x: float, y: float) -> tuple[float, float]:
        px = x * sw - sw / 2.0
        py = y * sh - sh / 2.0
        if mirrored:
            px = -px
        px *= scale
        py *= scale
        rx = px * cos - py * sin
        ry = px * sin + py * cos
        return art_w / 2.0 + rx + tx, art_h / 2.0 + ry + ty

    return to_art, scale


def card_frame_strokes() -> StrokeSet:
    """A rectangle inset 8 %: top, right, bottom, left."""
    lo, hi = _CARD_INSET, 1.0 - _CARD_INSET
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    edges = [Stroke(points=[corners[i], corners[(i + 1) % 4]]) for i in range(4)]
    return StrokeSet(width=828, height=850, strokes=edges)
