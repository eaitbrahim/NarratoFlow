"""Re-render extracted strokes as a channel-styled marker drawing.

All randomness comes from one seeded generator drawn in a fixed order over the
whole stroke set, so a partial render (`upto` < 1) is a prefix of the full one.
"""

import hashlib
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from narratoflow.models import Series
from narratoflow.visuals.strokes import StrokeSet

WOBBLE_SIGMA_PX = 0.6
WEIGHT_JITTER = 0.10

_Pt = tuple[float, float]


def video_id(series: Series) -> str:
    """The id audio, sketching and render all seed their randomness with."""
    return f"{series.channel_slug}/{series.title}"


def marker_seed(video_id: str, scene_id: str, attempt: int = 0) -> int:
    """Seed for (video, scene); a later `attempt` (regeneration) gets a different one."""
    text = f"{video_id}:{scene_id}" if attempt == 0 else f"{video_id}:{scene_id}:{attempt}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest, 16) % (2**63)


def _disc(draw: ImageDraw.ImageDraw, p: _Pt, diameter: float, ink: tuple[int, int, int]) -> None:
    r = diameter / 2.0
    draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=ink)


@dataclass(frozen=True)
class PreparedStroke:
    """One stroke in target pixels with its wobble applied; a dot keeps one point."""

    points: tuple[_Pt, ...]
    weight: float  # line weight, or the disc diameter for a dot
    dot: bool
    length: float  # polyline length, or the diameter for a dot


def prepare_marker(
    strokes: StrokeSet,
    size: tuple[int, int],
    *,
    line_weight_px: float,
    seed: int,
    to_pixels: Callable[[float, float], _Pt] | None = None,
    reverse: Sequence[bool] | None = None,
    dot_px_per_unit: float | None = None,
) -> list[PreparedStroke]:
    """Fixed-order draws for every stroke: noise then weight jitter, per stroke.

    `reverse[i]` flips stroke i after its noise is added, so it never changes the draws.
    A dot has no direction: `reverse` leaves it unchanged.
    """
    width, height = size
    if to_pixels is None:

        def to_pixels(x: float, y: float) -> _Pt:
            return (x * (width - 1), y * (height - 1))

    if reverse is not None and len(reverse) != len(strokes.strokes):
        raise ValueError(f"reverse has {len(reverse)} entries for {len(strokes.strokes)} strokes")
    per_unit = float(width) if dot_px_per_unit is None else dot_px_per_unit
    rng = np.random.default_rng(seed)
    out: list[PreparedStroke] = []
    for i, stroke in enumerate(strokes.strokes):
        noise = rng.normal(0.0, WOBBLE_SIGMA_PX, size=(len(stroke.points), 2))
        jitter = 1.0 + rng.uniform(-WEIGHT_JITTER, WEIGHT_JITTER)
        pts: list[_Pt] = []
        for (x, y), n in zip(stroke.points, noise, strict=True):
            px, py = to_pixels(x, y)
            pts.append((px + float(n[0]), py + float(n[1])))
        weight = line_weight_px * jitter
        if reverse is not None and reverse[i] and stroke.width is None:
            pts.reverse()
        if stroke.width is not None:
            diameter = max(stroke.width * per_unit, weight)
            out.append(PreparedStroke((pts[0],), diameter, True, diameter))
        else:
            seg = sum(math.dist(a, b) for a, b in itertools.pairwise(pts))
            out.append(PreparedStroke(tuple(pts), weight, False, seg))
    return out


def _walk(stroke: PreparedStroke, length: float) -> list[_Pt]:
    """The polyline cut after `length` px (the whole stroke once `length` covers it)."""
    if length >= stroke.length:
        return list(stroke.points)
    cut = [stroke.points[0]]
    remaining = length
    for a, b in itertools.pairwise(stroke.points):
        d = math.dist(a, b)
        if remaining >= d:
            cut.append(b)
            remaining -= d
        else:
            t = remaining / d if d > 0 else 0.0
            cut.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            break
    return cut


def _draw_prefix(
    draw: ImageDraw.ImageDraw, stroke: PreparedStroke, length: float, rgb: tuple[int, int, int]
) -> None:
    """Draw the first `length` px of `stroke`; a dot appears only once complete."""
    if stroke.dot:
        if length >= stroke.length:
            _disc(draw, stroke.points[0], stroke.weight, rgb)
        return
    cut = _walk(stroke, length)
    half = stroke.weight / 2.0
    for a, b in itertools.pairwise(cut):
        d = math.dist(a, b)
        if d > 0.0:
            nx, ny = -(b[1] - a[1]) / d * half, (b[0] - a[0]) / d * half
            draw.polygon(
                [
                    (a[0] + nx, a[1] + ny),
                    (b[0] + nx, b[1] + ny),
                    (b[0] - nx, b[1] - ny),
                    (a[0] - nx, a[1] - ny),
                ],
                fill=rgb,
            )
    for p in cut:
        _disc(draw, p, stroke.weight, rgb)


def draw_stroke(
    draw: ImageDraw.ImageDraw, stroke: PreparedStroke, fraction: float, rgb: tuple[int, int, int]
) -> None:
    """Draw the first `fraction` of the stroke's own length."""
    _draw_prefix(draw, stroke, fraction * stroke.length, rgb)


def stroke_point(stroke: PreparedStroke, fraction: float) -> _Pt:
    """Pen position after `fraction` of the stroke (clamped to [0, 1])."""
    if stroke.dot:
        return stroke.points[0]
    return _walk(stroke, min(max(fraction, 0.0), 1.0) * stroke.length)[-1]


def render_marker(
    strokes: StrokeSet,
    size: tuple[int, int],
    *,
    line_weight_px: float,
    ink: str,
    seed: int,
    upto: float = 1.0,
) -> Image.Image:
    if not 0.0 <= upto <= 1.0:
        raise ValueError(f"upto must be in [0, 1], got {upto}")
    colour = ImageColor.getrgb(ink)
    rgb = (colour[0], colour[1], colour[2])
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    prepared = prepare_marker(strokes, size, line_weight_px=line_weight_px, seed=seed)

    total = sum(p.length for p in prepared)
    budget = upto * total
    if upto <= 0.0:
        return img
    # At upto == 1 the float budget can fall an ulp short of the last stroke;
    # every stroke is complete then, so the budget check is skipped.
    complete = upto >= 1.0

    for stroke in prepared:
        if complete:
            _draw_prefix(draw, stroke, stroke.length, rgb)
            continue
        _draw_prefix(draw, stroke, budget, rgb)
        if budget < stroke.length:
            break
        budget -= stroke.length
    return img
