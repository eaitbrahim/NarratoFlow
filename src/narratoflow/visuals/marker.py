"""Re-render extracted strokes as a channel-styled marker drawing.

All randomness comes from one seeded generator drawn in a fixed order over the
whole stroke set, so a partial render (`upto` < 1) is a prefix of the full one.
"""

import hashlib
import itertools
import math

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from narratoflow.visuals.strokes import StrokeSet

WOBBLE_SIGMA_PX = 0.6
WEIGHT_JITTER = 0.10

_Pt = tuple[float, float]


def marker_seed(video_id: str, scene_id: str) -> int:
    digest = hashlib.sha256(f"{video_id}:{scene_id}".encode()).hexdigest()
    return int(digest, 16) % (2**63)


def _disc(draw: ImageDraw.ImageDraw, p: _Pt, diameter: float, ink: tuple[int, int, int]) -> None:
    r = diameter / 2.0
    draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=ink)


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
    width, height = size
    colour = ImageColor.getrgb(ink)
    rgb = (colour[0], colour[1], colour[2])
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)

    # Fixed-order draws for every stroke, independent of upto.
    prepared: list[tuple[list[_Pt], float, bool]] = []
    for stroke in strokes.strokes:
        noise = rng.normal(0.0, WOBBLE_SIGMA_PX, size=(len(stroke.points), 2))
        jitter = 1.0 + rng.uniform(-WEIGHT_JITTER, WEIGHT_JITTER)
        pts = [
            (x * (width - 1) + float(n[0]), y * (height - 1) + float(n[1]))
            for (x, y), n in zip(stroke.points, noise, strict=True)
        ]
        weight = line_weight_px * jitter
        if stroke.width is not None:
            prepared.append((pts[:1], max(stroke.width * width, weight), True))
        else:
            prepared.append((pts, weight, False))

    def length(pts: list[_Pt], diameter: float, dot: bool) -> float:
        if dot:
            return diameter
        return sum(math.dist(a, b) for a, b in itertools.pairwise(pts))

    total = sum(length(*item) for item in prepared)
    budget = upto * total
    if upto <= 0.0:
        return img

    for pts, weight, dot in prepared:
        seg = length(pts, weight, dot)
        if dot:
            if budget >= seg:
                _disc(draw, pts[0], weight, rgb)
                budget -= seg
                continue
            break
        cut = pts
        done = budget >= seg
        if not done:
            cut = [pts[0]]
            remaining = budget
            for a, b in itertools.pairwise(pts):
                d = math.dist(a, b)
                if remaining >= d:
                    cut.append(b)
                    remaining -= d
                else:
                    t = remaining / d if d > 0 else 0.0
                    cut.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                    break
            if len(cut) < 2:
                break
        half = weight / 2.0
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
            _disc(draw, p, weight, rgb)
        if not done:
            break
        budget -= seg
    return img
