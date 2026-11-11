"""Skeletonize -> polylines -> stroke ordering for draw-on animation.

Pipeline (all pure and deterministic, ties go to the lowest (y, x)). Pixel
thresholds are given at 512 px and scale with min(h, w) / 512.
  1. Fill pinholes: enclosed background components of at most
     PINHOLE_PX * scale^2 px (a thick or anti-aliased stroke leaves them, and
     each one would become a tiny skeleton loop).
  2. Skeletonize the ink mask.
  3. Classify skeleton pixels by 8-neighbour count: endpoint (1), path (2),
     junction (>= 3). Adjacent junction pixels form one junction cluster.
  4. Trace edges from endpoints and clusters along path pixels; node-less
     closed loops start at their lowest (y, x) pixel. Skeleton components of
     1-3 pixels are dots and become 2-point "tap" strokes whose `width` is the
     dot's ink diameter (Task 5 draws a filled disc).
  5. Prune spurs (endpoint-to-cluster edges shorter than MIN_SPUR_PX whose tip
     the main stroke's line width already covers) while the cluster keeps
     >= 3 edges, then join the two edges at every cluster left
     with exactly 2 (corners and staircase artefacts are not real junctions).
     Junctions of degree >= 3 are never merged through: a "+" is 4 strokes.
  6. Simplify with Douglas-Peucker (cv2.approxPolyDP, epsilon 1 px).
  7. Order greedily: the pen starts at the top-left corner (top-right when
     rtl) and each step takes the stroke whose nearer end is closest to the
     pen, reversed when its far end was the nearer one. So the first stroke is
     the one with an end nearest the corner, drawn from that end.
  8. Chain: a stroke starting within CHAIN_PX of the pen's end is appended to
     the current stroke (no pen lift). Dots are never chained.

Ink is never dropped. PACING_STROKES is only a hint for the renderer (Task 5
speeds up the reveal above it); nothing here enforces it.
"""

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, field_validator
from skimage.morphology import skeletonize

from narratoflow.errors import StrokeDataError

REFERENCE_PX = 512
MIN_SPUR_PX = 8
SPUR_REACH_PX = 2.0
CHAIN_PX = 2.0
PINHOLE_PX = 4
PACING_STROKES = 400
SIMPLIFY_EPSILON_PX = 1.0
MAX_DOT_PX = 3
DOT_TAP_PX = 0.5

Pixel = tuple[int, int]  # (y, x)
Point = tuple[float, float]  # (y, x) in pixels

_OFFSETS: tuple[Pixel, ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class Stroke(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[tuple[float, float]] = Field(min_length=2)
    # dot diameter normalized by image width; None for ordinary line strokes
    width: float | None = Field(default=None, gt=0.0, le=1.0)

    @field_validator("points")
    @classmethod
    def _normalized(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"point ({x}, {y}) is outside [0, 1]")
        return points


class StrokeSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1, le=1)
    width: PositiveInt
    height: PositiveInt
    strokes: list[Stroke]

    def to_json(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "StrokeSet":
        try:
            return cls.model_validate(json.loads(raw.decode("utf-8")))
        except (ValueError, ValidationError) as exc:
            raise StrokeDataError(f"invalid stroke data: {exc}") from exc


@dataclass(frozen=True)
class _Edge:
    a: int  # node id at pixels[0], -1 for a closed loop
    b: int  # node id at pixels[-1]
    pixels: list[Pixel]


@dataclass(frozen=True)
class _Line:
    points: list[Point]
    diameter: float | None = None  # pixels, dots only


def extract_strokes(ink: np.ndarray, *, rtl: bool = False) -> StrokeSet:
    if ink.ndim != 2 or ink.size == 0:
        raise StrokeDataError(f"ink mask must be a non-empty 2-D array, got shape {ink.shape}")
    height, width = ink.shape
    scale = min(height, width) / REFERENCE_PX
    filled = _fill_pinholes(ink.astype(bool), round(PINHOLE_PX * scale * scale))
    skel = skeletonize(filled).astype(bool)  # type: ignore[no-untyped-call]
    lines = _dots_and_edges(
        skel, filled, spur_px=MIN_SPUR_PX * scale, reach_px=SPUR_REACH_PX * scale
    )
    ordered = _order(lines, start=(0.0, float(width) if rtl else 0.0))
    return StrokeSet(
        width=width,
        height=height,
        strokes=[
            Stroke(
                points=[(x / width, y / height) for y, x in line.points],
                width=None if line.diameter is None else line.diameter / width,
            )
            for line in _chain(ordered, CHAIN_PX * scale)
        ],
    )


def _fill_pinholes(ink: np.ndarray, limit: int) -> np.ndarray:
    """Fill enclosed (not border-touching) 4-connected background holes <= limit px."""
    if limit < 1:
        return ink
    height, width = ink.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    right = left + stats[:, cv2.CC_STAT_WIDTH]
    bottom = top + stats[:, cv2.CC_STAT_HEIGHT]
    hole = (
        (stats[:, cv2.CC_STAT_AREA] <= limit)
        & (left > 0)
        & (top > 0)
        & (right < width)
        & (bottom < height)
    )
    hole[0] = False  # label 0 is the ink itself
    return np.asarray(ink | hole[labels])


def _dots_and_edges(
    skel: np.ndarray, ink: np.ndarray, *, spur_px: float, reach_px: float
) -> list[_Line]:
    """Skeleton -> simplified polylines in (y, x) pixel units."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8
    )
    lines: list[_Line] = []
    small = np.zeros(count, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= MAX_DOT_PX
    if small.any():
        _, blobs, blob_stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8
        )
    for label in np.flatnonzero(small):
        left, top, w, h = (int(v) for v in stats[label, :4])
        window = labels[top : top + h, left : left + w] == label
        ys, xs = np.nonzero(window)
        blob = int(blobs[top + int(ys[0]), left + int(xs[0])])
        # equivalent-area diameter of the dot's ink blob
        diameter = 2.0 * math.sqrt(float(blob_stats[blob, cv2.CC_STAT_AREA]) / math.pi)
        x, y = (float(v) for v in centroids[label])
        lines.append(_Line([(y, x), (y, x + DOT_TAP_PX)], max(1.0, diameter)))
    graph = skel & ~small[labels]
    depth = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    for edge in _merge(_prune(_trace(graph), depth, spur_px=spur_px, reach_px=reach_px)):
        lines.append(_Line(_simplify(edge.pixels)))
    return lines


def _trace(skel: np.ndarray) -> list[_Edge]:
    height, width = skel.shape
    padded = np.pad(skel, 1).astype(np.uint8)
    neighbours = sum(
        padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dy, dx in _OFFSETS
    )
    junction = skel & (neighbours >= 3)
    endpoint = skel & (neighbours == 1)
    _, clusters = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)

    node: dict[Pixel, int] = {}
    for y, x in zip(*np.nonzero(junction), strict=True):
        node[(int(y), int(x))] = int(clusters[y, x])
    next_id = int(clusters.max()) + 1
    endpoints: list[Pixel] = []
    for y, x in zip(*np.nonzero(endpoint), strict=True):
        node[(int(y), int(x))] = next_id
        endpoints.append((int(y), int(x)))
        next_id += 1

    pixels = {(int(y), int(x)) for y, x in zip(*np.nonzero(skel), strict=True)}

    def around(p: Pixel) -> list[Pixel]:
        return [(p[0] + dy, p[1] + dx) for dy, dx in _OFFSETS if (p[0] + dy, p[1] + dx) in pixels]

    visited: set[Pixel] = set()
    edges: list[_Edge] = []

    def walk(start: Pixel, first: Pixel) -> None:
        path = [start, first]
        visited.add(first)
        prev, cur = start, first
        while cur not in node:
            step = [n for n in around(cur) if n != prev]
            # a path pixel has exactly two neighbours, so step is one pixel
            prev, cur = cur, step[0]
            path.append(cur)
            visited.add(cur)
        edges.append(_Edge(node[start], node[cur], path))

    for start in endpoints:
        if start not in visited:
            visited.add(start)
            walk(start, around(start)[0])
    for start in sorted(p for p in node if junction[p]):
        for first in around(start):
            if first not in visited and not junction[first]:
                walk(start, first)
    for start in sorted(pixels - visited - node.keys()):
        if start in visited:
            continue
        # node-less closed loop: every pixel has exactly two neighbours
        path = [start]
        visited.add(start)
        prev, cur = start, around(start)[0]
        while cur != start:
            path.append(cur)
            visited.add(cur)
            prev, cur = cur, next(n for n in around(cur) if n != prev)
        path.append(start)
        edges.append(_Edge(-1, -1, path))
    return edges


def _degrees(edges: list[_Edge]) -> dict[int, int]:
    degree: dict[int, int] = {}
    for edge in edges:
        for end in (edge.a, edge.b):
            degree[end] = degree.get(end, 0) + 1
    return degree


def _prune(
    edges: list[_Edge], depth: np.ndarray, *, spur_px: float, reach_px: float
) -> list[_Edge]:
    """Drop short endpoint-to-junction spurs while the junction keeps >= 3 edges.

    Only spurs whose ink the remaining stroke redraws are dropped: the tip's
    ink (distance plus its own radius beyond one pixel) must lie within the
    line radius at the junction plus reach_px. A short real stroke (an
    ascender, a serif, a tick) reaches further and is kept, so pruning never
    deletes visible ink. _trace walks every endpoint first, so a spur's tip is
    always pixels[0] and its junction pixel pixels[-1].
    """
    degree = _degrees(edges)
    kept: list[_Edge] = []
    for edge in edges:
        is_spur = degree[edge.a] == 1 and degree[edge.b] >= 3
        if is_spur and len(edge.pixels) - 1 < spur_px and _redrawn(edge, depth, reach_px):
            degree[edge.a] -= 1
            degree[edge.b] -= 1
            continue
        kept.append(edge)
    return kept


def _redrawn(edge: _Edge, depth: np.ndarray, reach_px: float) -> bool:
    tip, attach = edge.pixels[0], edge.pixels[-1]
    covered = float(depth[attach]) + reach_px
    return math.dist(tip, attach) + float(depth[tip]) - 1.0 <= covered


def _merge(edges: list[_Edge]) -> list[_Edge]:
    """Join the two edges meeting at every degree-2 junction cluster.

    Builds new edges; the inputs are never mutated. Merging replaces two edges
    by one at the far nodes, so node degrees do not change and one pass over
    the degree-2 nodes in sorted order suffices.
    """
    pool: dict[int, _Edge] = dict(enumerate(edges))
    next_id = len(edges)
    incident: dict[int, list[int]] = {}
    for eid, edge in pool.items():
        for end in (edge.a, edge.b):
            if end >= 0:
                incident.setdefault(end, []).append(eid)
    for nid in sorted(n for n, ids in incident.items() if len(ids) == 2):
        first_id, second_id = incident[nid]
        if first_id == second_id:
            # a loop leaving and re-entering the same cluster: now closed
            loop = pool.pop(first_id)
            pixels = (
                loop.pixels if loop.pixels[0] == loop.pixels[-1] else [*loop.pixels, loop.pixels[0]]
            )
            pool[next_id] = _Edge(-1, -1, pixels)
            next_id += 1
            continue
        first, second = pool.pop(first_id), pool.pop(second_id)
        head = first.pixels if first.b == nid else first.pixels[::-1]
        tail = second.pixels if second.a == nid else second.pixels[::-1]
        start = first.a if first.b == nid else first.b
        end = second.b if second.a == nid else second.a
        pool[next_id] = _Edge(start, end, head + (tail[1:] if tail[0] == head[-1] else tail))
        for far, old in ((start, first_id), (end, second_id)):
            ids = incident[far]
            ids[ids.index(old)] = next_id
        next_id += 1
    return list(pool.values())


def _simplify(pixels: list[Pixel]) -> list[tuple[float, float]]:
    """Douglas-Peucker; a closed loop is split at its farthest pixel from the start
    so neither half is degenerate (start == end)."""
    if len(pixels) > 3 and pixels[0] == pixels[-1]:
        y0, x0 = pixels[0]
        far = max(
            range(len(pixels)), key=lambda i: (pixels[i][0] - y0) ** 2 + (pixels[i][1] - x0) ** 2
        )
        return _douglas_peucker(pixels[: far + 1]) + _douglas_peucker(pixels[far:])[1:]
    return _douglas_peucker(pixels)


def _douglas_peucker(pixels: list[Pixel]) -> list[tuple[float, float]]:
    contour = np.array([[[x, y]] for y, x in pixels], dtype=np.int32)
    approx = cv2.approxPolyDP(contour, SIMPLIFY_EPSILON_PX, False).reshape(-1, 2)
    return [(float(y), float(x)) for x, y in approx]


def _order(lines: list[_Line], *, start: Point) -> list[_Line]:
    """Greedy nearest-end ordering from the start corner; O(n^2) in numpy."""
    n = len(lines)
    if n == 0:
        return []
    # rows 0..n-1: each line's first point; rows n..2n-1: its last point
    heads = np.array([line.points[0] for line in lines] + [line.points[-1] for line in lines])
    alive = np.ones(2 * n, dtype=bool)
    ordered: list[_Line] = []
    pen = np.array(start)
    for _ in range(n):
        dist2 = ((heads - pen) ** 2).sum(axis=1)
        dist2[~alive] = np.inf
        best = np.flatnonzero(dist2 == dist2.min())
        # ties: lowest (y, x) of the candidate's start; lexsort is stable
        pick = int(best[np.lexsort((heads[best, 1], heads[best, 0]))[0]])
        index = pick % n
        alive[index] = alive[index + n] = False
        line = lines[index]
        if pick >= n:
            line = _Line(line.points[::-1], line.diameter)
        ordered.append(line)
        pen = np.array(line.points[-1])
    return ordered


def _chain(lines: list[_Line], chain_px: float) -> list[_Line]:
    """Append each stroke starting within chain_px of the pen to the current one."""
    chained: list[_Line] = []
    for line in lines:
        if chained:
            current = chained[-1]
            touching = math.dist(current.points[-1], line.points[0]) <= chain_px
            if touching and current.diameter is None and line.diameter is None:
                tail = line.points[1:] if line.points[0] == current.points[-1] else line.points
                chained[-1] = _Line(current.points + tail)
                continue
        chained.append(line)
    return chained
