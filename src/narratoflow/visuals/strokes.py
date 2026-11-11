"""Skeletonize -> polylines -> stroke ordering for draw-on animation.

Pipeline (all pure and deterministic, ties go to the lowest (y, x)):
  1. Skeletonize the ink mask.
  2. Classify skeleton pixels by 8-neighbour count: endpoint (1), path (2),
     junction (>= 3). Adjacent junction pixels form one junction cluster.
  3. Trace edges from endpoints and clusters along path pixels; node-less
     closed loops start at their lowest (y, x) pixel. Skeleton components of
     1-3 pixels are dots and become 2-point "tap" strokes.
  4. Prune spurs (endpoint-to-cluster edges shorter than MIN_SPUR_PX) while the
     cluster keeps >= 3 edges, then join the two edges at every cluster left
     with exactly 2 (corners and staircase artefacts are not real junctions).
     Junctions of degree >= 3 are never merged through: a "+" is 4 strokes.
  5. Simplify with Douglas-Peucker (cv2.approxPolyDP, epsilon 1 px).
  6. Cap at MAX_STROKES by dropping the shortest (polyline length) strokes.
  7. Order greedily: the pen starts at the top-left corner (top-right when
     rtl) and each step takes the stroke whose nearer end is closest to the
     pen, reversed when its far end was the nearer one. So the first stroke is
     the one with an end nearest the corner, drawn from that end.
"""

import itertools
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, field_validator
from skimage.morphology import skeletonize

from narratoflow.errors import StrokeDataError

MIN_SPUR_PX = 8
MAX_STROKES = 400
SIMPLIFY_EPSILON_PX = 1.0
MAX_DOT_PX = 3
DOT_TAP_PX = 0.5

Pixel = tuple[int, int]  # (y, x)

_OFFSETS: tuple[Pixel, ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class Stroke(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[tuple[float, float]] = Field(min_length=2)

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
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "StrokeSet":
        try:
            return cls.model_validate(json.loads(raw.decode("utf-8")))
        except (ValueError, ValidationError) as exc:
            raise StrokeDataError(f"invalid stroke data: {exc}") from exc


@dataclass
class _Edge:
    a: int  # node id at pixels[0], -1 for a closed node-less loop
    b: int  # node id at pixels[-1]
    pixels: list[Pixel]


def extract_strokes(ink: np.ndarray, *, rtl: bool = False) -> StrokeSet:
    if ink.ndim != 2 or ink.size == 0:
        raise StrokeDataError(f"ink mask must be a non-empty 2-D array, got shape {ink.shape}")
    height, width = ink.shape
    skel = skeletonize(ink.astype(bool)).astype(bool)  # type: ignore[no-untyped-call]
    polylines = _dots_and_edges(skel)
    if len(polylines) > MAX_STROKES:
        lengths = [_length(p) for p in polylines]
        keep = sorted(range(len(polylines)), key=lambda i: -lengths[i])[:MAX_STROKES]
        polylines = [polylines[i] for i in sorted(keep)]
    ordered = _order(polylines, start=(0.0, float(width) if rtl else 0.0))
    return StrokeSet(
        width=width,
        height=height,
        strokes=[Stroke(points=[(x / width, y / height) for y, x in line]) for line in ordered],
    )


def _dots_and_edges(skel: np.ndarray) -> list[list[tuple[float, float]]]:
    """Skeleton -> simplified polylines in (y, x) pixel units."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8
    )
    lines: list[list[tuple[float, float]]] = []
    small = np.zeros(count, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= MAX_DOT_PX
    dot_mask = small[labels]
    for label in np.flatnonzero(small):
        ys, xs = np.nonzero(labels == label)
        y, x = float(ys.mean()), float(xs.mean())
        lines.append([(y, x), (y, x + DOT_TAP_PX)])
    graph = skel & ~dot_mask
    for edge in _merge(_prune(_trace(graph))):
        lines.append(_simplify(edge.pixels))
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


def _prune(edges: list[_Edge]) -> list[_Edge]:
    """Drop short endpoint-to-junction spurs while the junction keeps >= 3 edges."""
    degree = _degrees(edges)
    kept: list[_Edge] = []
    for edge in edges:
        ends = (degree[edge.a], degree[edge.b])
        is_spur = edge.a != edge.b and 1 in ends and max(ends) >= 3
        if is_spur and len(edge.pixels) - 1 < MIN_SPUR_PX:
            degree[edge.a] -= 1
            degree[edge.b] -= 1
            continue
        kept.append(edge)
    return kept


def _merge(edges: list[_Edge]) -> list[_Edge]:
    """Join the two edges meeting at every degree-2 junction cluster."""
    alive = list(edges)
    changed = True
    while changed:
        changed = False
        degree = _degrees(alive)
        for nid in sorted(n for n, d in degree.items() if d == 2 and n >= 0):
            incident = [e for e in alive if nid in (e.a, e.b)]
            if len(incident) == 1:
                # a loop leaving and re-entering the same cluster: now closed
                loop = incident[0]
                loop.a = loop.b = -1
                if loop.pixels[0] != loop.pixels[-1]:
                    loop.pixels.append(loop.pixels[0])
                continue
            first, second = incident
            head = first.pixels if first.b == nid else first.pixels[::-1]
            tail = second.pixels if second.a == nid else second.pixels[::-1]
            start = first.a if first.b == nid else first.b
            end = second.b if second.a == nid else second.a
            joined = head + (tail[1:] if tail[0] == head[-1] else tail)
            alive = [e for e in alive if e is not first and e is not second]
            alive.append(_Edge(start, end, joined))
            changed = True
            break
    return alive


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


def _length(line: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(line))


def _order(
    lines: list[list[tuple[float, float]]], *, start: tuple[float, float]
) -> list[list[tuple[float, float]]]:
    """Greedy nearest-end ordering from the start corner; O(n^2)."""
    remaining = list(lines)
    ordered: list[list[tuple[float, float]]] = []
    pen = start
    while remaining:
        best_key: tuple[float, float, float] | None = None
        best_index = 0
        best_line: list[tuple[float, float]] = []
        for index, line in enumerate(remaining):
            for candidate in (line, line[::-1]):
                head = candidate[0]
                dy, dx = head[0] - pen[0], head[1] - pen[1]
                key = (dy * dy + dx * dx, head[0], head[1])
                if best_key is None or key < best_key:
                    best_key, best_index, best_line = key, index, candidate
        ordered.append(best_line)
        pen = best_line[-1]
        del remaining[best_index]
    return ordered
