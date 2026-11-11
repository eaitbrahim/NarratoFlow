"""Pinned 9:16 geometry (1080x1920), safe zones and the frame clock."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

FPS = 30
FRAME_W = 1080
FRAME_H = 1920
STAGE_PADDING_PX = 24


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def contains(self, other: Rect) -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def intersects(self, other: Rect) -> bool:
        return (
            self.x < other.right
            and other.x < self.right
            and self.y < other.bottom
            and other.y < self.bottom
        )

    def scaled(self, f: Fraction) -> Rect:
        x0 = round(self.x * f)
        y0 = round(self.y * f)
        x1 = round(self.right * f)
        y1 = round(self.bottom * f)
        return Rect(x0, y0, x1 - x0, y1 - y0)


PLATFORM_TOP = Rect(0, 0, 1080, 160)
RIGHT_RAIL = Rect(888, 768, 192, 1152)
PLATFORM_BOTTOM = Rect(0, 1440, 1080, 480)
LEFT_MARGIN = Rect(0, 0, 60, 1920)

TOP_BAR = Rect(60, 170, 960, 200)
STAGE = Rect(60, 390, 828, 850)
CAPTIONS = Rect(60, 1260, 828, 180)

_ALLOWED_SCALES = frozenset({Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)})


@dataclass(frozen=True)
class Canvas:
    scale: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if self.scale not in _ALLOWED_SCALES:
            raise ValueError(f"unsupported canvas scale {self.scale}; use 1, 1/2, 1/3 or 1/6")

    @property
    def width(self) -> int:
        return int(FRAME_W * self.scale)

    @property
    def height(self) -> int:
        return int(FRAME_H * self.scale)

    @property
    def stage(self) -> Rect:
        return STAGE.scaled(self.scale)

    @property
    def captions(self) -> Rect:
        return CAPTIONS.scaled(self.scale)

    @property
    def top_bar(self) -> Rect:
        return TOP_BAR.scaled(self.scale)


def frame_of(ms: int) -> int:
    """Round half up of ms * FPS / 1000."""
    return (ms * FPS * 2 + 1000) // 2000


def frame_time_ms(k: int) -> float:
    return k * 1000 / FPS


def frame_range(start_ms: int, end_ms: int) -> range:
    return range(frame_of(start_ms), frame_of(end_ms))
