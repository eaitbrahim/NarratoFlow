"""Compose one scene's frames (stage, camera, hand, fade) and write them as a segment.

Everything outside the stage rect is always paper: the art is warped into a
stage-sized buffer and the hand is composited into that buffer only, so numpy
slicing clips it to the stage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from narratoflow.render.assets import HandSprite
from narratoflow.render.camera import CameraTrack
from narratoflow.render.encoder import FrameSink
from narratoflow.render.layout import STAGE_PADDING_PX, Canvas, frame_time_ms
from narratoflow.render.reveal import RevealCanvas, pen_at
from narratoflow.render.variation import (
    ZOOM_MAX,
    Variation,
    art_size,
    card_frame_strokes,
    sketch_to_art,
)
from narratoflow.visuals.marker import PreparedStroke, prepare_marker, render_marker
from narratoflow.visuals.schedule import StrokeEvent, draw_schedule
from narratoflow.visuals.strokes import StrokeSet

PAPER = (255, 255, 255)
STAGE_FADE_MS = 200


@dataclass(frozen=True)
class SceneJob:
    scene_id: str
    kind: Literal["sketch", "text_card", "paper"]
    frames: range
    slot: tuple[int, int]
    draw: tuple[int, int] | None
    strokes: StrokeSet | None
    variation: Variation | None
    wobble_seed: int
    line_weight_px: float
    ink: str
    canvas: Canvas
    hand: HandSprite
    segment: Path
    ffmpeg: Path


def _frame_times(job: SceneJob) -> Iterator[float]:
    lo, hi = job.slot
    for k in job.frames:
        yield min(max(frame_time_ms(k), float(lo)), float(hi))


def _paper_frame(canvas: Canvas) -> np.ndarray:
    return np.full((canvas.height, canvas.width, 3), PAPER, dtype=np.uint8)


def _static_frames(job: SceneJob, stage: np.ndarray | None) -> Iterator[np.ndarray]:
    base = _paper_frame(job.canvas)
    if stage is not None:
        s = job.canvas.stage
        base[s.y : s.bottom, s.x : s.right] = stage
    for _ in job.frames:
        yield base.copy()


def _card_stage(job: SceneJob) -> np.ndarray:
    s = job.canvas.stage
    img = render_marker(
        card_frame_strokes(),
        (s.w, s.h),
        line_weight_px=job.line_weight_px * float(job.canvas.scale),
        ink=job.ink,
        seed=job.wobble_seed,
        upto=1.0,
    )
    return np.array(img, dtype=np.uint8)


def _paste_hand(
    stage: np.ndarray, sprite: np.ndarray, tip: tuple[int, int], at: tuple[float, float]
) -> None:
    """Alpha-composite `sprite` with its tip at `at` (stage px), clipped to `stage`."""
    sh, sw = stage.shape[:2]
    hh, hw = sprite.shape[:2]
    x0 = round(at[0]) - tip[0]
    y0 = round(at[1]) - tip[1]
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + hw, sw), min(y0 + hh, sh)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    patch = sprite[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    dst = stage[cy0:cy1, cx0:cx1].astype(np.float32)
    out = patch[:, :, :3].astype(np.float32) * alpha + dst * (1.0 - alpha)
    stage[cy0:cy1, cx0:cx1] = np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _fade(t: float, start: float, last: float) -> float:
    """0 before `start`, rising linearly to exactly 1 at `last`."""
    if t < start:
        return 0.0
    if t >= last:
        return 1.0
    return (t - start) / (last - start)


def _sketch_frames(job: SceneJob) -> Iterator[np.ndarray]:
    if job.draw is None:
        raise ValueError(f"scene {job.scene_id}: a sketch scene needs draw times")
    if job.strokes is None or job.variation is None:
        raise ValueError(f"scene {job.scene_id}: a sketch scene needs strokes and a variation")
    strokes, variation, canvas = job.strokes, job.variation, job.canvas
    if not strokes.strokes:
        yield from _static_frames(job, None)
        return
    s = canvas.stage
    art = art_size(canvas)
    pad = STAGE_PADDING_PX * float(canvas.scale)
    to_art, combined = sketch_to_art(strokes, variation, art, pad)
    reversed_set = set(variation.reversed_strokes)
    prepared: list[PreparedStroke] = prepare_marker(
        strokes,
        art,
        line_weight_px=job.line_weight_px
        * variation.line_weight_factor
        * ZOOM_MAX
        * float(canvas.scale),
        seed=job.wobble_seed,
        to_pixels=to_art,
        reverse=[i in reversed_set for i in range(len(strokes.strokes))],
        dot_px_per_unit=combined * strokes.width,
    )
    events: list[StrokeEvent] = draw_schedule(strokes, *job.draw)
    reveal = RevealCanvas(prepared, events, art, job.ink)
    camera = CameraTrack(prepared, events, art, job.draw)
    sprite, tip = job.hand.image(canvas.scale)
    fade_start = float(max(job.slot[1] - STAGE_FADE_MS, job.draw[1]))
    # The fade ends on the slot's last frame (clamped like every frame time).
    last = min(frame_time_ms(job.frames[-1]), float(job.slot[1])) if job.frames else 0.0
    paper = np.array(PAPER, dtype=np.float32)
    for t in _frame_times(job):
        matrix = camera.affine(camera.at(t), (s.w, s.h))
        # The reveal image may be shared and redrawn by the next at(): warp it now.
        art_rgb = np.asarray(reveal.at(t))
        stage = cv2.warpAffine(
            art_rgb,
            matrix,
            (s.w, s.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=PAPER,
        )
        del art_rgb
        pen = pen_at(prepared, events, t)
        if pen is not None:
            px = matrix[0, 0] * pen[0] + matrix[0, 1] * pen[1] + matrix[0, 2]
            py = matrix[1, 0] * pen[0] + matrix[1, 1] * pen[1] + matrix[1, 2]
            _paste_hand(stage, sprite, tip, (float(px), float(py)))
        f = _fade(t, fade_start, last)
        if f > 0.0:
            mixed = stage.astype(np.float32) * (1.0 - f) + paper * f
            stage = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        frame = _paper_frame(canvas)
        frame[s.y : s.bottom, s.x : s.right] = stage
        yield frame


def scene_frames(job: SceneJob) -> Iterator[np.ndarray]:
    """`len(job.frames)` HxWx3 uint8 frames, each at its frame time clamped to the slot."""
    if job.kind == "sketch":
        yield from _sketch_frames(job)
    elif job.kind == "text_card":
        yield from _static_frames(job, _card_stage(job))
    else:
        yield from _static_frames(job, None)


def render_segment(job: SceneJob) -> int:
    """Write the scene into `job.segment`; returns the frames written. Picklable target."""
    canvas = job.canvas
    with FrameSink(job.ffmpeg, job.segment, (canvas.width, canvas.height)) as sink:
        for frame in scene_frames(job):
            sink.write(frame)
    return sink.frames
