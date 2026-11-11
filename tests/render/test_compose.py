import math
import pickle
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from narratoflow.render import compose
from narratoflow.render.assets import default_hand
from narratoflow.render.camera import ZOOM_OUT_MS
from narratoflow.render.compose import (
    PAPER,
    STAGE_FADE_MS,
    SceneJob,
    render_segment,
    scene_frames,
)
from narratoflow.render.layout import STAGE_PADDING_PX, Canvas, frame_range, frame_time_ms
from narratoflow.render.reveal import RevealCanvas
from narratoflow.render.variation import (
    ZOOM_MAX,
    Variation,
    art_size,
    choose_variation,
    sketch_to_art,
)
from narratoflow.visuals.marker import prepare_marker
from narratoflow.visuals.schedule import draw_schedule
from narratoflow.visuals.strokes import Stroke, StrokeSet

SLOT = (0, 3800)
DRAW = (200, 2800)
INK = "#0000ff"  # the hand sprite is grey, so blue ink tells them apart


def _sketch() -> StrokeSet:
    return StrokeSet(
        width=400,
        height=400,
        strokes=[
            Stroke(points=[(0.1, 0.2), (0.5, 0.2), (0.9, 0.5)]),
            Stroke(points=[(0.1, 0.8), (0.9, 0.8)]),
            Stroke(points=[(0.2, 0.3), (0.3, 0.7), (0.6, 0.6)]),
            Stroke(points=[(0.5, 0.5), (0.5, 0.5)], width=0.05),
        ],
    )


def _edge_sketch() -> StrokeSet:
    return StrokeSet(
        width=400,
        height=400,
        strokes=[
            Stroke(points=[(1.0, 0.0), (1.0, 1.0)]),
            Stroke(points=[(0.0, 1.0), (1.0, 1.0)]),
            Stroke(points=[(1.0, 1.0), (0.95, 0.95)]),
        ],
    )


def _plain_variation(n: int, **update: object) -> Variation:
    base = Variation(
        mirrored=False,
        rotation_deg=0.0,
        scale=1.0,
        dx=0.0,
        dy=0.0,
        line_weight_factor=1.0,
        reversed_strokes=[],
        wobble_seed=7,
    )
    return base.model_copy(update=update) if update else base


def _job(
    canvas: Canvas,
    *,
    kind: str = "sketch",
    strokes: StrokeSet | None = None,
    slot: tuple[int, int] = SLOT,
    draw: tuple[int, int] | None = DRAW,
    variation: Variation | None = None,
    line_weight_px: float = 6.0,
) -> SceneJob:
    if kind == "sketch":
        strokes = strokes if strokes is not None else _sketch()
        if variation is None:
            variation = choose_variation(
                "ch/T", "s1", mirror_safe=True, stroke_count=len(strokes.strokes)
            )
    return SceneJob(
        scene_id="s1",
        kind=kind,  # type: ignore[arg-type]
        frames=frame_range(*slot),
        slot=slot,
        draw=draw if kind == "sketch" else None,
        strokes=strokes if kind == "sketch" else None,
        variation=variation if kind == "sketch" else None,
        wobble_seed=12345,
        line_weight_px=line_weight_px,
        ink=INK,
        canvas=canvas,
        hand=default_hand(),
        segment=Path("seg-000.mkv"),
        ffmpeg=Path("/nonexistent/ffmpeg"),
    )


def _stage(frame: np.ndarray, canvas: Canvas) -> np.ndarray:
    s = canvas.stage
    return frame[s.y : s.bottom, s.x : s.right]


def _ink(region: np.ndarray) -> int:
    return int((region.astype(np.int32).sum(axis=2) < 3 * 128).sum())


def _hand(region: np.ndarray) -> int:
    """Hand pixels: the sprite is dark grey, the ink is blue."""
    r = region.astype(np.int32)
    spread = r.max(axis=2) - r.min(axis=2)
    return int(((spread <= 12) & (r.max(axis=2) < 200)).sum())


def _outside_is_paper(frame: np.ndarray, canvas: Canvas) -> bool:
    s = canvas.stage
    mask = np.ones(frame.shape[:2], dtype=bool)
    mask[s.y : s.bottom, s.x : s.right] = False
    return bool((frame[mask] == np.array(PAPER, dtype=np.uint8)).all())


def _is_paper(region: np.ndarray) -> bool:
    return bool((region == np.array(PAPER, dtype=np.uint8)).all())


def test_constants() -> None:
    assert PAPER == (255, 255, 255)
    assert STAGE_FADE_MS == 200


def test_job_is_picklable(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas)
    assert pickle.loads(pickle.dumps(job)) == job  # noqa: S301


def test_frame_count_equals_range(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas)
    frames = list(scene_frames(job))
    assert len(frames) == len(job.frames)
    for f in frames:
        assert f.shape == (tiny_canvas.height, tiny_canvas.width, 3)
        assert f.dtype == np.uint8


def test_nothing_outside_stage_ever(tiny_canvas: Canvas, monkeypatch: pytest.MonkeyPatch) -> None:
    strokes = _edge_sketch()
    variation = _plain_variation(len(strokes.strokes), dx=0.03, dy=0.03, scale=1.0)
    job = _job(tiny_canvas, strokes=strokes, variation=variation)
    overflowed: list[bool] = []
    real_paste = compose._paste_hand

    def spy(
        stage: np.ndarray, sprite: np.ndarray, tip: tuple[int, int], at: tuple[float, float]
    ) -> None:
        x0, y0 = round(at[0]) - tip[0], round(at[1]) - tip[1]
        sh, sw = stage.shape[:2]
        overflowed.append(x0 + sprite.shape[1] > sw or y0 + sprite.shape[0] > sh)
        real_paste(stage, sprite, tip, at)

    monkeypatch.setattr(compose, "_paste_hand", spy)
    saw_hand = False
    for frame in scene_frames(job):
        assert _outside_is_paper(frame, tiny_canvas)
        saw_hand = saw_hand or _hand(_stage(frame, tiny_canvas)) > 0
    assert saw_hand
    # The unclipped sprite would reach past the stage, so the clip is exercised.
    assert any(overflowed)


def test_hand_visible_only_while_drawing(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas)
    for k, frame in zip(job.frames, scene_frames(job), strict=True):
        t = frame_time_ms(k)
        hand = _hand(_stage(frame, tiny_canvas))
        if DRAW[0] < t < DRAW[1]:
            assert hand > 0, t
        elif t < DRAW[0] or t > DRAW[1]:
            assert hand == 0, t


def _overview_oracle(job: SceneJob) -> np.ndarray:
    """The complete sketch on the art canvas, scaled straight onto the stage (zoom 1)."""
    assert job.strokes is not None and job.variation is not None and job.draw is not None
    canvas, strokes, v = job.canvas, job.strokes, job.variation
    art = art_size(canvas)
    to_art, combined = sketch_to_art(strokes, v, art, STAGE_PADDING_PX * float(canvas.scale))
    prepared = prepare_marker(
        strokes,
        art,
        line_weight_px=job.line_weight_px * v.line_weight_factor * ZOOM_MAX * float(canvas.scale),
        seed=job.wobble_seed,
        to_pixels=to_art,
        reverse=[i in v.reversed_strokes for i in range(len(strokes.strokes))],
        dot_px_per_unit=combined * strokes.width,
    )
    events = draw_schedule(strokes, *job.draw)
    full = np.asarray(RevealCanvas(prepared, events, art, job.ink).at(float(job.draw[1])))
    s = canvas.stage
    matrix = np.array([[s.w / art[0], 0.0, 0.0], [0.0, s.h / art[1], 0.0]])
    return cv2.warpAffine(full, matrix, (s.w, s.h), borderValue=PAPER)


def test_hold_shows_complete_sketch_then_fades(tiny_canvas: Canvas) -> None:
    assert DRAW[1] + ZOOM_OUT_MS <= SLOT[1] - STAGE_FADE_MS
    job = _job(tiny_canvas)
    frames = dict(zip(job.frames, scene_frames(job), strict=True))
    hold_k = next(k for k in job.frames if frame_time_ms(k) >= DRAW[1] + ZOOM_OUT_MS)
    got = _ink(_stage(frames[hold_k], tiny_canvas))
    want = _ink(_overview_oracle(job))
    assert want > 0
    assert abs(got - want) <= 0.02 * want
    last = frames[job.frames[-1]]
    assert _is_paper(last)
    before_fade = max(k for k in job.frames if frame_time_ms(k) <= SLOT[1] - STAGE_FADE_MS)
    assert _ink(_stage(frames[before_fade], tiny_canvas)) > 0
    mid = [frames[k] for k in job.frames if frame_time_ms(k) > SLOT[1] - STAGE_FADE_MS]
    inks = [_stage(f, tiny_canvas).astype(np.int32).sum() for f in mid]
    assert inks == sorted(inks)


def test_fade_starts_at_draw_end_when_later(tiny_canvas: Canvas) -> None:
    slot = (0, 1000)
    job = _job(tiny_canvas, slot=slot, draw=(0, 900))
    frames = list(scene_frames(job))
    assert _is_paper(frames[-1])
    at_draw_end = next(
        f for k, f in zip(job.frames, frames, strict=True) if frame_time_ms(k) >= 900
    )
    assert _ink(_stage(at_draw_end, tiny_canvas)) > 0


def test_text_card_and_paper_scenes_never_fade(tiny_canvas: Canvas) -> None:
    card = list(scene_frames(_job(tiny_canvas, kind="text_card")))
    assert _ink(_stage(card[-1], tiny_canvas)) > 0
    assert np.array_equal(card[0], card[-1])
    paper = list(scene_frames(_job(tiny_canvas, kind="paper")))
    assert np.array_equal(paper[0], paper[-1])


def test_text_card_frame_has_all_four_sides(tiny_canvas: Canvas) -> None:
    frame = next(iter(scene_frames(_job(tiny_canvas, kind="text_card"))))
    stage = _stage(frame, tiny_canvas)
    assert _outside_is_paper(frame, tiny_canvas)
    h, w = stage.shape[:2]
    band_h, band_w = max(2, h // 6), max(2, w // 6)
    assert _ink(stage[:band_h]) > 0
    assert _ink(stage[-band_h:]) > 0
    assert _ink(stage[:, :band_w]) > 0
    assert _ink(stage[:, -band_w:]) > 0
    assert _ink(stage[band_h:-band_h, band_w:-band_w]) == 0


def test_text_card_static_frame(tiny_canvas: Canvas) -> None:
    frames = list(scene_frames(_job(tiny_canvas, kind="text_card")))
    assert all(np.array_equal(frames[0], f) for f in frames)


def test_paper_scene_blank(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas, kind="paper")
    frames = list(scene_frames(job))
    assert len(frames) == len(job.frames)
    assert all(_is_paper(f) for f in frames)


def _dot_set(width: int, height: int) -> StrokeSet:
    return StrokeSet(
        width=width,
        height=height,
        strokes=[Stroke(points=[(0.5, 0.5), (0.5, 0.5)], width=0.2)],
    )


def test_dots_scale_with_sketch() -> None:
    canvas = Canvas(Fraction(1, 2))
    areas = []
    for size in ((100, 400), (200, 800)):
        strokes = _dot_set(*size)
        job = _job(
            canvas,
            strokes=strokes,
            variation=_plain_variation(1),
            draw=(0, 100),
            slot=(0, 1000),
            line_weight_px=1.0,
        )
        frame = list(scene_frames(job))[6]
        areas.append(_ink(_stage(frame, canvas)))
    assert areas[0] > 0
    assert abs(areas[0] - areas[1]) <= 0.05 * areas[0]
    # The sketch is height-bound: its width on the art is about a quarter of its height.
    art = art_size(canvas)
    sketch_w_stage = (
        (art[1] - 2 * STAGE_PADDING_PX * float(canvas.scale) * ZOOM_MAX) * 0.25 / ZOOM_MAX
    )
    want = math.pi / 4 * (0.2 * sketch_w_stage) ** 2
    assert want * 0.7 <= areas[0] <= want * 1.4


def test_zero_length_strokes_and_zero_draw_window_render(tiny_canvas: Canvas) -> None:
    strokes = StrokeSet(
        width=100,
        height=100,
        strokes=[
            Stroke(points=[(0.5, 0.5), (0.5, 0.5)]),
            Stroke(points=[(0.2, 0.2), (0.8, 0.2)]),
        ],
    )
    job = _job(tiny_canvas, strokes=strokes, draw=(1000, 1000), slot=(0, 2000))
    frames = list(scene_frames(job))
    assert len(frames) == len(job.frames)
    assert _is_paper(_stage(frames[0], tiny_canvas))
    after = next(f for k, f in zip(job.frames, frames, strict=True) if frame_time_ms(k) > 1000)
    assert _ink(_stage(after, tiny_canvas)) > 0
    assert all(_hand(_stage(f, tiny_canvas)) == 0 for f in frames)


def test_zero_strokes_is_paper(tiny_canvas: Canvas) -> None:
    strokes = StrokeSet(width=10, height=10, strokes=[])
    job = _job(tiny_canvas, strokes=strokes, slot=(0, 500), draw=(0, 400))
    assert all(_is_paper(f) for f in scene_frames(job))


def test_sketch_without_draw_times_is_a_precondition_error(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas, draw=None)
    with pytest.raises(ValueError, match="draw"):
        list(scene_frames(job))
    with pytest.raises(ValueError, match="strokes"):
        list(scene_frames(replace(_job(tiny_canvas), strokes=None)))


def test_time_is_clamped_to_slot(tiny_canvas: Canvas) -> None:
    # Frames past the slot end hold the slot's last state (paper after the fade).
    job = replace(_job(tiny_canvas, slot=(0, 1000), draw=(0, 600)), frames=range(0, 40))
    frames = list(scene_frames(job))
    assert len(frames) == 40
    assert all(_is_paper(f) for f in frames[30:])


def test_scene_frames_deterministic(tiny_canvas: Canvas) -> None:
    job = _job(tiny_canvas)
    a = list(scene_frames(job))
    b = list(scene_frames(job))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_reveal_frame_consumed_before_next_at(
    tiny_canvas: Canvas, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`RevealCanvas.at()` may hand back a shared image; frames must not alias it."""
    job = _job(tiny_canvas)
    clean = list(scene_frames(job))

    class Scribbling(RevealCanvas):
        last: Image.Image | None = None

        def at(self, t_ms: float) -> Image.Image:
            if self.last is not None:
                ImageDraw.Draw(self.last).rectangle((0, 0, 10**4, 10**4), fill=(255, 0, 0))
            self.last = super().at(t_ms).copy()
            return self.last

    monkeypatch.setattr(compose, "RevealCanvas", Scribbling)
    dirty = list(scene_frames(job))
    assert all(np.array_equal(x, y) for x, y in zip(clean, dirty, strict=True))


def test_render_segment_writes_every_frame(
    tiny_canvas: Canvas, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[np.ndarray] = []

    class FakeSink:
        def __init__(self, ffmpeg: Path, path: Path, size: tuple[int, int]) -> None:
            assert size == (tiny_canvas.width, tiny_canvas.height)
            self.frames = 0

        def __enter__(self) -> "FakeSink":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def write(self, frame: np.ndarray) -> None:
            written.append(frame)
            self.frames += 1

    monkeypatch.setattr(compose, "FrameSink", FakeSink)
    job = _job(tiny_canvas, kind="paper", slot=(0, 300))
    assert render_segment(job) == len(job.frames) == len(written)


@pytest.mark.parametrize("draw_end", [3800, 3790, 4500])
def test_draw_to_slot_end_still_ends_on_paper(tiny_canvas: Canvas, draw_end: int) -> None:
    """draw_fraction 1.0: draw_end >= the last frame's time; the last frame is still paper."""
    job = _job(tiny_canvas, slot=SLOT, draw=(DRAW[0], draw_end))
    frames = list(scene_frames(job))
    assert frame_time_ms(job.frames[-1]) <= draw_end
    assert _is_paper(frames[-1])
    assert _hand(_stage(frames[-1], tiny_canvas)) == 0
    assert _hand(_stage(frames[-2], tiny_canvas)) > 0


def test_normal_fade_start_unchanged(tiny_canvas: Canvas) -> None:
    """With draw_end before slot_end - STAGE_FADE_MS, the fade starts at slot_end - 200."""
    job = _job(tiny_canvas)
    start = SLOT[1] - STAGE_FADE_MS
    frames = dict(zip(job.frames, scene_frames(job), strict=True))
    hold = [k for k in job.frames if DRAW[1] + ZOOM_OUT_MS <= frame_time_ms(k) <= start]
    assert len(hold) > 1
    assert all(np.array_equal(frames[hold[0]], frames[k]) for k in hold)
    first_fading = hold[-1] + 1
    assert frame_time_ms(first_fading) > start
    lighter = _stage(frames[first_fading], tiny_canvas).astype(np.int64).sum()
    assert lighter > _stage(frames[hold[-1]], tiny_canvas).astype(np.int64).sum()


def test_paste_hand_fully_off_stage_leaves_stage_unchanged() -> None:
    stage = np.full((20, 20, 3), 200, dtype=np.uint8)
    sprite = np.full((8, 8, 4), 255, dtype=np.uint8)
    for at in [(-50.0, 5.0), (500.0, 5.0), (5.0, -50.0), (5.0, 500.0)]:
        compose._paste_hand(stage, sprite, (0, 0), at)
    assert (stage == 200).all()
