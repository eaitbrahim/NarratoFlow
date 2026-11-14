import io
from collections.abc import Callable
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pytest
from fixtures import images
from PIL import Image

from narratoflow.errors import SketchRejectedError
from narratoflow.visuals import lineart
from narratoflow.visuals.lineart import (
    binarize,
    check_density,
    ink_to_png,
    xdog_lineart,
)
from narratoflow.visuals.strokes import PACING_STROKES, StrokeSet, extract_strokes
from visuals.lineart_samples import SAMPLES, sample_png


def test_line_drawing_passes() -> None:
    report = check_density(binarize(images.line_drawing_scroll()))
    assert report.ok
    assert report.reason is None
    assert 0.0 < report.ink_ratio < 0.05
    assert 2.5 < report.mean_stroke_width < 4.0
    assert report.max_tile_ink < lineart.DENSE_TILE_INK
    assert report.dense_tiles == 0
    assert 0.0 < report.junction_clusters_per_kpx < 10.0


def test_filled_silhouette_rejected_by_stroke_width() -> None:
    report = check_density(binarize(images.filled_silhouette()))
    assert not report.ok
    assert report.reason is not None
    assert report.reason.startswith("stroke_width ")
    assert report.reason.endswith(" > 12")


def test_crosshatching_rejected_by_dense_tiles() -> None:
    report = check_density(binarize(images.crosshatched_box()))
    assert not report.ok
    assert report.ink_ratio <= lineart.MAX_INK_RATIO
    assert report.junction_clusters_per_kpx > 0.0
    assert report.reason is not None
    assert report.reason.startswith(f"dense_tiles {report.dense_tiles} >= ")


def test_ink_ratio_backstop_rejects_scribble() -> None:
    report = check_density(binarize(images.road_map(roads=80)))
    assert not report.ok
    assert report.reason is not None
    assert report.reason.startswith("ink_ratio ")


def test_stroke_width_limit_scales_with_line_weight() -> None:
    ink = binarize(images.scroll(size=1024, width=14))
    assert check_density(ink).reason is not None
    assert check_density(ink, line_weight_px=6.0).ok


def test_noise_specks_removed() -> None:
    clean = binarize(images.line_drawing_scroll())
    noisy = binarize(images.noisy_line_drawing())
    assert np.array_equal(clean & noisy, clean)  # strokes untouched
    # no free-standing speck survives: every component left holds stroke ink
    # (salt fused onto a stroke is indistinguishable from the stroke)
    count, labels = cv2.connectedComponents(noisy.astype(np.uint8), connectivity=8)
    assert set(np.unique(labels[clean])) == set(range(1, count))


@pytest.mark.parametrize("width", [1, 2])
def test_thin_line_art_survives(width: int) -> None:
    ink = binarize(images.scroll(width=width))
    assert ink.sum() > 0.8 * binarize(images.scroll()).sum() * width / 3
    assert check_density(ink).ok


_CROSS = (45, 135)
_GRID = (0, 90)
_MUST_REJECT_HATCH = [
    pytest.param(_CROSS, w, s, id=f"cross45-w{w}-s{s}") for w in (2, 3) for s in range(6, 11)
] + [pytest.param(_GRID, w, s, id=f"grid-w{w}-s{s}") for w in (2, 3) for s in range(6, 11)]
# single-direction parallel hatching (0/45/30 deg, 2-3 px, spacing 6-8) is deliberately
# left to the vision outlier check: no assertion either way (see the calibration table)


@pytest.mark.parametrize(("angles", "width", "spacing"), _MUST_REJECT_HATCH)
def test_sweep_hatching_rejected(angles: tuple[int, ...], width: int, spacing: int) -> None:
    report = check_density(binarize(images.hatch(spacing=spacing, width=width, angles=angles)))
    assert not report.ok, report
    assert report.second_orientation_ratio >= lineart.MIN_SECOND_ORIENTATION


def test_scene_with_crosshatched_shadow_rejected() -> None:
    report = check_density(binarize(images.scene_with_shadow()))
    assert report.reason is not None
    assert report.reason.startswith("dense_tiles ")


_RULINGS: dict[str, Callable[..., bytes]] = {
    "notebook-8": partial(images.notebook, spacing=8),
    "notebook-10": partial(images.notebook, spacing=10),
    "scroll-text-10": partial(images.scroll_text, spacing=10),
    "brick-20x10": partial(images.brick, brick_w=20, brick_h=10),
    "brick-24x12": partial(images.brick, brick_w=24, brick_h=12),
    "fence-8": partial(images.fence, gap=8),
    "fence-10": partial(images.fence, gap=10),
    "blinds-8": partial(images.blinds, slat=8),
    "blinds-10": partial(images.blinds, slat=10),
    "ladder-10": partial(images.ladder, rung=10),
    "stripes-8": partial(images.stripes, period=8),
}


@pytest.mark.parametrize("size", [512, 1024])
@pytest.mark.parametrize("name", list(_RULINGS))
def test_single_direction_rulings_pass(name: str, size: int) -> None:
    # ruled pages, fences, blinds, brick: one dominant orientation, not crosshatch
    report = check_density(binarize(_RULINGS[name](size=size)))
    assert report.ok, report
    assert report.second_orientation_ratio < 0.6 * lineart.MIN_SECOND_ORIENTATION


@pytest.mark.parametrize("size", [512, 1024, 2048])
def test_pen_dots_survive_speck_removal(size: int) -> None:
    ink = binarize(images.pen_dots(size=size))
    count = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)[0] - 1
    assert count == 1 + len(images.PEN_DOT_FACTORS)  # frame + every dot


_SPARSE_PATTERNS = [
    pytest.param(angles, w, s, id=f"{name}-w{w}-s{s}")
    for name, angles in (("cross45", _CROSS), ("grid", _GRID), ("parallel0", (0,)))
    for w in (2, 3)
    for s in (20, 24, 32, 40)
]


@pytest.mark.parametrize(("angles", "width", "spacing"), _SPARSE_PATTERNS)
def test_sparse_regular_patterns_pass(angles: tuple[int, ...], width: int, spacing: int) -> None:
    # lattices, fences and chart grids are legitimate; a FAIL here is unrecoverable
    report = check_density(binarize(images.hatch(spacing=spacing, width=width, angles=angles)))
    assert report.ok, report


@pytest.mark.parametrize("coverage", [0.10, 0.15, 0.20, 0.30])
def test_sweep_silhouettes_rejected(coverage: float) -> None:
    report = check_density(binarize(images.silhouette(coverage)))
    assert not report.ok
    assert report.reason is not None
    assert report.reason.startswith("stroke_width ")


_LEGIT: dict[str, Callable[[], bytes]] = {
    "scroll-512": images.scroll,
    "scroll-1024": partial(images.scroll, size=1024),
    "scroll-1024-w6": partial(images.scroll, size=1024, width=6),
    "scroll-512-w6": partial(images.scroll, width=6),
    "scroll-512-w1": partial(images.scroll, width=1),
    "scroll-512-w2": partial(images.scroll, width=2),
    "scroll-1024-w1": partial(images.scroll, size=1024, width=1),
    **{f"road-map-40-seed{n}": partial(images.road_map, seed=n) for n in range(1, 6)},
    "road-map-40-1024-w6": partial(images.road_map, size=1024, width=6),
    "label-map": images.label_map,
}


@pytest.mark.parametrize("name", list(_LEGIT))
def test_sweep_legit_drawings_pass_with_margin(name: str) -> None:
    report = check_density(binarize(_LEGIT[name]()))
    assert report.ok, report
    assert report.mean_stroke_width < 0.75 * lineart.MIN_FILL_WIDTH_PX
    assert report.ink_ratio < 0.75 * lineart.MAX_INK_RATIO
    # the hatch test needs all three; each legit drawing clears at least one widely
    assert (
        report.dense_tiles < lineart.MIN_DENSE_TILES // 2
        or report.block_periodicity < 0.75 * lineart.MIN_BLOCK_PERIODICITY
        or report.hatch_period_px > 2 * lineart.MAX_HATCH_PERIOD_PX
    )


def test_blank_raises() -> None:
    with pytest.raises(SketchRejectedError, match="empty"):
        binarize(images.blank())


def test_garbage_bytes_raise() -> None:
    with pytest.raises(SketchRejectedError):
        binarize(b"not a png")


def test_alpha_png_flattened_on_white() -> None:
    assert np.array_equal(
        binarize(images.line_drawing_scroll_rgba()), binarize(images.line_drawing_scroll())
    )


def _reencode(mode: str) -> bytes:
    image = Image.open(io.BytesIO(images.line_drawing_scroll())).convert("L")
    if mode == "I;16":
        image = image.point(lambda v: v * 257).convert("I;16")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["L", "I;16"])
def test_grayscale_and_16_bit_inputs(mode: str) -> None:
    assert np.array_equal(binarize(_reencode(mode)), binarize(images.line_drawing_scroll()))


def test_deterministic() -> None:
    png = images.noisy_line_drawing()
    assert np.array_equal(binarize(png), binarize(png))
    assert check_density(binarize(png)) == check_density(binarize(png))


def test_adaptive_method_keeps_lines() -> None:
    ink = binarize(images.line_drawing_scroll(), method="adaptive")
    assert ink.dtype == bool
    assert check_density(ink).ok


def test_empty_skeleton_reports_zero() -> None:
    report = check_density(np.zeros((8, 8), dtype=bool))
    assert report.ok
    assert report.mean_stroke_width == 0.0
    assert report.junction_clusters_per_kpx == 0.0
    assert report.dense_tiles == 0
    assert report.block_periodicity == 0.0
    assert report.hatch_period_px == 0.0


def test_solid_block_has_zero_periodicity() -> None:
    report = check_density(np.ones((64, 64), dtype=bool))
    assert report.max_tile_ink == 1.0
    assert report.dense_tiles == 64
    assert report.block_periodicity == 0.0
    assert report.hatch_period_px == 0.0
    assert report.reason is not None
    assert report.reason.startswith("stroke_width ")


def test_tiny_image_does_not_crash() -> None:
    report = check_density(np.ones((4, 4), dtype=bool))
    assert report.ink_ratio == 1.0
    assert report.hatch_period_px == 0.0


def test_hatch_period_scales_with_resolution() -> None:
    # spacing 20 at 1024 is spacing 10 at 512: rejected; at 512 it passes
    big = check_density(
        binarize(images.hatch(spacing=20, width=4, angles=_CROSS, size=1024, box=600))
    )
    assert big.reason is not None
    assert big.reason.startswith("dense_tiles ")
    assert 19.0 < big.hatch_period_px < 21.0
    assert check_density(binarize(images.hatch(spacing=20, width=2, angles=_CROSS))).ok


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow"
    for path in (root / "visuals" / "lineart.py", root / "errors.py"):
        assert path.read_bytes().isascii()


def test_tiny_image_has_no_hatch_frequencies() -> None:
    # at 16 px the period limit is under a pixel, so no bin is a hatch frequency
    y, x = np.indices((16, 16))
    report = check_density((y + x) % 2 == 1)
    assert report.dense_tiles == 64
    assert report.block_periodicity > 0.0
    assert report.second_orientation_ratio == 0.0


@pytest.mark.parametrize("name", [sample.name for sample in SAMPLES])
def test_xdog_sample_set_feeds_strokes(name: str) -> None:
    raw = sample_png(name)
    mask = xdog_lineart(raw)
    png = ink_to_png(mask)
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "L"
        pixels = np.asarray(image)
    assert set(np.unique(pixels).tolist()) <= {0, 255}
    if name == "camera":
        assert pixels[0, 0] == 255
    assert np.array_equal(binarize(png), mask)
    assert check_density(mask).ok
    strokes = extract_strokes(mask)
    assert 1 <= len(strokes.strokes) <= PACING_STROKES
    assert StrokeSet.from_json(strokes.to_json()) == strokes
    assert ink_to_png(xdog_lineart(raw)) == png


@pytest.mark.parametrize("colour", [255, 0, 128])
def test_xdog_flat_images_are_empty(colour: int) -> None:
    flat = Image.new("L", (256, 256), colour)
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    with pytest.raises(SketchRejectedError, match=r"^empty$"):
        xdog_lineart(buffer.getvalue())


@pytest.mark.parametrize(("size", "message"), [((16, 16), "too small"), ((9000, 10), "too large")])
def test_xdog_size_limits(size: tuple[int, int], message: str) -> None:
    buffer = io.BytesIO()
    Image.new("L", size, 255).save(buffer, format="PNG")
    with pytest.raises(SketchRejectedError, match=message):
        xdog_lineart(buffer.getvalue())


def test_xdog_undecodable() -> None:
    with pytest.raises(SketchRejectedError, match=r"^undecodable image$"):
        xdog_lineart(b"not an image")


def test_xdog_rgba_composited_on_white() -> None:
    mask = xdog_lineart(images.line_drawing_scroll_rgba())
    assert mask.any()
    assert check_density(mask).ok


def test_xdog_upscales_small_images() -> None:
    small = Image.open(io.BytesIO(images.line_drawing_scroll())).convert("L").resize((100, 80))
    buffer = io.BytesIO()
    small.save(buffer, format="PNG")
    mask = xdog_lineart(buffer.getvalue())
    assert max(mask.shape) == lineart.XDOG_LONG_SIDE_PX


def test_ink_to_png_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ink_to_png(np.zeros((4, 4, 3), bool))


def test_xdog_sample_licences_recorded() -> None:
    assert len(SAMPLES) == 7
    for sample in SAMPLES:
        assert sample.licence in {"CC0-1.0", "public-domain"}
        assert sample.source


def test_xdog_smooth_gradient_has_no_edges() -> None:
    ramp = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (256, 1))
    buffer = io.BytesIO()
    Image.fromarray(ramp).save(buffer, format="PNG")
    with pytest.raises(SketchRejectedError, match=r"^empty$"):
        xdog_lineart(buffer.getvalue())


def test_xdog_decompression_bomb_is_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    buffer = io.BytesIO()
    Image.new("L", (400, 400), 255).save(buffer, format="PNG")
    with pytest.raises(SketchRejectedError, match=r"^too large$"):
        xdog_lineart(buffer.getvalue())


def test_xdog_image_at_working_size_is_not_resized() -> None:
    scroll = Image.open(io.BytesIO(images.line_drawing_scroll())).convert("L")
    assert max(scroll.size) == lineart.XDOG_LONG_SIDE_PX
    assert xdog_lineart(images.line_drawing_scroll()).shape == scroll.size[::-1]
