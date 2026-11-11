import io
from pathlib import Path

import numpy as np
import pytest
from fixtures import images
from PIL import Image

from narratoflow.errors import SketchRejectedError
from narratoflow.visuals import lineart
from narratoflow.visuals.lineart import binarize, check_density


def test_line_drawing_passes() -> None:
    report = check_density(binarize(images.line_drawing_scroll()))
    assert report.ok
    assert report.reason is None
    assert 0.0 < report.ink_ratio < 0.05
    assert report.branch_points_per_kpx < lineart.MAX_BRANCH_DENSITY
    assert 0.0 < report.endpoint_ratio < 1.0


def test_filled_silhouette_rejected_by_ink_ratio() -> None:
    report = check_density(binarize(images.filled_silhouette()))
    assert not report.ok
    assert report.branch_points_per_kpx < lineart.MAX_BRANCH_DENSITY
    assert report.reason is not None
    assert report.reason.startswith("ink_ratio 0.30 > ")


def test_crosshatching_rejected_by_branch_density() -> None:
    report = check_density(binarize(images.crosshatched_box()))
    assert not report.ok
    assert report.ink_ratio <= lineart.MAX_INK_RATIO
    assert report.reason is not None
    assert report.reason.startswith("branch_density ")


def test_noise_specks_removed_by_open() -> None:
    clean = binarize(images.line_drawing_scroll())
    noisy = binarize(images.noisy_line_drawing())
    assert np.array_equal(clean, noisy)


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
    assert report.branch_points_per_kpx == 0.0
    assert report.endpoint_ratio == 0.0


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow"
    for path in (root / "visuals" / "lineart.py", root / "errors.py"):
        assert path.read_bytes().isascii()
