"""Stored reveal frames at 0%, 50% and 100%, compared with a tolerance.

Regenerate only as a reviewed diff, never in CI:
    uv run python tests/render/test_reveal_golden.py --update
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from narratoflow.render.reveal import RevealCanvas
from narratoflow.visuals.marker import prepare_marker
from narratoflow.visuals.schedule import draw_schedule
from narratoflow.visuals.strokes import Stroke, StrokeSet

SIZE = (200, 120)
INK = "#112233"
GOLDEN_DIR = Path(__file__).parent / "golden"
KEYS = ("000", "050", "100")
MAX_DIFFERING_FRACTION = 0.005
CHANNEL_TOLERANCE = 32
INK_TOLERANCE = 0.02


def _set() -> StrokeSet:
    return StrokeSet(
        width=200,
        height=120,
        strokes=[
            Stroke(points=[(0.1, 0.2), (0.5, 0.2), (0.9, 0.5)]),
            Stroke(points=[(0.1, 0.8), (0.9, 0.8)]),
            Stroke(points=[(0.5, 0.5), (0.5, 0.5)], width=0.05),
        ],
    )


def _ink(img: Image.Image) -> int:
    return int((np.asarray(img).sum(axis=2) < 3 * 255).sum())


def frames() -> dict[str, Image.Image]:
    events = draw_schedule(_set(), 0, 3000)
    times = {"000": 0, "050": 1500, "100": events[-1].end_ms}
    out: dict[str, Image.Image] = {}
    for key, t_ms in times.items():
        prepared = prepare_marker(_set(), SIZE, line_weight_px=4.0, seed=1)
        out[key] = RevealCanvas(prepared, events, SIZE, INK).at(t_ms).copy()
    return out


def close_enough(got: Image.Image, want: Image.Image) -> str | None:
    if got.size != want.size:
        return f"size {got.size} != {want.size}"
    diff = np.abs(np.asarray(got.convert("RGB"), int) - np.asarray(want.convert("RGB"), int))
    differing = int((diff.max(axis=2) > CHANNEL_TOLERANCE).sum())
    allowed = MAX_DIFFERING_FRACTION * diff.shape[0] * diff.shape[1]
    if differing > allowed:
        return f"{differing} pixels differ by more than {CHANNEL_TOLERANCE} (allowed {allowed:.0f})"
    got_ink, want_ink = _ink(got), _ink(want)
    if abs(got_ink - want_ink) > INK_TOLERANCE * want_ink:
        return f"ink pixels {got_ink} vs golden {want_ink} (allowed {INK_TOLERANCE:.0%})"
    return None


def _golden(key: str) -> Image.Image:
    return Image.open(GOLDEN_DIR / f"reveal-{key}.png").convert("RGB")


@pytest.mark.parametrize("key", KEYS)
def test_reveal_matches_golden(key: str) -> None:
    assert close_enough(frames()[key], _golden(key)) is None


def test_close_enough_rejects_a_missing_stroke() -> None:
    assert close_enough(frames()["100"], _golden("050")) is not None


def test_close_enough_accepts_small_drift() -> None:
    want = _golden("050")
    pixels = np.asarray(want).copy()
    ys, xs = np.nonzero(pixels.sum(axis=2) < 3 * 255)
    n = int(0.002 * pixels.shape[0] * pixels.shape[1])
    assert len(ys) > n
    pixels[ys[:n], xs[:n]] = np.clip(pixels[ys[:n], xs[:n]].astype(int) - 20, 0, 255)
    assert close_enough(Image.fromarray(pixels), want) is None


def _flip_ink_pixels(want: Image.Image, fraction: float) -> Image.Image:
    """Change `fraction` of all pixels by 100 levels while keeping each one ink."""
    pixels = np.asarray(want).copy()
    ys, xs = np.nonzero(pixels.sum(axis=2) < 3 * 255)
    n = round(fraction * pixels.shape[0] * pixels.shape[1])
    assert len(ys) > n
    old = pixels[ys[:n], xs[:n]].astype(int)
    pixels[ys[:n], xs[:n]] = np.where(old >= 128, old - 100, old + 100)
    return Image.fromarray(pixels)


def test_close_enough_rejects_a_same_ink_shift() -> None:
    want = _golden("100")
    shifted = Image.fromarray(np.roll(np.asarray(want), 6, axis=1))
    assert _ink(shifted) == _ink(want)
    assert close_enough(shifted, want) is not None


def test_close_enough_accepts_sparse_large_changes() -> None:
    want = _golden("050")
    got = _flip_ink_pixels(want, 0.004)
    assert _ink(got) == _ink(want)
    assert close_enough(got, want) is None


def test_close_enough_rejects_too_many_large_changes() -> None:
    want = _golden("050")
    got = _flip_ink_pixels(want, 0.006)
    assert _ink(got) == _ink(want)
    assert close_enough(got, want) is not None


def test_zero_percent_is_nearly_blank() -> None:
    assert _ink(_golden("000")) <= 2 * 4 * 4


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        raise SystemExit("usage: test_reveal_golden.py --update")
    GOLDEN_DIR.mkdir(exist_ok=True)
    for name, frame in frames().items():
        frame.save(GOLDEN_DIR / f"reveal-{name}.png", optimize=True)
    print(f"wrote {GOLDEN_DIR}")
