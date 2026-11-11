import numpy as np
import pytest
from PIL import Image

from narratoflow.visuals.marker import marker_seed, render_marker
from narratoflow.visuals.strokes import Stroke, StrokeSet

SIZE = (200, 120)


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


def _ink(img: Image.Image) -> np.ndarray:
    return np.asarray(img).sum(axis=2) < 3 * 255


def _render(**kw: object) -> Image.Image:
    args: dict[str, object] = {"line_weight_px": 4.0, "ink": "#112233", "seed": 1}
    args.update(kw)
    return render_marker(_set(), SIZE, **args)  # type: ignore[arg-type]


def test_deterministic() -> None:
    assert _render().tobytes() == _render().tobytes()
    assert _render().mode == "RGB" and _render().size == SIZE


def test_different_seed_same_coverage() -> None:
    a, b = _render(seed=1), _render(seed=2)
    assert a.tobytes() != b.tobytes()
    ca, cb = _ink(a).sum(), _ink(b).sum()
    assert abs(ca - cb) / ca <= 0.15


def test_upto_zero_blank() -> None:
    assert not _ink(_render(upto=0.0)).any()


def test_upto_half_about_half_ink() -> None:
    ratio = _ink(_render(upto=0.5)).sum() / _ink(_render()).sum()
    assert 0.35 < ratio < 0.65


def test_partial_is_subset_of_full() -> None:
    half, full = _ink(_render(upto=0.5)), _ink(_render())
    extra = (half & ~full).sum()
    assert extra <= 0.02 * half.sum()


def test_upto_validated() -> None:
    with pytest.raises(ValueError):
        _render(upto=1.5)
    with pytest.raises(ValueError):
        _render(upto=-0.1)


def test_ink_colour() -> None:
    img = _render()
    assert tuple(np.asarray(img)[_ink(img)][0]) != (255, 255, 255)
    colours = {tuple(p) for p in np.asarray(img)[_ink(img)]}
    assert (0x11, 0x22, 0x33) in colours


def test_weight_monotonic() -> None:
    cov = [_ink(_render(line_weight_px=w)).sum() for w in (2.0, 4.0, 8.0)]
    assert cov[0] < cov[1] < cov[2]


def test_dot_is_filled_disc_not_thinner_than_line() -> None:
    dot_only = StrokeSet(
        width=200, height=120, strokes=[Stroke(points=[(0.5, 0.5)] * 2, width=0.1)]
    )
    img = render_marker(dot_only, SIZE, line_weight_px=2.0, ink="#000000", seed=3)
    n = _ink(img).sum()
    assert 0.7 * np.pi * 10**2 < n < 1.3 * np.pi * 10**2


def test_marker_seed() -> None:
    assert marker_seed("v", "s1") == marker_seed("v", "s1")
    assert marker_seed("v", "s1") != marker_seed("v", "s2")
    assert 0 <= marker_seed("v", "s1") < 2**63
