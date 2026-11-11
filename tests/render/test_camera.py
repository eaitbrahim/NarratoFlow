import numpy as np

from narratoflow.render import variation
from narratoflow.render.camera import (
    CAMERA_MIN_DRAW_MS,
    ZOOM_DRAW,
    ZOOM_OUT_MS,
    CameraTrack,
)
from narratoflow.visuals.marker import PreparedStroke
from narratoflow.visuals.schedule import StrokeEvent

ART = (1076, 1105)


def _stroke(a: tuple[float, float], b: tuple[float, float]) -> PreparedStroke:
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return PreparedStroke(points=(a, b), weight=3.0, dot=False, length=length)


def _track(
    pts: list[tuple[float, float]], start: int = 1000, per: int = 1000
) -> tuple[CameraTrack, int, int]:
    prepared = [_stroke(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    events = [StrokeEvent(i, start + i * per, start + (i + 1) * per) for i in range(len(pts))]
    end = events[-1].end_ms
    return CameraTrack(prepared, events, ART, (start, end)), start, end


def test_zoom_draw_is_zoom_max() -> None:
    assert ZOOM_DRAW == variation.ZOOM_MAX


def test_smoothed_pen_clamps_last_samples_onto_pen() -> None:
    track, _, end = _track([(900.0, 900.0), (1000.0, 1000.0), (1060.0, 1090.0)])
    v = track.at(end - 1)
    assert v.zoom == ZOOM_DRAW
    assert v.cx > ART[0] * 0.6 and v.cy > ART[1] * 0.6


def test_overview_outside_window_and_for_short_draws() -> None:
    track, start, end = _track([(100.0, 100.0), (900.0, 900.0), (200.0, 800.0)])
    for t in (0, start - 1, end + ZOOM_OUT_MS + 1, 10**6):
        v = track.at(t)
        assert (v.cx, v.cy, v.zoom) == (ART[0] / 2, ART[1] / 2, 1.0)
    prepared = [_stroke((0.0, 0.0), (500.0, 500.0))]
    events = [StrokeEvent(0, 0, CAMERA_MIN_DRAW_MS - 1)]
    short = CameraTrack(prepared, events, ART, (0, CAMERA_MIN_DRAW_MS - 1))
    assert short.at(1000).zoom == 1.0
    assert CameraTrack(prepared, events, ART, None).at(1000).zoom == 1.0
    assert CameraTrack(prepared, [], ART, (0, 9000)).at(1000).zoom == 1.0


def test_zoom_eases_in_and_out_continuously() -> None:
    track, start, end = _track([(100.0, 100.0), (900.0, 900.0), (200.0, 800.0)])
    prev = track.at(start - 100).zoom
    t = start - 100.0
    while t < end + ZOOM_OUT_MS + 200:
        z = track.at(t).zoom
        assert abs(z - prev) <= 0.05
        prev = z
        t += 1000 / 30


def test_view_window_never_leaves_art() -> None:
    rng = np.random.default_rng(3)
    pts = [(float(rng.choice([0.0, 1076.0])), float(rng.choice([0.0, 1105.0]))) for _ in range(6)]
    track, _, end = _track(pts)
    t = 0.0
    while t < end + 2000:
        v = track.at(t)
        hw, hh = ART[0] / v.zoom / 2, ART[1] / v.zoom / 2
        assert v.cx - hw >= -1e-9 and v.cx + hw <= ART[0] + 1e-9
        assert v.cy - hh >= -1e-9 and v.cy + hh <= ART[1] + 1e-9
        t += 33.0


def test_centre_follows_pen_with_lag_bounded() -> None:
    from narratoflow.render.reveal import pen_at

    pts = [(100.0, 100.0), (900.0, 300.0), (300.0, 900.0), (800.0, 800.0)]
    track, start, end = _track(pts)
    for t in range(start + 800, end - 200, 100):
        pen = pen_at(track._prepared, track._events, t)
        if pen is None:
            continue
        v = track.at(t)
        assert abs(v.cx - pen[0]) <= 0.25 * ART[0]


def test_affine_maps_view_window_to_stage_corners() -> None:
    track, _, _ = _track([(100.0, 100.0), (900.0, 900.0), (200.0, 800.0)])
    from narratoflow.render.camera import View

    view = View(400.0, 500.0, 1.25)
    m = track.affine(view, (828, 850))
    vw, vh = ART[0] / 1.25, ART[1] / 1.25
    tl = m @ np.array([400.0 - vw / 2, 500.0 - vh / 2, 1.0])
    br = m @ np.array([400.0 + vw / 2, 500.0 + vh / 2, 1.0])
    assert np.allclose(tl, [0, 0]) and np.allclose(br, [828, 850])


def test_pure_function_order_independent() -> None:
    track, _, end = _track([(100.0, 100.0), (900.0, 900.0), (200.0, 800.0)])
    times = [float(t) for t in range(0, end + 1500, 137)]
    fwd = [track.at(t) for t in times]
    rev = [track.at(t) for t in reversed(times)][::-1]
    assert fwd == rev


def test_all_samples_in_a_gap_fall_back_to_overview_centre() -> None:
    prepared = [_stroke((100.0, 100.0), (200.0, 200.0)), _stroke((900.0, 900.0), (800.0, 800.0))]
    events = [StrokeEvent(0, 0, 500), StrokeEvent(1, 5500, 6000)]
    track = CameraTrack(prepared, events, ART, (0, 6000))
    v = track.at(3000)
    assert (v.cx, v.cy) == (ART[0] / 2, ART[1] / 2)
