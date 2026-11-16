"""The 16:9 producers: the panorama re-composition (and the pillarbox until #194)."""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from fixtures.parts import add_sketch, make_series, narr, render, write_timelines

from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.errors import RenderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.publish.landscape import (
    CANVAS_SCALES,
    LANDSCAPE_BOTTOM,
    LANDSCAPE_SIZE,
    landscape_name,
    landscape_size,
    panorama_layout,
    pillarbox_export,
)
from narratoflow.render.ffmpeg import RenderTools, run_tool
from narratoflow.render.layout import CAPTIONS, STAGE, TOP_BAR, Canvas, Rect
from narratoflow.settings import Settings


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


def _row(tools: RenderTools, video: Path, crop: str) -> np.ndarray:
    raw = run_tool(
        tools.ffmpeg,
        ["-v", "error", "-i", str(video), "-vf", f"crop={crop},format=rgb24",
         "-frames:v", "1", "-f", "rawvideo", "-"],
    )  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8)


def test_landscape_name_is_zero_padded() -> None:
    assert landscape_name(3) == "part-03.16x9.mp4"
    assert LANDSCAPE_SIZE == (1920, 1080)


def test_pillarbox_is_1920x1080_same_frames(
    channel: ChannelConfig,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    sid = add_sketch(library)
    series = make_series([narr("s1", "alpha beta", sketch=sid), narr("s2", "gamma")])
    write_timelines(series, tmp_path / "audio", library)
    built = render(
        series, channel, Settings(data_dir=tmp_path / "data"), tmp_path, render_tools,
        library=library,
    )  # fmt: skip
    [part] = built.parts
    out = tmp_path / "export" / landscape_name(1)
    got = pillarbox_export(
        render_tools,
        part.video_path,
        out,
        frames=part.report.frames,
        duration_ms=part.report.duration_ms,
    )
    assert got == out and out.is_file()
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]  # no temp file left
    left = _row(render_tools, out, "100:1080:0:0")
    right = _row(render_tools, out, "100:1080:1820:0")
    assert left.mean() > 250 and right.mean() > 250


def test_pillarbox_failure_leaves_no_file(render_tools: RenderTools, tmp_path: Path) -> None:
    out = tmp_path / "export" / landscape_name(1)
    with pytest.raises(RenderError):
        pillarbox_export(render_tools, tmp_path / "missing.mp4", out, frames=60, duration_ms=2000)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_pillarbox_wrong_frame_count_is_refused_and_removed(
    channel: ChannelConfig,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    sid = add_sketch(library)
    series = make_series([narr("s1", "alpha beta", sketch=sid)])
    write_timelines(series, tmp_path / "audio", library)
    [part] = render(
        series, channel, Settings(data_dir=tmp_path / "data"), tmp_path, render_tools,
        library=library,
    ).parts  # fmt: skip
    out = tmp_path / "export" / landscape_name(1)
    with pytest.raises(RenderError, match="frames"):
        pillarbox_export(
            render_tools,
            part.video_path,
            out,
            frames=part.report.frames + 5,
            duration_ms=part.report.duration_ms,
        )
    assert list(out.parent.iterdir()) == []


def test_layout_scale_1_matches_the_design_table() -> None:
    layout = panorama_layout(Fraction(1))
    got = {z.name: (z.src, z.dst) for z in layout.zones}
    assert got == {
        "stage": (STAGE, Rect(48, 115, 828, 850)),
        "top_bar": (TOP_BAR, Rect(912, 115, 960, 200)),
        "captions": (CAPTIONS, Rect(978, 450, 828, 180)),
    }
    assert all((z.dst.w, z.dst.h) == (z.src.w, z.src.h) for z in layout.zones)


@pytest.mark.parametrize("scale", CANVAS_SCALES)
def test_layout_bled_rects_are_disjoint_and_inside(scale: Fraction) -> None:
    layout = panorama_layout(scale)
    canvas = Canvas(scale)
    assert layout.source == (canvas.width, canvas.height)
    src_frame = Rect(0, 0, canvas.width, canvas.height)
    dst_frame = Rect(0, 0, *landscape_size(scale))
    for z in layout.zones:
        assert src_frame.contains(z.src_bled) and dst_frame.contains(z.dst_bled)
        assert (z.dst.w, z.dst.h) == (z.src.w, z.src.h)  # 1:1 at every scale
        assert z.bleed == max(1, round(8 * scale))
    for a, b in itertools.combinations(layout.zones, 2):
        assert not a.src_bled.intersects(b.src_bled)
        assert not a.dst_bled.intersects(b.dst_bled)


def test_layout_keeps_the_bottom_and_side_margins_clear() -> None:
    for z in panorama_layout(Fraction(1)).zones:  # unbled zones
        assert not z.dst.intersects(LANDSCAPE_BOTTOM)
        assert z.dst.x >= 48 and 1920 - z.dst.right >= 48


def test_landscape_size_follows_the_canvas_scale() -> None:
    assert landscape_size(Fraction(1)) == (1920, 1080) == LANDSCAPE_SIZE
    assert landscape_size(Fraction(1, 6)) == (320, 180)
    with pytest.raises(ValueError, match="scale"):
        landscape_size(Fraction(1, 4))
