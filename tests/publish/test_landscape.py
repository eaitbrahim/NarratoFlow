"""panorama_export: the checked 9:16 part re-composed side by side, guarded and verified."""

from __future__ import annotations

import dataclasses
import itertools
import os
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from fixtures.parts import add_sketch, make_series, narr, render, verse, write_timelines

from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.errors import RenderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import Scene
from narratoflow.publish import landscape
from narratoflow.publish.landscape import (
    CANVAS_SCALES,
    LANDSCAPE_BOTTOM,
    LANDSCAPE_SIZE,
    _compose_graph,
    _first_ink,
    landscape_name,
    landscape_size,
    panorama_export,
    panorama_layout,
)
from narratoflow.render.build import PartVideo
from narratoflow.render.encoder import X264_ARGS
from narratoflow.render.ffmpeg import RenderTools, probe, run_tool
from narratoflow.render.layout import CAPTIONS, STAGE, TOP_BAR, Canvas, Rect
from narratoflow.settings import Settings


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


def test_landscape_name_is_zero_padded() -> None:
    assert landscape_name(3) == "part-03.16x9.mp4"
    assert LANDSCAPE_SIZE == (1920, 1080)


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
        assert (
            z.bleed
            == {Fraction(1): 8, Fraction(1, 2): 4, Fraction(1, 3): 3, Fraction(1, 6): 1}[scale]
        )
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


# --- the ink guard and panorama_export -------------------------------------------------

MARK = Rect(165, 100, 3, 3)
"""At scale 1/6: right of the stage and below the top bar, outside every bled zone."""


def _part(
    channel: ChannelConfig,
    library: SketchLibrary,
    tmp_path: Path,
    tools: RenderTools,
    scenes: list[Scene],
) -> PartVideo:
    series = make_series(scenes)
    write_timelines(series, tmp_path / "audio", library)
    built = render(
        series, channel, Settings(data_dir=tmp_path / "data"), tmp_path, tools, library=library
    )
    [part] = built.parts
    return part


def _sketch_part(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, tools: RenderTools
) -> PartVideo:
    sid = add_sketch(library)
    return _part(
        channel,
        library,
        tmp_path,
        tools,
        [narr("s1", "alpha beta", sketch=sid), narr("s2", "gamma")],
    )


def _verse_part(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, tools: RenderTools
) -> PartVideo:
    return _part(channel, library, tmp_path, tools, [verse("v1"), narr("s1", "alpha")])


def _frames(tools: RenderTools, video: Path, pix: str = "rgb24") -> np.ndarray:
    found = probe(tools, video)
    assert found.width and found.height
    raw = run_tool(
        tools.ffmpeg,
        ["-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", pix, "-"],
    )
    shape = (-1, found.height, found.width) + ((3,) if pix == "rgb24" else ())
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _reencode(tools: RenderTools, src: Path, dst: Path, vf: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        tools.ffmpeg,
        ["-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0", "-vf", vf,
         *X264_ARGS, "-c:a", "copy", str(dst)],
    )  # fmt: skip
    return dst


def _marked(tools: RenderTools, video: Path, tmp_path: Path) -> Path:
    layout = panorama_layout(Fraction(1, 6))
    assert not any(z.src_bled.intersects(MARK) for z in layout.zones)
    vf = f"drawbox=x={MARK.x}:y={MARK.y}:w={MARK.w}:h={MARK.h}:color=black:t=fill"
    return _reencode(tools, video, tmp_path / "src" / "marked.mp4", vf)


def _white_clip(tools: RenderTools, path: Path, vf: str = "null") -> Path:
    """A 3-frame all-white clip at the 1/6 canvas size, with `vf` applied."""
    run_tool(
        tools.ffmpeg,
        ["-v", "error", "-f", "lavfi", "-i", "color=white:s=180x320:d=0.1:r=30",
         "-vf", vf, "-y", str(path)],
    )  # fmt: skip
    return path


@pytest.mark.parametrize(
    ("text", "frames"),
    [
        ("", 2),  # nothing written
        ("frame:0 pts:0 pts_time:0\nlavfi.signalstats.YMIN=235\n", 2),  # a frame missing
        ("frame:0 pts:0 pts_time:0\nlavfi.signalstats.YMIN=abc\n", 1),  # not an int
        ("frame:0 pts:0 pts_time:0\nlavfi.signalstats.YMIN=300\n", 1),  # out of range
        ("frame:0 pts:0 pts_time:0\n", 1),  # no value
        ("frame:1 pts:0 pts_time:0\nlavfi.signalstats.YMIN=235\n", 1),  # wrong numbering
        ("lavfi.signalstats.YMIN=235\n", 1),  # value before frame
        ("garbage\n", 1),
    ],
)
def test_first_ink_rejects_unreadable_output(text: str, frames: int) -> None:
    with pytest.raises(RenderError, match="unreadable"):
        _first_ink(text, frames)


def test_first_ink_reports_the_first_dark_frame() -> None:
    rec = "frame:{0} pts:{0} pts_time:0\nlavfi.signalstats.YMIN={1}\n"
    assert _first_ink(rec.format(0, 235) + rec.format(1, 194) + rec.format(2, 16), 3) == (1, 194)
    assert _first_ink(rec.format(0, 195), 1) is None  # 195 is not ink
    assert _first_ink(rec.format(0, 255), 1) is None  # 255 is a valid luma


def _crop(a: np.ndarray, r: Rect) -> np.ndarray:
    return (
        a[..., r.y : r.bottom, r.x : r.right] if a.ndim == 2 else a[r.y : r.bottom, r.x : r.right]
    )


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _ink_mass(gray: np.ndarray) -> float:
    """Darkness below the crop's own paper (its median), in black-pixel equivalents."""
    paper = int(np.median(gray))
    return float(np.clip(paper - gray.astype(np.int16), 0, None).sum()) / max(paper, 1)


def _export(tools: RenderTools, part: PartVideo, tmp_path: Path) -> Path:
    return panorama_export(
        tools,
        part.video_path,
        tmp_path / "export" / landscape_name(1),
        frames=part.report.frames,
        duration_ms=part.report.duration_ms,
    )


def test_compose_graph_at_scale_1_is_the_design_graph() -> None:
    assert _compose_graph(panorama_layout(Fraction(1))) == (
        "[0:v]format=yuv444p,split=3[s][t][c];"
        "[s]crop=844:866:52:382:exact=1,pad=1920:1080:40:107:color=white[p0];"
        "[t]crop=976:216:52:162:exact=1[t1];[p0][t1]overlay=904:107:format=yuv444[p1];"
        "[c]crop=844:196:52:1252:exact=1[c1];[p1][c1]overlay=970:442:format=yuv444[p2];"
        "[p2]format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709"
        ":colorspace=bt709:range=tv[v]"
    )


def test_panorama_has_landscape_size_and_frames(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = tmp_path / "export" / landscape_name(1)
    got = _export(render_tools, part, tmp_path)
    found = probe(render_tools, out)
    assert got == out
    assert (found.width, found.height) == landscape_size(Fraction(1, 6))
    assert found.frames == part.report.frames
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]


def test_panorama_carries_stage_and_captions_on_paper(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = _export(render_tools, part, tmp_path)
    k = 9  # 300 ms: mid-draw of s1 (0-600 ms)
    src = _frames(render_tools, part.video_path)[k].astype(np.int16)
    dst = _frames(render_tools, out)[k].astype(np.int16)
    zones = {z.name: z for z in panorama_layout(Fraction(1, 6)).zones}
    for name in ("stage", "captions"):
        z = zones[name]
        assert np.abs(_crop(dst, z.dst) - _crop(src, z.src)).mean() < 4
    assert _crop(src, zones["stage"].src).min() < 128  # the frame really is mid-draw
    outside = np.ones(dst.shape[:2], dtype=bool)
    for z in zones.values():
        r = z.dst_bled
        outside[r.y : r.bottom, r.x : r.right] = False
    assert dst[outside].mean() > 250


def test_verse_ink_is_carried_whole(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _verse_part(channel, library, tmp_path, render_tools)
    out = _export(render_tools, part, tmp_path)
    stage = next(z for z in panorama_layout(Fraction(1, 6)).zones if z.name == "stage")
    src = _frames(render_tools, part.video_path, "gray")
    dst = _frames(render_tools, out, "gray")
    counts = [(_crop(f, stage.src) < 128).sum() for f in src[:30]]  # the verse slot
    k = int(np.argmax(counts))  # the fully lit page
    s_ink = _crop(src[k], stage.src) < 128
    d_ink = _crop(dst[k], stage.dst) < 128
    assert s_ink.sum() > 0
    assert np.abs(np.array(_box(d_ink)) - np.array(_box(s_ink))).max() <= 2
    # Ink pixel count, weighted by coverage: the second x264 pass softens 1/6-scale glyph
    # edges, so a hard threshold flips edge pixels (about 8 %) while the ink mass holds.
    s_mass = _ink_mass(_crop(src[k], stage.src))
    assert abs(_ink_mass(_crop(dst[k], stage.dst)) - s_mass) <= 0.02 * s_mass


def _old_target(tmp_path: Path) -> Path:
    out = tmp_path / "export" / landscape_name(1)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"old")
    return out


def _refused(out: Path) -> None:
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_marked_source_is_refused_and_leaves_nothing(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    marked = _marked(render_tools, part.video_path, tmp_path)
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match=r"frame \d+"):
        panorama_export(
            render_tools,
            marked,
            out,
            frames=part.report.frames,
            duration_ms=part.report.duration_ms,
        )
    _refused(out)


def test_non_canvas_source_is_refused(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    odd = _reencode(render_tools, part.video_path, tmp_path / "src" / "odd.mp4", "scale=200:356")
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match="not a 9:16 canvas size"):
        panorama_export(
            render_tools, odd, out, frames=part.report.frames, duration_ms=part.report.duration_ms
        )
    _refused(out)


def test_missing_source_is_refused(render_tools: RenderTools, tmp_path: Path) -> None:
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match=r"missing\.mp4"):
        panorama_export(render_tools, tmp_path / "missing.mp4", out, frames=60, duration_ms=2000)
    _refused(out)


def test_wrong_frame_count_is_refused(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match=r"frames is \d+, expected"):
        panorama_export(
            render_tools,
            part.video_path,
            out,
            frames=part.report.frames + 5,
            duration_ms=part.report.duration_ms,
        )
    _refused(out)


def test_wrong_duration_is_refused_by_verify(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match="video_ms"):
        panorama_export(
            render_tools,
            part.video_path,
            out,
            frames=part.report.frames,
            duration_ms=part.report.duration_ms + 500,
        )
    _refused(out)


def test_source_without_audio_is_refused(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    mute = tmp_path / "src" / "mute.mp4"
    mute.parent.mkdir()
    run_tool(
        render_tools.ffmpeg,
        [
            "-v",
            "error",
            "-y",
            "-i",
            str(part.video_path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            str(mute),
        ],
    )
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match="matches no streams"):
        panorama_export(
            render_tools, mute, out, frames=part.report.frames, duration_ms=part.report.duration_ms
        )
    _refused(out)


def test_source_changed_during_export_is_refused(
    channel: ChannelConfig,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    src = tmp_path / "src" / "a.mp4"
    src.parent.mkdir()
    shutil.copy(part.video_path, src)
    real = landscape._check_ink

    def touch_after_guard(
        tools: RenderTools, path: Path, layout: landscape.PanoramaLayout, *, frames: int
    ) -> None:
        real(tools, path, layout, frames=frames)
        os.utime(src, ns=(1, 1))

    monkeypatch.setattr(landscape, "_check_ink", touch_after_guard)
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match="changed during"):
        panorama_export(
            render_tools, src, out, frames=part.report.frames, duration_ms=part.report.duration_ms
        )
    _refused(out)


def test_ink_error_names_the_frame_and_luma_found(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    vf = f"drawbox=x={MARK.x}:y={MARK.y}:w={MARK.w}:h={MARK.h}:color=black:t=fill"
    src = _white_clip(render_tools, tmp_path / "x.mp4", vf)
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match=r"at frame 0 \(luma \d+ < 195\)"):
        panorama_export(render_tools, src, out, frames=3, duration_ms=100)
    _refused(out)


def test_ink_guard_that_cannot_run_is_an_error(render_tools: RenderTools, tmp_path: Path) -> None:
    src = _white_clip(render_tools, tmp_path / "x.mp4")
    broken = dataclasses.replace(render_tools, ffmpeg=tmp_path / "no-ffmpeg")
    out = _old_target(tmp_path)
    with pytest.raises(RenderError, match="cannot run"):
        panorama_export(broken, src, out, frames=3, duration_ms=100)
    _refused(out)


def test_panorama_creates_missing_target_directories(
    channel: ChannelConfig, library: SketchLibrary, tmp_path: Path, render_tools: RenderTools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = tmp_path / "a" / "b" / landscape_name(1)
    panorama_export(
        render_tools,
        part.video_path,
        out,
        frames=part.report.frames,
        duration_ms=part.report.duration_ms,
    )
    assert out.is_file()
