import json
import os
import stat
from collections.abc import Iterator, Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.audio import tone

from narratoflow.audio.io import encode_flac
from narratoflow.captions.ass import AssDocument, AssEvent, build_styles
from narratoflow.errors import RenderError
from narratoflow.models import Language
from narratoflow.render import encoder
from narratoflow.render.assets import default_fonts
from narratoflow.render.encoder import (
    AAC_ARGS,
    AV_TOLERANCE_MS,
    FINAL_TIMEOUT_S,
    SEGMENT_TIMEOUT_S,
    X264_ARGS,
    FrameSink,
    encode_part,
    verify_mp4,
    write_concat_list,
)
from narratoflow.render.ffmpeg import MediaProbe, RenderTools, probe, run_tool
from narratoflow.render.layout import Canvas, frame_of, frame_range

W, H = 32, 24


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-ffmpeg"
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _frames(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(n)]


def test_segment_timeout() -> None:
    assert SEGMENT_TIMEOUT_S == 600


def test_frame_sink_roundtrip_lossless(render_tools: RenderTools, tmp_path: Path) -> None:
    out = tmp_path / "seg-000.mkv"
    frames = _frames(5)
    with FrameSink(render_tools.ffmpeg, out, (W, H)) as sink:
        for f in frames:
            sink.write(f)
    assert sink.frames == 5
    raw = run_tool(
        render_tools.ffmpeg,
        ["-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
    )
    assert raw == b"".join(f.tobytes() for f in frames)


def test_frame_sink_writes_exact_path(render_tools: RenderTools, tmp_path: Path) -> None:
    out = tmp_path / "seg-000.mkv"
    with FrameSink(render_tools.ffmpeg, out, (W, H)) as sink:
        sink.write(_frames(1)[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seg-000.mkv"]


def test_frame_sink_passes_the_pinned_command(tmp_path: Path) -> None:
    log = tmp_path / "args.txt"
    ffmpeg = _script(tmp_path, f'echo "$@" > "{log}"\ncat > /dev/null')
    out = tmp_path / "seg-001.mkv"
    with FrameSink(ffmpeg, out, (W, H)) as sink:
        sink.write(_frames(1)[0])
    assert log.read_text(encoding="utf-8").split() == [
        *["-hide_banner", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"],
        *f"-s {W}x{H} -framerate 30 -i pipe:0 -an -c:v ffv1 -level 3".split(),
        *["-pix_fmt", "bgr0"],
        str(out),
    ]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((H, W + 1, 3), dtype=np.uint8),
        np.zeros((H, W), dtype=np.uint8),
        np.zeros((H, W, 4), dtype=np.uint8),
        np.zeros((H, W, 3), dtype=np.float32),
    ],
)
def test_frame_sink_bad_shape_raises(tmp_path: Path, bad: np.ndarray) -> None:
    ffmpeg = _script(tmp_path, "cat > /dev/null")
    with pytest.raises(ValueError), FrameSink(ffmpeg, tmp_path / "s.mkv", (W, H)) as sink:
        sink.write(bad)
    assert sink.frames == 0


def test_frame_sink_ffmpeg_failure_raises_render_error(tmp_path: Path) -> None:
    body = "cat > /dev/null\n" + "\n".join(f"echo line{i} >&2" for i in range(8)) + "\nexit 3"
    ffmpeg = _script(tmp_path, body)
    with (
        pytest.raises(RenderError, match="exit 3") as info,
        FrameSink(ffmpeg, tmp_path / "s.mkv", (W, H)) as sink,
    ):
        sink.write(_frames(1)[0])
    msg = str(info.value)
    assert "line7" in msg and "line3" in msg and "line2" not in msg


def test_frame_sink_early_exit_raises_render_error(tmp_path: Path) -> None:
    ffmpeg = _script(tmp_path, "echo cannot open output >&2\nexit 1")
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    with (
        pytest.raises(RenderError, match="cannot open output"),
        FrameSink(ffmpeg, tmp_path / "s.mkv", (640, 480)) as sink,
    ):
        for _ in range(50):
            sink.write(big)


def test_frame_sink_body_exception_kills_ffmpeg(tmp_path: Path) -> None:
    ffmpeg = _script(tmp_path, "sleep 30")
    with pytest.raises(KeyError), FrameSink(ffmpeg, tmp_path / "s.mkv", (W, H)) as sink:
        raise KeyError("boom")
    assert sink.returncode is not None and sink.returncode != 0


def test_frame_sink_missing_tool_raises_render_error(tmp_path: Path) -> None:
    with pytest.raises(RenderError, match="cannot run"):
        FrameSink(tmp_path / "nope", tmp_path / "s.mkv", (W, H)).__enter__()


def test_frame_sink_timeout_kills_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.render import encoder

    monkeypatch.setattr(encoder, "SEGMENT_TIMEOUT_S", 0.2)
    ffmpeg = _script(tmp_path, "exec sleep 30")
    with (
        pytest.raises(RenderError, match="timed out"),
        FrameSink(ffmpeg, tmp_path / "s.mkv", (W, H)),
    ):
        pass


def test_frame_sink_write_outside_with_block(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path / "ffmpeg", tmp_path / "s.mkv", (W, H))
    with pytest.raises(RuntimeError, match="with"):
        sink.write(np.zeros((H, W, 3), dtype=np.uint8))


def test_frame_sink_clean_exit_without_reading_raises(tmp_path: Path) -> None:
    ffmpeg = _script(tmp_path, "exit 0")
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    with (
        pytest.raises(RenderError, match="stopped reading"),
        FrameSink(ffmpeg, tmp_path / "s.mkv", (640, 480)) as sink,
    ):
        for _ in range(50):
            sink.write(big)


# --- Task 9: final encode, mux and verification ---

SMALL = (180, 320)


def test_final_encode_constants() -> None:
    assert [
        *["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high"],
        *["-level:v", "4.2", "-pix_fmt", "yuv420p", "-g", "60", "-bf", "2"],
        *["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"],
        *["-color_range", "tv"],
    ] == X264_ARGS
    assert AAC_ARGS == ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    assert AV_TOLERANCE_MS == 50
    assert FINAL_TIMEOUT_S == 1800


def _color(i: int, *, paper: bool = False) -> np.ndarray:
    rgb = (255, 255, 255) if paper else ((i * 67) % 256, (i * 131 + 40) % 256, (i * 29 + 90) % 256)
    frame = np.empty((SMALL[1], SMALL[0], 3), dtype=np.uint8)
    frame[:, :] = rgb
    return frame


def _work(
    tools: RenderTools,
    tmp_path: Path,
    lengths: Sequence[int],
    events: Sequence[AssEvent] = (),
    *,
    paper: bool = False,
) -> Path:
    """A work dir as the orchestrator builds it: segments, segments.txt, captions.ass, fonts."""
    work = tmp_path / ".render-part-01"
    work.mkdir()
    fonts = default_fonts()
    fonts.copy_to(work / "fonts")
    doc = AssDocument(styles=build_styles(Language.ENGLISH, "#111111", fonts), events=[*events])
    doc.write(work / "captions.ass")
    segments: list[Path] = []
    for i, n in enumerate(lengths):
        seg = work / f"seg-{i:03d}.mkv"
        frame = _color(i, paper=paper)
        with FrameSink(tools.ffmpeg, seg, SMALL) as sink:
            for _ in range(n):
                sink.write(frame)
        segments.append(seg)
    write_concat_list(segments, work / "segments.txt")
    return work


def _flac(tmp_path: Path, ms: int) -> Path:
    path = tmp_path / "part-01.flac"
    encode_flac(tone(ms), path)
    return path


def test_concat_list_writes_bare_names(tmp_path: Path) -> None:
    segs = [tmp_path / "seg-000.mkv", tmp_path / "seg-001.mkv"]
    write_concat_list(segs, tmp_path / "segments.txt")
    text = (tmp_path / "segments.txt").read_text(encoding="utf-8")
    assert text == "file 'seg-000.mkv'\nfile 'seg-001.mkv'\n"


@pytest.mark.parametrize(
    "name", ["a'b.mkv", "seg:0.mkv", "file:seg.mkv", "x\ny.mkv", "a b.mkv", "-x.mkv", ".."]
)
def test_concat_list_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unsafe segment name"):
        write_concat_list([tmp_path / name], tmp_path / "segments.txt")
    assert not (tmp_path / "segments.txt").exists()


@pytest.mark.parametrize("sub", ["other", "sub/deeper"])
def test_concat_list_rejects_segment_outside_list_dir(tmp_path: Path, sub: str) -> None:
    with pytest.raises(ValueError, match="not in"):
        write_concat_list([tmp_path / sub / "seg-000.mkv"], tmp_path / "segments.txt")


def test_concat_list_accepts_relative_and_absolute_spellings(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()  # the autouse fixture makes tmp_path the cwd
    write_concat_list([Path("work/seg-000.mkv")], tmp_path.absolute() / "work" / "segments.txt")
    write_concat_list([tmp_path.absolute() / "work" / "seg-001.mkv"], Path("work/list.txt"))
    write_concat_list([Path("seg-002.mkv")], Path("work/bare.txt"))
    assert (tmp_path / "work" / "list.txt").read_text(encoding="utf-8") == "file 'seg-001.mkv'\n"


def test_concat_list_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no segments"):
        write_concat_list([], tmp_path / "segments.txt")


def _probe_script(tmp_path: Path, frames: str) -> Path:
    """A fake ffprobe that logs its args and cwd and prints a packet count."""
    path = tmp_path / "fake-ffprobe"
    log = tmp_path / "probe-args.txt"
    path.write_text(f'#!/bin/sh\necho "$@" > "{log}"\npwd >> "{log}"\necho {frames}\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def umask_022() -> Iterator[None]:
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_encode_part_passes_the_pinned_command(tmp_path: Path, umask_022: None) -> None:
    log = tmp_path / "args.txt"
    cwd_log = tmp_path / "cwd.txt"
    body = f'echo "$@" > "{log}"\npwd > "{cwd_log}"\nfor a; do last="$a"; done\necho mp4 > "$last"'
    ffprobe = _probe_script(tmp_path, "42")
    tools = RenderTools(ffmpeg=_script(tmp_path, body), ffprobe=ffprobe, version="")
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out" / "part-01.mp4"
    out.parent.mkdir()
    encode_part(tools, work, frames=42, audio=Path("part-01.flac"), out=out)
    args = log.read_text(encoding="utf-8").split()
    vf = (
        "ass=filename=captions.ass:fontsdir=fonts:shaping=complex,"
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    )
    assert args[:-1] == [
        *["-hide_banner", "-nostdin", "-v", "error", "-xerror", "-y"],
        *["-f", "concat", "-safe", "0", "-i", "segments.txt"],
        *["-i", str((tmp_path / "part-01.flac").absolute())],
        *["-map", "0:v:0", "-map", "1:a:0", "-vf", vf],
        *["-r", "30", "-fps_mode", "cfr", "-frames:v", "42"],
        *X264_ARGS,
        *AAC_ARGS,
        *["-movflags", "+faststart"],
    ]
    tmp_out = Path(args[-1])
    probe_log = (tmp_path / "probe-args.txt").read_text(encoding="utf-8").splitlines()
    assert probe_log[0].split() == [
        *["-v", "error", "-f", "concat", "-safe", "0", "-i", "segments.txt"],
        *["-select_streams", "v:0", "-count_packets"],
        *["-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"],
    ]
    assert Path(probe_log[1]).resolve() == work.resolve()
    assert tmp_out.parent == out.parent and tmp_out.suffix == ".mp4" and tmp_out != out
    assert Path(cwd_log.read_text(encoding="utf-8").strip()).resolve() == work.resolve()
    assert out.read_text(encoding="utf-8") == "mp4\n"
    assert sorted(p.name for p in out.parent.iterdir()) == ["part-01.mp4"]
    assert out.stat().st_mode & 0o777 == 0o644


def test_encode_part_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(encoder, "FINAL_TIMEOUT_S", 0.2)
    ffprobe = _probe_script(tmp_path, "1")
    tools = RenderTools(ffmpeg=_script(tmp_path, "exec sleep 30"), ffprobe=ffprobe, version="")
    out = tmp_path / "part-01.mp4"
    with pytest.raises(RenderError, match="timed out"):
        encode_part(tools, tmp_path, frames=1, audio=tmp_path / "a.flac", out=out)
    assert not out.exists()


def test_encode_part_failure_leaves_no_output(tmp_path: Path) -> None:
    """A failed final pass removes its temp file and any stale MP4 at the final path."""
    body = 'for a; do last="$a"; done\necho partial > "$last"\necho broken segment >&2\nexit 1'
    ffprobe = _probe_script(tmp_path, "1")
    tools = RenderTools(ffmpeg=_script(tmp_path, body), ffprobe=ffprobe, version="")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "part-01.mp4"
    out.write_text("an older render", encoding="utf-8")
    with pytest.raises(RenderError, match="broken segment"):
        encode_part(tools, tmp_path, frames=1, audio=tmp_path / "a.flac", out=out)
    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("output", ["41", "garbage"])
def test_encode_part_refuses_wrong_segment_frames(tmp_path: Path, output: str) -> None:
    ran = tmp_path / "ran"
    ffmpeg = _script(tmp_path, f'touch "{ran}"')
    tools = RenderTools(ffmpeg=ffmpeg, ffprobe=_probe_script(tmp_path, output), version="")
    out = tmp_path / "part-01.mp4"
    with pytest.raises(RenderError, match=output):
        encode_part(tools, tmp_path, frames=42, audio=tmp_path / "a.flac", out=out)
    assert not ran.exists() and not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".mp4") == []


@pytest.mark.parametrize("frames", [0, -1])
def test_encode_part_rejects_no_frames(tmp_path: Path, frames: int) -> None:
    tools = RenderTools(ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path, version="")
    with pytest.raises(ValueError, match="frames"):
        encode_part(tools, tmp_path, frames=frames, audio=tmp_path / "a", out=tmp_path / "o.mp4")


_GOOD = MediaProbe(
    width=180,
    height=320,
    fps=Fraction(30),
    frames=60,
    video_ms=2000,
    audio_ms=2021,
    video_codec="h264",
    pix_fmt="yuv420p",
    audio_codec="aac",
    audio_rate=48000,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 181),
        ("height", None),
        ("fps", Fraction(30000, 1001)),
        ("fps", None),
        ("frames", 59),
        ("video_codec", "hevc"),
        ("pix_fmt", "yuv444p"),
        ("audio_codec", "opus"),
        ("audio_rate", 44100),
        ("video_ms", 2051),
        ("video_ms", None),
        ("audio_ms", 1949),
        ("audio_ms", None),
    ],
)
def test_verify_mp4_names_the_bad_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    monkeypatch.setattr(encoder, "probe", lambda _t, _p: replace(_GOOD, **{field: value}))
    tools = RenderTools(ffmpeg=tmp_path, ffprobe=tmp_path, version="")
    with pytest.raises(RenderError, match=field):
        verify_mp4(tools, tmp_path / "p.mp4", size=SMALL, frames=60, duration_ms=2000)


def test_verify_mp4_accepts_within_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encoder, "probe", lambda _t, _p: _GOOD)
    tools = RenderTools(ffmpeg=tmp_path, ffprobe=tmp_path, version="")
    got = verify_mp4(tools, tmp_path / "p.mp4", size=SMALL, frames=60, duration_ms=2050)
    assert got == _GOOD


def test_concat_segments_exact_frame_count(render_tools: RenderTools, tmp_path: Path) -> None:
    lengths = [1, 5, 2, 13, 3, 8, 4]
    total = sum(lengths)
    work = _work(render_tools, tmp_path, lengths)
    duration_ms = total * 1000 // 30
    out = tmp_path / "part-01.mp4"
    encode_part(render_tools, work, frames=total, audio=_flac(tmp_path, duration_ms), out=out)
    got = verify_mp4(render_tools, out, size=SMALL, frames=total, duration_ms=duration_ms)
    assert got.frames == total


def test_long_part_no_av_drift(render_tools: RenderTools, tmp_path: Path) -> None:
    """90 s as 37 scenes with ms boundaries off the frame grid: no dup, no drop, no drift."""
    duration_ms = 90_000
    rng = np.random.default_rng(9)
    inner = sorted(int(x) for x in rng.choice(np.arange(1, duration_ms), 36, replace=False))
    bounds = [0, *inner, duration_ms]
    assert any(b * 30 % 1000 for b in inner)  # boundaries off the frame grid
    lengths = [len(frame_range(a, b)) for a, b in pairwise(bounds)]
    frames = frame_of(duration_ms)
    assert sum(lengths) == frames == 2700
    kept = [n for n in lengths if n > 0]
    work = _work(render_tools, tmp_path, kept)
    out = tmp_path / "part-01.mp4"
    encode_part(render_tools, work, frames=frames, audio=_flac(tmp_path, duration_ms), out=out)
    got = verify_mp4(render_tools, out, size=SMALL, frames=frames, duration_ms=duration_ms)
    assert got.video_ms is not None and got.audio_ms is not None
    assert abs(got.video_ms - got.audio_ms) <= AV_TOLERANCE_MS
    # Every decoded frame (averaged to 1x1) shows its own segment's flat colour, so a
    # matched drop + dup at any boundary fails even though the count still matches.
    args = [
        "-v", "error", "-i", str(out), "-vf", "scale=1:1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip
    raw = run_tool(render_tools.ffmpeg, args)
    shown = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    palette = np.array([_color(i)[0, 0] for i in range(len(kept))], dtype=np.int32)
    nearest = ((shown[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    assert nearest.tolist() == np.repeat(np.arange(len(kept)), kept).tolist()


def test_output_tagged_bt709_high_42(render_tools: RenderTools, tmp_path: Path) -> None:
    work = _work(render_tools, tmp_path, [20, 10])
    out = tmp_path / "part-01.mp4"
    encode_part(render_tools, work, frames=30, audio=_flac(tmp_path, 1000), out=out)
    args = [
        "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
        "stream=color_primaries,color_transfer,color_space,color_range,profile,level",
        str(out),
    ]  # fmt: skip
    raw = run_tool(render_tools.ffprobe, args)
    stream = json.loads(raw)["streams"][0]
    assert stream == {
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_space": "bt709",
        "color_range": "tv",
        "profile": "High",
        "level": 42,
    }


def _frame_at(tools: RenderTools, path: Path, t_ms: int) -> np.ndarray:
    w, h = SMALL
    raw = run_tool(
        tools.ffmpeg,
        [
            "-v", "error", "-i", str(path), "-ss", f"{t_ms / 1000}", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
    )  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def test_ass_burned_into_output(render_tools: RenderTools, tmp_path: Path) -> None:
    caption = AssEvent(500, 1_500, "Caption", "Hello world")
    work = _work(render_tools, tmp_path, [60], events=[caption], paper=True)
    out = tmp_path / "part-01.mp4"
    encode_part(render_tools, work, frames=60, audio=_flac(tmp_path, 2000), out=out)
    band = Canvas(Fraction(1, 6)).captions
    lit = _frame_at(render_tools, out, 1_000).mean(axis=2)
    dark = lit < 128
    assert dark[band.y : band.bottom, band.x : band.right].sum() > 20
    assert dark[: band.y].sum() == 0 and dark[band.bottom :].sum() == 0
    before = _frame_at(render_tools, out, 100).mean(axis=2)
    assert (before < 128).sum() == 0


def test_output_atomic_on_failure(render_tools: RenderTools, tmp_path: Path) -> None:
    work = _work(render_tools, tmp_path, [3, 4])
    (work / "seg-001.mkv").write_bytes(b"\x1a\x45\xdf\xa3 not really a matroska file")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "part-01.mp4"
    with pytest.raises(RenderError):
        encode_part(render_tools, work, frames=7, audio=_flac(tmp_path, 234), out=out)
    assert list(out_dir.iterdir()) == []


def test_encode_part_refuses_truncated_segment(render_tools: RenderTools, tmp_path: Path) -> None:
    """A killed FrameSink leaves a valid-looking but short segment; concat exits 0 on it."""
    work = _work(render_tools, tmp_path, [3, 40, 3])
    seg = work / "seg-001.mkv"
    seg.unlink()
    rng = np.random.default_rng(5)
    with FrameSink(render_tools.ffmpeg, seg, SMALL) as sink:
        for _ in range(40):  # noise, so the frame data dominates the file and is cut
            sink.write(rng.integers(0, 256, size=(SMALL[1], SMALL[0], 3), dtype=np.uint8))
    seg.write_bytes(seg.read_bytes()[: seg.stat().st_size // 3])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "part-01.mp4"
    with pytest.raises(RenderError, match="expected 46"):
        encode_part(render_tools, work, frames=46, audio=_flac(tmp_path, 1534), out=out)
    assert list(out_dir.iterdir()) == []


def test_verify_mp4_rejects_wrong_size_and_frames(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    work = _work(render_tools, tmp_path, [10, 5])
    out = tmp_path / "part-01.mp4"
    encode_part(render_tools, work, frames=15, audio=_flac(tmp_path, 500), out=out)
    assert probe(render_tools, out).frames == 15
    with pytest.raises(RenderError, match="width"):
        verify_mp4(render_tools, out, size=(1080, 1920), frames=15, duration_ms=500)
    with pytest.raises(RenderError, match="frames"):
        verify_mp4(render_tools, out, size=SMALL, frames=16, duration_ms=500)
    verify_mp4(render_tools, out, size=SMALL, frames=15, duration_ms=500)


def test_encode_part_missing_out_dir_raises_render_error(tmp_path: Path) -> None:
    tools = RenderTools(ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path, version="")
    out = tmp_path / "missing" / "part-01.mp4"
    with pytest.raises(RenderError, match="cannot write"):
        encode_part(tools, tmp_path, frames=1, audio=tmp_path / "a.flac", out=out)


def test_encode_part_rename_failure_cleans_up(tmp_path: Path) -> None:
    body = 'for a; do last="$a"; done\necho mp4 > "$last"'
    tools = RenderTools(
        ffmpeg=_script(tmp_path, body), ffprobe=_probe_script(tmp_path, "1"), version=""
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "part-01.mp4"
    (out / "blocker").mkdir(parents=True)  # a non-empty directory: os.replace fails
    with pytest.raises(RenderError, match="cannot write"):
        encode_part(tools, tmp_path, frames=1, audio=tmp_path / "a.flac", out=out)
    assert sorted(p.name for p in out_dir.iterdir()) == ["part-01.mp4"]
    assert out.is_dir()
