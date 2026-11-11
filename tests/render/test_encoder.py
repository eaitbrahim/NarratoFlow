import stat
from pathlib import Path

import numpy as np
import pytest

from narratoflow.errors import RenderError
from narratoflow.render.encoder import SEGMENT_TIMEOUT_S, FrameSink
from narratoflow.render.ffmpeg import RenderTools, run_tool

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
