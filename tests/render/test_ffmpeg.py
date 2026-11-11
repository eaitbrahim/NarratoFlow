import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from narratoflow.errors import RenderError
from narratoflow.render import ffmpeg as F
from narratoflow.render.ffmpeg import (
    RenderTools,
    find_render_tools,
    probe,
    run_tool,
    run_tool_capture,
)
from narratoflow.settings import Settings


def _fake_tool(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_missing_ass_filter_message_names_ffmpeg_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ff = _fake_tool(tmp_path / "ffmpeg", "true\n")
    _fake_tool(tmp_path / "ffprobe", "true\n")

    def fake_run(tool: Path, args: list[str], **_: object) -> bytes:
        if "-filters" in args:
            return b" ... scale V->V\n"
        if "-encoders" in args:
            return b" V....D libx264\n V....D ffv1\n A....D aac\n"
        return b"ffmpeg version x\n"

    monkeypatch.setattr(F, "run_tool", fake_run)
    with pytest.raises(RenderError, match="brew install ffmpeg-full") as ei:
        find_render_tools(Settings(ffmpeg=ff))
    assert "NARRATOFLOW_FFMPEG" in str(ei.value)


def test_missing_encoder_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ff = _fake_tool(tmp_path / "ffmpeg", "true\n")
    _fake_tool(tmp_path / "ffprobe", "true\n")

    def fake_run(tool: Path, args: list[str], **_: object) -> bytes:
        if "-filters" in args:
            return b" ... ass V->V\n"
        if "-encoders" in args:
            return b" V....D libx264\n"
        return b"ffmpeg version x\n"

    monkeypatch.setattr(F, "run_tool", fake_run)
    with pytest.raises(RenderError, match="missing encoder ffv1"):
        find_render_tools(Settings(ffmpeg=ff))


def test_settings_ffmpeg_override_used(tmp_path: Path) -> None:
    ff = _fake_tool(
        tmp_path / "ffmpeg",
        'case "$*" in\n'
        '  *-filters*) echo " ... ass              V->V       libass";;\n'
        '  *-encoders*) printf " V....D libx264\\n V....D ffv1\\n A....D aac\\n";;\n'
        '  *) echo "ffmpeg version fake-1.0";;\nesac\n',
    )
    probe_bin = _fake_tool(tmp_path / "ffprobe", "true\n")
    tools = find_render_tools(Settings(ffmpeg=ff))
    assert tools.ffmpeg == ff
    assert tools.ffprobe == probe_bin
    assert tools.version == "ffmpeg version fake-1.0"


def test_ffmpeg_not_found(tmp_path: Path) -> None:
    with pytest.raises(RenderError, match="not found"):
        find_render_tools(Settings(ffmpeg=tmp_path / "nope"))


def test_render_tools_fixture_reads_owner_ffmpeg_at_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_conftest = str(Path(__file__).parents[1] / "conftest.py")
    render_conftest = next(
        m for m in list(sys.modules.values()) if getattr(m, "__file__", None) == root_conftest
    )
    sentinel = tmp_path / "owner-ffmpeg"
    monkeypatch.setattr(render_conftest, "_OWNER_FFMPEG", sentinel)
    seen: list[Path | None] = []

    def spy(settings: Settings) -> RenderTools:
        seen.append(settings.ffmpeg)
        return RenderTools(settings.ffmpeg or Path("x"), Path("probe"), "v")

    monkeypatch.setattr(render_conftest, "find_render_tools", spy)
    # Bypass pytest's fixture wrapper: call the underlying function.
    tools = render_conftest.render_tools.__wrapped__()
    assert tools.ffmpeg == sentinel
    assert seen == [sentinel]


def test_run_tool_nonzero_raises_with_stderr_tail(tmp_path: Path) -> None:
    sh = _fake_tool(tmp_path / "t", "for i in 1 2 3 4 5 6 7; do echo line$i >&2; done\nexit 3\n")
    with pytest.raises(RenderError) as ei:
        run_tool(sh, [])
    msg = str(ei.value)
    assert "line7" in msg and "line3" in msg and "line2" not in msg


def test_run_tool_timeout(tmp_path: Path) -> None:
    sh = _fake_tool(tmp_path / "t", "sleep 5\n")
    with pytest.raises(RenderError, match="timed out"):
        run_tool(sh, [], timeout_s=0.2)


def test_run_tool_oserror(tmp_path: Path) -> None:
    with pytest.raises(RenderError, match="cannot run"):
        run_tool(tmp_path / "missing", [])


def test_run_tool_returns_stdout_and_input(tmp_path: Path) -> None:
    sh = _fake_tool(tmp_path / "t", "cat\n")
    assert run_tool(sh, [], input=b"abc", cwd=tmp_path) == b"abc"


def test_run_tool_capture_returns_stderr(probe_tools: RenderTools) -> None:
    out, err = run_tool_capture(probe_tools.ffmpeg, ["-hide_banner", "-v", "info", "-version"])
    assert out.startswith(b"ffmpeg version")
    assert isinstance(err, str)
    _, err = run_tool_capture(
        probe_tools.ffmpeg,
        ["-hide_banner", "-nostdin", "-v", "info", "-f", "lavfi", "-i", "anullsrc=d=0.1",
         "-af", "volumedetect", "-f", "null", "-"],
    )  # fmt: skip
    assert "mean_volume" in err
    with pytest.raises(RenderError, match="failed"):
        run_tool_capture(probe_tools.ffmpeg, ["-hide_banner", "-i", "/nonexistent.mp4"])


def test_probe_bad_json(tmp_path: Path) -> None:
    sh = _fake_tool(tmp_path / "ffprobe", "echo notjson\n")
    with pytest.raises(RenderError, match="invalid JSON"):
        probe(RenderTools(sh, sh, "v"), tmp_path / "x.mp4")


def test_probe_parses_fake_output(tmp_path: Path) -> None:
    sh = _fake_tool(
        tmp_path / "ffprobe",
        "cat <<'EOF'\n"
        '{"streams":[{"codec_type":"video","width":4,"height":6,"avg_frame_rate":"30/1",'
        '"nb_read_frames":"3","duration":"0.1","codec_name":"h264","pix_fmt":"yuv420p"},'
        '{"codec_type":"audio","duration":"0.2","codec_name":"aac","sample_rate":"48000"}]}\n'
        "EOF\n",
    )
    p = probe(RenderTools(sh, sh, "v"), tmp_path / "x.mp4")
    assert (p.width, p.height, p.frames, p.video_ms, p.audio_ms) == (4, 6, 3, 100, 200)
    assert p.audio_rate == 48000 and p.fps is not None and int(p.fps) == 30


def test_probe_reads_streams(render_tools: RenderTools, tmp_path: Path) -> None:
    out = tmp_path / "t.mp4"
    subprocess.run(  # noqa: S603
        [
            str(render_tools.ffmpeg), "-v", "error", "-f", "lavfi", "-i",
            "testsrc=size=64x64:rate=30:duration=1", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )  # fmt: skip
    p = probe(render_tools, out)
    assert p.frames == 30
    assert p.audio_ms is not None and abs(p.audio_ms - 1000) <= 30
    assert p.video_codec == "h264" and p.audio_codec == "aac"


def test_render_ass_still_draws_ink(render_tools: RenderTools, tmp_path: Path) -> None:
    (tmp_path / "fonts").mkdir()
    (tmp_path / "c.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 64\nPlayResY: 48\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour\n"
        "Style: Default,Sans,10,&H00000000\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,{\\an5\\fs40}###\n"
    )
    img = F.render_ass_still(render_tools, tmp_path, "c.ass", (64, 48), 100)
    assert img.shape == (48, 64, 3) and img.dtype.name == "uint8"
    assert int(img.min()) < 255


def _ok_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(tool: Path, args: list[str], **_: object) -> bytes:
        if "-filters" in args:
            return b" ... ass V->V\n"
        if "-encoders" in args:
            return b" V....D libx264\n V....D ffv1\n A....D aac\n"
        return b"ffmpeg version x\n"

    monkeypatch.setattr(F, "run_tool", fake_run)


def test_ffmpeg_not_on_path_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RenderError, match="ffmpeg not found; install it"):
        find_render_tools(Settings())


def test_ffprobe_taken_from_path_when_not_beside_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "bin").mkdir()
    ff = _fake_tool(tmp_path / "bin" / "ffmpeg", "true\n")
    probe = _fake_tool(tmp_path / "ffprobe", "true\n")
    _ok_run(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: str(probe) if name == "ffprobe" else None)
    tools = find_render_tools(Settings(ffmpeg=ff))
    assert tools.ffprobe == probe


def test_ffprobe_missing_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ff = _fake_tool(tmp_path / "ffmpeg", "true\n")
    _ok_run(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RenderError, match="ffprobe not found beside ffmpeg or on PATH"):
        find_render_tools(Settings(ffmpeg=ff))
