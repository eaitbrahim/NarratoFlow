"""Render tool discovery, subprocess wrapper, media probing and libass stills."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from narratoflow.errors import RenderError
from narratoflow.settings import Settings

_ASS_FIX = (
    "this ffmpeg has no 'ass' filter (libass); on macOS run `brew install ffmpeg-full` and set "
    "NARRATOFLOW_FFMPEG=$(brew --prefix ffmpeg-full)/bin/ffmpeg"
)
_ENCODERS = ("libx264", "ffv1", "aac")


@dataclass(frozen=True)
class RenderTools:
    ffmpeg: Path
    ffprobe: Path
    version: str


@dataclass(frozen=True)
class MediaProbe:
    width: int | None
    height: int | None
    fps: Fraction | None
    frames: int | None
    video_ms: int | None
    audio_ms: int | None
    video_codec: str | None
    pix_fmt: str | None
    audio_codec: str | None
    audio_rate: int | None


def run_tool(
    tool: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    input: bytes | None = None,
    timeout_s: float = 120,
) -> bytes:
    try:
        proc = subprocess.run(  # noqa: S603
            [str(tool), *args],
            cwd=cwd,
            input=input,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"{tool.name} timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise RenderError(f"cannot run {tool}: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.decode("utf-8", errors="replace").splitlines()[-5:])
        raise RenderError(f"{tool.name} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def find_render_tools(settings: Settings) -> RenderTools:
    if settings.ffmpeg is not None:
        ffmpeg: Path | None = settings.ffmpeg
        if not settings.ffmpeg.is_file():
            raise RenderError(f"ffmpeg not found at {settings.ffmpeg} (NARRATOFLOW_FFMPEG)")
    else:
        found = shutil.which("ffmpeg")
        ffmpeg = Path(found) if found else None
    if ffmpeg is None:
        raise RenderError(
            "ffmpeg not found; install it (brew install ffmpeg-full / apt install ffmpeg)"
        )
    ffmpeg = ffmpeg.absolute()
    beside = ffmpeg.with_name("ffprobe")
    if beside.is_file():
        ffprobe: Path | None = beside
    else:
        found_probe = shutil.which("ffprobe")
        ffprobe = Path(found_probe) if found_probe else None
    if ffprobe is None:
        raise RenderError("ffprobe not found beside ffmpeg or on PATH")
    ffprobe = ffprobe.absolute()

    missing: list[str] = []
    filters = run_tool(ffmpeg, ["-hide_banner", "-filters"]).decode("utf-8", errors="replace")
    if not any(len(p) > 1 and p[1] == "ass" for p in (ln.split() for ln in filters.splitlines())):
        missing.append(_ASS_FIX)
    encoders = run_tool(ffmpeg, ["-hide_banner", "-encoders"]).decode("utf-8", errors="replace")
    names = {p[1] for p in (ln.split() for ln in encoders.splitlines()) if len(p) > 1}
    missing.extend(f"missing encoder {enc}" for enc in _ENCODERS if enc not in names)
    if missing:
        raise RenderError(f"{ffmpeg} is not usable for render: " + "; ".join(missing))

    out = run_tool(ffmpeg, ["-version"]).decode("utf-8", errors="replace")
    version = out.splitlines()[0] if out else ""
    return RenderTools(ffmpeg=ffmpeg, ffprobe=ffprobe, version=version)


def _ms(value: object) -> int | None:
    try:
        return round(float(str(value)) * 1000)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def probe(tools: RenderTools, path: Path) -> MediaProbe:
    raw = run_tool(
        tools.ffprobe,
        [
            "-v", "error", "-show_streams", "-show_format", "-of", "json", "-count_frames",
            str(path),
        ],
    )  # fmt: skip
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RenderError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps: Fraction | None = None
    if video is not None:
        try:
            fps = Fraction(str(video.get("avg_frame_rate", "")))
        except (ValueError, ZeroDivisionError):
            fps = None
        if fps == 0:
            fps = None
    video = video or {}
    audio = audio or {}
    return MediaProbe(
        width=_int(video.get("width")),
        height=_int(video.get("height")),
        fps=fps,
        frames=_int(video.get("nb_read_frames")),
        video_ms=_ms(video.get("duration")),
        audio_ms=_ms(audio.get("duration")),
        video_codec=video.get("codec_name"),
        pix_fmt=video.get("pix_fmt"),
        audio_codec=audio.get("codec_name"),
        audio_rate=_int(audio.get("sample_rate")),
    )


def render_ass_still(
    tools: RenderTools, work_dir: Path, ass_name: str, size: tuple[int, int], t_ms: int
) -> np.ndarray:
    w, h = size
    dur = (t_ms + 200) / 1000
    args = [
        "-hide_banner", "-nostdin", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=white:s={w}x{h}:r=30:d={dur}",
        "-vf", f"ass=filename={ass_name}:fontsdir=fonts:shaping=complex",
        "-ss", f"{t_ms / 1000}", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip
    out = run_tool(tools.ffmpeg, args, cwd=work_dir)
    if len(out) != w * h * 3:
        raise RenderError(f"ass still has {len(out)} bytes, expected {w * h * 3}")
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3).copy()
