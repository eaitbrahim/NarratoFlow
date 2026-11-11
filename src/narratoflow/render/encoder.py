"""FFV1 segment writer, then the single final pass: libass burn, FLAC mux, H.264 encode."""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np

from narratoflow.errors import RenderError
from narratoflow.render.ffmpeg import MediaProbe, RenderTools, probe, run_tool
from narratoflow.render.layout import FPS

SEGMENT_TIMEOUT_S = 600

X264_ARGS = [
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
    "-level:v", "4.2", "-pix_fmt", "yuv420p", "-g", "60", "-bf", "2",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-color_range", "tv",
]  # fmt: skip
AAC_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
AV_TOLERANCE_MS = 50
FINAL_TIMEOUT_S = 1800

# Relative names only: the final pass runs with cwd=<work dir>, so no path is interpolated.
_FINAL_VF = (
    "ass=filename=captions.ass:fontsdir=fonts:shaping=complex,"
    "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
)


class FrameSink:
    """Context manager writing HxWx3 uint8 RGB frames to exactly `path` (FFV1, bgr0)."""

    def __init__(self, ffmpeg: Path, path: Path, size: tuple[int, int]) -> None:
        self._ffmpeg = ffmpeg
        self._path = path
        self._size = size
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: IO[bytes] | None = None
        self.frames = 0
        self.returncode: int | None = None

    def _args(self) -> list[str]:
        w, h = self._size
        return [
            str(self._ffmpeg),
            *("-hide_banner", "-nostdin", "-v", "error"),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}"),
            *("-framerate", str(FPS), "-i", "pipe:0", "-an"),
            *("-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"),
            str(self._path),
        ]

    def __enter__(self) -> FrameSink:
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                self._args(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except OSError as exc:
            self._stderr.close()
            raise RenderError(f"cannot run {self._ffmpeg}: {exc}") from exc
        return self

    def _running(self) -> subprocess.Popen[bytes]:
        if self._proc is None:
            raise RuntimeError("FrameSink is used outside its `with` block")
        return self._proc

    def _tail(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        text = self._stderr.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-5:])

    def _fail(self) -> RenderError:
        return RenderError(f"{self._ffmpeg.name} failed (exit {self.returncode}):\n{self._tail()}")

    def _finish(self) -> None:
        """Close stdin and wait; raise RenderError on a timeout or a non-zero exit."""
        proc = self._running()
        if proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError):
                proc.stdin.close()
        try:
            self.returncode = proc.wait(timeout=SEGMENT_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            self._kill()
            raise RenderError(
                f"{self._ffmpeg.name} timed out after {SEGMENT_TIMEOUT_S}s writing {self._path}"
            ) from exc
        if self.returncode != 0:
            raise self._fail()

    def _kill(self) -> None:
        proc = self._running()
        proc.kill()
        if proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError):
                proc.stdin.close()
        self.returncode = proc.wait()

    def write(self, frame: np.ndarray) -> None:
        w, h = self._size
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"frame must be ({h}, {w}, 3) uint8, got {frame.shape} {frame.dtype}")
        stdin = self._running().stdin
        if stdin is None:  # pragma: no cover - Popen was given stdin=PIPE
            raise RuntimeError("ffmpeg has no stdin pipe")
        try:
            stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self._finish()
            raise RenderError(f"{self._ffmpeg.name} stopped reading frames") from None
        self.frames += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._proc is None:
                return
            if exc_type is not None:
                if self.returncode is None:
                    self._kill()
                return
            if self.returncode is None:
                self._finish()
        finally:
            if self._stderr is not None:
                self._stderr.close()


def write_concat_list(segments: Sequence[Path], path: Path) -> None:
    """Write an ffconcat list of bare segment names; each segment must sit beside `path`."""
    if not segments:
        raise ValueError("no segments to concatenate")
    lines: list[str] = []
    for seg in segments:
        name = seg.name
        if seg.parent not in (Path("."), path.parent):
            raise ValueError(f"segment {seg} is not in {path.parent}")
        if not name or any(c in name for c in "'/\n\r"):
            raise ValueError(f"unsafe segment name {name!r}")
        lines.append(f"file '{name}'\n")
    path.write_text("".join(lines), encoding="utf-8")


def concat_frames(tools: RenderTools, work_dir: Path) -> int:
    """Count the video packets of `work_dir/segments.txt` as concatenated (no decode)."""
    raw = run_tool(
        tools.ffprobe,
        [
            "-v", "error", "-f", "concat", "-safe", "0", "-i", "segments.txt",
            "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
        ],
        cwd=work_dir,
        timeout_s=FINAL_TIMEOUT_S,
    )  # fmt: skip
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        return int(text)
    except ValueError:
        raise RenderError(f"ffprobe returned {text!r} counting segment frames") from None


def encode_part(tools: RenderTools, work_dir: Path, *, frames: int, audio: Path, out: Path) -> None:
    """Burn captions.ass, mux the audio and encode H.264/AAC to `out`, atomically.

    The MP4 is written to a temp name beside `out` and renamed only on success. On any
    failure the temp file and any older file at `out` are removed, so a failed render never
    leaves a misleading MP4 behind.

    The concat demuxer does not fail on a bad segment: an unreadable one silently ends the
    input (exit 0 without `-xerror`), and a truncated one (a killed FrameSink) keeps its
    header duration, so `-fps_mode cfr` pads the gap with duplicated frames and the frame
    count still comes out right. So before encoding, the video packets of the whole concat
    input are counted (demux only, no decode) and must equal `frames`; `-xerror` is set too.
    A stale or partial segment can therefore never reach `out`.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    try:
        fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}.", suffix=".mp4")
        os.close(fd)
    except OSError as exc:
        raise RenderError(f"cannot write {out}: {exc}") from exc
    tmp = Path(tmp_name).absolute()
    args = [
        "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-y",
        "-f", "concat", "-safe", "0", "-i", "segments.txt",
        "-i", str(audio.absolute()),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", _FINAL_VF,
        "-r", str(FPS), "-fps_mode", "cfr", "-frames:v", str(frames),
        *X264_ARGS, *AAC_ARGS,
        "-movflags", "+faststart",
        str(tmp),
    ]  # fmt: skip
    try:
        got = concat_frames(tools, work_dir)
        if got != frames:
            raise RenderError(
                f"{out.name}: segments hold {got} frames, expected {frames} "
                "(a segment is short or corrupt)"
            )
        run_tool(tools.ffmpeg, args, cwd=work_dir, timeout_s=FINAL_TIMEOUT_S)
        try:
            tmp.chmod(0o644)
            os.replace(tmp, out)
        except OSError as exc:
            raise RenderError(f"cannot write {out}: {exc}") from exc
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
        raise


def _check(path: Path, field: str, got: object, want: object) -> None:
    if got != want:
        raise RenderError(f"{path.name}: {field} is {got}, expected {want}")


def _check_ms(path: Path, field: str, got: int | None, duration_ms: int) -> None:
    if got is None or abs(got - duration_ms) > AV_TOLERANCE_MS:
        raise RenderError(
            f"{path.name}: {field} is {got}, expected {duration_ms} +/- {AV_TOLERANCE_MS} ms"
        )


def verify_mp4(
    tools: RenderTools, path: Path, *, size: tuple[int, int], frames: int, duration_ms: int
) -> MediaProbe:
    """Probe `path` and raise RenderError naming the first field that is off spec."""
    got = probe(tools, path)
    _check(path, "width", got.width, size[0])
    _check(path, "height", got.height, size[1])
    _check(path, "fps", got.fps, Fraction(FPS))
    _check(path, "frames", got.frames, frames)
    _check(path, "video_codec", got.video_codec, "h264")
    _check(path, "pix_fmt", got.pix_fmt, "yuv420p")
    _check(path, "audio_codec", got.audio_codec, "aac")
    _check(path, "audio_rate", got.audio_rate, 48000)
    _check_ms(path, "video_ms", got.video_ms, duration_ms)
    _check_ms(path, "audio_ms", got.audio_ms, duration_ms)
    return got
