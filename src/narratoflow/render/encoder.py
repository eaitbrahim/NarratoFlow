"""Pipe raw RGB frames into FFmpeg as a lossless FFV1 segment."""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np

from narratoflow.errors import RenderError
from narratoflow.render.layout import FPS

SEGMENT_TIMEOUT_S = 600


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
