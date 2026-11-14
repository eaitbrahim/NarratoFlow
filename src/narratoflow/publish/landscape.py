"""The 16:9 export: a pillarboxed copy of an exported 9:16 part (D31).

The compositor draws 9:16 only (ADR 0002), so a 16:9 file is the same video centred on
white bars, re-encoded with the project's H.264 settings and the audio copied. A native
panorama layout is a separate project; only this producer would change.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from narratoflow.errors import RenderError
from narratoflow.render.encoder import FINAL_TIMEOUT_S, X264_ARGS, verify_mp4
from narratoflow.render.ffmpeg import RenderTools, run_tool

LANDSCAPE_SIZE = (1920, 1080)
_PILLARBOX_VF = "scale=-2:1080:flags=lanczos,pad=1920:1080:(ow-iw)/2:0:color=white"


def landscape_name(index: int) -> str:
    """File name of part `index`'s 16:9 copy, beside its 9:16 export."""
    return f"part-{index:02d}.16x9.mp4"


def pillarbox_export(
    tools: RenderTools, src: Path, out: Path, *, frames: int, duration_ms: int
) -> Path:
    """Write a 1920x1080 pillarboxed copy of `src` to `out`; `RenderError` on any failure.

    The video goes to a temp file beside `out` and is renamed into place only after
    `verify_mp4` accepts it (1920x1080, `frames` frames, `duration_ms`). On failure the
    temp file and any older `out` are removed.
    """
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}.", suffix=".mp4")
        os.close(fd)
    except OSError as exc:
        raise RenderError(f"cannot write {out}: {exc}") from exc
    tmp = Path(name)
    args = [
        "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-y",
        "-i", str(src.absolute()),
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", _PILLARBOX_VF,
        *X264_ARGS, "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]  # fmt: skip
    try:
        run_tool(tools.ffmpeg, args, timeout_s=FINAL_TIMEOUT_S)
        verify_mp4(tools, tmp, size=LANDSCAPE_SIZE, frames=frames, duration_ms=duration_ms)
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
    return out
