"""The 16:9 export: a pillarboxed copy of an exported 9:16 part (D31).

The compositor draws 9:16 only (ADR 0002), so a 16:9 file is the same video centred on
white bars, re-encoded with the project's H.264 settings and the audio copied. A native
panorama layout is a separate project; only this producer would change.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from narratoflow.errors import RenderError
from narratoflow.render.encoder import FINAL_TIMEOUT_S, X264_ARGS, verify_mp4
from narratoflow.render.ffmpeg import RenderTools, run_tool
from narratoflow.render.layout import Canvas, Rect

LANDSCAPE_SIZE = (1920, 1080)
STAGE_16X9 = Rect(48, 115, 828, 850)
TOP_BAR_16X9 = Rect(912, 115, 960, 200)
CAPTIONS_16X9 = Rect(978, 450, 828, 180)
LANDSCAPE_BOTTOM = Rect(0, 972, 1920, 108)
BLEED_PX = 8
INK_LUMA_MAX = 195
CANVAS_SCALES = (Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
_PILLARBOX_VF = "scale=-2:1080:flags=lanczos,pad=1920:1080:(ow-iw)/2:0:color=white"


@dataclass(frozen=True)
class Zone:
    """One 9:16 zone and where it lands in the wide frame (both at the source's scale)."""

    name: str
    src: Rect
    dst: Rect
    bleed: int

    @property
    def src_bled(self) -> Rect:
        return _grow(self.src, self.bleed)

    @property
    def dst_bled(self) -> Rect:
        return _grow(self.dst, self.bleed)


@dataclass(frozen=True)
class PanoramaLayout:
    scale: Fraction
    source: tuple[int, int]
    size: tuple[int, int]
    zones: tuple[Zone, Zone, Zone]


def _grow(r: Rect, bleed: int) -> Rect:
    return Rect(r.x - bleed, r.y - bleed, r.w + 2 * bleed, r.h + 2 * bleed)


def landscape_size(scale: Fraction) -> tuple[int, int]:
    """Size of the 16:9 file for a source at canvas `scale`; `ValueError` on a bad scale."""
    Canvas(scale)
    return (int(LANDSCAPE_SIZE[0] * scale), int(LANDSCAPE_SIZE[1] * scale))


def panorama_layout(scale: Fraction) -> PanoramaLayout:
    """Where the stage, top bar and captions zones go in the wide frame, bled, at 1:1."""
    canvas = Canvas(scale)
    bleed = max(1, round(BLEED_PX * scale))

    def zone(name: str, src: Rect, dst: Rect) -> Zone:
        return Zone(
            name, src, Rect(round(dst.x * scale), round(dst.y * scale), src.w, src.h), bleed
        )

    return PanoramaLayout(
        scale=scale,
        source=(canvas.width, canvas.height),
        size=landscape_size(scale),
        zones=(
            zone("stage", canvas.stage, STAGE_16X9),
            zone("top_bar", canvas.top_bar, TOP_BAR_16X9),
            zone("captions", canvas.captions, CAPTIONS_16X9),
        ),
    )


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
