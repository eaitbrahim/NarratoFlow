"""The 16:9 export, made from an exported 9:16 part.

`panorama_export` re-composes the part for the wide frame (#163): the stage, top bar and
captions zones are cropped whole at 1:1 with a bleed and placed side by side in a frame
1920x1080 times the source canvas scale, in one ffmpeg pass with the project's H.264 settings
and the audio copied. An ink guard refuses a source with ink outside those zones, so verse
pixels are always the certified 9:16 burn.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from narratoflow.errors import RenderError
from narratoflow.render.encoder import FINAL_TIMEOUT_S, X264_ARGS, verify_mp4
from narratoflow.render.ffmpeg import RenderTools, probe, run_tool
from narratoflow.render.layout import Canvas, Rect

LANDSCAPE_SIZE = (1920, 1080)
STAGE_16X9 = Rect(48, 115, 828, 850)
TOP_BAR_16X9 = Rect(912, 115, 960, 200)
CAPTIONS_16X9 = Rect(978, 450, 828, 180)
LANDSCAPE_BOTTOM = Rect(0, 972, 1920, 108)
BLEED_PX = 8
INK_LUMA_MAX = 195
CANVAS_SCALES = (Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))


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
    bleed = round(BLEED_PX * scale)

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


_FRAME_LINE = re.compile(r"frame:(\d+)\s")
_YMIN_LINE = re.compile(r"lavfi\.signalstats\.YMIN=(\d+)")


def _ink_filter(layout: PanoramaLayout) -> str:
    """Mask the bled source zones white, then write each frame's darkest luma to `ink.txt`."""
    masks = [
        f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}:color=white:t=fill"
        for r in (z.src_bled for z in layout.zones)
    ]
    return ",".join(
        [*masks, "signalstats", "metadata=mode=print:key=lavfi.signalstats.YMIN:file=ink.txt"]
    )


def _unreadable(reason: str) -> RenderError:
    return RenderError(f"ink guard output is unreadable ({reason})")


def _record_luma(head: str, value: str, n: int) -> int:
    """The darkest luma of frame `n` from its two lines, or `RenderError`."""
    framed, valued = _FRAME_LINE.match(head), _YMIN_LINE.fullmatch(value)
    if framed is None or int(framed.group(1)) != n:
        raise _unreadable(f"expected a record for frame {n}, got {head!r}")
    if valued is None or int(valued.group(1)) > 255:
        raise _unreadable(f"frame {n} has no luma value, got {value!r}")
    return int(valued.group(1))


def _first_ink(text: str, frames: int) -> tuple[int, int] | None:
    """The first `(frame, darkest luma)` below `INK_LUMA_MAX`, or None.

    The whole text is read and must hold exactly `frames` records numbered `0..frames-1`;
    anything else is `RenderError`, so a short or garbled file is never a pass.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 2 * frames:
        raise _unreadable(f"{len(lines)} lines for {frames} frames")
    lumas = [_record_luma(lines[2 * n], lines[2 * n + 1], n) for n in range(frames)]
    return next(((n, v) for n, v in enumerate(lumas) if v < INK_LUMA_MAX), None)


def _check_ink(tools: RenderTools, src: Path, layout: PanoramaLayout, *, frames: int) -> None:
    """`RenderError` if any frame of `src` has ink outside the bled zones, or the guard fails."""
    with tempfile.TemporaryDirectory(prefix="narratoflow-ink-") as name:
        work = Path(name)
        args = [
            "-hide_banner", "-nostdin", "-v", "error", "-xerror",
            "-i", str(src.absolute()),
            "-map", "0:v:0", "-vf", _ink_filter(layout),
            "-f", "null", "-",
        ]  # fmt: skip
        run_tool(tools.ffmpeg, args, cwd=work, timeout_s=FINAL_TIMEOUT_S)
        try:
            text = (work / "ink.txt").read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as exc:
            raise RenderError(f"ink guard wrote no readable output for {src.name}: {exc}") from exc
    hit = _first_ink(text, frames)
    if hit is not None:
        raise RenderError(
            f"{src.name}: ink outside the 16:9 zones at frame {hit[0]} "
            f"(luma {hit[1]} < {INK_LUMA_MAX}); "
            "the 9:16 layout lets text or drawing leave its zone"
        )


def _compose_graph(layout: PanoramaLayout) -> str:
    """One filtergraph: crop the three bled zones and place them on white at their spots."""
    s, t, c = (z.src_bled for z in layout.zones)
    sd, td, cd = (z.dst_bled for z in layout.zones)
    w, h = layout.size

    def crop(r: Rect) -> str:
        return f"crop={r.w}:{r.h}:{r.x}:{r.y}:exact=1"

    return (
        "[0:v]format=yuv444p,split=3[s][t][c];"
        f"[s]{crop(s)},pad={w}:{h}:{sd.x}:{sd.y}:color=white[p0];"
        f"[t]{crop(t)}[t1];[p0][t1]overlay={td.x}:{td.y}:format=yuv444[p1];"
        f"[c]{crop(c)}[c1];[p1][c1]overlay={cd.x}:{cd.y}:format=yuv444[p2];"
        "[p2]format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709"
        ":colorspace=bt709:range=tv[v]"
    )


def _source_scale(src: Path, width: int | None, height: int | None) -> Fraction:
    for scale in CANVAS_SCALES:
        canvas = Canvas(scale)
        if (canvas.width, canvas.height) == (width, height):
            return scale
    raise RenderError(f"{src.name}: size {width}x{height} is not a 9:16 canvas size")


def _identity(path: Path) -> tuple[int, int, int, int]:
    try:
        st = path.stat()
    except OSError as exc:
        raise RenderError(f"cannot read {path}: {exc}") from exc
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def panorama_export(
    tools: RenderTools, src: Path, out: Path, *, frames: int, duration_ms: int
) -> Path:
    """Write the side-by-side 16:9 copy of `src` to `out`; `RenderError` on any failure.

    The ink guard runs before the encode. The video goes to a temp file beside `out` and
    is renamed into place only after `verify_mp4` accepts it and `src` is seen unchanged.
    On failure the temp file and any older `out` are removed.
    """
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}.", suffix=".mp4")
        os.close(fd)
    except OSError as exc:
        raise RenderError(f"cannot write {out}: {exc}") from exc
    tmp = Path(name)
    try:
        before = _identity(src)
        found = probe(tools, src)
        layout = panorama_layout(_source_scale(src, found.width, found.height))
        if found.frames != frames:
            raise RenderError(f"{src.name}: frames is {found.frames}, expected {frames}")
        _check_ink(tools, src, layout, frames=frames)
        args = [
            "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-y",
            "-i", str(src.absolute()),
            "-filter_complex", _compose_graph(layout),
            "-map", "[v]", "-map", "0:a:0",
            *X264_ARGS, "-c:a", "copy",
            "-movflags", "+faststart",
            str(tmp),
        ]  # fmt: skip
        run_tool(tools.ffmpeg, args, timeout_s=FINAL_TIMEOUT_S)
        verify_mp4(tools, tmp, size=layout.size, frames=frames, duration_ms=duration_ms)
        if _identity(src) != before:
            raise RenderError(f"{src.name} changed during the 16:9 export")
        try:
            tmp.chmod(0o644)
            os.replace(tmp, out)
        except OSError as exc:
            raise RenderError(f"cannot write {out}: {exc}") from exc
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        with contextlib.suppress(OSError):
            out.unlink()
        raise
    return out


def landscape_name(index: int) -> str:
    """File name of part `index`'s 16:9 copy, beside its 9:16 export."""
    return f"part-{index:02d}.16x9.mp4"
