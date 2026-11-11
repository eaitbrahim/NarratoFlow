"""ASS subtitles in logical-order Unicode (libass shapes Arabic; no reshaper).

Every author or overlay string goes through `escape_text`, so it can never open an
override block; only the tags this module emits (``{\\k..}``) reach libass as tags.
Karaoke uses ``\\k`` and never ``\\kf``: libass sweeps ``\\kf`` left to right even over
right-to-left text.
"""

from __future__ import annotations

import itertools
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from narratoflow.audio.timeline import SceneTiming
from narratoflow.errors import RenderError
from narratoflow.models import Language, SceneKind, WordTiming
from narratoflow.render.assets import FontPack
from narratoflow.render.layout import FRAME_H, FRAME_W

CAPTION_MAX_WORDS = 6
CAPTION_MAX_CHARS = 28  # characters of the page text, joining spaces included
CAPTION_GAP_MS = 700
CAPTION_HOLD_MS = 400
SENTENCE_END = (".", "!", "?", "\u061f", "\u2026")

_MODE = 0o644
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
_NEWLINES = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
_ESCAPES = str.maketrans({"\\": "\u29f5", "{": "\uff5b", "}": "\uff5d"})

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """``#RRGGBB`` to ``&HAABBGGRR``; alpha 0 is opaque and 255 transparent."""
    if not _HEX.fullmatch(hex_rgb):
        raise ValueError(f"colour must be #RRGGBB, got {hex_rgb!r}")
    if not 0 <= alpha <= 255:
        raise ValueError(f"alpha must be 0..255, got {alpha}")
    rgb = hex_rgb[1:].upper()
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def _cs(ms: int) -> int:
    """Milliseconds to centiseconds, rounded half up."""
    return (ms * 2 + 10) // 20


def ass_time(ms: int) -> str:
    """``H:MM:SS.cc``, rounded half up to centiseconds."""
    if ms < 0:
        raise ValueError(f"ASS time cannot be negative, got {ms}")
    cs = _cs(ms)
    h, rest = divmod(cs, 360_000)
    m, rest = divmod(rest, 6_000)
    s, c = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def escape_text(s: str) -> str:
    """Neutralise override syntax in user text: no braces, backslashes or newlines."""
    return _NEWLINES.sub(" ", s.translate(_ESCAPES))


def karaoke(words: Sequence[tuple[str, int]], event_start_ms: int) -> str:
    """``\\k`` karaoke for ``(text, reveal_ms)`` words; each word lights at its reveal.

    Durations come from cumulative centisecond offsets, so a reveal stays within 5 ms
    of its time however long the page is.
    """
    prev: int | None = None
    offsets: list[int] = []
    for _, reveal in words:
        if prev is not None and reveal < prev:
            raise ValueError("karaoke reveal times must be non-decreasing")
        prev = reveal
        offsets.append(max(0, _cs(reveal - event_start_ms)))
    if not offsets:
        return ""
    durations = [b - a for a, b in itertools.pairwise(offsets)] + [0]
    body = " ".join(
        f"{{\\k{d}}}{escape_text(text)}" for (text, _), d in zip(words, durations, strict=True)
    )
    lead = f"{{\\k{offsets[0]}}}" if offsets[0] > 0 else ""
    return lead + body


@dataclass(frozen=True)
class AssStyle:
    name: str
    family: str
    size: int
    bold: bool
    primary: str
    primary_alpha: int
    secondary: str
    secondary_alpha: int
    outline: str
    outline_px: float
    alignment: int
    margin_l: int
    margin_r: int
    margin_v: int
    encoding: int

    def line(self) -> str:
        fields = [
            self.name,
            self.family,
            str(self.size),
            ass_color(self.primary, self.primary_alpha),
            ass_color(self.secondary, self.secondary_alpha),
            ass_color(self.outline),
            ass_color("#000000", 0xFF),  # BackColour: unused (no shadow, no box)
            "-1" if self.bold else "0",
            "0",  # Italic
            "0",  # Underline
            "0",  # StrikeOut
            "100",  # ScaleX
            "100",  # ScaleY
            "0",  # Spacing
            "0",  # Angle
            "1",  # BorderStyle: outline
            f"{self.outline_px:g}",
            "0",  # Shadow
            str(self.alignment),
            str(self.margin_l),
            str(self.margin_r),
            str(self.margin_v),
            str(self.encoding),
        ]
        return "Style: " + ",".join(fields)


@dataclass(frozen=True)
class AssEvent:
    start_ms: int
    end_ms: int
    style: str
    text: str
    layer: int = 0

    def line(self) -> str:
        return (
            f"Dialogue: {self.layer},{ass_time(self.start_ms)},{ass_time(self.end_ms)},"
            f"{self.style},,0,0,0,,{self.text}"
        )


@dataclass
class AssDocument:
    styles: list[AssStyle]
    events: list[AssEvent]

    def to_text(self) -> str:
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {FRAME_W}",
            f"PlayResY: {FRAME_H}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
            "[V4+ Styles]",
            _STYLE_FORMAT,
            *(s.line() for s in self.styles),
            "",
            "[Events]",
            _EVENT_FORMAT,
            *(e.line() for e in sorted(self.events, key=lambda e: (e.start_ms, e.layer))),
        ]
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        """Write UTF-8 (no BOM) atomically: a temp file beside `path`, then `os.replace`."""
        text = self.to_text()
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                tmp.write(text)
                tmp.flush()
                os.fchmod(tmp.fileno(), _MODE)
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        except OSError as exc:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
            raise RenderError(f"cannot write subtitles {path}: {exc}") from exc


# name: (font weight, size ar, size en, bold, alignment, (margin L, R, V)).
# Sizes are the starting values; when a zone test fails, lower the size here and never
# move or grow a zone. The Verse style always uses the Quran font.
_STYLE_TABLE: dict[str, tuple[str, int, int, bool, int, tuple[int, int, int]]] = {
    "Caption": ("bold", 64, 60, True, 8, (60, 192, 1260)),
    "Verse": ("quran", 76, 76, False, 5, (84, 216, 0)),
    "VerseRef": ("regular", 40, 40, False, 2, (84, 216, 696)),
    "TopLeft": ("bold", 36, 36, True, 7, (60, 60, 172)),
    "TopRight": ("regular", 36, 36, False, 9, (60, 60, 172)),
    "Headline": ("bold", 46, 46, True, 8, (60, 60, 240)),
    "Card": ("bold", 88, 88, True, 5, (84, 216, 0)),
}
_FONT_IDS = {
    Language.ARABIC: {"bold": "amiri-bold", "regular": "amiri-regular", "quran": "amiri-quran"},
    Language.ENGLISH: {
        "bold": "notosans-bold",
        "regular": "notosans-regular",
        "quran": "amiri-quran",
    },
}
_WHITE = "#FFFFFF"
_CAPTION_SECONDARY_ALPHA = 0x99


def build_styles(language: Language, ink: str, fonts: FontPack) -> list[AssStyle]:
    """The fixed style set; families come from the font pack, never literals."""
    styles: list[AssStyle] = []
    for name, (weight, size_ar, size_en, bold, align, (ml, mr, mv)) in _STYLE_TABLE.items():
        verse = name == "Verse"
        secondary_alpha = 0
        if name == "Caption":
            secondary_alpha = _CAPTION_SECONDARY_ALPHA
        elif verse:
            secondary_alpha = 0xFF
        styles.append(
            AssStyle(
                name=name,
                family=fonts.get(_FONT_IDS[language][weight]).family,
                size=size_ar if language is Language.ARABIC else size_en,
                bold=bold,
                primary=ink,
                primary_alpha=0,
                secondary=ink,
                secondary_alpha=secondary_alpha,
                outline=_WHITE,
                outline_px=0 if verse else 3,
                alignment=align,
                margin_l=ml,
                margin_r=mr,
                margin_v=mv,
                encoding=-1 if verse or language is Language.ARABIC else 1,
            )
        )
    return styles


def caption_pages(words: Sequence[WordTiming]) -> list[list[WordTiming]]:
    """Greedy, in-order pages bounded by word count, characters, sentence ends and gaps."""
    pages: list[list[WordTiming]] = []
    page: list[WordTiming] = []
    chars = 0
    for w in words:
        if page:
            prev = page[-1]
            if (
                len(page) + 1 > CAPTION_MAX_WORDS
                or chars + 1 + len(w.text) > CAPTION_MAX_CHARS
                or prev.text.endswith(SENTENCE_END)
                or w.start_ms - prev.start_ms > CAPTION_GAP_MS
            ):
                pages.append(page)
                page, chars = [], 0
        chars += len(w.text) + (1 if page else 0)
        page.append(w)
    if page:
        pages.append(page)
    return pages


def caption_events(scene: SceneTiming) -> list[AssEvent]:
    """One word-timed ``Caption`` event per page; none for a verse scene."""
    if scene.kind is SceneKind.VERSE or not scene.words:
        return []
    pages = caption_pages(scene.words)
    events: list[AssEvent] = []
    for i, page in enumerate(pages):
        start = page[0].start_ms
        if i + 1 < len(pages):
            end = pages[i + 1][0].start_ms
        else:
            end = min(max(w.end_ms for w in page) + CAPTION_HOLD_MS, scene.end_ms)
        text = karaoke([(w.text, w.start_ms) for w in page], start)
        events.append(AssEvent(start, end, "Caption", text))
    return events
