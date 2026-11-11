"""ASS subtitles in logical-order Unicode (libass shapes Arabic; no reshaper).

Every author or overlay string goes through `escape_text`, so it can never open an
override block; only the tags this module emits (``{\\k..}``) reach libass as tags.
Karaoke uses ``\\k`` and never ``\\kf``: libass sweeps ``\\kf`` left to right even over
right-to-left text. It never emits ``\\k0`` either: libass does not start a syllable at a
zero-length ``\\k``, so such a word would light with the syllable before it.

Style sizes are target em pixels; `FontPack.fontsize_for_em` turns them into a per-font
ASS Fontsize, because libass makes Fontsize the font's win-metric line height.
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from narratoflow.audio.timeline import SceneTiming
from narratoflow.errors import RenderError
from narratoflow.models import Language, SceneKind, WordTiming
from narratoflow.render.assets import FontPack
from narratoflow.render.layout import FRAME_H, FRAME_W

CAPTION_MAX_WORDS = 6
CAPTION_MAX_CHARS = 28  # characters of the page text, joining spaces included
# Arabic pages stay on one line: at the Caption em, two Amiri lines (line height = the
# win-metric Fontsize) do not fit the 180 px CAPTIONS band.
CAPTION_MAX_CHARS_AR = 28  # widest letters (seen, sad) measure ~27.5 px each: 28 fit 828 px
CAPTION_GAP_MS = 700
CAPTION_HOLD_MS = 400
SENTENCE_END = (".", "!", "?", "\u061f", "\u2026")

_MODE = 0o644
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
_NEWLINES = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
_ESCAPES = str.maketrans({"\\": "/", "{": "(", "}": ")"})  # glyphs every bundled font has
_TAGS = re.compile(r"\{[^}]*\}")
_TIME = re.compile(r"(\d+):([0-5]\d):([0-5]\d)\.(\d\d)")
_DIALOGUE = "Dialogue:"

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """``#RRGGBB`` to ``&HAABBGGRR``; alpha 0 is opaque and 255 transparent."""
    if not _HEX.fullmatch(hex_rgb):
        raise RenderError(f"ASS colour must be #RRGGBB, got {hex_rgb!r}")
    if not 0 <= alpha <= 255:
        raise RenderError(f"ASS alpha must be 0..255, got {alpha}")
    rgb = hex_rgb[1:].upper()
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def _cs(ms: int) -> int:
    """Milliseconds to centiseconds, rounded half up."""
    return (ms * 2 + 10) // 20


def ass_round_ms(ms: int) -> int:
    """The milliseconds `ass_time` writes for `ms` (rounded half up to centiseconds)."""
    return _cs(ms) * 10


def ass_time(ms: int) -> str:
    """``H:MM:SS.cc``, rounded half up to centiseconds."""
    if ms < 0:
        raise RenderError(f"ASS time cannot be negative, got {ms}")
    cs = _cs(ms)
    h, rest = divmod(cs, 360_000)
    m, rest = divmod(rest, 6_000)
    s, c = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def escape_text(s: str) -> str:
    """Neutralise override syntax in user text: no braces, backslashes or newlines."""
    return _NEWLINES.sub(" ", s.translate(_ESCAPES))


def strip_tags(text: str, sep: str = "") -> str:
    """`text` with every ``{...}`` override block replaced by `sep`."""
    return _TAGS.sub(sep, text)


def karaoke(words: Sequence[tuple[str, int]], event_start_ms: int) -> str:
    """``\\k`` karaoke for ``(text, reveal_ms)`` words; each word lights at its reveal.

    Offsets are ``c_i = cs(reveal_i) - cs(event_start)`` clamped at 0, so a reveal stays
    within 5 ms of its time however long the page is. Words sharing an offset share one
    syllable (libass ignores ``\\k0``), the last syllable lasts 1 cs, and a leading empty
    ``{\\k<c_0>}`` delays the first word when it is revealed after the event start.
    """
    prev: int | None = None
    groups: list[tuple[int, list[str]]] = []
    for text, reveal in words:
        if prev is not None and reveal < prev:
            raise RenderError("karaoke reveal times must be non-decreasing")
        prev = reveal
        offset = max(0, _cs(reveal) - _cs(event_start_ms))
        if groups and groups[-1][0] == offset:
            groups[-1][1].append(escape_text(text))
        else:
            groups.append((offset, [escape_text(text)]))
    if not groups:
        return ""
    parts: list[str] = []
    for i, (offset, texts) in enumerate(groups):
        d = groups[i + 1][0] - offset if i + 1 < len(groups) else 1
        parts.append(f"{{\\k{d}}}" + " ".join(texts))
    lead = f"{{\\k{groups[0][0]}}}" if groups[0][0] > 0 else ""
    return lead + " ".join(parts)


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


def _parse_time(field: str) -> int:
    m = _TIME.fullmatch(field.strip())
    if m is None:
        raise ValueError(f"bad ASS time {field!r}")
    h, mi, sec, cs = (int(g) for g in m.groups())
    return ((h * 60 + mi) * 60 + sec) * 1000 + cs * 10


def parse_events(text: str) -> list[AssEvent]:
    """The ``Dialogue:`` events of the ``[Events]`` section, in file order.

    Inverts `AssEvent.line` (times come back at centisecond precision); the text field is
    the tenth and may contain commas. A malformed line raises `ValueError` naming it.
    """
    events: list[AssEvent] = []
    in_events = False
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line == "[Events]"
            continue
        if not in_events or not line.startswith(_DIALOGUE):
            continue
        fields = raw.lstrip()[len(_DIALOGUE) :].lstrip().split(",", 9)
        try:
            if len(fields) != 10:
                raise ValueError(f"{len(fields)} fields, expected 10")
            event = AssEvent(
                start_ms=_parse_time(fields[1]),
                end_ms=_parse_time(fields[2]),
                style=fields[3].strip(),
                text=fields[9],
                layer=int(fields[0]),
            )
        except ValueError as exc:
            raise ValueError(f"malformed ASS event on line {n}: {raw!r}: {exc}") from None
        events.append(event)
    return events


# name: (font weight, em px, bold, alignment, (margin L, R, V)).
# Sizes are target em pixels (the Noto Sans em at the brief's English Fontsizes 60, 76,
# 40, 36, 36, 46, 88), converted per font by `FontPack.fontsize_for_em`, so English output
# is unchanged and Arabic gets the same em. When a zone test fails, lower the size here
# and never move or grow a zone. The Verse style always uses the Quran font.
_STYLE_TABLE: dict[str, tuple[str, Fraction, bool, int, tuple[int, int, int]]] = {
    "Caption": ("bold", Fraction("39.5"), True, 8, (60, 192, 1260)),
    "Verse": ("quran", Fraction("50.0"), False, 5, (84, 216, 0)),
    "VerseRef": ("regular", Fraction("26.3"), False, 2, (84, 216, 696)),
    "TopLeft": ("bold", Fraction("23.7"), True, 7, (60, 60, 172)),
    "TopRight": ("regular", Fraction("23.7"), False, 9, (60, 60, 172)),
    "Headline": ("bold", Fraction("30.3"), True, 8, (60, 60, 240)),
    "Card": ("bold", Fraction("57.9"), True, 5, (84, 216, 0)),
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


def style_font_id(language: Language, style: str) -> str:
    """The font pack id a style renders with in `language`."""
    return _FONT_IDS[language][_STYLE_TABLE[style][0]]


_CAPTION_SECONDARY_ALPHA = 0x99


def build_styles(language: Language, ink: str, fonts: FontPack) -> list[AssStyle]:
    """The fixed style set; families come from the font pack, never literals."""
    styles: list[AssStyle] = []
    for name, (weight, em_px, bold, align, (ml, mr, mv)) in _STYLE_TABLE.items():
        verse = name == "Verse"
        font_id = _FONT_IDS[language][weight]
        secondary_alpha = 0
        if name == "Caption":
            secondary_alpha = _CAPTION_SECONDARY_ALPHA
        elif verse:
            secondary_alpha = 0xFF
        styles.append(
            AssStyle(
                name=name,
                family=fonts.get(font_id).family,
                size=fonts.fontsize_for_em(font_id, em_px),
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


def caption_max_chars(language: Language) -> int:
    """The page character limit: one line for Arabic, up to two lines for English."""
    return CAPTION_MAX_CHARS_AR if language is Language.ARABIC else CAPTION_MAX_CHARS


EMOJI_MODIFIERS = range(0x1F3FB, 0x1F400)


def display_units(ch: str, language: Language) -> int:
    """Display width in quarter characters: marks and joiners take none; Latin letters in
    an Arabic style (Amiri) take 5/4 of an Arabic letter."""
    if unicodedata.category(ch) in ("Mn", "Me", "Cf") or ord(ch) in EMOJI_MODIFIERS:
        return 0
    if language is Language.ARABIC and ord(ch) < 0x0600 and not ch.isspace():
        return 5
    return 4


def _width(text: str, language: Language) -> int:
    """`display_units` of `text`: tashkeel takes no width, Latin in Arabic takes 5/4."""
    return sum(display_units(ch, language) for ch in text)


def caption_pages(
    words: Sequence[WordTiming],
    max_chars: int = CAPTION_MAX_CHARS,
    language: Language = Language.ENGLISH,
) -> list[list[WordTiming]]:
    """Greedy, in-order pages bounded by word count, width, sentence ends and gaps.

    The width is counted like `overlays.truncate` counts it (`display_units`, in quarter
    characters against ``4 * max_chars``), so Latin-heavy Arabic pages stay one line.
    """
    room = 4 * max_chars
    space = display_units(" ", language)
    pages: list[list[WordTiming]] = []
    page: list[WordTiming] = []
    used = 0
    for w in words:
        if page:
            prev = page[-1]
            if (
                len(page) + 1 > CAPTION_MAX_WORDS
                or used + space + _width(w.text, language) > room
                or prev.text.endswith(SENTENCE_END)
                or w.start_ms - prev.start_ms > CAPTION_GAP_MS
            ):
                pages.append(page)
                page, used = [], 0
        used += _width(w.text, language) + (space if page else 0)
        page.append(w)
    if page:
        pages.append(page)
    return pages


def caption_events(scene: SceneTiming, language: Language) -> list[AssEvent]:
    """One word-timed ``Caption`` event per page; none for a verse scene.

    A page that would last 0 ms (the next page starts at the same time) is dropped.
    """
    if scene.kind is SceneKind.VERSE or not scene.words:
        return []
    pages = caption_pages(scene.words, caption_max_chars(language), language)
    events: list[AssEvent] = []
    for i, page in enumerate(pages):
        start = page[0].start_ms
        if i + 1 < len(pages):
            end = pages[i + 1][0].start_ms
        else:
            end = min(max(w.end_ms for w in page) + CAPTION_HOLD_MS, scene.end_ms)
        if end <= start:
            continue
        text = karaoke([(w.text, w.start_ms) for w in page], start)
        events.append(AssEvent(start, end, "Caption", text))
    return events
