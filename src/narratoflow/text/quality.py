"""Input quality gate: reject PDF copy-paste corruption, never repair it."""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from narratoflow.errors import CorruptInputError
from narratoflow.models import Language

_EXCERPT_MAX = 40
_MAX_REPORTED = 10
_FRAGMENT_SHORT = 3
_FRAGMENT_RATIO = 0.2

_INVISIBLE_AND_TATWEEL = dict.fromkeys(map(ord, "‎‏؜ـ"))
# Harakat only: U+064B-U+065F and U+0670. Deliberately not tatweel or Qur'anic marks.
_HARAKAT = re.compile("[ً-ٰٟ]")
_ARABIC_STRETCH = re.compile("([ء-غف-ي])\\1{2,}")
_LATIN_STRETCH = re.compile("([A-Za-z])\\1{3,}")
_PAGE_NUMBER = re.compile("[\\d\u0660-\u0669\u06f0-\u06f9\\W_]+")
_PRIVATE_USE = re.compile("[-\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


class IssueKind(StrEnum):
    STRETCHED_LETTERS = "STRETCHED_LETTERS"
    PAGE_NUMBER = "PAGE_NUMBER"
    FRAGMENTED_LINES = "FRAGMENTED_LINES"
    REPLACEMENT_CHAR = "REPLACEMENT_CHAR"


@dataclass(frozen=True)
class QualityIssue:
    kind: IssueKind
    line: int
    excerpt: str


@dataclass(frozen=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def normalize_input(text: str) -> str:
    """Strip BOM, CRLF, bidi marks and tatweel; apply NFC. Letters and harakat are kept."""
    text = text.removeprefix("﻿").replace("\r\n", "\n")
    text = unicodedata.normalize("NFC", text)
    return text.translate(_INVISIBLE_AND_TATWEEL)


def _excerpt(line: str) -> str:
    return line.strip()[:_EXCERPT_MAX]


def check_input(text: str, language: Language) -> QualityReport:
    """Report corruption with 1-based line numbers. `language` is currently unused."""
    del language  # both stretch patterns are cheap and safe on any text
    lines = normalize_input(text).split("\n")
    issues: list[QualityIssue] = []
    non_empty = 0
    short: list[tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if "�" in line or _PRIVATE_USE.search(line):
            issues.append(QualityIssue(IssueKind.REPLACEMENT_CHAR, number, _excerpt(line)))
        if not stripped:
            continue
        non_empty += 1
        if len(stripped) <= _FRAGMENT_SHORT:
            short.append((number, stripped))
        if _PAGE_NUMBER.fullmatch(stripped):
            issues.append(QualityIssue(IssueKind.PAGE_NUMBER, number, _excerpt(line)))
        bare = _HARAKAT.sub("", line)
        if _ARABIC_STRETCH.search(bare) or _LATIN_STRETCH.search(bare):
            issues.append(QualityIssue(IssueKind.STRETCHED_LETTERS, number, _excerpt(line)))
    if non_empty and len(short) / non_empty > _FRAGMENT_RATIO:
        number, excerpt = short[0]
        issues.append(QualityIssue(IssueKind.FRAGMENTED_LINES, number, excerpt))
    issues.sort(key=lambda i: i.line)
    return QualityReport(tuple(issues))


def require_clean(text: str, language: Language) -> str:
    """Return the normalized text, or raise CorruptInputError listing up to 10 issues."""
    report = check_input(text, language)
    if not report.ok:
        shown = report.issues[:_MAX_REPORTED]
        detail = "\n".join(f"line {i.line}: {i.kind} «{i.excerpt}»" for i in shown)
        extra = len(report.issues) - len(shown)
        more = f"\n... and {extra} more" if extra else ""
        raise CorruptInputError(f"Input text looks corrupted:\n{detail}{more}")
    return normalize_input(text)
