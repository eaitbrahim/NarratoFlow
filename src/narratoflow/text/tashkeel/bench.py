"""DER/WER scoring and gold-set loading for the diacritizer benchmark.

Definitions follow Fadel et al. 2019 (`diacritization_stat.py`): only Arabic letters are
scored; a letter's class is the set of harakat that follow it (shadda and a vowel in either
order are one class); a gold letter with no harakat has class "". The "without case ending"
variant drops each word's last letter. See docs/adr/0005-diacritizer-benchmark.md.
"""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from narratoflow.errors import ProviderError
from narratoflow.text.tashkeel.engines import _is_neutral
from narratoflow.text.tashkeel.verify import SkeletonMismatchError, is_diacritic, strip_diacritics

__all__ = [
    "Counts",
    "EngineResult",
    "GoldManifest",
    "LineScore",
    "load_manifest",
    "make_manifest",
    "manifest_toml",
    "not_run",
    "results_markdown",
    "run_engine",
    "score_line",
    "select_gold",
]


@dataclass(frozen=True)
class Counts:
    letters: int
    letter_errors: int
    words: int
    word_errors: int


@dataclass(frozen=True)
class LineScore:
    with_case: Counts
    without_case: Counts
    rejected: bool


@dataclass(frozen=True)
class EngineResult:
    engine: str
    lines: int
    rejected_lines: int
    der: float
    der_no_case: float
    wer: float
    wer_no_case: float
    seconds: float
    not_run: str | None = None


@dataclass(frozen=True)
class GoldManifest:
    source_url: str
    commit: str
    file_sha256: str
    licence_notes: str
    rule: str
    lines: list[tuple[int, str]]


def _is_letter(ch: str) -> bool:
    return 0x0621 <= ord(ch) <= 0x063A or 0x0641 <= ord(ch) <= 0x064A


def _is_haraka(ch: str) -> bool:
    return 0x064B <= ord(ch) <= 0x0652


def _words(text: str) -> list[list[str]]:
    """Diacritic class of every Arabic letter, grouped by whitespace token (no letters: dropped)."""
    words: list[list[str]] = []
    for token in unicodedata.normalize("NFC", text).split():
        classes: list[str] = []
        after_letter = False
        for ch in token:
            if _is_letter(ch):
                classes.append("")
                after_letter = True
            elif _is_haraka(ch):
                if after_letter:
                    classes[-1] += ch
            elif not is_diacritic(ch):
                after_letter = False
        if classes:
            words.append(["".join(sorted(c)) for c in classes])
    return words


def _bases(text: str) -> list[str]:
    return [
        ch
        for ch in unicodedata.normalize("NFC", text)
        if not _is_neutral(ch) and not is_diacritic(ch)
    ]


def _count(gold: list[list[str]], predicted: list[str], *, drop_last: bool) -> Counts:
    """Compare class by class; the word grouping is the gold's."""
    letters = letter_errors = word_errors = 0
    start = 0
    for word in gold:
        scored = len(word) - 1 if drop_last else len(word)
        wrong = sum(
            g != p for g, p in zip(word[:scored], predicted[start : start + scored], strict=False)
        )
        letters += scored
        letter_errors += wrong
        word_errors += wrong > 0
        start += len(word)
    return Counts(letters, letter_errors, len(gold), word_errors)


def _all_wrong(counts: Counts) -> Counts:
    return Counts(counts.letters, counts.letters, counts.words, counts.words)


def score_line(gold: str, predicted: str) -> LineScore:
    """Score one line. Output whose base letters differ from the gold is rejected: all wrong."""
    words = _words(gold)
    if _bases(gold) != _bases(predicted):
        return LineScore(
            _all_wrong(_count(words, [], drop_last=False)),
            _all_wrong(_count(words, [], drop_last=True)),
            rejected=True,
        )
    flat = list(chain.from_iterable(_words(predicted)))
    return LineScore(
        _count(words, flat, drop_last=False), _count(words, flat, drop_last=True), rejected=False
    )


def _rate(errors: int, total: int) -> float:
    return errors / total * 100 if total else 0.0


def run_engine(
    name: str,
    diacritize: Callable[[str], str],
    gold: Sequence[str],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> EngineResult:
    """Run an engine over bare gold lines. A failed or skeleton-changing output is rejected."""
    totals = [0] * 8
    rejected_lines = 0
    started = clock()
    for line in gold:
        try:
            predicted = diacritize(strip_diacritics(line))
        except (SkeletonMismatchError, ProviderError):
            predicted = ""
        score = score_line(line, predicted)
        rejected_lines += score.rejected
        for i, c in enumerate((*_fields(score.with_case), *_fields(score.without_case))):
            totals[i] += c
    seconds = clock() - started
    return EngineResult(
        engine=name,
        lines=len(gold),
        rejected_lines=rejected_lines,
        der=_rate(totals[1], totals[0]),
        der_no_case=_rate(totals[5], totals[4]),
        wer=_rate(totals[3], totals[2]),
        wer_no_case=_rate(totals[7], totals[6]),
        seconds=seconds,
    )


def _fields(counts: Counts) -> tuple[int, int, int, int]:
    return counts.letters, counts.letter_errors, counts.words, counts.word_errors


def not_run(name: str, reason: str) -> EngineResult:
    return EngineResult(name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, not_run=reason)


def results_markdown(results: Sequence[EngineResult]) -> str:
    rows = [
        "| engine | lines | rejected | DER | DER (no case ending) | WER | WER (no case ending) "
        "| runtime |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    notes = []
    for r in results:
        if r.not_run is not None:
            rows.append(f"| {r.engine} | " + " | ".join(["not run"] * 6) + " | - |")
            notes.append(f"{r.engine}: {r.not_run}")
        else:
            rows.append(
                f"| {r.engine} | {r.lines} | {r.rejected_lines} | {r.der:.2f} | {r.der_no_case:.2f}"
                f" | {r.wer:.2f} | {r.wer_no_case:.2f} | {r.seconds:.1f} s |"
            )
    return "\n".join([*rows, "", *notes])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_manifest(
    text: str,
    *,
    source_url: str,
    commit: str,
    licence_notes: str,
    count: int = 300,
    min_words: int = 5,
    max_words: int = 60,
) -> GoldManifest:
    """The first `count` lines with `min_words` to `max_words` whitespace tokens (1-based)."""
    picked: list[tuple[int, str]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if len(picked) == count:
            break
        if min_words <= len(line.split()) <= max_words:
            picked.append((number, _sha256(line)))
    rule = (
        f"first {count} lines of the file, in order, with {min_words} to {max_words} "
        "whitespace-separated tokens; line numbers are 1-based"
    )
    return GoldManifest(source_url, commit, _sha256(text), licence_notes, rule, picked)


def manifest_toml(manifest: GoldManifest) -> str:
    """ASCII TOML; json.dumps yields valid basic strings for ASCII text."""
    head = [
        f"{key} = {json.dumps(value)}"
        for key, value in (
            ("source_url", manifest.source_url),
            ("commit", manifest.commit),
            ("file_sha256", manifest.file_sha256),
            ("licence_notes", manifest.licence_notes),
            ("rule", manifest.rule),
        )
    ]
    body = [f'[[lines]]\nn = {n}\nsha256 = "{digest}"' for n, digest in manifest.lines]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


def load_manifest(path: Path) -> GoldManifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return GoldManifest(
        source_url=data["source_url"],
        commit=data["commit"],
        file_sha256=data["file_sha256"],
        licence_notes=data["licence_notes"],
        rule=data["rule"],
        lines=[(row["n"], row["sha256"]) for row in data["lines"]],
    )


def select_gold(text: str, manifest: GoldManifest) -> list[str]:
    """The manifest's lines from the downloaded file, each checked against its hash."""
    rows = text.split("\n")
    out: list[str] = []
    for number, digest in manifest.lines:
        line = rows[number - 1] if number <= len(rows) else ""
        if _sha256(line) != digest:
            raise ValueError(f"gold line {number}: sha256 mismatch")
        out.append(line)
    return out
