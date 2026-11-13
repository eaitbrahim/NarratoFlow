"""Spike: how far do edge-tts Arabic word boundaries drift from a CTC alignment?

Voices fixed diacritized sentences with edge-tts, times the same audio with the CTC
aligner, and prints a Markdown report. CTC is a reference, not ground truth.
Run: uv sync --extra tts-edge --extra align && uv run python scripts/spike_word_timing.py

Capture/analyse split (docs/spikes/2026-11-13-word-onset-ground-truth.md):
  --save DIR  one networked run that stores the audio and spans per sentence;
  --from DIR  offline analysis against waveform energy and optional labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

from narratoflow.audio.io import SAMPLE_RATE, Samples, decode
from narratoflow.audio.mp3info import DECODER_DELAY_SAMPLES, read_mp3_info
from narratoflow.audio.onsets import voiced_regions
from narratoflow.audio.words import TimedSpan, timings_from_spans, tokens
from narratoflow.models import WordTiming

VOICE = "ar-SA-HamedNeural"
EXTRAS_HINT = "uv sync --extra tts-edge --extra align"
EDGE_HINT = "uv sync --extra tts-edge"
CAPTURE_VERSION = 1
ENCODER_DELAY_SAMPLES = 576  # libmp3lame, as measured in docs/spikes/2026-11-13-*
PRIMING_RATE = 24000  # Hz, the edge voice's MP3 sample rate
SIGN_RULE = "estimate minus waveform (positive = the estimate is later)"

SENTENCES: tuple[str, ...] = (
    (
        "\u0648\u064e\u0630\u064e\u0647\u064e\u0628\u064e \u0627\u0644\u0652"
        "\u0648\u064e\u0644\u064e\u062f\u064f \u0625\u0650\u0644\u064e\u0649"
        " \u0627\u0644\u0652\u0645\u064e\u062f\u0652\u0631\u064e\u0633\u064e"
        "\u0629\u0650 \u0635\u064e\u0628\u064e\u0627\u062d\u064b\u0627."
    ),
    (
        "\u0641\u064e\u0643\u064e\u062a\u064e\u0628\u064e \u0627\u0644\u0637"
        "\u064e\u0651\u0627\u0644\u0650\u0628\u064f \u0631\u0650\u0633\u064e"
        "\u0627\u0644\u064e\u0629\u064b \u0637\u064e\u0648\u0650\u064a\u0644"
        "\u064e\u0629\u064b\u060c \u062b\u064f\u0645\u064e\u0651 "
        "\u0623\u064e\u0631\u0652\u0633\u064e\u0644\u064e\u0647\u064e\u0627 "
        "\u0625\u0650\u0644\u064e\u0649 \u0635\u064e\u062f\u0650\u064a\u0642"
        "\u0650\u0647\u0650."
    ),
    (
        "\u0628\u0650\u0627\u0644\u0635\u064e\u0651\u0628\u0652\u0631\u0650 "
        "\u0648\u064e\u0627\u0644\u0652\u0639\u064e\u0645\u064e\u0644\u0650 "
        "\u062a\u064e\u062a\u064e\u062d\u064e\u0642\u064e\u0651\u0642\u064f "
        "\u0627\u0644\u0652\u0623\u064e\u062d\u0652\u0644\u064e\u0627\u0645\u064f"
        "\u061b \u0648\u064e\u0628\u0650\u0627\u0644\u0652\u0643\u064e\u0633"
        "\u064e\u0644\u0650 \u062a\u064e\u0636\u0650\u064a\u0639\u064f "
        "\u0627\u0644\u0652\u0641\u064f\u0631\u064e\u0635\u064f."
    ),
    (
        "\u0644\u0650\u0645\u064e\u0627\u0630\u064e\u0627 \u062a\u064e\u0623"
        "\u064e\u062e\u064e\u0651\u0631\u064e \u0627\u0644\u0652\u0642\u0650"
        "\u0637\u064e\u0627\u0631\u064f \u0639\u064e\u0646\u0652 "
        "\u0645\u064e\u0648\u0652\u0639\u0650\u062f\u0650\u0647\u0650 "
        "\u0627\u0644\u0652\u0645\u064f\u0639\u0652\u062a\u064e\u0627\u062f\u0650"
        "\u061f"
    ),
    (
        "\u0643\u064e\u0627\u0644\u0634\u064e\u0651\u0645\u0652\u0633\u0650 "
        "\u0641\u0650\u064a \u0627\u0644\u0633\u064e\u0651\u0645\u064e\u0627"
        "\u0621\u0650 \u064a\u064e\u0628\u0652\u062f\u064f\u0648 "
        "\u0648\u064e\u062c\u0652\u0647\u064f\u0647\u064f \u0627\u0644\u0652"
        "\u062c\u064e\u0645\u0650\u064a\u0644\u064f."
    ),
    (
        "\u0643\u0650\u062a\u064e\u0627\u0628\u064f\u0647\u064f "
        "\u062c\u064e\u062f\u0650\u064a\u062f\u064c \u0648\u064e\u0642\u064e"
        "\u0644\u064e\u0645\u064f\u0647\u064e\u0627 \u0642\u064e\u062f\u0650"
        "\u064a\u0645\u064c \u0648\u064e\u0628\u064e\u064a\u0652\u062a\u064f"
        "\u0646\u064e\u0627 \u0643\u064e\u0628\u0650\u064a\u0631\u064c."
    ),
    (
        "\u0642\u064e\u0627\u0644\u064e \u0627\u0644\u0652\u062d\u064e\u0643"
        "\u0650\u064a\u0645\u064f: \u00ab\u0627\u0644\u0652\u0639\u0650"
        "\u0644\u0652\u0645\u064f \u0646\u064f\u0648\u0631\u064c "
        "\u0648\u064e\u0627\u0644\u0652\u062c\u064e\u0647\u0652\u0644\u064f "
        "\u0638\u064e\u0644\u064e\u0627\u0645\u064c\u00bb."
    ),
    (
        "\u0627\u0634\u0652\u062a\u064e\u0631\u064e\u064a\u0652\u062a\u064f "
        "\u062e\u064e\u0645\u0652\u0633\u064e\u0629\u064e \u0639\u064e\u0634"
        "\u064e\u0631\u064e \u0643\u0650\u062a\u064e\u0627\u0628\u064b\u0627"
        " \u0645\u0650\u0646\u064e \u0627\u0644\u0652\u0645\u064e\u0643"
        "\u0652\u062a\u064e\u0628\u064e\u0629\u0650 \u0627\u0644\u0652\u0643"
        "\u064e\u0628\u0650\u064a\u0631\u064e\u0629\u0650."
    ),
    (
        "\u0648\u064e\u0644\u0650\u0644\u0652\u0645\u064e\u062f\u0650\u064a\u0646"
        "\u064e\u0629\u0650 \u0627\u0644\u0652\u0642\u064e\u062f\u0650\u064a"
        "\u0645\u064e\u0629\u0650 \u0623\u064e\u0633\u0652\u0648\u064e\u0627"
        "\u0631\u064c \u0639\u064e\u0627\u0644\u0650\u064a\u064e\u0629\u064c"
        " \u0648\u064e\u0623\u064e\u0628\u0652\u0648\u064e\u0627\u0628\u064c"
        " \u0643\u064e\u062b\u0650\u064a\u0631\u064e\u0629\u064c."
    ),
    (
        "\u0641\u0650\u064a \u0627\u0644\u0635\u064e\u0651\u0628\u064e\u0627"
        "\u062d\u0650 \u0627\u0644\u0652\u0628\u064e\u0627\u0643\u0650\u0631"
        "\u0650 \u062e\u064e\u0631\u064e\u062c\u064e \u0627\u0644\u0652"
        "\u0645\u064f\u0633\u064e\u0627\u0641\u0650\u0631\u064f\u0648\u0646\u064e"
        " \u0645\u0650\u0646\u064e \u0627\u0644\u0652\u0641\u064f\u0646"
        "\u0652\u062f\u064f\u0642\u0650 \u0648\u064e\u0633\u064e\u0627\u0631"
        "\u064f\u0648\u0627 \u0639\u064e\u0628\u0652\u0631\u064e "
        "\u0627\u0644\u0652\u0623\u064e\u0632\u0650\u0642\u064e\u0651\u0629\u0650"
        " \u0627\u0644\u0636\u064e\u0651\u064a\u0650\u0651\u0642\u064e\u0629"
        "\u0650 \u062d\u064e\u062a\u064e\u0651\u0649 \u0648\u064e\u0635"
        "\u064e\u0644\u064f\u0648\u0627 \u0625\u0650\u0644\u064e\u0649 "
        "\u0627\u0644\u0633\u064f\u0651\u0648\u0642\u0650 \u0627\u0644\u0652"
        "\u0643\u064e\u0628\u0650\u064a\u0631\u0650 \u0641\u064e\u0627\u0634"
        "\u0652\u062a\u064e\u0631\u064e\u0648\u0652\u0627 \u0645\u064e\u0627"
        " \u064a\u064e\u062d\u0652\u062a\u064e\u0627\u062c\u064f\u0648\u0646"
        "\u064e \u0625\u0650\u0644\u064e\u064a\u0652\u0647\u0650 "
        "\u0645\u0650\u0646\u064e \u0627\u0644\u0652\u0641\u064e\u0648\u064e"
        "\u0627\u0643\u0650\u0647\u0650 \u0648\u064e\u0627\u0644\u0652\u062e"
        "\u064f\u0636\u064e\u0631\u0650 \u062b\u064f\u0645\u064e\u0651 "
        "\u0639\u064e\u0627\u062f\u064f\u0648\u0627 \u0645\u064e\u0633\u0652"
        "\u0631\u064f\u0648\u0631\u0650\u064a\u0646\u064e."
    ),
    (
        "\u0648\u064e\u064a\u064e\u0633\u0652\u0623\u064e\u0644\u064f\u0648\u0646"
        "\u064e: \u0647\u064e\u0644\u0652 \u0633\u064e\u062a\u064e"
        "\u0623\u0652\u062a\u0650\u064a \u063a\u064e\u062f\u064b\u0627\u061f"
        " \u0641\u064e\u0623\u064f\u062c\u0650\u064a\u0628\u064f: "
        "\u00ab\u0625\u0650\u0646\u0652 \u0634\u064e\u0627\u0621\u064e "
        "\u0627\u0644\u0644\u064e\u0651\u0647\u064f\u00bb\u060c "
        "\u0648\u064e\u0623\u064e\u0628\u0652\u062a\u064e\u0633\u0650\u0645\u064f"
        "."
    ),
)


@dataclass(frozen=True)
class DriftStats:
    words: int
    median_start_ms: float
    p95_start_ms: float
    max_start_ms: int
    median_end_ms: float
    max_end_ms: int


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def drift_stats(native: Sequence[WordTiming], reference: Sequence[WordTiming]) -> DriftStats:
    """Absolute per-word differences between two timings of the same words."""
    if len(native) != len(reference):
        raise ValueError(f"length differs: {len(native)} vs {len(reference)}")
    if not native:
        raise ValueError("no words to compare")
    if any(a.text != b.text for a, b in zip(native, reference, strict=True)):
        raise ValueError("word text differs between the two timings")
    starts = [abs(a.start_ms - b.start_ms) for a, b in zip(native, reference, strict=True)]
    ends = [abs(a.end_ms - b.end_ms) for a, b in zip(native, reference, strict=True)]
    return DriftStats(
        words=len(native),
        median_start_ms=float(statistics.median(starts)),
        p95_start_ms=_percentile(starts, 0.95),
        max_start_ms=max(starts),
        median_end_ms=float(statistics.median(ends)),
        max_end_ms=max(ends),
    )


@dataclass(frozen=True)
class OffsetStats:
    """Signed start (or end) deltas, native minus reference."""

    mean_ms: float
    median_ms: float
    std_ms: float
    negatives: int


def signed_deltas(
    native: Sequence[WordTiming], reference: Sequence[WordTiming]
) -> tuple[list[int], list[int]]:
    """Per-word (start, end) deltas, native minus reference; positive means native is later."""
    drift_stats(native, reference)
    starts = [a.start_ms - b.start_ms for a, b in zip(native, reference, strict=True)]
    ends = [a.end_ms - b.end_ms for a, b in zip(native, reference, strict=True)]
    return starts, ends


def offset_stats(deltas: Sequence[float]) -> OffsetStats:
    if not deltas:
        raise ValueError("no words to compare")
    return OffsetStats(
        mean_ms=statistics.fmean(deltas),
        median_ms=float(statistics.median(deltas)),
        std_ms=statistics.pstdev(deltas),
        negatives=sum(1 for d in deltas if d < 0),
    )


def residual(deltas: Sequence[float], offset_ms: float) -> tuple[float, float]:
    """Median and p95 of the absolute deltas left after subtracting `offset_ms`."""
    if not deltas:
        raise ValueError("no words to compare")
    left = sorted(abs(d - offset_ms) for d in deltas)
    pos = (len(left) - 1) * 0.95
    low = math.floor(pos)
    high = math.ceil(pos)
    return float(statistics.median(left)), left[low] + (left[high] - left[low]) * (pos - low)


def worst_end_gap(end_deltas: Sequence[float]) -> tuple[int, float]:
    """Index and signed value of the end delta with the largest magnitude."""
    if not end_deltas:
        raise ValueError("no words to compare")
    index = max(range(len(end_deltas)), key=lambda i: abs(end_deltas[i]))
    return index, end_deltas[index]


def _first_difference(words: list[str], spans_text: list[str]) -> str:
    from narratoflow.audio.words import norm_for_timing

    a = "".join(norm_for_timing(w) for w in words)
    b = "".join(norm_for_timing(t) for t in spans_text)
    i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    return f"at {i}: words {a[i : i + 12]!r} vs spans {b[i : i + 12]!r} (len {len(a)} vs {len(b)})"


def _row(index: int, cells: list[str]) -> str:
    return "| " + " | ".join([str(index), *cells]) + " |"


def _legacy() -> int:
    if (
        importlib.util.find_spec("edge_tts") is None
        or importlib.util.find_spec("ctc_forced_aligner") is None
    ):
        print(f"missing extras: run {EXTRAS_HINT}", file=sys.stderr)
        return 2
    from narratoflow.align.ctc import CtcAligner
    from narratoflow.models import Language
    from narratoflow.tts import SpeechRequest
    from narratoflow.tts.edge import EdgeProvider

    provider = EdgeProvider()
    aligner = CtcAligner()
    pairs: list[tuple[int, list[WordTiming], list[WordTiming]]] = []
    native_failures: list[str] = []
    ctc_failures: list[str] = []
    print("| # | words | spans | native mapped | ctc mapped | abs start median/p95/max ms |")
    print("|---|---|---|---|---|---|")
    for n, sentence in enumerate(SENTENCES, 1):
        # The raw sentence is its spoken form here (no digits, empty lexicon), as in production.
        words = tokens(sentence)
        speech = provider.synthesize(SpeechRequest(sentence, VOICE, Language.ARABIC))
        native = timings_from_spans(words, speech.spans) if speech.spans else None
        aligned = aligner.align(decode(speech.audio), SAMPLE_RATE, sentence, Language.ARABIC)
        reference = timings_from_spans(words, aligned)
        if native is None:
            why = _first_difference(words, [s.text for s in speech.spans])
            native_failures.append(f"- sentence {n}: {why}")
        if reference is None:
            why = _first_difference(words, [s.text for s in aligned])
            ctc_failures.append(f"- sentence {n}: {why}")
        cells = [str(len(words)), str(len(speech.spans)), "yes" if native else "NO"]
        cells.append("yes" if reference else "NO")
        if native is not None and reference is not None:
            d = drift_stats(native, reference)
            pairs.append((n, native, reference))
            cells.append(f"{d.median_start_ms:.0f} / {d.p95_start_ms:.0f} / {d.max_start_ms}")
        else:
            cells.append("n/a")
        print(_row(n, cells))
    _report(pairs)
    for title, items in (("Native", native_failures), ("CTC", ctc_failures)):
        print(f"\n{title} mapping failures:" if items else f"\n{title} mapping failures: none")
        for line in items:
            print(line)
    return 0


def _report(pairs: list[tuple[int, list[WordTiming], list[WordTiming]]]) -> None:
    if not pairs:
        return
    all_native = [w for _, nat, _ in pairs for w in nat]
    all_ref = [w for _, _, ref in pairs for w in ref]
    t = drift_stats(all_native, all_ref)
    print(
        f"\nAbsolute totals over {t.words} words: start median {t.median_start_ms:.0f} ms, "
        f"p95 {t.p95_start_ms:.0f} ms, max {t.max_start_ms} ms; "
        f"end median {t.median_end_ms:.0f} ms, max {t.max_end_ms} ms."
    )
    pooled_start, pooled_end = signed_deltas(all_native, all_ref)
    pooled = offset_stats(pooled_start)
    print("\nSigned start delta (native minus CTC, positive = edge later), residual after")
    print("subtracting the sentence's own median and the single pooled median.\n")
    print(
        "| # | mean | median | std | negatives | own-median resid med/p95 | pooled resid med/p95 |"
    )
    print("|---|---|---|---|---|---|---|")
    medians: list[float] = []
    for n, nat, ref in pairs:
        starts, _ = signed_deltas(nat, ref)
        s = offset_stats(starts)
        medians.append(s.median_ms)
        own = residual(starts, s.median_ms)
        glob = residual(starts, pooled.median_ms)
        print(
            f"| {n} | {s.mean_ms:.0f} | {s.median_ms:.0f} | {s.std_ms:.0f} | {s.negatives} "
            f"| {own[0]:.0f} / {own[1]:.0f} | {glob[0]:.0f} / {glob[1]:.0f} |"
        )
    own_all = [
        d - m
        for (_, nat, ref), m in zip(pairs, medians, strict=True)
        for d in signed_deltas(nat, ref)[0]
    ]
    own_res = residual(own_all, 0.0)
    pooled_res = residual(pooled_start, pooled.median_ms)
    print(
        f"\nPooled signed start: mean {pooled.mean_ms:.0f}, median {pooled.median_ms:.0f}, "
        f"std {pooled.std_ms:.0f}, negatives {pooled.negatives} of {len(pooled_start)}."
    )
    print(f"Residual after per-sentence medians: median {own_res[0]:.0f}, p95 {own_res[1]:.0f}.")
    print(f"Residual after the pooled median: median {pooled_res[0]:.0f}, p95 {pooled_res[1]:.0f}.")
    consistent = all(m > 0 for m in medians) or all(m < 0 for m in medians)
    print(f"Sign of per-sentence median consistent: {'yes' if consistent else 'no'}.")
    print("\nLargest signed end delta per sentence (native minus CTC):\n")
    print("| # | word index | last word | signed end delta ms |")
    print("|---|---|---|---|")
    for n, nat, ref in pairs:
        index, value = worst_end_gap(signed_deltas(nat, ref)[1])
        print(f"| {n} | {index} | {'yes' if index == len(nat) - 1 else 'no'} | {value} |")
    print(f"\nPooled signed end median: {offset_stats(pooled_end).median_ms:.0f} ms.")


@dataclass(frozen=True)
class Capture:
    index: int
    sentence: str
    audio: bytes
    words: list[str]
    edge: list[TimedSpan]
    ctc: list[TimedSpan] | None


def _spans_json(spans: Sequence[TimedSpan]) -> list[list[object]]:
    return [[s.text, s.start_ms, s.end_ms] for s in spans]


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def save_capture(directory: Path, capture: Capture) -> None:
    """Write one sentence's files: NN.mp3, NN.edge.json, NN.words.json and NN.ctc.json."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{capture.index:02d}"
    stem.with_suffix(".mp3").write_bytes(capture.audio)
    _dump(stem.with_suffix(".edge.json"), _spans_json(capture.edge))
    _dump(stem.with_suffix(".words.json"), capture.words)
    if capture.ctc is not None:
        _dump(stem.with_suffix(".ctc.json"), _spans_json(capture.ctc))


def write_manifest(
    directory: Path,
    *,
    voice: str,
    edge_tts_version: str,
    captured_at: str,
    sentences: Sequence[str] = SENTENCES,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": CAPTURE_VERSION,
        "voice": voice,
        "edge_tts_version": edge_tts_version,
        "captured_at": captured_at,
        "sentences": list(sentences),
    }
    _dump(directory / "manifest.json", manifest)


def edge_tts_version() -> str:
    try:
        return metadata.version("edge-tts")
    except metadata.PackageNotFoundError:
        return "unknown"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path.name}: file is missing") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name}: malformed JSON ({exc.__class__.__name__})") from None


def _read_spans(path: Path) -> list[TimedSpan]:
    raw = _read_json(path)
    try:
        return [TimedSpan(str(t), int(a), int(b)) for t, a, b in raw]
    except (TypeError, ValueError):
        raise ValueError(f"{path.name}: expected [[text, start_ms, end_ms], ...]") from None


def _load_one(directory: Path, index: int, sentence: str) -> Capture:
    stem = directory / f"{index:02d}"
    mp3 = stem.with_suffix(".mp3")
    if not mp3.exists():
        raise ValueError(f"{mp3.name}: file is missing")
    words_path = stem.with_suffix(".words.json")
    words = _read_json(words_path)
    if words != tokens(sentence):
        raise ValueError(f"{words_path.name}: words differ from the manifest sentence")
    ctc_path = stem.with_suffix(".ctc.json")
    ctc = _read_spans(ctc_path) if ctc_path.exists() else None
    edge = _read_spans(stem.with_suffix(".edge.json"))
    return Capture(index, sentence, mp3.read_bytes(), words, edge, ctc)


def load_captures(directory: Path) -> list[Capture]:
    """Every sentence saved in `directory`; ValueError names the offending file."""
    manifest = _read_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
        raise ValueError("manifest.json: expected an object with an integer version")
    if manifest["version"] != CAPTURE_VERSION:
        raise ValueError(
            f"manifest.json: version {manifest['version']}, expected {CAPTURE_VERSION}"
        )
    sentences = manifest.get("sentences")
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise ValueError("manifest.json: expected a list of sentences")
    return [_load_one(directory, n, s) for n, s in enumerate(sentences, 1)]


@dataclass(frozen=True)
class WaveTruth:
    first_onset_ms: float
    last_offset_ms: float
    pause_onsets_ms: list[float]


def wave_truth(samples: Samples, sample_rate: int, *, pause_gap_ms: int = 150) -> WaveTruth | None:
    """Energy onsets the waveform can vouch for; None for a silent clip."""
    regions = voiced_regions(samples, sample_rate, min_gap_ms=pause_gap_ms)
    if not regions:
        return None
    return WaveTruth(
        first_onset_ms=regions[0].start_ms,
        last_offset_ms=regions[-1].end_ms,
        pause_onsets_ms=[r.start_ms for r in regions[1:]],
    )


@dataclass(frozen=True)
class WaveDeltas:
    """Signed deltas, estimate minus waveform; positive means the estimate is later."""

    first_start_ms: float
    last_end_ms: float
    pause_starts_ms: list[tuple[int, float]]


def match_pause_words(
    estimates: Mapping[str, Sequence[WordTiming]], onsets: Sequence[float], window_ms: int
) -> list[int | None]:
    """The word each pause onset belongs to, chosen once for every estimator.

    Rule: among words 1.. (word 0 is the sentence start), take the one whose start
    has the smallest summed distance to the onset over all estimators, provided
    every estimator has it within `window_ms`; None when no word qualifies. An
    estimator that runs early cannot pull the match to the next word on its own.
    """
    if not estimates:
        return [None] * len(onsets)
    count = min(len(t) for t in estimates.values())
    out: list[int | None] = []
    for onset in onsets:
        distance = {
            i: [abs(t[i].start_ms - onset) for t in estimates.values()] for i in range(1, count)
        }
        ok = {i: sum(d) for i, d in distance.items() if max(d) <= window_ms}
        out.append(min(ok, key=lambda i: ok[i]) if ok else None)
    return out


def wave_deltas(
    timings: Sequence[WordTiming],
    truth: WaveTruth,
    *,
    window_ms: int = 300,
    words: Sequence[int | None] | None = None,
) -> WaveDeltas:
    """Estimate minus waveform at the sentence start, each post-pause word and the end.

    `words` is the matched word per pause onset (see `match_pause_words`); when
    omitted it is matched against these timings alone.
    """
    if words is None:
        words = match_pause_words({"": timings}, truth.pause_onsets_ms, window_ms)
    pauses = [
        (word, timings[word].start_ms - onset)
        for word, onset in zip(words, truth.pause_onsets_ms, strict=True)
        if word is not None
    ]
    return WaveDeltas(
        first_start_ms=timings[0].start_ms - truth.first_onset_ms,
        last_end_ms=timings[-1].end_ms - truth.last_offset_ms,
        pause_starts_ms=pauses,
    )


def _label_entry(raw: object, known: int) -> tuple[int, int, int]:
    if not isinstance(raw, dict):
        raise ValueError("labels: each entry must be an object")
    sentence, word, onset = raw.get("sentence"), raw.get("word"), raw.get("onset_ms")
    if not all(type(v) is int for v in (sentence, word, onset)):  # bool is an int subclass
        raise ValueError(f"labels: sentence, word and onset_ms must be integers: {raw}")
    if not 1 <= sentence <= known:
        raise ValueError(f"labels: unknown sentence {sentence}")
    if word < 0 or onset < 0:
        raise ValueError(f"labels: negative word or onset_ms in {raw}")
    return sentence, word, onset


def load_labels(path: Path, *, known: int = len(SENTENCES)) -> dict[int, list[tuple[int, int]]]:
    """Sentence number to [(word index, onset_ms)] from `labels.json`."""
    body = _read_json(path)
    if not isinstance(body, dict) or body.get("version") != CAPTURE_VERSION:
        raise ValueError(f"labels: {path.name} must have version {CAPTURE_VERSION}")
    out: dict[int, list[tuple[int, int]]] = {}
    entries = body.get("labels", [])
    if not isinstance(entries, list):
        raise ValueError("labels: 'labels' must be a list")
    for raw in entries:
        sentence, word, onset = _label_entry(raw, known)
        out.setdefault(sentence, []).append((word, onset))
    return out


def label_deltas(timings: Sequence[WordTiming], labels: Sequence[tuple[int, int]]) -> list[int]:
    """Estimate minus label for each labelled word."""
    out: list[int] = []
    for word, onset in labels:
        if word >= len(timings):
            raise ValueError(f"labels: word index {word} is out of range ({len(timings)} words)")
        out.append(timings[word].start_ms - onset)
    return out


@dataclass(frozen=True)
class _Scored:
    index: int
    timings: dict[str, list[WordTiming]]
    deltas: dict[str, WaveDeltas]


def _ctc_spans(captures: Sequence[Capture]) -> list[list[TimedSpan] | None]:
    saved = [c.ctc for c in captures]
    if all(s is not None for s in saved) or importlib.util.find_spec("ctc_forced_aligner") is None:
        return saved
    from narratoflow.align.ctc import CtcAligner
    from narratoflow.models import Language

    aligner = CtcAligner()
    return [
        c.ctc
        if c.ctc is not None
        else aligner.align(decode(c.audio), SAMPLE_RATE, c.sentence, Language.ARABIC)
        for c in captures
    ]


def _score(capture: Capture, ctc: list[TimedSpan] | None) -> _Scored:
    spans = {"edge": capture.edge, "ctc": ctc}
    timings: dict[str, list[WordTiming]] = {}
    for name, found in spans.items():
        mapped = timings_from_spans(capture.words, found) if found is not None else None
        if mapped:
            timings[name] = mapped
    truth = wave_truth(decode(capture.audio), SAMPLE_RATE)
    deltas: dict[str, WaveDeltas] = {}
    if truth:
        words = match_pause_words(timings, truth.pause_onsets_ms, 300)
        deltas = {n: wave_deltas(t, truth, words=words) for n, t in timings.items()}
    return _Scored(capture.index, timings, deltas)


def _print_mp3(captures: Sequence[Capture], out: TextIO) -> None:
    print("## MP3 header (decoder priming)\n", file=out)
    print("| # | tag | encoder | encoder_delay | encoder_padding |", file=out)
    print("|---|---|---|---|---|", file=out)
    for c in captures:
        info = read_mp3_info(c.audio)
        cells = [info.tag, info.encoder, info.encoder_delay, info.encoder_padding]
        print(f"| {c.index} | " + " | ".join(str(x) for x in cells) + " |", file=out)
    late_ms = (ENCODER_DELAY_SAMPLES + DECODER_DELAY_SAMPLES) * 1000 / PRIMING_RATE
    print(
        "\nWith an Info tag declaring the delay, ffmpeg trims the priming: a burst authored at "
        "200 ms decodes at 200.02 ms. With no tag the decoded audio is "
        f"({ENCODER_DELAY_SAMPLES} + {DECODER_DELAY_SAMPLES}) / {PRIMING_RATE} s = "
        f"{late_ms:.2f} ms late (burst at {200 + late_ms:.2f} ms). "
        "Measured on ffmpeg 9.0.2, libmp3lame, 24 kHz mono.\n",
        file=out,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0f}"


def _pause_cell(deltas: WaveDeltas | None) -> str:
    if deltas is None or not deltas.pause_starts_ms:
        return "-"
    return ", ".join(f"w{i}:{d:+.0f}" for i, d in deltas.pause_starts_ms)


def _columns(names: Sequence[str]) -> list[tuple[str, str]]:
    cols = [(n, "first") for n in names] + [(n, "last") for n in names]
    return cols + [(n, "pause") for n in names]


def _column_title(name: str, kind: str) -> str:
    return {
        "first": f"{name} first - onset",
        "last": f"{name} last end - offset",
        "pause": f"{name} post-pause starts - onset",
    }[kind]


def _cell(scored: _Scored, name: str, kind: str) -> str:
    deltas = scored.deltas.get(name)
    if kind == "pause":
        return _pause_cell(deltas)
    if deltas is None:
        return "n/a"
    return _fmt(deltas.first_start_ms if kind == "first" else deltas.last_end_ms)


def _pooled(scored: Sequence[_Scored], name: str, kind: str) -> list[float]:
    found = [s.deltas[name] for s in scored if name in s.deltas]
    if kind == "first":
        return [d.first_start_ms for d in found]
    if kind == "last":
        return [d.last_end_ms for d in found]
    return [d for f in found for _, d in f.pause_starts_ms]


def _print_wave_table(scored: Sequence[_Scored], names: Sequence[str], out: TextIO) -> None:
    cols = _columns(names)
    print("## edge vs waveform\n", file=out)
    print(f"All deltas in ms, {SIGN_RULE}.\n", file=out)
    print("| # | " + " | ".join(_column_title(n, k) for n, k in cols) + " |", file=out)
    print("|---|" + "---|" * len(cols), file=out)
    for s in scored:
        print(f"| {s.index} | " + " | ".join(_cell(s, n, k) for n, k in cols) + " |", file=out)
    medians = []
    for n, k in cols:
        values = _pooled(scored, n, k)
        medians.append(_fmt(statistics.median(values)) if values else "n/a")
    print("| median | " + " | ".join(medians) + " |", file=out)


def _print_labels(
    scored: Sequence[_Scored],
    names: Sequence[str],
    labels: dict[int, list[tuple[int, int]]],
    out: TextIO,
) -> None:
    print("\n## Labelled words\n", file=out)
    print("Deltas in ms, estimate minus label (positive = the estimate is later).\n", file=out)
    for name in names:
        deltas: list[int] = []
        for s in scored:
            if name in s.timings:
                deltas += label_deltas(s.timings[name], labels.get(s.index, []))
        if not deltas:
            continue
        absolute = [abs(d) for d in deltas]
        print(
            f"- {name}: labelled words: {len(deltas)}, signed median "
            f"{statistics.median(deltas):+.0f}, abs median {statistics.median(absolute):.0f}, "
            f"abs p95 {_percentile(absolute, 0.95):.0f}",
            file=out,
        )


def analyse(directory: Path, out: TextIO) -> int:
    """The `--from` body: MP3 headers, edge/CTC against the waveform, optional labels."""
    captures = load_captures(directory)
    _print_mp3(captures, out)
    ctc = _ctc_spans(captures)
    scored = [_score(c, s) for c, s in zip(captures, ctc, strict=True)]
    names = ["edge", "ctc"] if any(s is not None for s in ctc) else ["edge"]
    if len(names) == 1:
        print("CTC: not available (no saved NN.ctc.json, align extra not installed)\n", file=out)
    _print_wave_table(scored, names, out)
    labels_path = directory / "labels.json"
    if labels_path.exists():
        _print_labels(scored, names, load_labels(labels_path, known=len(captures)), out)
    return 0


def _aligner() -> Any:
    if importlib.util.find_spec("ctc_forced_aligner") is None:
        return None
    from narratoflow.align.ctc import CtcAligner

    return CtcAligner()


def _save(directory: Path) -> int:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        # Stale NN.ctc.json files would otherwise mix with the new audio.
        print(f"error: {directory} exists and is not empty; use a new directory", file=sys.stderr)
        return 2
    if importlib.util.find_spec("edge_tts") is None:
        print(f"missing extra: run {EDGE_HINT}", file=sys.stderr)
        return 2
    from narratoflow.models import Language
    from narratoflow.tts import SpeechRequest
    from narratoflow.tts.edge import EdgeProvider

    provider = EdgeProvider()
    aligner = _aligner()
    for n, sentence in enumerate(SENTENCES, 1):
        speech = provider.synthesize(SpeechRequest(sentence, VOICE, Language.ARABIC))
        audio = decode(speech.audio)
        ctc = aligner.align(audio, SAMPLE_RATE, sentence, Language.ARABIC) if aligner else None
        capture = Capture(n, sentence, speech.audio, tokens(sentence), list(speech.spans), ctc)
        save_capture(directory, capture)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    write_manifest(directory, voice=VOICE, edge_tts_version=edge_tts_version(), captured_at=stamp)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="DIR", type=Path, help="capture once (networked)")
    group.add_argument("--from", dest="source", metavar="DIR", type=Path, help="analyse offline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.save is not None:
        return _save(args.save)
    if args.source is not None:
        try:
            return analyse(args.source, sys.stdout)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return _legacy()


if __name__ == "__main__":
    sys.exit(main())
