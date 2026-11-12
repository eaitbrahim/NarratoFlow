"""Spike: how far do edge-tts Arabic word boundaries drift from a CTC alignment?

Voices fixed diacritized sentences with edge-tts, times the same audio with the CTC
aligner, and prints a Markdown report. CTC is a reference, not ground truth.
Run: uv sync --extra tts-edge --extra align && uv run python scripts/spike_word_timing.py
"""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from narratoflow.models import WordTiming

VOICE = "ar-SA-HamedNeural"
EXTRAS_HINT = "uv sync --extra tts-edge --extra align"

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


def _percentile(values: Sequence[int], q: float) -> float:
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


def offset_stats(deltas: Sequence[int]) -> OffsetStats:
    if not deltas:
        raise ValueError("no words to compare")
    return OffsetStats(
        mean_ms=statistics.fmean(deltas),
        median_ms=float(statistics.median(deltas)),
        std_ms=statistics.pstdev(deltas),
        negatives=sum(1 for d in deltas if d < 0),
    )


def residual(deltas: Sequence[int], offset_ms: float) -> tuple[float, float]:
    """Median and p95 of the absolute deltas left after subtracting `offset_ms`."""
    if not deltas:
        raise ValueError("no words to compare")
    left = sorted(abs(d - offset_ms) for d in deltas)
    pos = (len(left) - 1) * 0.95
    low = math.floor(pos)
    high = math.ceil(pos)
    return float(statistics.median(left)), left[low] + (left[high] - left[low]) * (pos - low)


def worst_end_gap(end_deltas: Sequence[int]) -> tuple[int, int]:
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


def main() -> int:
    import importlib.util

    if (
        importlib.util.find_spec("edge_tts") is None
        or importlib.util.find_spec("ctc_forced_aligner") is None
    ):
        print(f"missing extras: run {EXTRAS_HINT}", file=sys.stderr)
        return 2
    from narratoflow.align.ctc import CtcAligner
    from narratoflow.audio.io import SAMPLE_RATE, decode
    from narratoflow.audio.words import timings_from_spans, tokens
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


if __name__ == "__main__":
    sys.exit(main())
