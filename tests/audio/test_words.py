from __future__ import annotations

import itertools

import pytest

from narratoflow.audio.words import (
    TimedSpan,
    estimate_timings,
    norm_for_timing,
    shift,
    split_ms,
    timings_from_spans,
    tokens,
)
from narratoflow.models import WordTiming

# "qala al-rajulu." with harakat and a trailing full stop.
ARABIC = "\u0642\u064e\u0627\u0644\u064e \u0627\u0644\u0631\u0651\u064e\u062c\u064f\u0644\u064f."


def char_spans(text: str, step: int = 50) -> list[TimedSpan]:
    return [TimedSpan(ch, i * step, i * step + step) for i, ch in enumerate(text)]


def spans_of(word_timings: list[WordTiming]) -> list[tuple[str, int, int]]:
    return [(w.text, w.start_ms, w.end_ms) for w in word_timings]


def test_tokens_are_whitespace_tokens() -> None:
    assert tokens("  a  b\tc\n") == ["a", "b", "c"]


def test_norm_for_timing_drops_marks_punct_symbols_space() -> None:
    assert norm_for_timing("Hello, World!") == "helloworld"
    assert norm_for_timing("\u0642\u064e\u0627\u0644\u064e\u060c") == "\u0642\u0627\u0644"
    assert norm_for_timing("$5 + 3") == "53"
    assert norm_for_timing("\u2014") == ""


def test_char_spans_map_to_words() -> None:
    words = tokens(ARABIC)
    out = timings_from_spans(words, char_spans(ARABIC))
    assert out is not None
    # Word 1: letters at char 0..3 (char 4 is a fatha, skipped). Word 2: chars 6..13.
    assert spans_of(out) == [(words[0], 0, 200), (words[1], 300, 700)]


def test_digit_token_maps_to_one_word() -> None:
    text = "in 1924, the"
    out = timings_from_spans(tokens(text), char_spans(text))
    assert out is not None
    assert spans_of(out) == [("in", 0, 100), ("1924,", 150, 350), ("the", 450, 600)]

    arabic_digits = "\u0641\u064a \u0661\u0669\u0662\u0664\u060c"
    out_ar = timings_from_spans(tokens(arabic_digits), char_spans(arabic_digits))
    assert out_ar is not None
    assert len(out_ar) == 2
    assert (out_ar[1].start_ms, out_ar[1].end_ms) == (150, 350)


def test_edge_style_boundaries_without_punctuation_map() -> None:
    words = ["Hello,", "world.", "Again!"]
    spans = [
        TimedSpan("Hello", 0, 400),
        TimedSpan("world", 450, 900),
        TimedSpan("again", 950, 1300),
    ]
    out = timings_from_spans(words, spans)
    assert out is not None
    assert spans_of(out) == [("Hello,", 0, 400), ("world.", 450, 900), ("Again!", 950, 1300)]


def test_spans_split_differently_from_words_still_map() -> None:
    words = ["sunflower", "seeds"]
    spans = [TimedSpan("sun", 0, 200), TimedSpan("flowers", 200, 700), TimedSpan("eeds", 700, 900)]
    out = timings_from_spans(words, spans)
    assert out is not None
    assert spans_of(out) == [("sunflower", 0, 700), ("seeds", 200, 900)]


def test_mismatch_returns_none() -> None:
    words = ["hello", "world"]
    spans = [TimedSpan("hello", 0, 400), TimedSpan("word", 450, 900)]
    assert timings_from_spans(words, spans) is None


def test_empty_norm_word_is_zero_width() -> None:
    words = ["\u2014", "one", "-", "two"]
    spans = [TimedSpan("one", 100, 300), TimedSpan("two", 400, 600)]
    out = timings_from_spans(words, spans)
    assert out is not None
    assert spans_of(out) == [
        ("\u2014", 0, 0),
        ("one", 100, 300),
        ("-", 300, 300),
        ("two", 400, 600),
    ]


def test_starts_are_made_non_decreasing() -> None:
    words = ["ab", "c"]
    spans = [TimedSpan("a", 100, 200), TimedSpan("b", 200, 500), TimedSpan("c", 50, 80)]
    out = timings_from_spans(words, spans)
    assert out is not None
    assert spans_of(out) == [("ab", 100, 500), ("c", 100, 100)]


def test_empty_inputs() -> None:
    assert timings_from_spans([], []) == []
    assert timings_from_spans([], [TimedSpan(" ", 0, 10)]) == []


def test_split_ms_contiguous_and_exact_end() -> None:
    out = split_ms([1, 1, 1], 0, 100)
    assert out == [(0, 33), (33, 67), (67, 100)]
    out = split_ms([3, 1, 0, 2.5], 1000, 1777)
    assert out[0][0] == 1000
    assert out[-1][1] == 1777
    assert all(a[1] == b[0] for a, b in itertools.pairwise(out))
    assert all(s <= e for s, e in out)
    assert out[2][0] == out[2][1]
    assert split_ms([], 0, 100) == []
    assert split_ms([0, 0], 0, 10) == [(0, 5), (5, 10)]
    assert split_ms([2], 5, 5) == [(5, 5)]


@pytest.mark.parametrize(
    ("weights", "start", "end"),
    [([1.0], 10, 5), ([-1.0, 2.0], 0, 10), ([float("nan")], 0, 10), ([float("inf")], 0, 10)],
)
def test_split_ms_rejects_bad_input(weights: list[float], start: int, end: int) -> None:
    with pytest.raises(ValueError):
        split_ms(weights, start, end)


def test_estimate_proportional_and_exact_end() -> None:
    words = ["a", "abcd", "\u2014", "abc,"]
    out = estimate_timings(words, 200, 1103)
    # weights 1, 4, 1, 3 over 903 ms.
    assert [w.text for w in out] == words
    assert out[0].start_ms == 200
    assert out[-1].end_ms == 1103
    assert all(a.end_ms == b.start_ms for a, b in itertools.pairwise(out))
    durations = [w.end_ms - w.start_ms for w in out]
    assert durations == [100, 402, 100, 301]
    assert estimate_timings([], 0, 100) == []


def test_shift_clamps() -> None:
    words = [
        WordTiming(text="a", start_ms=0, end_ms=100),
        WordTiming(text="b", start_ms=900, end_ms=1200),
    ]
    assert spans_of(shift(words, 50, limit_ms=1000)) == [("a", 50, 150), ("b", 950, 1000)]
    assert spans_of(shift(words, -50, limit_ms=1000)) == [("a", 0, 50), ("b", 850, 1000)]
    assert spans_of(shift(words, 5000, limit_ms=1000)) == [("a", 1000, 1000), ("b", 1000, 1000)]


def test_tatweel_in_author_text_maps() -> None:
    author = ["\u0642\u0640\u0640\u0627\u0644", "\u0644\u0647"]
    spans = [TimedSpan("\u0642\u0627\u0644", 0, 300), TimedSpan("\u0644\u0647", 300, 500)]
    out = timings_from_spans(author, spans)
    assert out is not None
    assert spans_of(out) == [(author[0], 0, 300), (author[1], 300, 500)]


def test_diacritized_author_vs_bare_provider_maps() -> None:
    author = tokens(ARABIC)
    spans = [
        TimedSpan("\u0642\u0627\u0644", 0, 300),
        TimedSpan("\u0627\u0644\u0631\u062c\u0644", 350, 900),
    ]
    out = timings_from_spans(author, spans)
    assert out is not None
    assert spans_of(out) == [(author[0], 0, 300), (author[1], 350, 900)]


def test_curly_apostrophe_contraction_maps() -> None:
    author = ["don\u2019t", "go"]
    spans = [TimedSpan("don't", 0, 300), TimedSpan("go", 300, 500)]
    out = timings_from_spans(author, spans)
    assert out is not None
    assert spans_of(out) == [("don\u2019t", 0, 300), ("go", 300, 500)]


def test_provider_dropping_or_adding_a_repeated_word_is_none() -> None:
    author = ["very", "very", "good"]
    dropped = [TimedSpan("very", 0, 300), TimedSpan("good", 300, 600)]
    added = [
        TimedSpan("very", 0, 300),
        TimedSpan("very", 300, 600),
        TimedSpan("very", 600, 900),
        TimedSpan("good", 900, 1200),
    ]
    assert timings_from_spans(author, dropped) is None
    assert timings_from_spans(author, added) is None


def test_decomposed_hamza_and_accent_map() -> None:
    # Author: precomposed alef-with-hamza-above and e-acute; provider: decomposed.
    author = ["\u0623\u062d\u0645\u062f", "caf\u00e9"]
    spans = [TimedSpan("\u0627\u0654\u062d\u0645\u062f", 0, 400), TimedSpan("cafe\u0301", 400, 800)]
    out = timings_from_spans(author, spans)
    assert out is not None
    assert spans_of(out) == [(author[0], 0, 400), ("caf\u00e9", 400, 800)]
    assert norm_for_timing("\u0623") == norm_for_timing("\u0627\u0654")


def test_zero_width_joiners_are_dropped_on_both_sides() -> None:
    author = ["\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"]
    spans = [TimedSpan("\u0645\u06cc\u062e\u0648\u0627\u0647\u0645\u200d", 0, 500)]
    out = timings_from_spans(author, spans)
    assert out is not None
    assert spans_of(out) == [(author[0], 0, 500)]
    assert norm_for_timing("a\u200cb\ufeff") == "ab"


def test_split_ms_overflow_is_value_error() -> None:
    with pytest.raises(ValueError):
        split_ms([1e308, 1e308], 0, 10)
