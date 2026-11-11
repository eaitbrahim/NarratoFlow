from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.quran import detect as detect_module
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.detect import VerseHit, VerseIndex, detect_verses
from narratoflow.quran.normalize import match_normalize

OPEN = "\ufd3f"
CLOSE = "\ufd3e"

AYAH_31_33 = (
    "يَا أَيُّهَا النَّاسُ اتَّقُوا رَبَّكُمْ وَاخْشَوْا يَوْمًا لَا يَجْزِي وَالِدٌ عَنْ وَلَدِهِ"
    " وَلَا مَوْلُودٌ هُوَ جَازٍ عَنْ وَالِدِهِ شَيْئًا إِنَّ وَعْدَ اللَّهِ حَقٌّ فَلَا تَغُرَّنَّكُمُ"
    " الْحَيَاةُ الدُّنْيَا وَلَا يَغُرَّنَّكُمْ بِاللَّهِ الْغَرُورُ"
)


@pytest.fixture(scope="module")
def corpus() -> QuranCorpus:
    q = FIXTURES / "quran"
    return QuranCorpus.from_files(q / "quran-simple-clean.txt", q / "quran-uthmani.txt")


def _span(text: str, hit: VerseHit) -> str:
    return match_normalize(text[hit.start : hit.end])


def test_bracketed_full_ayah_certain(corpus: QuranCorpus) -> None:
    text = f"قال تعالى {OPEN}{AYAH_31_33}{CLOSE} ..."
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (
        31,
        33,
        33,
        True,
    )
    assert _span(text, hit) == match_normalize(corpus.ayah(31, 33).simple)
    assert text[hit.start] != OPEN
    assert text[hit.end - 1] != CLOSE
    assert hit.ref.uthmani_text == corpus.text_range(31, 33, 33)


def test_bracketed_partial_ayah_long_is_certain(corpus: QuranCorpus) -> None:
    quote = "يَوْمًا لَا يَجْزِي وَالِدٌ عَنْ وَلَدِهِ وَلَا مَوْلُودٌ هُوَ جَازٍ عَنْ وَالِدِهِ شَيْئًا"
    text = f"قال تعالى {OPEN}{quote}{CLOSE} ..."
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (
        31,
        33,
        33,
        True,
    )
    assert text[hit.start : hit.end] == quote
    # partial quote: the verified text is still the whole ayah
    assert hit.ref.uthmani_text == corpus.ayah(31, 33).uthmani


def test_unbracketed_quote_with_variant_spelling_detected(corpus: QuranCorpus) -> None:
    quote = "وآتيناه الحكم صبيا وحنانا من لدنا وزكاة وكان تقيا"
    text = f'قال الله تعالى "{quote}" فتأمل'
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end) == (19, 12, 13)
    assert hit.ref.certain
    assert text[hit.start : hit.end] == quote
    assert hit.ref.uthmani_text == corpus.text_range(19, 12, 13)


@pytest.mark.parametrize(
    "quote",
    [
        "وأتيناه الحكم صبيا وحنانا من لدنا وزكاة وكان تقيا",
        "واتيناه الْحُكْمَ صَبِيًّا وَحَنَانًا مِن لَدُنَّا وَزَكَاةً وكان تقيا",
        "وإتيناه الحكم صبيا وحنانا من لدنا وزكاه وكان تقيا",
    ],
)
def test_variant_alef_and_partial_harakat(corpus: QuranCorpus, quote: str) -> None:
    text = f"قال: \u00ab{quote}\u00bb."
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (
        19,
        12,
        13,
        True,
    )
    assert text[hit.start : hit.end] == quote


def test_uthmani_waw_spelling_breaks_run(corpus: QuranCorpus) -> None:
    # Uthmani waw spelling of zakah does not fold to the imla'i form (accepted v1 limit):
    # the run breaks there, and the 3+3-word piece before it is trimmed below min_words.
    text = "قال: وأتيناه الحكم صبيا وحنانا من لدنا وزكوة وكان تقيا."
    assert detect_verses(text, corpus) == []


def test_undelimited_short_start_piece_is_trimmed(corpus: QuranCorpus) -> None:
    quote = "وأتيناه الحكم صبيا وحنانا من لدنا وزكاة وكان تقيا"
    [hit] = detect_verses(quote, corpus)
    # only 3 words of 19:12 and no opening quote: they stay the author's narration
    assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (13, 13, True)
    assert quote[hit.start : hit.end] == "وحنانا من لدنا وزكاة وكان تقيا"


def test_narrator_word_after_ayah_is_not_swallowed(corpus: QuranCorpus) -> None:
    for tail in ("قال", "قال رب", "قال رب إني"):
        text = f"إذ نادى ربه نداء خفيا {tail} كذا"
        [hit] = detect_verses(text, corpus)
        assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (3, 3, True)
        assert text[hit.start : hit.end] == "إذ نادى ربه نداء خفيا"
        assert hit.ref.uthmani_text == corpus.ayah(19, 3).uthmani


def test_four_words_into_next_ayah_are_kept(corpus: QuranCorpus) -> None:
    text = "إذ نادى ربه نداء خفيا قال رب إني وهن"
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.end) == (3, 4, len(text))


def test_narrator_words_before_ayah_are_not_swallowed(corpus: QuranCorpus) -> None:
    text = "ربك عبده زكريا إذ نادى ربه نداء خفيا"
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (3, 3, True)
    assert text[hit.start : hit.end] == "إذ نادى ربه نداء خفيا"


def test_trim_below_min_words_drops_run(corpus: QuranCorpus) -> None:
    assert detect_verses("ربك عبده زكريا إذ نادى", corpus) == []


def test_ayah_number_digits_split_tokens(corpus: QuranCorpus) -> None:
    text = "إذ نادى ربه نداء خفيا\u06dd\u0663"
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.ayah_start, hit.ref.certain) == (3, True)
    assert text[hit.start : hit.end] == "إذ نادى ربه نداء خفيا"
    # numbers between ayat do not break a multi-ayah quote
    text = "إذ نادى ربه نداء خفيا (\u0663) قال رب إني وهن العظم"
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.ayah_start, hit.ref.ayah_end) == (3, 4)


def test_short_overlap_is_uncertain(corpus: QuranCorpus) -> None:
    for quote in ("وسلام عليه يوم ولد", "وسلام عليه يوم ولد ويوم"):
        [hit] = detect_verses(f"كما قيل {quote} في الكتاب", corpus)
        assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.certain) == (19, 15, False)
        assert hit.ref.uthmani_text == corpus.ayah(19, 15).uthmani


def test_short_full_ayah_is_certain(corpus: QuranCorpus) -> None:
    [hit] = detect_verses("إذ نادى ربه نداء خفيا", corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (
        19,
        3,
        3,
        True,
    )


def test_certain_words_threshold(corpus: QuranCorpus) -> None:
    quote = "يوما لا يجزي والد عن ولده"
    [hit] = detect_verses(quote, corpus)
    assert hit.ref.certain
    [hit] = detect_verses(quote, corpus, certain_words=7)
    assert not hit.ref.certain


def test_min_words(corpus: QuranCorpus) -> None:
    assert detect_verses("وسلام عليه يوم ولد", corpus, min_words=5) == []
    with pytest.raises(ValueError, match="min_words"):
        detect_verses("وسلام عليه يوم", corpus, min_words=3)


def test_three_words_ignored(corpus: QuranCorpus) -> None:
    assert detect_verses("وسلام عليه يوم", corpus) == []


def test_english_text_no_hits(corpus: QuranCorpus) -> None:
    assert detect_verses("The day when no father can avail his son.", corpus) == []
    assert detect_verses("", corpus) == []


def test_arabic_non_verse_text_no_hits(corpus: QuranCorpus) -> None:
    assert detect_verses("ذهب الولد إلى المدرسة في الصباح الباكر", corpus) == []


def test_several_hits_sorted_by_start(corpus: QuranCorpus) -> None:
    text = "إذ نادى ربه نداء خفيا ثم قال وسلام عليه يوم ولد ويوم يموت ويوم يبعث حيا"
    hits = detect_verses(text, corpus)
    assert [(h.ref.ayah_start, h.ref.certain) for h in hits] == [(3, True), (15, True)]
    assert hits[0].start < hits[1].start


def test_punctuation_glued_inside_token(corpus: QuranCorpus) -> None:
    text = "قال:وسلام عليه يوم ولد"
    [hit] = detect_verses(text, corpus)
    assert text[hit.start : hit.end] == "وسلام عليه يوم ولد"


def test_harakat_only_token_dropped(corpus: QuranCorpus) -> None:
    text = "وسلام \u064e عليه يوم ولد"
    [hit] = detect_verses(text, corpus)
    assert (hit.start, hit.end) == (0, len(text))


# --- synthetic corpus for overlap, tie and boundary rules ---


def _corpus(tmp_path: Path, rows: list[str]) -> QuranCorpus:
    path = tmp_path / "c.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return QuranCorpus.from_files(path, path)


def test_longest_run_wins_and_tie_goes_to_earliest(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["1|5|بب تت ثث جج خخ", "2|1|بب تت ثث جج حح"])
    [hit] = detect_verses("بب تت ثث جج حح", c)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.certain) == (2, 1, True)
    [hit] = detect_verses("بب تت ثث جج", c)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.certain) == (1, 5, False)


def test_overlapping_runs_keep_longest(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["3|1|سس شش صص ضض طط", "3|9|ضض طط ظظ عع غغ فف"])
    [hit] = detect_verses("سس شش صص ضض طط ظظ عع غغ فف", c)
    assert (hit.ref.surah, hit.ref.ayah_start) == (3, 9)


def test_no_extension_across_surahs(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["1|7|قق كك لل مم", "2|1|نن شش وو يي"])
    hits = detect_verses("قق كك لل مم نن شش وو يي", c)
    assert [(h.ref.surah, h.ref.ayah_start) for h in hits] == [(1, 7), (2, 1)]
    assert detect_verses("لل مم نن شش", c) == []


def test_extension_across_ayat_is_full_range(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["4|1|بب تت ثث جج", "4|2|خخ دد سس شش"])
    [hit] = detect_verses("بب تت ثث جج خخ دد سس شش", c)
    assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (1, 2, True)


def test_whole_short_edge_ayat_are_kept(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["4|1|بب تت", "4|2|ثث جج"])
    [hit] = detect_verses("بب تت ثث جج", c)
    assert (hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (1, 2, True)


def test_single_word_edge_ayah_is_trimmed(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["4|1|بب", "4|2|تت ثث جج خخ", "4|3|دد"])
    text = "بب تت ثث جج خخ دد"
    [hit] = detect_verses(text, c)
    assert (hit.ref.ayah_start, hit.ref.ayah_end) == (2, 2)
    assert text[hit.start : hit.end] == "تت ثث جج خخ"
    # an author-placed bracket still keeps them
    [hit] = detect_verses(f"\ufd3f{text}\ufd3e", c)
    assert (hit.ref.ayah_start, hit.ref.ayah_end) == (1, 3)


def test_fatiha_with_ayah_numbers_keeps_whole_short_ayat() -> None:
    q = FIXTURES / "quran"
    c = QuranCorpus.from_files(q / "fatiha-simple-clean.txt", q / "fatiha-uthmani.txt")
    text = (
        "الحمد لله رب العالمين \u06dd\u0662 الرحمن الرحيم \u06dd\u0663 مالك يوم الدين \u06dd\u0664"
    )
    [hit] = detect_verses(text, c)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (1, 2, 4, True)
    assert text[hit.start : hit.end].endswith("الدين")
    assert hit.ref.uthmani_text == c.text_range(1, 2, 4)


def test_non_contiguous_ayat_do_not_join(tmp_path: Path) -> None:
    c = _corpus(tmp_path, ["4|1|بب تت ثث جج", "4|5|خخ دد سس شش"])
    hits = detect_verses("بب تت ثث جج خخ دد سس شش", c)
    assert [(h.ref.ayah_start, h.ref.ayah_end) for h in hits] == [(1, 1), (5, 5)]
    assert detect_verses("ثث جج خخ دد", c) == []


def test_index_built_once_per_corpus(corpus: QuranCorpus) -> None:
    assert VerseIndex.for_corpus(corpus) is VerseIndex.for_corpus(corpus)


def test_detect_source_is_ascii() -> None:
    assert Path(detect_module.__file__).read_bytes().isascii()


def test_identical_run_at_two_locations_is_uncertain(tmp_path: Path) -> None:
    six = "بب تت ثث جج خخ دد"
    c = _corpus(tmp_path, [f"5|1|{six}", "5|2|سس شش", f"7|3|{six}"])
    [hit] = detect_verses(six, c)
    # full ayah and 6 words, but ambiguous: earliest location kept, marked uncertain
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (
        5,
        1,
        1,
        False,
    )
    assert hit.ref.uthmani_text == c.ayah(5, 1).uthmani


def test_ambiguity_from_duplicated_fixture_ayah(tmp_path: Path) -> None:
    q = FIXTURES / "quran"
    simple = (q / "quran-simple-clean.txt").read_text(encoding="utf-8")
    uthmani = (q / "quran-uthmani.txt").read_text(encoding="utf-8")
    dup_s = next(ln for ln in simple.splitlines() if ln.startswith("19|3|")).replace(
        "19|3|", "20|1|"
    )
    dup_u = next(ln for ln in uthmani.splitlines() if ln.startswith("19|3|")).replace(
        "19|3|", "20|1|"
    )
    (tmp_path / "s.txt").write_text(simple + "\n" + dup_s + "\n", encoding="utf-8")
    (tmp_path / "u.txt").write_text(uthmani + "\n" + dup_u + "\n", encoding="utf-8")
    c = QuranCorpus.from_files(tmp_path / "s.txt", tmp_path / "u.txt")
    [hit] = detect_verses("إذ نادى ربه نداء خفيا", c)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.certain) == (19, 3, False)


def test_duplicate_with_different_shape_is_uncertain(tmp_path: Path) -> None:
    six = "بب تت ثث جج خخ دد"
    # suffix of 8:1 and exactly 9:2: ambiguous, and the exact (full-ayah) location wins
    c = _corpus(tmp_path, [f"8|1|سس {six}", f"9|2|{six}"])
    [hit] = detect_verses(six, c)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.certain) == (9, 2, False)
