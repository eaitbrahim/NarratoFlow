import re
from pathlib import Path

import pytest
from conftest import FIXTURES
from llm.fakes import FakeLLM

from narratoflow.errors import SegmentationError
from narratoflow.models import SceneKind, VerseRef
from narratoflow.planning import prompts as prompts_module
from narratoflow.planning import segment as segment_module
from narratoflow.planning.prompts import SEGMENT_SYSTEM
from narratoflow.planning.segment import SegmentOut, SegmentSpan, collapse_ws, segment_script
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.detect import VerseHit, detect_verses

OPEN = "\ufd3f"
CLOSE = "\ufd3e"
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def _ref(surah: int = 31, start: int = 33, end: int = 33) -> VerseRef:
    return VerseRef(surah=surah, ayah_start=start, ayah_end=end, certain=True, uthmani_text="x")


def _out(*texts: str, title: str = "t") -> SegmentOut:
    return SegmentOut(scenes=[SegmentSpan(text=t, title=title) for t in texts])


def _hit(text: str, quote: str, ref: VerseRef | None = None) -> VerseHit:
    start = text.index(quote)
    return VerseHit(start=start, end=start + len(quote), ref=ref or _ref())


def test_verbatim_segmentation_and_verse_cut() -> None:
    quote = "يوم لا يجزي والد عن ولده"
    text = f"العلم نور. والجهل ظلام. قال تعالى {OPEN}{quote}{CLOSE} وهذا حق."
    hit = _hit(text, quote)
    llm = FakeLLM(
        [
            _out("العلم نور.", "والجهل ظلام.", f"قال تعالى {OPEN}", title="first"),
            _out(f"{CLOSE} وهذا حق.", title="second"),
        ]
    )
    scenes = segment_script(text, [hit], llm)

    assert [s.id for s in scenes] == ["s001", "s002", "s003", "s004", "s005"]
    assert [s.kind for s in scenes] == [SceneKind.NARRATION] * 3 + [
        SceneKind.VERSE,
        SceneKind.NARRATION,
    ]
    assert "".join(s.narration.source_text for s in scenes) == text
    verse = scenes[3]
    assert verse.narration.source_text == quote
    assert verse.narration.protected is True
    assert verse.verse == hit.ref
    assert verse.title == "31:33"
    assert scenes[2].narration.source_text == f"قال تعالى {OPEN}"
    assert scenes[4].narration.source_text == f"{CLOSE} وهذا حق."
    assert scenes[0].narration.source_text == "العلم نور. "
    assert [s.title for s in scenes] == ["first", "first", "first", "31:33", "second"]
    assert not any(s.narration.protected for s in scenes if s.kind is SceneKind.NARRATION)
    assert all(s.verse is None for s in scenes if s.kind is SceneKind.NARRATION)
    # one call per non-verse stretch, with the segmentation system prompt
    assert len(llm.calls) == 2
    assert all(call[0] == SEGMENT_SYSTEM and call[2] is SegmentOut for call in llm.calls)
    assert "وهذا حق." in llm.calls[1][1]
    assert quote not in llm.calls[0][1] and quote not in llm.calls[1][1]


def test_unmatched_verse_brackets_flagged() -> None:
    text = f"قال تعالى {OPEN}نص غير معروف{CLOSE} انتهى."
    scenes = segment_script(text, [], FakeLLM([_out(text)]))
    assert scenes[0].flags == ["unmatched_verse_brackets"]
    plain = segment_script("العلم نور.", [], FakeLLM([_out("العلم نور.")]))
    assert plain[0].flags == []


def test_brackets_flush_with_a_matched_verse_are_not_flagged() -> None:
    quote = "يوم لا يجزي والد"
    text = f"قال {OPEN}{quote}{CLOSE} انتهى."
    llm = FakeLLM([_out(f"قال {OPEN}"), _out(f"{CLOSE} انتهى.")])
    scenes = segment_script(text, [_hit(text, quote)], llm)
    assert [s.flags for s in scenes] == [[], [], []]


def test_reversed_brackets_between_verses_are_flagged() -> None:
    verse_a = "يوم لا يجزي والد"
    verse_b = "ولا مولود هو جاز"
    text = f"{verse_a} {OPEN}نص مجهول{CLOSE} {verse_b}"
    hits = [_hit(text, verse_a), _hit(text, verse_b)]
    middle = text[len(verse_a) : text.index(verse_b)]
    scenes = segment_script(text, hits, FakeLLM([_out(middle.strip())]))
    assert [s.kind for s in scenes] == [SceneKind.VERSE, SceneKind.NARRATION, SceneKind.VERSE]
    assert scenes[1].flags == ["unmatched_verse_brackets"]


def test_bracket_separated_by_space_is_flagged() -> None:
    quote = "يوم لا يجزي والد"
    text = f"قال {OPEN} {quote}{CLOSE} انتهى."
    llm = FakeLLM([_out(f"قال {OPEN}"), _out(f"{CLOSE} انتهى.")])
    scenes = segment_script(text, [_hit(text, quote)], llm)
    assert [s.flags for s in scenes] == [["unmatched_verse_brackets"], [], []]


def test_verse_range_title_and_punctuation_only_stretches_attach() -> None:
    quote = "يوم لا يجزي والد"
    text = f"{OPEN}{quote}{CLOSE} ... العلم نور.\n\n{OPEN}{quote}{CLOSE}."
    first = text.index(quote)
    second = text.index(quote, first + 1)
    hits = [
        VerseHit(first, first + len(quote), _ref(31, 33, 34)),
        VerseHit(second, second + len(quote), _ref(2, 5, 5)),
    ]
    middle = text[first + len(quote) : second]
    llm = FakeLLM([_out(middle.strip())])
    scenes = segment_script(text, hits, llm)

    assert len(llm.calls) == 1  # the bracket-only stretches never reach the LLM
    assert [s.kind for s in scenes] == [SceneKind.VERSE, SceneKind.NARRATION, SceneKind.VERSE]
    assert [s.title for s in scenes] == ["31:33-34", "t", "2:5"]
    assert [s.id for s in scenes] == ["s001", "s002", "s003"]
    assert "".join(s.narration.source_text for s in scenes) == text
    assert scenes[1].narration.source_text == middle
    assert scenes[1].flags == []
    # leading filler joins the next scene, trailing filler the previous one
    assert scenes[0].narration.source_text == f"{OPEN}{quote}"
    assert scenes[2].narration.source_text == f"{quote}{CLOSE}."


def test_whitespace_only_stretch_between_verses() -> None:
    quote = "يوم لا يجزي والد"
    text = f"{quote} \n {quote}"
    second = text.index(quote, 1)
    hits = [VerseHit(0, len(quote), _ref()), VerseHit(second, second + len(quote), _ref())]
    scenes = segment_script(text, hits, FakeLLM([]))
    assert [s.kind for s in scenes] == [SceneKind.VERSE, SceneKind.VERSE]
    assert "".join(s.narration.source_text for s in scenes) == text


def test_inter_span_whitespace_goes_to_preceding_scene() -> None:
    text = "  العلم نور.\n\n  والجهل   ظلام.  "
    llm = FakeLLM([_out("العلم نور.", "والجهل ظلام.")])
    scenes = segment_script(text, [], llm)
    assert [s.narration.source_text for s in scenes] == [
        "  العلم نور.\n\n  ",
        "والجهل   ظلام.  ",
    ]


def test_rewrite_rejected_after_one_retry() -> None:
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM(
        [
            SegmentOut(
                scenes=[
                    SegmentSpan(text="العلم نور.", title="t"),
                    SegmentSpan(text=" والجهل ظلم.", title="t"),
                ]
            )
        ]
        * 2
    )
    with pytest.raises(SegmentationError, match="offset"):
        segment_script(text, [], llm)
    assert len(llm.calls) == 2


def test_rewrite_fixed_on_retry_names_offset() -> None:
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM([_out("العلم نور.", "والجهل ظلم."), _out("العلم نور.", "والجهل ظلام.")])
    scenes = segment_script(text, [], llm)
    assert len(scenes) == 2
    bad_at = text.index("ظلام") + 2  # first character that differs
    retry_user = llm.calls[1][1]
    assert llm.calls[0][1] in retry_user
    assert f"offset {bad_at}" in retry_user


def test_mid_word_cut_rejected() -> None:
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM([_out("العل", "م نور. والجهل ظلام.")] * 2)
    with pytest.raises(SegmentationError, match=f"offset {text.index('م ')}"):
        segment_script(text, [], llm)
    assert len(llm.calls) == 2


def test_full_stop_moved_into_next_scene_rejected() -> None:
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM([_out("العلم نور", ". والجهل ظلام.")] * 2)
    with pytest.raises(SegmentationError, match=f"offset {text.index('.')}"):
        segment_script(text, [], llm)


def test_cut_after_full_stop_and_space_accepted() -> None:
    text = "العلم نور. والجهل ظلام."
    scenes = segment_script(text, [], FakeLLM([_out("العلم نور.", "والجهل ظلام.")]))
    assert [s.narration.source_text for s in scenes] == ["العلم نور. ", "والجهل ظلام."]


@pytest.mark.parametrize("mark", ["\u060c", "\u061b", ":", "\u2026", "\u00bb", '"'])
def test_cut_right_after_closing_punctuation_accepted(mark: str) -> None:
    text = f"العلم نور{mark}والجهل ظلام."
    llm = FakeLLM([_out(f"العلم نور{mark}", "والجهل ظلام.")])
    scenes = segment_script(text, [], llm)
    assert scenes[1].narration.source_text == "والجهل ظلام."


def test_cut_between_two_punctuation_marks_rejected() -> None:
    text = "العلم نور.. والجهل ظلام."
    llm = FakeLLM([_out("العلم نور.", ". والجهل ظلام.")] * 2)
    with pytest.raises(SegmentationError, match=f"offset {text.index('.') + 1}"):
        segment_script(text, [], llm)


def test_spans_that_stop_short_are_rejected() -> None:
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM([_out("العلم نور.")] * 2)
    with pytest.raises(SegmentationError, match=f"offset {text.index('والجهل')}"):
        segment_script(text, [], llm)


def test_spans_that_run_past_the_stretch_are_rejected() -> None:
    text = "العلم نور."
    llm = FakeLLM([_out("العلم نور. زيادة")] * 2)
    with pytest.raises(SegmentationError, match=f"offset {len(text)}"):
        segment_script(text, [], llm)


def test_empty_spans_are_ignored() -> None:
    scenes = segment_script("العلم نور.", [], FakeLLM([_out("", "العلم نور.", "  ")]))
    assert [s.narration.source_text for s in scenes] == ["العلم نور."]


def test_overlapping_hits_are_rejected() -> None:
    text = "يوم لا يجزي والد عن ولده"
    hits = [VerseHit(0, 10, _ref()), VerseHit(5, 12, _ref())]
    with pytest.raises(SegmentationError, match="overlap"):
        segment_script(text, hits, FakeLLM([]))
    with pytest.raises(SegmentationError, match="range"):
        segment_script(text, [VerseHit(0, len(text) + 1, _ref())], FakeLLM([]))


def test_text_without_letters_is_rejected() -> None:
    with pytest.raises(SegmentationError):
        segment_script(" ... ", [], FakeLLM([]))


def _sentence_split(stretch: str) -> SegmentOut:
    # spans are stripped, so the guard must match them whitespace-insensitively
    parts = [p.strip() for p in _SENTENCE_END.split(stretch) if p.strip()]
    return _out(*parts)


def test_join_invariant_on_real_fixture() -> None:
    text = (FIXTURES / "input" / "clean_ar.txt").read_text(encoding="utf-8")
    q = FIXTURES / "quran"
    corpus = QuranCorpus.from_files(q / "quran-simple-clean.txt", q / "quran-uthmani.txt")
    hits = detect_verses(text, corpus)
    assert hits
    stretches: list[str] = []
    cursor = 0
    for hit in hits:
        stretches.append(text[cursor : hit.start])
        cursor = hit.end
    stretches.append(text[cursor:])
    responses = [_sentence_split(s) for s in stretches if any(c.isalnum() for c in s)]
    llm = FakeLLM(list(responses))

    scenes = segment_script(text, hits, llm)

    joined = "".join(s.narration.source_text for s in scenes)
    assert collapse_ws(joined) == collapse_ws(text)
    assert joined == text
    assert not llm.responses
    assert [s.id for s in scenes] == [f"s{n:03d}" for n in range(1, len(scenes) + 1)]
    [verse] = [s for s in scenes if s.kind is SceneKind.VERSE]
    [hit] = hits
    # the leading bracket has no letters, so it joins the verse scene
    assert scenes[0] is verse
    # ...and so do the closing bracket and "..." the LLM cut off after it
    assert verse.narration.source_text.startswith(OPEN + text[hit.start : hit.end] + CLOSE)
    assert not any(c.isalnum() for c in verse.narration.source_text[hit.end - hit.start + 1 :])
    first_narration = scenes[1]
    assert first_narration.kind is SceneKind.NARRATION
    # the first narration starts after the quoted verse, so it swallows none of it
    assert len(verse.narration.source_text) >= hit.end
    assert len(scenes) > 4
    assert all(not s.flags for s in scenes)


def testcollapse_ws() -> None:
    assert collapse_ws("  a \n\t b  ") == "a b"


def test_prompts_mention_verbatim_rule() -> None:
    assert "exact" in SEGMENT_SYSTEM.lower()
    assert "title" in SEGMENT_SYSTEM.lower()


def test_sources_are_ascii() -> None:
    for module in (segment_module, prompts_module):
        assert Path(module.__file__ or "").read_bytes().isascii()


def test_letterless_llm_span_merges_into_previous_verse_scene() -> None:
    # The reference script opens with a verse, the closing bracket and "...", then prose.
    text = (FIXTURES / "input" / "clean_ar.txt").read_text(encoding="utf-8")
    q = FIXTURES / "quran"
    corpus = QuranCorpus.from_files(q / "quran-simple-clean.txt", q / "quran-uthmani.txt")
    [hit] = detect_verses(text, corpus)
    answer = _sentence_split(text[hit.end :])
    # the sentence split cuts the bracket and the ellipsis off as their own span
    assert not any(c.isalnum() for c in answer.scenes[0].text)

    scenes = segment_script(text, [hit], FakeLLM([answer]))

    assert "".join(s.narration.source_text for s in scenes) == text
    assert all(any(c.isalnum() for c in s.narration.source_text) for s in scenes)
    verse = scenes[0]
    assert verse.kind is SceneKind.VERSE
    assert verse.narration.source_text.startswith(text[: hit.end])
    assert answer.scenes[0].text in verse.narration.source_text
    assert all(s.kind is SceneKind.NARRATION for s in scenes[1:])
    assert [s.id for s in scenes] == [f"s{n:03d}" for n in range(1, len(scenes) + 1)]


def test_leading_letterless_span_merges_into_next_scene() -> None:
    text = "... Hello there. Goodbye now."
    llm = FakeLLM([_out("...", "Hello there.", "Goodbye now.")])
    scenes = segment_script(text, [], llm)
    assert [s.narration.source_text for s in scenes] == ["... Hello there. ", "Goodbye now."]
    assert [s.id for s in scenes] == ["s001", "s002"]


def test_several_leading_letterless_spans_merge_into_next_scene() -> None:
    text = "... -- Hello there."
    scenes = segment_script(text, [], FakeLLM([_out("...", "--", "Hello there.")]))
    assert [s.narration.source_text for s in scenes] == [text]
