"""The exact-verse publishing check: corpus text, render report and the shipped ASS."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from fixtures.parts import (
    CORPUS,
    GENERATED_PROVENANCE,
    add_sketch,
    make_series,
    narr,
    verse,
    write_timelines,
)
from fixtures.publish import VOICE, make_inputs

from narratoflow.audio.recitation import display_words
from narratoflow.audio.timeline import (
    AyahTiming,
    PartTimeline,
    SceneTiming,
    TimelineWord,
    TimingSource,
)
from narratoflow.captions.ass import parse_events
from narratoflow.captions.verse import shown_text
from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import VerseCorpusError
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import SceneKind, Series
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.verse import expected_verse_text, verse_checks
from narratoflow.quran.corpus import QuranCorpus

UTHMANI = CORPUS.ayah(31, 33).uthmani
OK = "verse text matches the corpus exactly"


def _corpus() -> QuranCorpus:
    return CORPUS


def _no_corpus() -> QuranCorpus:
    raise AssertionError("the corpus must not be loaded")


def _fails(results: list[CheckResult]) -> list[CheckResult]:
    assert all(r.check is CheckId.VERSE_TEXT for r in results)
    return [r for r in results if r.severity is Severity.FAIL]


def _series(library: SketchLibrary) -> Series:
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    return make_series([narr("s1", "alpha beta gamma", sketch=sid), verse("v1")])


def _segments(start: int, end: int) -> SceneTiming:
    """A SEGMENTS verse slot for 31:33: each display word gets an equal share."""
    words = display_words(UTHMANI)
    step = (end - start) // len(words)
    timed = [
        TimelineWord(text=w, start_ms=start + i * step, end_ms=start + (i + 1) * step)
        for i, w in enumerate(words)
    ]
    return SceneTiming(
        scene_id="v1",
        kind=SceneKind.VERSE,
        start_ms=start,
        end_ms=end,
        speech_start_ms=start,
        speech_end_ms=end,
        words=[],
        timing_source=TimingSource.SEGMENTS,
        ayat=[AyahTiming(surah=31, ayah=33, start_ms=start, end_ms=end, words=timed)],
    )


def _verse_inputs(
    tmp_path: Path,
    library: SketchLibrary,
    channel: ChannelConfig,
    *,
    verse_start: int = 1000,
) -> PartInputs:
    """A narration scene then a SEGMENTS verse scene starting at `verse_start`."""
    series = _series(library)
    base = write_timelines(series, tmp_path / "audio", library, flac=False, voice=VOICE)[0]
    narration = base.scenes[0].model_copy(update={"end_ms": verse_start})
    end = verse_start + 4000
    timeline: PartTimeline = base.model_copy(
        update={"scenes": [narration, _segments(verse_start, end)], "duration_ms": end}
    )
    return make_inputs(tmp_path, library, channel, series=series, timeline=timeline)


def _ass(inputs: PartInputs) -> Path:
    return inputs.render_dir / inputs.render.captions_file


def test_expected_text_is_the_global_formula() -> None:
    a, b = CORPUS.ayah(19, 2).uthmani, CORPUS.ayah(19, 3).uthmani
    want = " ".join(" ".join(t.split()) for t in (a, b))
    assert expected_verse_text(CORPUS, [(19, 2), (19, 3)]) == want
    assert expected_verse_text(CORPUS, []) == ""


def test_verse_ok_31_33(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    events = parse_events(_ass(inputs).read_text(encoding="utf-8"))
    assert shown_text(events) == expected_verse_text(CORPUS, [(31, 33)])
    results = verse_checks(inputs, _corpus)
    assert [(r.check, r.severity, r.reason) for r in results] == [
        (CheckId.VERSE_TEXT, Severity.PASS, OK)
    ]


def test_ayah_timed_verse_passes(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel, series=_series(library))
    assert [r.reason for r in verse_checks(inputs, _corpus)] == [OK]


def test_no_verse_part_never_loads_corpus(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    results = verse_checks(inputs, _no_corpus)
    assert [(r.severity, r.reason) for r in results] == [
        (Severity.PASS, "no Qur'anic verse in this part")
    ]


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"[Events]\nDialogue: 0,not a time\n",
        b"\xff\xfe not utf-8",
    ],
)
def test_missing_or_unparseable_ass_fails_without_raising(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig, content: bytes | None
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    if content is None:
        _ass(inputs).unlink()
    else:
        _ass(inputs).write_bytes(content)
    fails = _fails(verse_checks(inputs, _corpus))
    assert len(fails) == 1
    assert "part-01.ass" in fails[0].reason
    assert "re-render" in fails[0].reason
    assert not fails[0].reason.startswith("FAIL")


def test_edited_ass_verse_word_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    word = display_words(UTHMANI)[1]
    text = _ass(inputs).read_text(encoding="utf-8")
    assert word in text
    _ass(inputs).write_text(
        text.replace(word, "\u0627\u0644\u0646\u0627\u0633", 1), encoding="utf-8"
    )
    fails = _fails(verse_checks(inputs, _corpus))
    assert [(f.scene_id, f.reason.split(":")[0]) for f in fails] == [("v1", "verse 31")]
    assert "subtitles" in fails[0].reason
    assert "re-render" in fails[0].reason


def test_verse_event_outside_slot_is_not_counted(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    lines = _ass(inputs).read_text(encoding="utf-8").splitlines()
    # Move every Verse event to 0:00:00.00 (the narration slot).
    moved = [
        ",".join([ln.split(",", 2)[0], "0:00:00.00", *ln.split(",", 2)[2:]])
        if ln.startswith("Dialogue:") and ",Verse," in ln
        else ln
        for ln in lines
    ]
    _ass(inputs).write_text("\n".join(moved) + "\n", encoding="utf-8")
    fails = _fails(verse_checks(inputs, _corpus))
    assert [f.scene_id for f in fails] == ["v1"]
    assert "subtitles" in fails[0].reason


def test_report_hash_mismatch_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    bad = inputs.render.verses[0].model_copy(update={"shown_sha256": "0" * 64})
    inputs = replace(inputs, render=inputs.render.model_copy(update={"verses": [bad]}))
    fails = _fails(verse_checks(inputs, _corpus))
    assert len(fails) == 1
    assert fails[0].reason.startswith("verse 31:33")
    assert "sha256" in fails[0].reason


def test_ayat_range_mismatch_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    bad = inputs.render.verses[0].model_copy(update={"ayat": ["31:33", "31:34"]})
    inputs = replace(inputs, render=inputs.render.model_copy(update={"verses": [bad]}))
    fails = _fails(verse_checks(inputs, _corpus))
    assert len(fails) == 1
    assert fails[0].reason.startswith("verse 31:33")
    assert "31:33, 31:34" in fails[0].reason


def test_missing_verse_render_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    inputs = replace(inputs, render=inputs.render.model_copy(update={"verses": []}))
    fails = _fails(verse_checks(inputs, _corpus))
    assert len(fails) == 1
    assert fails[0].reason.startswith("verse 31:33")
    assert fails[0].scene_id == "v1"


def test_missing_timeline_slot_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    timeline = inputs.timeline.model_copy(update={"scenes": inputs.timeline.scenes[:1]})
    fails = _fails(verse_checks(replace(inputs, timeline=timeline), _corpus))
    assert len(fails) == 1
    assert fails[0].reason.startswith("verse 31:33")
    assert "timeline" in fails[0].reason


def test_slot_bounds_use_ass_rounding(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel, verse_start=1004)
    assert "0:00:01.00," in _ass(inputs).read_text(encoding="utf-8")
    assert [r.reason for r in verse_checks(inputs, _corpus)] == [OK]


def test_corpus_errors_fail_without_raising(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)

    def broken() -> QuranCorpus:
        raise VerseCorpusError("corpus files missing")

    fails = _fails(verse_checks(inputs, broken))
    assert len(fails) == 1
    assert "corpus files missing" in fails[0].reason

    empty = QuranCorpus({})
    fails = _fails(verse_checks(inputs, lambda: empty))
    assert len(fails) == 1
    assert "31:33" in fails[0].reason


def test_report_hash_is_sha256_of_expected_text(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    text = expected_verse_text(CORPUS, [(31, 33)])
    assert inputs.render.verses[0].shown_sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_verse_scene_without_reference_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    scenes = [*inputs.part.scenes[:1], inputs.part.scenes[1].model_copy(update={"verse": None})]
    part = inputs.part.model_copy(update={"scenes": scenes})
    fails = _fails(verse_checks(replace(inputs, part=part), _corpus))
    assert [(f.scene_id, f.reason) for f in fails] == [
        ("v1", "verse scene v1 has no verse reference")
    ]
