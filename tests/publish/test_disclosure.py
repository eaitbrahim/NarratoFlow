"""AI-disclosure text and the disclosure-set check."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fixtures.parts import GENERATED_PROVENANCE, add_sketch, make_series, narr, reciter, verse
from fixtures.publish import VOICE, make_inputs, write_rules

from narratoflow.audio.timeline import RecitationUsed
from narratoflow.channels.config import ChannelConfig
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchOrigin
from narratoflow.models import Language, RightsBasis, ScriptSource
from narratoflow.publish.checks import CheckId, PartInputs, Severity
from narratoflow.publish.disclosure import disclosure_checks, disclosure_text, platform_labels
from narratoflow.publish.rules import load_rules
from narratoflow.quran.reciters import Reciter


def _with_sketch(inputs: PartInputs, **provenance: object) -> PartInputs:
    [(sid, record)] = inputs.sketches.items()
    assert record is not None
    new = record.model_copy(update={"provenance": record.provenance.model_copy(update=provenance)})
    return replace(inputs, sketches={sid: new})


def _recitation(r: Reciter, ayat: list[str]) -> RecitationUsed:
    return RecitationUsed(
        reciter_id=r.id,
        name=r.name,
        riwaya=r.riwaya,
        license=r.license,
        license_url=r.license_url,
        source=r.source,
        segments_source=r.segments_source,
        checked=r.checked,
        ayat=ayat,
    )


def _source(inputs: PartInputs, **changes: object) -> PartInputs:
    source = ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN).model_copy(
        update=changes
    )
    return replace(inputs, series=inputs.series.model_copy(update={"source": source}))


def _english(inputs: PartInputs) -> PartInputs:
    return replace(inputs, series=inputs.series.model_copy(update={"language": Language.ENGLISH}))


def test_english_disclosure_lines_exact(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _english(make_inputs(tmp_path, library, channel))
    assert disclosure_text(inputs) == (
        "AI disclosure:\n"
        "- The narration voice is AI-generated (elevenlabs).\n"
        "- The sketches were generated with AI image models (openai/gpt-image-1).\n"
        "- Script: Scenes, a.\n"
    )


def test_arabic_disclosure_lines_exact(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    assert disclosure_text(inputs) == (
        "\u0625\u0641\u0635\u0627\u062d \u0639\u0646 \u0627\u0633"
        "\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u0630\u0643"
        "\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627"
        "\u0639\u064a:\n"
        "- \u0635\u0648\u062a \u0627\u0644\u0633\u0631\u062f \u0645"
        "\u0648\u0644\u062f \u0628\u0627\u0644\u0630\u0643\u0627"
        "\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639"
        "\u064a (elevenlabs).\n"
        "- \u0627\u0644\u0631\u0633\u0648\u0645\u0627\u062a \u0645"
        "\u0648\u0644\u062f\u0629 \u0628\u0646\u0645\u0627\u0630"
        "\u062c \u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646"
        "\u0627\u0639\u064a (openai/gpt-image-1).\n"
        "- \u0627\u0644\u0646\u0635: Scenes\u060c a.\n"
    )


def _verse_only(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> PartInputs:
    from fixtures.parts import write_timelines

    series = make_series([verse("v1")])
    [timeline] = write_timelines(series, tmp_path / "audio", library, flac=False)
    timeline = timeline.model_copy(update={"recitations": [_recitation(reciter(), ["31:33"])]})
    return make_inputs(tmp_path, library, channel, series=series, timeline=timeline)


def test_verse_only_part_says_none_and_names_reciter(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _english(_verse_only(tmp_path, library, channel))
    assert inputs.timeline.voice is None
    assert disclosure_text(inputs) == (
        "AI disclosure:\n"
        "- No AI-generated voice or images are used in this part.\n"
        "- Qur'an recitation: a recording by Files, used under CC-BY-4.0.\n"
        "- Script: Scenes, a.\n"
    )


def test_icon_plus_voice_gives_voice_line_and_no_sketches_line(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _english(make_inputs(tmp_path, library, channel))
    inputs = _with_sketch(inputs, provider=None, model=None, usage_terms=None)
    inputs = replace(
        inputs,
        sketches={
            k: v.model_copy(update={"origin": SketchOrigin.BUNDLED})
            for k, v in inputs.sketches.items()
            if v is not None
        },
    )
    text = disclosure_text(inputs)
    assert "narration voice is AI-generated" in text
    assert "sketches" not in text
    assert "No AI-generated" not in text


def test_blank_script_source_lines(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    base = _english(make_inputs(tmp_path, library, channel))
    assert disclosure_text(_source(base, author="", work="")).endswith("- Script: not recorded.\n")
    assert disclosure_text(_source(base, author="", work="W")).endswith("- Script: W.\n")
    assert disclosure_text(_source(base, author="A", work=" ")).endswith("- Script: A.\n")
    ar = make_inputs(tmp_path, library, channel)
    assert disclosure_text(_source(ar, author="", work="")).endswith(
        "- \u0627\u0644\u0646\u0635: \u063a\u064a\u0631 \u0645\u0633\u062c\u0644.\n"
    )


def test_models_sorted_unique(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    a = add_sketch(library, provenance=GENERATED_PROVENANCE, key="a|x")
    b = add_sketch(
        library,
        provenance=GENERATED_PROVENANCE.model_copy(update={"provider": "zeta", "model": "m"}),
        key="b|x",
    )
    series = make_series(
        [narr("s1", "alpha", sketch=b), narr("s2", "beta", sketch=a), narr("s3", "gamma", sketch=a)]
    )
    inputs = _english(make_inputs(tmp_path, library, channel, series=series))
    assert "(openai/gpt-image-1, zeta/m)" in disclosure_text(inputs)
    assert "(openai/gpt-image-1, zeta/m)" in disclosure_text(_english(inputs)).replace(", ", ", ")


def test_disclosure_off_fails_no_override(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    ok = disclosure_checks(inputs, disclosure_text(inputs))
    assert [r.severity for r in ok] == [Severity.PASS]
    off = replace(inputs, series=inputs.series.model_copy(update={"ai_disclosure": False}))
    [bad] = disclosure_checks(off, disclosure_text(off))
    assert bad.check is CheckId.AI_DISCLOSURE
    assert bad.severity is Severity.FAIL
    assert '"ai_disclosure": true' in bad.reason
    [blank] = disclosure_checks(inputs, "  \n")
    assert blank.severity is Severity.FAIL


def test_platform_labels_from_rules(tmp_path: Path) -> None:
    rules = load_rules(write_rules(tmp_path))
    labels = platform_labels(rules)
    assert set(labels) == set(rules.file.platforms)
    assert all(labels.values())
    assert VOICE.provider == "elevenlabs"


def test_generated_sketch_without_model_and_repeated_reciter(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _english(make_inputs(tmp_path, library, channel))
    inputs = _with_sketch(inputs, provider=None, model=None)
    assert "(unrecorded)" in disclosure_text(inputs)
    verse_inputs = _english(_verse_only(tmp_path / "v", library, channel))
    twice = [*verse_inputs.timeline.recitations, *verse_inputs.timeline.recitations]
    again = replace(
        verse_inputs, timeline=verse_inputs.timeline.model_copy(update={"recitations": twice})
    )
    assert disclosure_text(again).count("Qur'an recitation") == 1
