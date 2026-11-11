"""Render currency and stored sketch likeness/style publishing checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fixtures.parts import GENERATED_PROVENANCE, add_sketch, make_series, narr
from fixtures.publish import make_inputs

from narratoflow.channels.config import ChannelConfig
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchStatus
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.consistency import RERENDER, render_current_checks, sketch_checks


def _inputs(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> PartInputs:
    return make_inputs(tmp_path, library, channel)


def _fails(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.severity is Severity.FAIL]


def _restate(inputs: PartInputs, status: SketchStatus, reason: str | None) -> PartInputs:
    sid = inputs.part.scenes[0].sketch_id
    assert sid is not None
    record = inputs.sketches[sid]
    assert record is not None
    new = record.model_copy(update={"status": status, "status_reason": reason})
    return replace(inputs, sketches={**inputs.sketches, sid: new})


def test_current_render_passes(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    results = render_current_checks(_inputs(tmp_path, library, channel))
    assert [(r.check, r.severity) for r in results] == [(CheckId.RENDER_CURRENT, Severity.PASS)]
    assert results[0].reason == "the video matches the series and its audio timeline"


def test_swapped_sketch_fails_naming_scene_and_rerender(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    old = inputs.part.scenes[0].sketch_id
    new = add_sketch(library, key="other|key", provenance=GENERATED_PROVENANCE)
    scenes = [inputs.part.scenes[0].model_copy(update={"sketch_id": new}), *inputs.part.scenes[1:]]
    swapped = replace(inputs, part=inputs.part.model_copy(update={"scenes": scenes}))
    fails = _fails(render_current_checks(swapped))
    assert len(fails) == 1
    assert fails[0].reason.startswith(
        f"scene s1: the video draws sketch {old}, the series now has sketch {new}"
    )
    assert fails[0].reason.endswith("; " + RERENDER)


def test_text_card_swap_after_render_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    # drawn at render, now a text card
    card = narr("s1", "alpha beta gamma", title="One")
    scenes = [card, *inputs.part.scenes[1:]]
    now_card = replace(inputs, part=inputs.part.model_copy(update={"scenes": scenes}))
    assert _fails(render_current_checks(now_card))
    # card at render, now drawn
    inputs2 = _inputs(tmp_path, library, channel)
    scenes2 = [
        inputs2.part.scenes[0],
        inputs2.part.scenes[1].model_copy(
            update={
                "visual": inputs2.part.scenes[0].visual,
                "sketch_id": inputs2.part.scenes[0].sketch_id,
            }
        ),
    ]
    now_drawn = replace(inputs2, part=inputs2.part.model_copy(update={"scenes": scenes2}))
    fails = _fails(render_current_checks(now_drawn))
    assert fails
    assert all(r.reason.endswith("; " + RERENDER) for r in fails)


def test_replanned_scene_list_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    scenes = [inputs.part.scenes[0], inputs.part.scenes[1].model_copy(update={"id": "s9"})]
    replanned = replace(inputs, part=inputs.part.model_copy(update={"scenes": scenes}))
    fails = _fails(render_current_checks(replanned))
    assert len(fails) == 1
    assert fails[0].reason.endswith("; " + RERENDER)


def test_identity_or_duration_mismatch_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    for update in (
        {"series_title": "Other"},
        {"channel_slug": "x"},
        {"part_index": 2},
        {"part_count": 3},
    ):
        bad = replace(inputs, render=inputs.render.model_copy(update=update))
        assert _fails(render_current_checks(bad)), update
    tl = replace(inputs, timeline=inputs.timeline.model_copy(update={"part_count": 3}))
    assert _fails(render_current_checks(tl))
    dur = replace(
        inputs,
        render=inputs.render.model_copy(update={"duration_ms": inputs.render.duration_ms + 1}),
    )
    fails = _fails(render_current_checks(dur))
    assert fails
    assert all(r.reason.endswith("; " + RERENDER) for r in fails)


def test_missing_video_or_captions_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    (tmp_path / inputs.render.video_file).unlink()
    (tmp_path / inputs.render.captions_file).unlink()
    fails = _fails(render_current_checks(inputs))
    assert len(fails) == 2
    assert all(str(tmp_path) not in r.reason for r in fails)
    assert {inputs.render.video_file, inputs.render.captions_file} == {
        next(n for n in (inputs.render.video_file, inputs.render.captions_file) if n in r.reason)
        for r in fails
    }


def test_quarantined_after_render_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _restate(_inputs(tmp_path, library, channel), SketchStatus.FAILED, "face")
    results = sketch_checks(inputs)
    likeness = [r for r in results if r.check is CheckId.SKETCH_LIKENESS]
    assert [r.severity for r in likeness] == [Severity.FAIL]
    assert likeness[0].reason.startswith("scene s1: sketch ")
    assert likeness[0].reason.endswith("failed the vision check (face)")
    assert likeness[0].scene_id == "s1"


def test_unchecked_warns_never_passes(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _restate(
        _inputs(tmp_path, library, channel), SketchStatus.UNCHECKED, "vision: not checked"
    )
    likeness = [r for r in sketch_checks(inputs) if r.check is CheckId.SKETCH_LIKENESS]
    assert [r.severity for r in likeness] == [Severity.WARN]
    assert likeness[0].reason.endswith(
        "was not checked for faces, figures or logos (vision: not checked)"
    )


def test_style_outlier_warns(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _restate(
        _inputs(tmp_path, library, channel), SketchStatus.PASSED, "warn: style outlier; busy"
    )
    results = sketch_checks(inputs)
    style = [r for r in results if r.check is CheckId.SKETCH_STYLE]
    assert [(r.severity, r.reason) for r in style] == [(Severity.WARN, "warn: style outlier; busy")]
    likeness = [r for r in results if r.check is CheckId.SKETCH_LIKENESS]
    assert likeness[0].severity is Severity.PASS
    assert likeness[0].reason == "1 sketch passed the face/figure/logo check"


def test_no_sketch_part_passes_both(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    series = make_series([narr("s1", "alpha beta"), narr("s2", "gamma delta")])
    results = sketch_checks(make_inputs(tmp_path, library, channel, series=series))
    assert [(r.check, r.severity, r.reason) for r in results] == [
        (CheckId.SKETCH_LIKENESS, Severity.PASS, "no sketches in this part"),
        (CheckId.SKETCH_STYLE, Severity.PASS, "no style outliers"),
    ]


def test_clean_sketch_passes_and_missing_record_is_skipped(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    assert {r.severity for r in sketch_checks(inputs)} == {Severity.PASS}
    gone = replace(inputs, sketches={sid: None for sid in inputs.sketches})
    assert [r.reason for r in sketch_checks(gone)] == [
        "no sketches in this part",
        "no style outliers",
    ]
