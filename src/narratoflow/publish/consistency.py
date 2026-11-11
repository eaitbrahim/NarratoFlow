"""Render currency and stored sketch likeness/style publishing checks.

`render_current_checks` compares the render report with the series and the audio
timeline as they are now. Limit (G18): narration text edited after render, with the
same scene ids, is not detected here, the same as render's `check_timeline`; the
render report records no narration hash. The stale content review WARNs on it.
"""

from __future__ import annotations

from narratoflow.library.schemas import SketchStatus
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity, summarize
from narratoflow.render.build import visual_kind

RERENDER = "re-run narratoflow audio build and narratoflow render"
_STYLE_WARN = "warn:"


def _fail(reason: str, scene_id: str | None = None) -> CheckResult:
    return CheckResult(
        check=CheckId.RENDER_CURRENT,
        severity=Severity.FAIL,
        reason=f"{reason}; {RERENDER}",
        scene_id=scene_id,
    )


def render_current_checks(inputs: PartInputs) -> list[CheckResult]:
    """FAIL for each way the rendered video no longer matches the series or audio."""
    render, series, part, timeline = inputs.render, inputs.series, inputs.part, inputs.timeline
    out: list[CheckResult] = []
    want = (series.title, series.channel_slug, part.index, part.count)
    if (render.series_title, render.channel_slug, render.part_index, render.part_count) != want:
        out.append(_fail("the render is for a different series or part than the series now has"))
    if (
        timeline.series_title,
        timeline.channel_slug,
        timeline.part_index,
        timeline.part_count,
    ) != want:
        out.append(_fail("the audio timeline is for a different series or part"))
    if render.duration_ms != timeline.duration_ms:
        out.append(
            _fail(
                f"the video is {render.duration_ms} ms long, the audio timeline "
                f"{timeline.duration_ms} ms"
            )
        )
    rendered = [(s.scene_id, s.kind) for s in render.scenes]
    planned = [(s.id, s.kind) for s in part.scenes]
    if rendered != planned:
        out.append(_fail("the scenes of the video differ from the scenes of the series"))
    else:
        for drawn, scene in zip(render.scenes, part.scenes, strict=True):
            want_kind = visual_kind(scene)
            if drawn.visual != want_kind:
                out.append(
                    _fail(
                        f"scene {scene.id}: the video shows a {drawn.visual}, "
                        f"the series now has a {want_kind}",
                        scene.id,
                    )
                )
            elif drawn.visual == "sketch" and drawn.sketch_id != scene.sketch_id:
                out.append(
                    _fail(
                        f"scene {scene.id}: the video draws sketch {drawn.sketch_id}, "
                        f"the series now has sketch {scene.sketch_id}",
                        scene.id,
                    )
                )
    for name in (render.video_file, render.captions_file):
        if not (inputs.render_dir / name).is_file():
            out.append(_fail(f"{name} is missing from {inputs.render_dir}"))
    return summarize(
        CheckId.RENDER_CURRENT, out, "the video matches the series and its audio timeline"
    )


def sketch_checks(inputs: PartInputs) -> list[CheckResult]:
    """Stored vision (likeness) and style results of the sketches the video draws."""
    likeness: list[CheckResult] = []
    style: list[CheckResult] = []
    checked = 0
    for scene in inputs.render.scenes:
        if scene.visual != "sketch" or scene.sketch_id is None:
            continue
        record = inputs.sketches.get(scene.sketch_id)
        if record is None:
            continue
        n = scene.sketch_id
        reason = record.status_reason or ""
        if record.status is SketchStatus.FAILED:
            likeness.append(
                CheckResult(
                    check=CheckId.SKETCH_LIKENESS,
                    severity=Severity.FAIL,
                    reason=f"scene {scene.scene_id}: sketch {n} failed the vision check ({reason})",
                    scene_id=scene.scene_id,
                )
            )
            continue
        if record.status is SketchStatus.UNCHECKED:
            likeness.append(
                CheckResult(
                    check=CheckId.SKETCH_LIKENESS,
                    severity=Severity.WARN,
                    reason=(
                        f"scene {scene.scene_id}: sketch {n} was not checked for faces, "
                        f"figures or logos ({reason})"
                    ),
                    scene_id=scene.scene_id,
                )
            )
            continue
        checked += 1
        if reason.startswith(_STYLE_WARN):
            style.append(
                CheckResult(
                    check=CheckId.SKETCH_STYLE,
                    severity=Severity.WARN,
                    reason=reason,
                    scene_id=scene.scene_id,
                )
            )
    ok = (
        f"{checked} sketches passed the face/figure/logo check"
        if checked
        else ("no sketches in this part")
    )
    return [
        *summarize(CheckId.SKETCH_LIKENESS, likeness, ok),
        *summarize(CheckId.SKETCH_STYLE, style, "no style outliers"),
    ]
