"""Exact-verse publishing check: the corpus text, the render report and the shipped ASS.

The expected text is the corpus Uthmani text of the series scene's `VerseRef` range,
whitespace-collapsed and joined (equal to `captions.verse.shown_text` of a correct
render). It must match the render report's hash and the ``Verse`` events burned from the
part's ``.ass`` file in the scene's slot. Any difference is a FAIL; nothing overrides it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from narratoflow.audio.timeline import SceneTiming
from narratoflow.captions.ass import AssEvent, ass_round_ms, parse_events
from narratoflow.captions.verse import shown_text, verse_ref_label
from narratoflow.errors import NarratoFlowError
from narratoflow.models import Language, Scene, SceneKind, VerseRef
from narratoflow.publish.checks import (
    CheckId,
    CheckResult,
    PartInputs,
    Severity,
    bare_paths,
    summarize,
)
from narratoflow.quran.corpus import QuranCorpus

RERENDER = "re-render: narratoflow render"
OK = "verse text matches the corpus exactly"
NO_VERSE = "no Qur'anic verse in this part"


def expected_verse_text(corpus: QuranCorpus, ayat: Sequence[tuple[int, int]]) -> str:
    """The on-screen text a correct render shows for `ayat`."""
    return " ".join(" ".join(corpus.ayah(s, a).uthmani.split()) for s, a in ayat)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fail(reason: str, scene_id: str | None = None) -> CheckResult:
    return CheckResult(
        check=CheckId.VERSE_TEXT, severity=Severity.FAIL, reason=reason, scene_id=scene_id
    )


def _ayat(ref: VerseRef) -> list[tuple[int, int]]:
    return [(ref.surah, a) for a in range(ref.ayah_start, ref.ayah_end + 1)]


def _load_events(inputs: PartInputs) -> tuple[list[AssEvent] | None, str]:
    """The shipped ASS events, or None and the reason they cannot be read."""
    name = inputs.render.captions_file
    try:
        text = (inputs.render_dir / name).read_text(encoding="utf-8")
        return parse_events(text), ""
    except FileNotFoundError:
        return None, f"captions file {name} is missing; {RERENDER}"
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f"captions file {name} cannot be read ({bare_paths(str(exc))}); {RERENDER}"


def _burned(events: list[AssEvent], slot: SceneTiming) -> str:
    lo, hi = ass_round_ms(slot.start_ms), ass_round_ms(slot.end_ms)
    inside = [e for e in events if e.style == "Verse" and lo <= e.start_ms < hi]
    return shown_text(sorted(inside, key=lambda e: (e.start_ms, e.layer)))


def _scene_findings(
    inputs: PartInputs,
    scene: Scene,
    corpus: QuranCorpus,
    events: list[AssEvent] | None,
) -> list[CheckResult]:
    ref = scene.verse
    if ref is None:
        return [_fail(f"verse scene {scene.id} has no verse reference", scene.id)]
    label = f"verse {verse_ref_label(ref, Language.ENGLISH)}"
    out: list[CheckResult] = []

    def fail(reason: str) -> None:
        out.append(_fail(f"{label}: {reason}", scene.id))

    try:
        expected = expected_verse_text(corpus, _ayat(ref))
    except NarratoFlowError as exc:
        fail(f"cannot read the corpus text ({exc})")
        return out
    report = next((v for v in inputs.render.verses if v.scene_id == scene.id), None)
    if report is None:
        fail(f"the render report has no verse entry for scene {scene.id}; {RERENDER}")
    else:
        want = [f"{s}:{a}" for s, a in _ayat(ref)]
        if report.ayat != want:
            fail(
                f"the render report shows ayat {', '.join(report.ayat) or 'none'}, "
                f"not {', '.join(want)}; {RERENDER}"
            )
        if report.shown_sha256 != _sha256(expected):
            fail(f"the render report's shown_sha256 differs from the corpus text; {RERENDER}")
    if events is not None:
        slot = next((t for t in inputs.timeline.scenes if t.scene_id == scene.id), None)
        if slot is None:
            fail(f"scene {scene.id} has no slot in the audio timeline; {RERENDER}")
        elif _burned(events, slot) != expected:
            fail(
                f"the subtitles in {inputs.render.captions_file} differ from the corpus "
                f"text; {RERENDER}"
            )
    return out


def verse_checks(inputs: PartInputs, corpus: Callable[[], QuranCorpus]) -> list[CheckResult]:
    """`VERSE_TEXT` results; the corpus is loaded only for a part with a verse scene."""
    scenes = [s for s in inputs.part.scenes if s.kind is SceneKind.VERSE]
    if not scenes:
        return summarize(CheckId.VERSE_TEXT, [], NO_VERSE)
    try:
        loaded = corpus()
    except NarratoFlowError as exc:
        return [
            _fail(f"cannot load the Quran corpus to check the verse text: {bare_paths(str(exc))}")
        ]
    events, problem = _load_events(inputs)
    findings = [_fail(problem)] if events is None else []
    for scene in scenes:
        findings += _scene_findings(inputs, scene, loaded, events)
    return summarize(CheckId.VERSE_TEXT, findings, OK)
