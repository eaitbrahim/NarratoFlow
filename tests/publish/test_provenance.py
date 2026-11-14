"""Asset provenance and script-source publishing checks."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from fixtures.parts import (
    GENERATED_PROVENANCE,
    add_sketch,
    make_series,
    narr,
    reciter,
    verse,
    write_timelines,
)
from fixtures.publish import PACKAGED_RULES, dumps_toml, make_inputs, write_rules

from narratoflow.audio.timeline import (
    AssetUse,
    PartTimeline,
    RecitationUsed,
    SfxHit,
    TimingSource,
    VoiceUsed,
)
from narratoflow.channels.config import ChannelConfig
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord
from narratoflow.models import RightsBasis, ScriptSource
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.provenance import asset_records, provenance_checks, script_checks
from narratoflow.publish.rules import Rules, load_rules
from narratoflow.quran.reciters import Reciter, ReciterIndex

EMPTY = ReciterIndex([])


def _rules(tmp_path: Path, **changes: object) -> Rules:
    return load_rules(write_rules(tmp_path, **changes))


def _voice_pass(tmp_path: Path) -> Rules:
    return _rules(tmp_path, **{"licences.voice.elevenlabs-terms": {"verdict": "pass"}})


def _with_sketch(inputs: PartInputs, **provenance: object) -> PartInputs:
    """`inputs` with its one sketch's provenance (and `origin`) changed."""
    origin = provenance.pop("origin", None)
    [(sid, record)] = inputs.sketches.items()
    assert record is not None
    update: dict[str, object] = {"provenance": record.provenance.model_copy(update=provenance)}
    if origin is not None:
        update["origin"] = origin
    return replace(inputs, sketches={sid: record.model_copy(update=update)})


def _sketch(inputs: PartInputs) -> SketchRecord:
    [record] = inputs.sketches.values()
    assert record is not None
    return record


def _fails(results: list[CheckResult]) -> list[str]:
    return [r.reason for r in results if r.severity is Severity.FAIL]


def _warns(results: list[CheckResult]) -> list[str]:
    return [r.reason for r in results if r.severity is Severity.WARN]


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


def _verse_inputs(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig, *, recited: bool = True
) -> PartInputs:
    series = make_series([verse("v1")])
    [timeline] = write_timelines(series, tmp_path / "audio", library, flac=False)
    if recited:
        timeline = timeline.model_copy(update={"recitations": [_recitation(reciter(), ["31:33"])]})
    return make_inputs(tmp_path, library, channel, series=series, timeline=timeline)


# -- fixtures ----------------------------------------------------------------------------


def test_write_rules_round_trips_and_applies_dotted_changes(tmp_path: Path) -> None:
    packaged = tomllib.loads(PACKAGED_RULES.read_text(encoding="utf-8"))
    assert tomllib.loads(dumps_toml(packaged)) == packaged
    path = write_rules(
        tmp_path,
        **{
            "licences.assets.CC-BY-4.0.attribution": False,
            "licences.assets.GPL-3.0": {"verdict": "pass"},
            "licences.assets.MIT": None,
            "version": "t.1",
        },
    )
    rules = load_rules(path).file
    assert rules.version == "t.1"
    assert rules.licences.assets["CC-BY-4.0"].attribution is False
    assert rules.licences.assets["GPL-3.0"].verdict == "pass"
    assert "MIT" not in rules.licences.assets


def test_make_inputs_writes_render_dir(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel, spent_usd=1)
    assert inputs.render_dir == tmp_path
    assert (tmp_path / "part-01.mp4").is_file()
    assert (tmp_path / "part-01.ass").read_text(encoding="utf-8").startswith("[Script Info]")
    assert inputs.part.spent_usd == 1
    assert inputs.timeline.voice is not None
    assert _sketch(inputs).provenance == GENERATED_PROVENANCE


# -- provenance --------------------------------------------------------------------------


def test_narration_part_passes_with_empty_reciter_index(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    [result] = provenance_checks(inputs, _voice_pass(tmp_path), EMPTY)
    n = len(asset_records(inputs))
    assert result == CheckResult(
        check=CheckId.ASSET_PROVENANCE,
        severity=Severity.PASS,
        reason=f"{n} assets, each with a recorded source and an allowed licence",
    )


def test_generated_sketch_missing_model_or_terms_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    sid = _sketch(inputs).id
    for field in ("provider", "model", "usage_terms"):
        bad = _with_sketch(inputs, **{field: "  " if field == "model" else None})
        [reason] = _fails(provenance_checks(bad, rules, EMPTY))
        assert reason.startswith(f"sketch {sid}") and field in reason


def test_generated_origin_without_provider_is_sketch_checked_as_generated(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _with_sketch(
        make_inputs(tmp_path, library, channel), provider=None, origin=SketchOrigin.GENERATED
    )
    [record] = [r for r in asset_records(inputs) if r.kind in ("sketch", "icon")]
    assert record.kind == "sketch"
    [reason] = _fails(provenance_checks(inputs, _voice_pass(tmp_path), EMPTY))
    assert "provider" in reason  # openai-terms passes as generated terms: only provider fails
    bundled_with_provider = _with_sketch(inputs, provider="openai", origin=SketchOrigin.BUNDLED)
    assert next(r.kind for r in asset_records(bundled_with_provider)) == "sketch"


def test_unknown_generated_terms_fail_and_local_model_warns(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    unknown = _with_sketch(inputs, license="mystery-terms")
    [reason] = _fails(provenance_checks(unknown, rules, EMPTY))
    assert "licence mystery-terms is not allowed by platform rules 2026-11-14.1" in reason
    local = _with_sketch(inputs, license="local-model-license")
    results = provenance_checks(local, rules, EMPTY)
    assert _fails(results) == []
    [warn] = _warns(results)
    assert "local-model-license" in warn
    assert "check the local model's licence allows commercial use of outputs" in warn


def test_provenance_attribution_default_empty() -> None:
    old = {"source": "s", "license": "CC0-1.0", "created_at": "2026-01-01T00:00:00Z"}
    assert Provenance.model_validate(old).attribution == ""


def test_upload_cc_by_needs_attribution(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    base = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    upload = dict(origin=SketchOrigin.UPLOADED, provider=None, model=None, usage_terms=None)
    bare = _with_sketch(base, source="my photo", license="CC-BY-4.0", **upload)
    assert next(r.kind for r in asset_records(bare)) == "icon"
    [reason] = _fails(provenance_checks(bare, rules, EMPTY))
    assert "CC-BY-4.0 needs an attribution; none is recorded" in reason
    credited = _with_sketch(
        base, source="my photo", license="CC-BY-4.0", attribution="Ada, CC BY 4.0", **upload
    )
    record = next(r for r in asset_records(credited) if r.kind == "icon")
    assert record.attribution == "Ada, CC BY 4.0"
    assert [r.severity for r in provenance_checks(credited, rules, EMPTY)] == [Severity.PASS]


def test_icon_cc_by_without_attribution_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    base = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    icon = dict(origin=SketchOrigin.BUNDLED, provider=None, model=None, usage_terms=None)
    inputs = _with_sketch(base, source="assets/icons", license="CC-BY-4.0", **icon)
    sid = _sketch(inputs).id
    assert next(r.kind for r in asset_records(inputs)) == "icon"
    [reason] = _fails(provenance_checks(inputs, rules, EMPTY))
    assert reason.startswith(f"icon {sid}")
    assert "CC-BY-4.0 needs an attribution; none is recorded" in reason
    cc0 = _with_sketch(base, source="assets/icons", license="CC0-1.0", **icon)
    assert [r.severity for r in provenance_checks(cc0, rules, EMPTY)] == [Severity.PASS]
    copied = _with_sketch(
        base, source="assets/icons", license="CC0-1.0", copied_from_sketch_id=3, **icon
    )
    copied = replace(
        copied, sketches={sid: _sketch(copied).model_copy(update={"origin": SketchOrigin.COPIED})}
    )
    [record] = [r for r in asset_records(copied) if r.kind == "icon"]
    assert record.copied_from == 3


def test_missing_library_record_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    sid = _sketch(inputs).id
    missing = replace(inputs, sketches={sid: None})
    [result] = provenance_checks(missing, _voice_pass(tmp_path), EMPTY)
    assert result.severity is Severity.FAIL and result.scene_id == "s1"
    assert result.reason.startswith(
        f"scene s1: sketch {sid} is not in the library; provenance unknown"
    )
    assert all(r.kind != "sketch" for r in asset_records(missing))
    absent = replace(inputs, sketches={})
    assert _fails(provenance_checks(absent, _voice_pass(tmp_path), EMPTY))


def test_sfx_font_hand_licences_checked(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    gpl = AssetUse(kind="sfx", asset_id="boom", source="https://sfx.example/", license="GPL-3.0")
    with_sfx = replace(inputs, timeline=inputs.timeline.model_copy(update={"assets": [gpl]}))
    [reason] = _fails(provenance_checks(with_sfx, rules, EMPTY))
    assert reason.startswith("sfx boom") and "licence GPL-3.0 is not allowed" in reason
    font, *rest = inputs.render.assets
    blank = font.model_copy(update={"license": " "})
    hand = rest[-1].model_copy(update={"source": ""})
    render = inputs.render.model_copy(update={"assets": [blank, *rest[:-1], hand]})
    reasons = _fails(provenance_checks(replace(inputs, render=render), rules, EMPTY))
    assert len(reasons) == 2
    assert reasons[0].startswith(f"font {font.asset_id}") and "licence" in reasons[0]
    assert reasons[1].startswith(f"hand {hand.asset_id}") and "source" in reasons[1]


def test_missing_voice_on_narration_part_fails_and_verse_only_part_needs_none(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    silent = replace(inputs, timeline=inputs.timeline.model_copy(update={"voice": None}))
    assert _fails(provenance_checks(silent, rules, EMPTY)) == [
        "the narration voice is not recorded; rebuild audio"
    ]
    verse_only = _verse_inputs(tmp_path / "v", library, channel)
    assert verse_only.timeline.voice is None
    results = provenance_checks(verse_only, rules, ReciterIndex([reciter()]))
    assert [r.severity for r in results] == [Severity.PASS]


def test_edge_voice_terms_warn_with_note(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    edge = VoiceUsed(provider="edge", model="edge-tts", voice_id="ar-SA-x", terms="edge-tts-terms")
    inputs = replace(inputs, timeline=inputs.timeline.model_copy(update={"voice": edge}))
    results = provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY)
    assert _fails(results) == []
    [warn] = _warns(results)
    assert warn.startswith("voice edge/ar-SA-x")
    assert "Microsoft's read-aloud service grants no explicit commercial licence" in warn


def _piper_inputs(inputs: PartInputs, **declared: object) -> PartInputs:
    piper = VoiceUsed.model_validate(
        {
            "provider": "piper",
            "model": "m",
            "voice_id": "piper",
            "terms": "piper-model-license",
            **declared,
        }
    )
    return replace(inputs, timeline=inputs.timeline.model_copy(update={"voice": piper}))


_SRC = "https://example.org/voice"


@pytest.mark.parametrize(
    ("declared", "severity", "text"),
    [
        ({"model_license": "MIT", "model_source": _SRC}, None, ""),
        (
            {"model_license": "CC-BY-4.0", "model_source": _SRC, "model_attribution": "By A"},
            None,
            "",
        ),
        (
            {"model_license": "CC-BY-4.0", "model_source": _SRC},
            Severity.FAIL,
            "needs an attribution",
        ),
        (
            {"model_license": "CC-BY-NC-4.0", "model_source": _SRC},
            Severity.FAIL,
            "not allowed",
        ),
        ({}, Severity.WARN, "voice.piper_model_license"),
    ],
)
def test_piper_declared_licence_verdicts(
    tmp_path: Path,
    library: SketchLibrary,
    channel: ChannelConfig,
    declared: dict[str, object],
    severity: Severity | None,
    text: str,
) -> None:
    inputs = _piper_inputs(make_inputs(tmp_path, library, channel), **declared)
    results = provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY)
    voice_findings = [
        r for r in results if r.severity is not Severity.PASS and r.reason.startswith("voice ")
    ]
    if severity is None:
        assert voice_findings == []
        return
    [finding] = voice_findings
    assert finding.severity is severity
    assert text in finding.reason


def test_piper_record_uses_declared_source_and_attribution(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _piper_inputs(
        make_inputs(tmp_path, library, channel),
        model_license="CC-BY-4.0",
        model_source=_SRC,
        model_attribution="By A",
    )
    voice = next(r for r in asset_records(inputs) if r.kind == "voice")
    assert (voice.license, voice.source, voice.attribution) == ("CC-BY-4.0", _SRC, "By A")


def test_edge_voice_unchanged(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    edge = VoiceUsed(provider="edge", model="edge-tts", voice_id="ar-SA-x", terms="edge-tts-terms")
    inputs = replace(inputs, timeline=inputs.timeline.model_copy(update={"voice": edge}))
    [warn] = _warns(provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY))
    assert warn == (
        "voice edge/ar-SA-x: licence edge-tts-terms: "
        "Microsoft's read-aloud service grants no explicit commercial licence"
    )


def test_verse_without_recitation_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel, recited=False)
    [result] = provenance_checks(inputs, _voice_pass(tmp_path), ReciterIndex([reciter()]))
    assert result.severity is Severity.FAIL and result.scene_id == "v1"
    assert result.reason.startswith(
        "verse 31:33 has no licensed recitation recorded (unlicensed recitation)"
    )


def test_reciter_removed_or_terms_changed_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _verse_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    [reason] = _fails(provenance_checks(inputs, rules, EMPTY))
    assert reason.startswith(
        "reciter files-hafs is not on the reciter allow-list (assets/reciters/index.toml)"
    )
    for field, value in (
        ("license", "CC0-1.0"),
        ("license_url", "https://example.org/other"),
        ("source", "https://other.example.org/"),
    ):
        changed = reciter().model_copy(update={field: value})
        assert _fails(provenance_checks(inputs, rules, ReciterIndex([changed]))) == [
            "reciter files-hafs: its terms changed since the audio build; rebuild audio"
        ]


def test_asset_records_one_per_asset_with_scene_ids(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    series = make_series(
        [
            verse("v1"),
            narr("s1", "alpha beta", sketch=sid),
            narr("s2", "gamma"),
            narr("s3", "delta", sketch=sid),
        ]
    )
    pen = AssetUse(kind="sfx", asset_id="pen", source="https://sfx.example/", license="CC0-1.0")
    [timeline] = write_timelines(series, tmp_path / "audio", library, flac=False)
    hits = [
        SfxHit(scene_id=s, stroke_index=0, at_ms=0, duration_ms=10, asset_id="pen", gain_db=0)
        for s in ("s1", "s3", "s1")
    ]
    timeline = timeline.model_copy(
        update={
            "assets": [pen, pen],
            "sfx": hits,
            "voice": VoiceUsed(provider="p", model="m", voice_id="v", terms="t"),
            "recitations": [_recitation(reciter(), ["31:33"])] * 2,
        }
    )
    inputs = make_inputs(tmp_path, library, channel, series=series, timeline=timeline)
    records = asset_records(inputs)
    kinds = [r.kind for r in records]
    fonts = [a.asset_id for a in inputs.render.assets if a.kind == "font"]
    assert kinds == ["sketch", "sfx", *["font"] * len(fonts), "hand", "voice", "recitation"]
    sketch, sfx = records[0], records[1]
    assert (sketch.asset_id, sketch.scene_ids) == (str(sid), ["s1", "s3"])
    assert (sketch.provider, sketch.model, sketch.usage_terms) == (
        "openai",
        "gpt-image-1",
        GENERATED_PROVENANCE.usage_terms,
    )
    assert sketch.created_at == GENERATED_PROVENANCE.created_at
    assert (sfx.asset_id, sfx.scene_ids) == ("pen", ["s1", "s3"])
    voice, recitation = records[-2], records[-1]
    assert (voice.asset_id, voice.source, voice.license) == ("p/v", "p", "t")
    assert voice.scene_ids == ["s1", "s2", "s3"]
    r = reciter()
    assert (recitation.asset_id, recitation.license, recitation.source) == (
        r.id,
        r.license,
        r.source,
    )
    assert (recitation.usage_terms, recitation.scene_ids) == (r.license_url, ["v1"])


# -- script source -----------------------------------------------------------------------


def _with_source(inputs: PartInputs, **fields: object) -> PartInputs:
    source = inputs.series.source.model_copy(update=fields)
    return replace(inputs, series=inputs.series.model_copy(update={"source": source}))


def test_script_third_party_and_bare_permission_fail_blank_author_warns(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    from narratoflow.errors import ScriptBlockedError
    from narratoflow.script.source import check_rights

    inputs = make_inputs(tmp_path, library, channel)
    for fields in (
        {"rights": RightsBasis.THIRD_PARTY_IN_COPYRIGHT},
        {"rights": RightsBasis.PERMISSION, "permission_note": "  "},
    ):
        bad = _with_source(inputs, **fields)
        try:
            check_rights(bad.series.source)
        except ScriptBlockedError as exc:
            expected = str(exc)
        [result] = script_checks(bad)
        assert (result.check, result.severity) == (CheckId.SCRIPT_SOURCE, Severity.FAIL)
        assert result.reason == expected
    for fields in ({"author": " "}, {"work": ""}):
        [result] = script_checks(_with_source(inputs, **fields))
        assert (result.severity, result.reason) == (
            Severity.WARN,
            "script source not recorded (author/work)",
        )
    noted = _with_source(inputs, rights=RightsBasis.PERMISSION, permission_note="granted")
    assert [r.severity for r in script_checks(noted)] == [Severity.PASS]


def test_ai_transcribed_named_in_pass_reason(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    source = ScriptSource(author="Ibn Kathir", work="Al-Bidaya", rights=RightsBasis.PUBLIC_DOMAIN)
    plain = replace(inputs, series=inputs.series.model_copy(update={"source": source}))
    [result] = script_checks(plain)
    assert result.reason == "script: Al-Bidaya, Ibn Kathir (public_domain)"
    transcribed = _with_source(plain, ai_transcribed=True)
    [result] = script_checks(transcribed)
    assert result.severity is Severity.PASS
    assert result.reason == (
        "script: Al-Bidaya, Ibn Kathir (public_domain); AI transcription - proofread"
    )


def test_sfx_hit_without_asset_record_fails(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    hits = [
        SfxHit(scene_id="s1", stroke_index=0, at_ms=0, duration_ms=10, asset_id="ghost", gain_db=0)
    ] * 2
    timeline = inputs.timeline.model_copy(update={"assets": [], "sfx": hits})
    [result] = provenance_checks(replace(inputs, timeline=timeline), _voice_pass(tmp_path), EMPTY)
    assert result.severity is Severity.FAIL and result.scene_id == "s1"
    assert result.reason == "sfx ghost: no provenance recorded; rebuild audio"


def test_warn_licence_with_blank_note_has_no_dangling_colon(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _rules(
        tmp_path, **{"licences.voice.elevenlabs-terms": {"verdict": "warn", "note": " "}}
    )
    [warn] = _warns(provenance_checks(inputs, rules, EMPTY))
    assert warn.endswith("licence elevenlabs-terms"), warn


@pytest.mark.parametrize(
    ("licence", "verdict"),
    [
        ("Apache-2.0", "pass"),
        ("MIT", "pass"),
        ("OpenRAIL++-M", "warn"),
        ("Apache-2.0 AND OpenRAIL++-M", "warn"),
        ("Apache-2.0 AND FLUX-1-dev-non-commercial", "fail"),
    ],
)
def test_generated_composite_licence_verdicts(
    tmp_path: Path,
    library: SketchLibrary,
    channel: ChannelConfig,
    licence: str,
    verdict: str,
) -> None:
    inputs = _with_sketch(make_inputs(tmp_path, library, channel), license=licence)
    results = provenance_checks(inputs, _voice_pass(tmp_path), EMPTY)
    fails, warns = _fails(results), _warns(results)
    assert (len(fails), len(warns)) == {"pass": (0, 0), "warn": (0, 1), "fail": (1, 0)}[verdict]
    if verdict == "fail":
        assert "licence FLUX-1-dev-non-commercial is not allowed" in fails[0]
    if verdict == "warn":
        assert "OpenRAIL++-M" in warns[0]


def test_unknown_composite_part_names_only_that_part(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _with_sketch(
        make_inputs(tmp_path, library, channel), license="mystery-a AND Apache-2.0 AND mystery-b"
    )
    reasons = _fails(provenance_checks(inputs, _voice_pass(tmp_path), EMPTY))
    assert len(reasons) == 2
    assert "licence mystery-a is not allowed" in reasons[0]
    assert "licence mystery-b is not allowed" in reasons[1]


# -- aligner (ctc-forced-aligner is non-commercial) -------------------------------------


def _timed(inputs: PartInputs, *sources: TimingSource) -> PartInputs:
    """`inputs` whose narration scenes take `sources` in order."""
    scenes = [
        scene.model_copy(update={"timing_source": src})
        for scene, src in zip(inputs.timeline.scenes, sources, strict=True)
    ]
    timeline: PartTimeline = inputs.timeline.model_copy(update={"scenes": scenes})
    return replace(inputs, timeline=timeline)


def test_aligned_scene_records_aligner_and_warns(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _timed(
        make_inputs(tmp_path, library, channel), TimingSource.ALIGNED, TimingSource.ESTIMATED
    )
    [aligner] = [r for r in asset_records(inputs) if r.kind == "aligner"]
    assert aligner.asset_id == "ctc-forced-aligner"
    assert aligner.license == "ctc-forced-aligner-terms"
    assert aligner.source == "ctc-forced-aligner 1.0.2 + deskpai/ctc_forced_aligner"
    assert aligner.scene_ids == ["s1"]
    results = provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY)
    assert _fails(results) == []
    [warn] = [r for r in results if r.severity is Severity.WARN and "aligner" in r.reason]
    assert warn.scene_id == "s1"
    assert warn.reason == (
        "aligner ctc-forced-aligner: licence ctc-forced-aligner-terms: word timings from "
        "the align extra (DOSL package licence, CC-BY-NC-4.0 default model): "
        "not cleared for commercial use"
    )


@pytest.mark.parametrize("source", [TimingSource.PROVIDER, TimingSource.ESTIMATED])
def test_no_aligned_scene_no_aligner_record(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig, source: TimingSource
) -> None:
    inputs = _timed(make_inputs(tmp_path, library, channel), source, source)
    assert [r for r in asset_records(inputs) if r.kind == "aligner"] == []
    results = provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY)
    assert not any("aligner" in r.reason for r in results)


def test_several_aligned_scenes_collapse_to_one_record(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _timed(
        make_inputs(tmp_path, library, channel), TimingSource.ALIGNED, TimingSource.ALIGNED
    )
    [aligner] = [r for r in asset_records(inputs) if r.kind == "aligner"]
    assert aligner.scene_ids == ["s1", "s2"]
    results = provenance_checks(inputs, load_rules(PACKAGED_RULES), EMPTY)
    [warn] = [r for r in results if r.severity is Severity.WARN and "aligner" in r.reason]
    assert warn.scene_id is None


# -- unknown licence: the remedy depends on the rules origin and the asset kind ---------


def _packaged_origin(rules: Rules) -> Rules:
    return replace(rules, origin="packaged")


def test_unknown_licence_with_override_rules_names_the_file_and_says_to_add_the_id(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _voice_pass(tmp_path)
    [reason] = _fails(
        provenance_checks(_with_sketch(inputs, license="mystery-terms"), rules, EMPTY)
    )
    assert rules.origin == str((tmp_path / "platform_rules.toml").resolve())
    assert f"add mystery-terms to {rules.origin}" in reason
    assert "merge" in reason


def test_unknown_licence_with_packaged_rules_keeps_the_replace_the_asset_text(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _packaged_origin(_voice_pass(tmp_path))
    [reason] = _fails(
        provenance_checks(_with_sketch(inputs, license="mystery-terms"), rules, EMPTY)
    )
    assert reason.endswith(
        "licence mystery-terms is not allowed by platform rules 2026-11-14.1; replace the asset"
    )


def test_unknown_aligner_licence_says_rebuild_without_the_align_extra(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _timed(
        make_inputs(tmp_path, library, channel), TimingSource.ALIGNED, TimingSource.ESTIMATED
    )
    rules = _packaged_origin(_rules(tmp_path, **{"licences.assets.ctc-forced-aligner-terms": None}))
    [reason] = [r for r in _fails(provenance_checks(inputs, rules, EMPTY)) if "aligner" in r]
    assert reason.endswith("; rebuild the audio without the align extra")
    assert "replace the asset" not in reason


def test_unknown_voice_licence_says_rebuild_with_an_allowed_voice(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = make_inputs(tmp_path, library, channel)
    rules = _packaged_origin(_rules(tmp_path, **{"licences.voice.elevenlabs-terms": None}))
    [reason] = [r for r in _fails(provenance_checks(inputs, rules, EMPTY)) if "voice" in r]
    assert reason.endswith("; rebuild the audio with an allowed voice")
    assert "replace the asset" not in reason
