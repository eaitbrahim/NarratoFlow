"""Asset provenance (spec 6.2) and script-source (spec 6.3) publishing checks."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from narratoflow.audio.timeline import TimingSource
from narratoflow.errors import ScriptBlockedError
from narratoflow.library.schemas import COMPOSITE_SEPARATOR, SketchOrigin, SketchRecord
from narratoflow.models import SceneKind
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity, summarize
from narratoflow.publish.rules import LicenceRule, Rules, home_free
from narratoflow.quran.reciters import ReciterIndex
from narratoflow.script.source import check_rights

AssetKind = Literal["sketch", "icon", "sfx", "font", "hand", "voice", "recitation", "aligner"]

_GENERATED_FIELDS = ("provider", "model", "created_at", "usage_terms")


class AssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AssetKind
    asset_id: str
    source: str
    license: str
    attribution: str = ""
    provider: str | None = None
    model: str | None = None
    created_at: datetime | None = None
    usage_terms: str | None = None
    copied_from: int | None = None
    scene_ids: list[str] = []


def is_generated(record: SketchRecord) -> bool:
    """A generated image: generated origin, or any record naming an image provider."""
    return record.origin is SketchOrigin.GENERATED or record.provenance.provider is not None


def _sketch_records(inputs: PartInputs) -> list[AssetRecord]:
    by_id: dict[int, AssetRecord] = {}
    for scene in inputs.render.scenes:
        if scene.sketch_id is None:
            continue
        found = by_id.get(scene.sketch_id)
        if found is not None:
            found.scene_ids.append(scene.scene_id)
            continue
        record = inputs.sketches.get(scene.sketch_id)
        if record is None:
            continue
        p = record.provenance
        by_id[scene.sketch_id] = AssetRecord(
            kind="sketch" if is_generated(record) else "icon",
            asset_id=str(scene.sketch_id),
            source=p.source,
            license=p.license,
            attribution=p.attribution,
            provider=p.provider,
            model=p.model,
            created_at=p.created_at,
            usage_terms=p.usage_terms,
            copied_from=p.copied_from_sketch_id,
            scene_ids=[scene.scene_id],
        )
    return list(by_id.values())


def _sfx_records(inputs: PartInputs) -> list[AssetRecord]:
    timeline = inputs.timeline
    out: dict[str, AssetRecord] = {}
    for use in timeline.assets:
        if use.asset_id in out:
            continue
        scenes = dict.fromkeys(h.scene_id for h in timeline.sfx if h.asset_id == use.asset_id)
        out[use.asset_id] = AssetRecord(
            kind="sfx",
            asset_id=use.asset_id,
            source=use.source,
            license=use.license,
            attribution=use.attribution,
            scene_ids=list(scenes),
        )
    return list(out.values())


def _render_records(inputs: PartInputs, kind: Literal["font", "hand"]) -> list[AssetRecord]:
    out: dict[str, AssetRecord] = {}
    for asset in inputs.render.assets:
        if asset.kind == kind and asset.asset_id not in out:
            out[asset.asset_id] = AssetRecord(
                kind=kind,
                asset_id=asset.asset_id,
                source=asset.source,
                license=asset.license,
                attribution=asset.attribution,
            )
    return list(out.values())


def _voice_records(inputs: PartInputs) -> list[AssetRecord]:
    voice = inputs.timeline.voice
    if voice is None:
        return []
    narrated = [s.scene_id for s in inputs.render.scenes if s.kind is SceneKind.NARRATION]
    return [
        AssetRecord(
            kind="voice",
            asset_id=f"{voice.provider}/{voice.voice_id}",
            source=voice.model_source or voice.provider,
            license=voice.model_license or voice.terms,
            attribution=voice.model_attribution,
            provider=voice.provider,
            model=voice.model,
            scene_ids=narrated,
        )
    ]


def _recitation_records(inputs: PartInputs) -> list[AssetRecord]:
    out: dict[str, AssetRecord] = {}
    for used in inputs.timeline.recitations:
        covered = set(used.ayat)
        scenes = [v.scene_id for v in inputs.render.verses if covered & set(v.ayat)]
        found = out.get(used.reciter_id)
        if found is not None:
            found.scene_ids.extend(s for s in scenes if s not in found.scene_ids)
            continue
        out[used.reciter_id] = AssetRecord(
            kind="recitation",
            asset_id=used.reciter_id,
            source=used.source,
            license=used.license,
            usage_terms=used.license_url,
            scene_ids=scenes,
        )
    return list(out.values())


def _aligner_records(inputs: PartInputs) -> list[AssetRecord]:
    """The CTC aligner, once, when any scene's word times came from it (the only `Aligner`)."""
    scenes = [s.scene_id for s in inputs.timeline.scenes if s.timing_source is TimingSource.ALIGNED]
    if not scenes:
        return []
    return [
        AssetRecord(
            kind="aligner",
            asset_id="ctc-forced-aligner",
            source="ctc-forced-aligner 1.0.2 + deskpai/ctc_forced_aligner",
            license="ctc-forced-aligner-terms",
            scene_ids=scenes,
        )
    ]


def asset_records(inputs: PartInputs) -> list[AssetRecord]:
    """One record per distinct asset: sketches/icons (scene order), SFX, fonts, hand,
    voice, recitations, aligner. A sketch missing from the library has no record."""
    return [
        *_sketch_records(inputs),
        *_sfx_records(inputs),
        *_render_records(inputs, "font"),
        *_render_records(inputs, "hand"),
        *_voice_records(inputs),
        *_recitation_records(inputs),
        *_aligner_records(inputs),
    ]


def _fail(reason: str, scene_id: str | None = None) -> CheckResult:
    return CheckResult(
        check=CheckId.ASSET_PROVENANCE, severity=Severity.FAIL, reason=reason, scene_id=scene_id
    )


def _label(record: AssetRecord) -> str:
    return f"{record.kind} {record.asset_id}"


def _scene(record: AssetRecord) -> str | None:
    return record.scene_ids[0] if len(record.scene_ids) == 1 else None


def _licence_table(rules: Rules, record: AssetRecord) -> dict[str, LicenceRule] | None:
    licences = rules.file.licences
    if record.kind == "sketch":
        return licences.generated
    if record.kind == "voice":
        return {**licences.assets, **licences.voice}
    if record.kind == "aligner":
        return licences.aligner
    if record.kind == "recitation":
        return None  # checked against the reciter allow-list instead
    return licences.assets


def _generated_field_findings(record: AssetRecord) -> list[CheckResult]:
    out: list[CheckResult] = []
    for field in _GENERATED_FIELDS:
        value = getattr(record, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            out.append(
                _fail(
                    f"{_label(record)}: generated image has no {field} recorded; "
                    "regenerate the sketch",
                    _scene(record),
                )
            )
    return out


def _record_findings(record: AssetRecord, rules: Rules) -> list[CheckResult]:
    label, scene = _label(record), _scene(record)
    out: list[CheckResult] = []
    if not record.source.strip():
        out.append(_fail(f"{label}: no source is recorded; record where it came from", scene))
    if record.kind == "sketch":
        out.extend(_generated_field_findings(record))
    licence = record.license.strip()
    if not licence:
        out.append(_fail(f"{label}: no licence is recorded; record its licence", scene))
        return out
    table = _licence_table(rules, record)
    if table is None:
        return out
    # a generated sketch's licence is "<model> AND <lora>": judge each id on its own
    ids = licence.split(COMPOSITE_SEPARATOR) if record.kind == "sketch" else [licence]
    for lid in ids:
        out.extend(_licence_findings(record, rules, table, lid.strip()))
    return out


_LICENCE_TABLE_NAME = {
    "sketch": "[licences.generated]",
    "voice": "[licences.voice]",
    "aligner": "[licences.aligner]",
}


_REBUILD_REMEDY = {
    "voice": "rebuild the audio with an allowed voice",
    "aligner": "rebuild the audio without the align extra",
}


def _unknown_licence_remedy(record: AssetRecord, rules: Rules, licence: str) -> str:
    remedy = _REBUILD_REMEDY.get(record.kind, "replace the asset")
    if rules.origin == "packaged":
        return remedy
    return (
        f"add {licence} to {home_free(rules.origin)} under "
        f"{_LICENCE_TABLE_NAME.get(record.kind, '[licences.assets]')} (merge it with the "
        f"packaged rules after an upgrade), or {remedy}"
    )


def _licence_findings(
    record: AssetRecord, rules: Rules, table: dict[str, LicenceRule], licence: str
) -> list[CheckResult]:
    label, scene = _label(record), _scene(record)
    rule = table.get(licence)
    if rule is None:
        return [
            _fail(
                f"{label}: licence {licence} is not allowed by platform rules "
                f"{rules.file.version}; {_unknown_licence_remedy(record, rules, licence)}",
                scene,
            )
        ]
    if rule.attribution and not record.attribution.strip():
        return [
            _fail(
                f"{label}: {licence} needs an attribution; none is recorded; "
                "record the attribution or replace the asset",
                scene,
            )
        ]
    if rule.verdict != "warn":
        return []
    note = rule.note.strip()
    return [
        CheckResult(
            check=CheckId.ASSET_PROVENANCE,
            severity=Severity.WARN,
            reason=f"{label}: licence {licence}" + (f": {note}" if note else ""),
            scene_id=scene,
        )
    ]


def _recitation_findings(inputs: PartInputs, reciters: ReciterIndex) -> list[CheckResult]:
    out: list[CheckResult] = []
    for used in inputs.timeline.recitations:
        listed = reciters.get(used.reciter_id)
        if listed is None:
            out.append(
                _fail(
                    f"reciter {used.reciter_id} is not on the reciter allow-list "
                    "(assets/reciters/index.toml); choose a listed reciter and rebuild audio"
                )
            )
        elif (used.license, used.license_url, used.source) != (
            listed.license,
            listed.license_url,
            listed.source,
        ):
            out.append(
                _fail(
                    f"reciter {used.reciter_id}: its terms changed since the audio build; "
                    "rebuild audio"
                )
            )
    covered = {ayah for used in inputs.timeline.recitations for ayah in used.ayat}
    for shown in inputs.render.verses:
        for ayah in shown.ayat:
            if ayah not in covered:
                out.append(
                    _fail(
                        f"verse {ayah} has no licensed recitation recorded "
                        "(unlicensed recitation); rebuild audio with a listed reciter",
                        shown.scene_id,
                    )
                )
    return out


def provenance_checks(
    inputs: PartInputs, rules: Rules, reciters: ReciterIndex
) -> list[CheckResult]:
    """Every asset the video uses has a recorded source and a licence the rules allow."""
    findings: list[CheckResult] = []
    for scene in inputs.render.scenes:
        if scene.sketch_id is not None and inputs.sketches.get(scene.sketch_id) is None:
            findings.append(
                _fail(
                    f"scene {scene.scene_id}: sketch {scene.sketch_id} is not in the library; "
                    "provenance unknown; re-run sketch for this scene",
                    scene.scene_id,
                )
            )
    narrated = any(s.kind is SceneKind.NARRATION for s in inputs.part.scenes)
    if narrated and inputs.timeline.voice is None:
        findings.append(_fail("the narration voice is not recorded; rebuild audio"))
    listed = {use.asset_id for use in inputs.timeline.assets}
    unlisted: dict[str, str] = {}
    for hit in inputs.timeline.sfx:
        if hit.asset_id not in listed:
            unlisted.setdefault(hit.asset_id, hit.scene_id)
    for asset_id, scene_id in unlisted.items():
        findings.append(_fail(f"sfx {asset_id}: no provenance recorded; rebuild audio", scene_id))
    records = asset_records(inputs)
    for record in records:
        findings += _record_findings(record, rules)
    findings += _recitation_findings(inputs, reciters)
    return summarize(
        CheckId.ASSET_PROVENANCE,
        findings,
        f"{len(records)} assets, each with a recorded source and an allowed licence",
    )


def script_checks(inputs: PartInputs) -> list[CheckResult]:
    """The script's declared source: rights via `check_rights`, then author and work."""
    source = inputs.series.source
    try:
        check_rights(source)
    except ScriptBlockedError as exc:
        return [CheckResult(check=CheckId.SCRIPT_SOURCE, severity=Severity.FAIL, reason=str(exc))]
    if not source.author.strip() or not source.work.strip():
        return [
            CheckResult(
                check=CheckId.SCRIPT_SOURCE,
                severity=Severity.WARN,
                reason="script source not recorded (author/work)",
            )
        ]
    reason = f"script: {source.work}, {source.author} ({source.rights.value})"
    if source.ai_transcribed:
        reason += "; AI transcription - proofread"
    return [CheckResult(check=CheckId.SCRIPT_SOURCE, severity=Severity.PASS, reason=reason)]
