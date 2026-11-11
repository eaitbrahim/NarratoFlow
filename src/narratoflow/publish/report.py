"""The per-part production report (``part-NN.publish.json``/``.md``) and the series summary.

Both are strict models (unknown keys and non-finite numbers are refused), written
atomically and loaded strictly, following `RenderReport`. Money is `Decimal` and is
serialized as an exact string. The Markdown views escape every free-text value
(`md_cell`) so a title, reason or model finding cannot break the layout.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError, model_validator

from narratoflow.audio.timeline import PartTimeline, VoiceUsed
from narratoflow.channels.config import ChannelConfig, style_fingerprint, vision_model
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import PlanningError, PublishError
from narratoflow.models import AspectRatio, Part, SceneKind, ScriptSource, Series, VerseRef
from narratoflow.money import format_usd
from narratoflow.publish.checks import CheckResult, PartInputs, Severity, worst
from narratoflow.publish.provenance import AssetRecord, asset_records
from narratoflow.publish.rules import Rules
from narratoflow.publish.technical import MediaFacts
from narratoflow.render.variation import Variation

PUBLISH_SCHEMA_VERSION: Literal[1] = 1
SAVINGS_NOTE = "estimated: images not bought at today's price"
NO_PRICE_NOTE = "no image price"
NOT_CHECKED = "not checked"
EARLIER_RUN = "from an earlier run"

_MODE = 0o644
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)
_REUSED = frozenset({"reused", "bundled"})
_ORDER = {Severity.FAIL: 0, Severity.WARN: 1, Severity.PASS: 2}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|!~])")


def md_cell(s: str) -> str:
    """`s` safe for a Markdown table cell or line: every newline becomes a space and
    Markdown-significant characters (including `|`) are backslash-escaped."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(s.splitlines()))


def _home_free(s: str) -> str:
    """`s` with a leading home-directory prefix shown as ``~``."""
    home = str(Path.home())
    if home not in ("", "/") and (s == home or s.startswith(home + os.sep)):
        return "~" + s[len(home) :]
    return s


def _bare(name: str) -> bool:
    return bool(name) and name not in (".", "..") and not any(c in name for c in "/\\\0")


def _usd(value: Decimal) -> str:
    return format_usd(value)


# -- atomic save / strict load, as `RenderReport` --------------------------------------


def _save(model: BaseModel, path: Path, label: str) -> None:
    text = json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(text)
            tmp.flush()
            os.fchmod(tmp.fileno(), _MODE)
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise PublishError(f"cannot write {label} {path}: {exc}") from exc


def _load[M: BaseModel](cls: type[M], path: Path, label: str) -> M:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PublishError(f"cannot read {label} {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PublishError(f"{label} {path} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise PublishError(f"{label} {path} is nested too deeply") from exc
    found = raw.get("schema_version") if isinstance(raw, dict) else None
    if type(found) is not int or found != PUBLISH_SCHEMA_VERSION:
        raise PublishError(
            f"unsupported {label} schema_version {found!r}; expected {PUBLISH_SCHEMA_VERSION}"
        )
    try:
        return cls.model_validate_json(text, strict=True)
    except ValidationError as exc:
        raise PublishError(f"{label} {path} is invalid:\n{exc}") from exc


# -- models ----------------------------------------------------------------------------


class SceneEntry(BaseModel):
    model_config = _STRICT

    scene_id: str
    kind: SceneKind
    visual: Literal["sketch", "text_card", "verse"]
    idea_key: str | None
    match: Literal["reused", "bundled", "new"] | None
    sketch_id: int | None
    variation: Variation | None
    verse: str | None  # "31:33" or "31:33-35"


class CostSummary(BaseModel):
    model_config = _STRICT

    estimate_usd: Decimal
    spent_usd: Decimal
    by_provider: dict[str, Decimal]
    unattributed_usd: Decimal
    cost_estimated: bool
    savings_usd: Decimal | None
    savings_note: str


class ModelsUsed(BaseModel):
    model_config = _STRICT

    llm: str
    vision: str | None  # configured now, not recorded at generation time
    voice: VoiceUsed | None
    image_models: list[str]
    style_fingerprint: str  # configured now, not recorded at generation time
    style_version_ids: list[int]
    reciters: list[str]


def _verse_label(ref: VerseRef) -> str:
    if ref.ayah_end == ref.ayah_start:
        return f"{ref.surah}:{ref.ayah_start}"
    return f"{ref.surah}:{ref.ayah_start}-{ref.ayah_end}"


def cost_summary(
    part: Part, timeline: PartTimeline, channel: ChannelConfig, prices: PriceTable
) -> CostSummary:
    """Estimate vs actual spend, the per-provider ledger and the savings from reuse."""
    attributed = sum(part.spend.values(), Decimal(0))
    reused = sum(1 for s in part.scenes if s.match in _REUSED)
    style = channel.style
    savings: Decimal | None
    try:
        savings = reused * prices.image(style.image_provider, style.image_model)
        note = SAVINGS_NOTE
    except PlanningError:
        savings, note = None, NO_PRICE_NOTE
    return CostSummary(
        estimate_usd=part.est_cost_usd,
        spent_usd=part.spent_usd,
        by_provider=dict(part.spend),
        unattributed_usd=max(Decimal(0), part.spent_usd - attributed),
        cost_estimated=timeline.cost_estimated,
        savings_usd=savings,
        savings_note=note,
    )


def _scenes(inputs: PartInputs) -> list[SceneEntry]:
    by_id = {s.id: s for s in inputs.part.scenes}
    out: list[SceneEntry] = []
    for drawn in inputs.render.scenes:
        scene = by_id.get(drawn.scene_id)
        ref = scene.verse if scene is not None else None
        out.append(
            SceneEntry(
                scene_id=drawn.scene_id,
                kind=drawn.kind,
                visual=drawn.visual,
                idea_key=scene.idea_key if scene is not None else None,
                match=scene.match if scene is not None else None,
                sketch_id=drawn.sketch_id,
                variation=drawn.variation,
                verse=_verse_label(ref) if ref is not None else None,
            )
        )
    return out


def _models(inputs: PartInputs, assets: list[AssetRecord]) -> ModelsUsed:
    channel = inputs.channel
    vision = vision_model(channel)
    images = {
        "/".join(x for x in (a.provider, a.model) if x)
        for a in assets
        if a.kind == "sketch" and (a.provider or a.model)
    }
    versions = {
        record.style_version_id
        for s in inputs.render.scenes
        if s.sketch_id is not None and (record := inputs.sketches.get(s.sketch_id)) is not None
    }
    reciters = list(dict.fromkeys(r.reciter_id for r in inputs.timeline.recitations))
    return ModelsUsed(
        llm=f"{channel.llm.provider}/{channel.llm.model}",
        vision=f"{vision[0]}/{vision[1]}" if vision is not None else None,
        voice=inputs.timeline.voice,
        image_models=sorted(images),
        style_fingerprint=style_fingerprint(channel.style),
        style_version_ids=sorted(versions),
        reciters=reciters,
    )


class PublishReport(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = PUBLISH_SCHEMA_VERSION
    series_title: str
    channel_slug: str
    part_index: int
    part_count: int
    created_at: AwareDatetime
    rules_version: str
    rules_checked: date
    rules_sha256: str
    rules_origin: str
    video_file: str
    video_sha256: str | None
    duration_ms: int
    width: int | None
    height: int | None
    outputs: list[AspectRatio]
    scenes: list[SceneEntry]
    costs: CostSummary
    models: ModelsUsed
    assets: list[AssetRecord]
    script_source: ScriptSource
    script_ai_transcribed: bool
    checks: list[CheckResult]
    verdict: Severity
    exported: bool
    export_file: str | None
    export_error: str | None = None
    disclosure_text: str
    platform_labels: dict[str, str]
    warnings: list[str]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.verdict != worst(self.checks):
            raise ValueError(
                f"verdict {self.verdict} must equal the worst check result {worst(self.checks)}"
            )
        if self.exported and self.verdict is Severity.FAIL:
            raise ValueError("a FAIL report cannot be exported")
        if self.exported and self.export_error is not None:
            raise ValueError("an exported report cannot carry an export_error")
        if self.exported and self.export_file is None:
            raise ValueError("an exported report must name its export_file")
        for field in ("video_file", "export_file"):
            name = getattr(self, field)
            if name is not None and not _bare(name):
                raise ValueError(f"{field} must be a bare file name, got {name!r}")
        return self

    def save(self, path: Path) -> None:
        """Write atomically: a temp file in the same directory, then `os.replace`."""
        _save(self, path, "publish report")

    @classmethod
    def load(cls, path: Path) -> PublishReport:
        return _load(cls, path, "publish report")

    def to_markdown(self) -> str:
        out: list[str] = [
            f"# {md_cell(self.series_title)} - part {self.part_index} of {self.part_count}",
            "",
        ]
        if self.exported and self.export_file is not None:
            out.append(f"**Verdict: {self.verdict}** - exported as {md_cell(self.export_file)}")
        elif self.verdict is Severity.FAIL:
            out.append(f"**Verdict: {self.verdict}** - export blocked: fix every FAIL below")
        else:
            why = f" ({md_cell(self.export_error)})" if self.export_error else ""
            out.append(
                f"**Verdict: {self.verdict}** - not exported: the export copy failed{why}; "
                "fix the cause and run publish again"
            )
        out += self._md_checks()
        out += self._md_disclosure()
        out += self._md_scenes()
        out += self._md_costs()
        out += self._md_models()
        out += self._md_assets()
        out += self._md_script()
        out += ["", "## Warnings", ""]
        out += [f"- {md_cell(w)}" for w in self.warnings] or ["None."]
        return "\n".join(out) + "\n"

    def _md_checks(self) -> list[str]:
        out = [
            "",
            f"## Publishing check (rules {md_cell(self.rules_version)}, checked "
            f"{self.rules_checked.isoformat()}, {md_cell(self.rules_origin)}, "
            f"sha256 {self.rules_sha256[:12]})",
            "",
            "| Check | Result | Scene | Platform | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in sorted(self.checks, key=lambda c: _ORDER[c.severity]):
            out.append(
                f"| {c.check.value} | {c.severity.value} | {md_cell(c.scene_id or '-')} | "
                f"{md_cell(c.platform or '-')} | {md_cell(c.reason)} |"
            )
        return out

    def _md_disclosure(self) -> list[str]:
        text = self.disclosure_text.rstrip("\n")
        longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        out = ["", "## AI disclosure", "", fence, text, fence, "", "Platform labels:", ""]
        out += [f"- {md_cell(k)}: {md_cell(v)}" for k, v in self.platform_labels.items()]
        if not self.platform_labels:
            out.append("- none")
        return out

    def _md_scenes(self) -> list[str]:
        out = [
            "",
            "## Scenes",
            "",
            "| Scene | Kind | Visual | Idea | Match | Sketch | Verse |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in self.scenes:
            out.append(
                f"| {md_cell(s.scene_id)} | {s.kind.value} | {s.visual} | "
                f"{md_cell(s.idea_key or '-')} | {s.match or '-'} | "
                f"{s.sketch_id if s.sketch_id is not None else '-'} | {s.verse or '-'} |"
            )
        return out

    def _md_costs(self) -> list[str]:
        c = self.costs
        out = ["", "## Costs", ""]
        out += [f"- {md_cell(k)}: {_usd(v)}" for k, v in sorted(c.by_provider.items())]
        out.append(f"- unattributed: {_usd(c.unattributed_usd)}")
        estimated = " (some charges estimated)" if c.cost_estimated else ""
        out.append(f"- estimate {_usd(c.estimate_usd)} vs actual {_usd(c.spent_usd)}{estimated}")
        if c.savings_usd is None:
            out.append(f"- savings from reuse: unknown ({md_cell(c.savings_note)})")
        else:
            out.append(f"- savings from reuse: {_usd(c.savings_usd)} ({md_cell(c.savings_note)})")
        return out

    def _md_models(self) -> list[str]:
        m = self.models
        out = ["", "## Models and voices", "", f"- llm: {md_cell(m.llm)}"]
        out.append(f"- vision (configured): {md_cell(m.vision or 'off')}")
        if m.voice is None:
            out.append("- voice: none")
        else:
            v = m.voice
            name = "/".join(md_cell(x) for x in (v.provider, v.model) if x)
            line = f"- voice: {name}, voice {md_cell(v.voice_id)}, terms {md_cell(v.terms)}"
            if v.fallback_from is not None:
                line += f"; fallback from {md_cell(v.fallback_from)}"
                if v.fallback_reasons:
                    line += ": " + "; ".join(md_cell(r) for r in v.fallback_reasons)
            out.append(line)
        images = ", ".join(md_cell(x) for x in m.image_models) or "none"
        out.append(f"- image models: {images}")
        out.append(f"- style fingerprint (configured): {md_cell(m.style_fingerprint)}")
        versions = ", ".join(str(v) for v in m.style_version_ids) or "none"
        out.append(f"- style versions used: {versions}")
        reciters = ", ".join(md_cell(r) for r in m.reciters) or "none"
        out.append(f"- reciters: {reciters}")
        return out

    def _md_assets(self) -> list[str]:
        out = [
            "",
            "## Assets",
            "",
            "| Kind | Asset | Source | Licence | Attribution | Provider/model | Scenes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for a in self.assets:
            maker = "/".join(x for x in (a.provider, a.model) if x) or "-"
            out.append(
                f"| {a.kind} | {md_cell(a.asset_id)} | {md_cell(a.source or '-')} | "
                f"{md_cell(a.license or '-')} | {md_cell(a.attribution or '-')} | "
                f"{md_cell(maker)} | {md_cell(', '.join(a.scene_ids) or '-')} |"
            )
        return out

    def _md_script(self) -> list[str]:
        s = self.script_source
        out = ["", "## Script source", ""]
        out.append(f"- work: {md_cell(s.work or 'not recorded')}")
        out.append(f"- author: {md_cell(s.author or 'not recorded')}")
        if s.edition:
            out.append(f"- edition: {md_cell(s.edition)}")
        out.append(f"- rights: {s.rights.value}")
        if s.permission_note:
            out.append(f"- permission: {md_cell(s.permission_note)}")
        out.append(f"- AI transcribed: {'yes' if self.script_ai_transcribed else 'no'}")
        return out


def build_report(
    inputs: PartInputs,
    *,
    checks: list[CheckResult],
    facts: MediaFacts | None,
    disclosure: str,
    labels: dict[str, str],
    rules: Rules,
    prices: PriceTable,
    exported: bool,
    export_file: str | None,
    now: datetime,
    export_error: str | None = None,
) -> PublishReport:
    """The production report for one part; `style_fingerprint` may raise
    `ChannelConfigError` (`build_publish` calls it in preflight)."""
    series, part, render = inputs.series, inputs.part, inputs.render
    assets = asset_records(inputs)
    probe = facts.probe if facts is not None else None
    duration = probe.video_ms if probe is not None and probe.video_ms is not None else None
    return PublishReport(
        series_title=series.title,
        channel_slug=series.channel_slug,
        part_index=part.index,
        part_count=part.count,
        created_at=now,
        rules_version=rules.file.version,
        rules_checked=rules.file.checked,
        rules_sha256=rules.sha256,
        rules_origin=_home_free(rules.origin),
        video_file=render.video_file,
        video_sha256=facts.sha256 if facts is not None else None,
        duration_ms=duration if duration is not None else render.duration_ms,
        width=probe.width if probe is not None else None,
        height=probe.height if probe is not None else None,
        outputs=[AspectRatio.PORTRAIT],  # the only format v1 renders (spec 8)
        scenes=_scenes(inputs),
        costs=cost_summary(part, inputs.timeline, inputs.channel, prices),
        models=_models(inputs, assets),
        assets=assets,
        script_source=series.source,
        script_ai_transcribed=series.source.ai_transcribed,
        checks=list(checks),
        verdict=worst(checks),
        exported=exported,
        export_file=export_file,
        export_error=export_error,
        disclosure_text=disclosure,
        platform_labels=dict(labels),
        warnings=[*render.warnings, *inputs.timeline.warnings],
    )


# -- the series summary ----------------------------------------------------------------


class PartSummary(BaseModel):
    model_config = _STRICT

    index: int
    verdict: Severity | None
    exported: bool
    duration_ms: int | None
    spent_usd: Decimal
    fails: int
    warns: int
    checked_at: AwareDatetime | None
    note: str


class SeriesSummary(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = PUBLISH_SCHEMA_VERSION
    series_title: str
    channel_slug: str
    created_at: AwareDatetime
    rules_version: str
    parts: list[PartSummary]
    estimate_usd: Decimal
    spent_usd: Decimal
    planning_estimate_usd: Decimal
    savings_usd: Decimal | None

    def save(self, path: Path) -> None:
        """Write atomically: a temp file in the same directory, then `os.replace`."""
        _save(self, path, "series summary")

    @classmethod
    def load(cls, path: Path) -> SeriesSummary:
        return _load(cls, path, "series summary")

    def to_markdown(self) -> str:
        out = [
            f"# {md_cell(self.series_title)} - publishing summary",
            "",
            f"Rules {md_cell(self.rules_version)}; summary made {self.created_at.isoformat()}.",
            "",
            "| Part | Verdict | Exported | Duration | Spent | FAIL | WARN | Checked | Note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for p in self.parts:
            duration = f"{p.duration_ms / 1000:.1f} s" if p.duration_ms is not None else "-"
            checked = p.checked_at.isoformat() if p.checked_at is not None else "-"
            out.append(
                f"| {p.index} | {p.verdict.value if p.verdict is not None else '-'} | "
                f"{'yes' if p.exported else 'no'} | {duration} | {_usd(p.spent_usd)} | "
                f"{p.fails} | {p.warns} | {checked} | {md_cell(p.note) or '-'} |"
            )
        savings = _usd(self.savings_usd) if self.savings_usd is not None else "unknown"
        out += [
            "",
            "## Costs",
            "",
            f"- planning estimate: {_usd(self.planning_estimate_usd)}",
            f"- production estimate: {_usd(self.estimate_usd)}",
            f"- spent: {_usd(self.spent_usd)}",
            f"- savings from reuse (estimated): {savings}",
        ]
        return "\n".join(out) + "\n"


def _part_summary(part: Part, report: PublishReport | None, current: set[int]) -> PartSummary:
    if report is None:
        return PartSummary(
            index=part.index,
            verdict=None,
            exported=False,
            duration_ms=None,
            spent_usd=part.spent_usd,
            fails=0,
            warns=0,
            checked_at=None,
            note=NOT_CHECKED,
        )
    return PartSummary(
        index=part.index,
        verdict=report.verdict,
        exported=report.exported,
        duration_ms=report.duration_ms,
        spent_usd=part.spent_usd,
        fails=sum(1 for c in report.checks if c.severity is Severity.FAIL),
        warns=sum(1 for c in report.checks if c.severity is Severity.WARN),
        checked_at=report.created_at,
        note="" if part.index in current else EARLIER_RUN,
    )


def build_summary(
    series: Series,
    reports: Mapping[int, PublishReport | None],
    *,
    current: set[int],
    rules: Rules,
    now: datetime,
) -> SeriesSummary:
    """One line per series part. A report is "from an earlier run" when its part is not
    in `current` (the parts checked this run); `now` only stamps `created_at`."""
    priced = [
        r.costs.savings_usd
        for r in reports.values()
        if r is not None and r.costs.savings_usd is not None
    ]
    return SeriesSummary(
        series_title=series.title,
        channel_slug=series.channel_slug,
        created_at=now,
        rules_version=rules.file.version,
        parts=[_part_summary(p, reports.get(p.index), current) for p in series.parts],
        estimate_usd=sum((p.est_cost_usd for p in series.parts), Decimal(0)),
        spent_usd=sum((p.spent_usd for p in series.parts), Decimal(0)),
        planning_estimate_usd=series.est_planning_cost_usd,
        savings_usd=sum(priced, Decimal(0)) if priced else None,
    )
