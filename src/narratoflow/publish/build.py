"""Publish build: every selected part's checks, its production report and its export.

Everything that is an error rather than a check result is raised before any check
runs, in this order: the series is approved and belongs to the channel, every part
index exists, the export directory is outside the render directory (and does not
contain it), the platform rules load and the channel's style fingerprint computes,
each selected part has its render report and timeline, and the tools are found.

Then, per part: the sketch records, the MP4's facts (`inspect_video`; a video that
cannot be read is a TECHNICAL FAIL, and the run goes on), every check in `CheckId`
order and the report. A part with no FAIL is exported: the MP4 is copied into
`export_dir`, its sha256 is checked against the checked file's, and it is renamed into
place before ``part-NN.disclosure.txt`` and ``part-NN.publish.md`` are written. A
FAILing part is never exported, and any export left by an earlier run is removed;
there is no override. The part's ``part-NN.publish.json``/``.md`` are saved in
`render_dir` last.

A copy failure or a hash mismatch saves that part's report as not exported and then
raises `PublishError`; the series summary is not written. Otherwise the summary
(``series.publish.json``/``.md``, the ``.md`` also in `export_dir` when it exists)
covers every series part, reading the reports of earlier runs for unselected parts.

Publish never raises for a FAIL (the CLI does), never modifies the series, never
builds an LLM client and never spends. The clock is read exactly once per run.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from narratoflow.audio.timeline import PartTimeline
from narratoflow.channels.config import ChannelConfig, style_fingerprint
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.errors import AudioError, PlanningError, PublishError, RenderError
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.library.schemas import SketchRecord
from narratoflow.models import Part, Series
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity, worst
from narratoflow.publish.consistency import render_current_checks, sketch_checks
from narratoflow.publish.content import content_checks
from narratoflow.publish.disclosure import disclosure_checks, disclosure_text, platform_labels
from narratoflow.publish.provenance import provenance_checks, script_checks
from narratoflow.publish.report import PublishReport, build_report, build_summary
from narratoflow.publish.rules import Rules, default_rules, rules_checks
from narratoflow.publish.technical import MediaFacts, inspect_video, technical_checks
from narratoflow.publish.verse import verse_checks
from narratoflow.quran.corpus import QuranCorpus, default_corpus
from narratoflow.quran.reciters import ReciterIndex, default_reciters
from narratoflow.render.ffmpeg import RenderTools, find_render_tools
from narratoflow.render.report import RenderReport
from narratoflow.settings import Settings
from narratoflow.sketching.generate import require_series_channel

logger = logging.getLogger(__name__)

SUMMARY_STEM = "series.publish"
_MODE = 0o644
_CHUNK = 1 << 20
_POINTER = re.compile(r"\s*@\s*0x[0-9a-fA-F]+")
_ABS_PATH = re.compile(r"(?<![\w.])/(?:[^\s/'\"]+/)*([^\s/'\":]+)")
_MAX_DETAIL = 200


@dataclass(frozen=True)
class PartPublish:
    index: int
    report: PublishReport
    report_path: Path
    export_path: Path | None


@dataclass
class PublishResult:
    parts: list[PartPublish]
    summary_path: Path
    blocked: list[int]


def _utc_now() -> datetime:  # pragma: no cover - tests inject the clock
    return datetime.now(UTC)


def _stem(index: int) -> str:
    return f"part-{index:02d}"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a temp file in the same directory, then `os.replace`."""
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
        raise PublishError(f"cannot write {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _short_error(exc: Exception, video: Path) -> str:
    """`exc` as one short line for a report: what failed and the first meaningful line
    of its detail, with absolute paths cut to the file name and `@ 0x...` pointers
    dropped (a tool's stderr carries both)."""
    text = _ABS_PATH.sub(r"\1", _POINTER.sub("", str(exc).replace(str(video), video.name)))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    head = lines[0].rstrip(":").strip()
    if len(lines) == 1:
        return head[:_MAX_DETAIL]
    return f"{head}: {lines[1][:_MAX_DETAIL]}"


def _check_dirs(render_dir: Path, export_dir: Path) -> None:
    """Refuse an export directory that is, contains or is inside the render directory:
    exporting there would overwrite, or on a FAIL delete, the rendered files."""
    render, export = render_dir.resolve(), export_dir.resolve()
    if render == export or export.is_relative_to(render) or render.is_relative_to(export):
        raise PublishError(
            f"the export directory {export_dir} must be outside the render directory "
            f"{render_dir} (and not contain it)"
        )


def _selected(series: Series, parts: Sequence[int] | None) -> list[Part]:
    if parts is None:
        return list(series.parts)
    known = {p.index for p in series.parts}
    for n in parts:
        if n not in known:
            raise PlanningError(f"no part {n}; the series has parts {sorted(known)}")
    wanted = set(parts)
    return [p for p in series.parts if p.index in wanted]


@dataclass(frozen=True)
class _Loaded:
    part: Part
    render: RenderReport
    timeline: PartTimeline


def _load_part(part: Part, *, render_dir: Path, audio_dir: Path) -> _Loaded:
    n = part.index
    report_path = render_dir / f"{_stem(n)}.render.json"
    if not report_path.is_file():
        raise PublishError(f"part {n} has no render report; run narratoflow render")
    render = RenderReport.load(report_path)
    timeline_path = audio_dir / render.timeline_file
    if not timeline_path.is_file():
        raise PublishError(
            f"part {n}: {render.timeline_file} is missing; run narratoflow audio build"
        )
    return _Loaded(part, render, PartTimeline.load(timeline_path))


def _earlier_report(series: Series, part: Part, render_dir: Path) -> PublishReport | None:
    """An unselected part's report from an earlier run, if it is readable and is this
    series' (same title, channel and part)."""
    path = render_dir / f"{_stem(part.index)}.publish.json"
    if not path.is_file():
        return None
    try:
        report = PublishReport.load(path)
    except PublishError:
        return None
    named = (report.series_title, report.channel_slug, report.part_index)
    if named != (series.title, series.channel_slug, part.index):
        return None
    return report


@dataclass
class _Run:
    series: Series
    channel: ChannelConfig
    settings: Settings
    tools: RenderTools
    rules: Rules
    prices: PriceTable
    reciters: ReciterIndex
    library: SketchLibrary | None
    corpus: QuranCorpus | None
    render_dir: Path
    audio_dir: Path
    export_dir: Path
    now: datetime

    def get_library(self) -> SketchLibrary:
        if self.library is None:
            self.library = default_library(self.settings)
        return self.library

    def get_corpus(self) -> QuranCorpus:
        if self.corpus is None:
            self.corpus = default_corpus(self.settings)
        return self.corpus

    def sketches(self, render: RenderReport) -> dict[int, SketchRecord | None]:
        ids = [s.sketch_id for s in render.scenes if s.sketch_id is not None]
        if not ids:
            return {}
        library = self.get_library()
        return {sid: library.get_sketch(sid) for sid in ids}

    def facts(self, video: Path) -> tuple[MediaFacts | None, CheckResult | None]:
        """The MP4's facts, or None plus a TECHNICAL FAIL when it cannot be read."""
        if not video.is_file():
            return None, None
        try:
            return inspect_video(self.tools, video), None
        except (RenderError, AudioError) as exc:
            reason = f"{video.name} cannot be read: {_short_error(exc, video)}"
            return None, CheckResult(check=CheckId.TECHNICAL, severity=Severity.FAIL, reason=reason)

    def checks(
        self, inputs: PartInputs, facts: MediaFacts | None, unreadable: CheckResult | None
    ) -> tuple[list[CheckResult], str]:
        text = disclosure_text(inputs)
        checks = [
            *rules_checks(self.rules, self.now.date()),
            *render_current_checks(inputs),
            *provenance_checks(inputs, self.rules, self.reciters),
            *script_checks(inputs),
            *sketch_checks(inputs),
            *verse_checks(inputs, self.get_corpus),
            *disclosure_checks(inputs, text),
            *content_checks(inputs.part, self.rules),
        ]
        if unreadable is not None:
            checks.append(unreadable)
        else:
            checks += technical_checks(inputs, facts, self.rules)
        return checks, text

    def export_files(self, n: int) -> list[Path]:
        stem = _stem(n)
        return [
            self.export_dir / f"{stem}.mp4",
            self.export_dir / f"{stem}.disclosure.txt",
            self.export_dir / f"{stem}.publish.md",
        ]

    def remove_exports(self, n: int) -> None:
        for path in self.export_files(n):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise PublishError(f"part {n}: cannot remove the old export {path}: {exc}") from exc

    def copy_video(self, n: int, video: Path, facts: MediaFacts) -> Path:
        """Copy `video` into `export_dir` and rename it into place only if the copy's
        sha256 equals the checked file's."""
        target = self.export_files(n)[0]
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PublishError(
                f"part {n}: cannot create the export directory {self.export_dir}: {exc}"
            ) from exc
        tmp: Path | None = None
        try:
            fd, name = tempfile.mkstemp(dir=self.export_dir, prefix=f".{target.name}.")
            tmp = Path(name)
            with os.fdopen(fd, "wb") as out, video.open("rb") as src:
                shutil.copyfileobj(src, out, _CHUNK)
                out.flush()
                os.fchmod(out.fileno(), _MODE)
                os.fsync(out.fileno())
            copied = _file_sha256(tmp)
            if copied != facts.sha256:
                raise PublishError(
                    f"part {n}: the exported copy's sha256 {copied} differs from the checked "
                    f"file's {facts.sha256}; nothing was exported"
                )
            os.replace(tmp, target)
        except OSError as exc:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            raise PublishError(f"part {n}: cannot copy {video.name} to {target}: {exc}") from exc
        except PublishError:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            raise
        return target

    def save(self, report: PublishReport, n: int) -> Path:
        stem = _stem(n)
        path = self.render_dir / f"{stem}.publish.json"
        report.save(path)
        _write_text(self.render_dir / f"{stem}.publish.md", report.to_markdown())
        return path

    def part(self, loaded: _Loaded) -> PartPublish:
        part, render = loaded.part, loaded.render
        n = part.index
        inputs = PartInputs(
            series=self.series,
            part=part,
            channel=self.channel,
            render=render,
            timeline=loaded.timeline,
            sketches=self.sketches(render),
            render_dir=self.render_dir,
        )
        video = self.render_dir / render.video_file
        facts, unreadable = self.facts(video)
        checks, text = self.checks(inputs, facts, unreadable)
        labels = platform_labels(self.rules)
        export_name = f"{_stem(n)}.mp4"

        def report(exported: bool) -> PublishReport:
            return build_report(
                inputs,
                checks=checks,
                facts=facts,
                disclosure=text,
                labels=labels,
                rules=self.rules,
                prices=self.prices,
                exported=exported,
                export_file=export_name if exported else None,
                now=self.now,
            )

        if worst(checks) is Severity.FAIL:
            self.remove_exports(n)
            blocked = report(False)
            return PartPublish(n, blocked, self.save(blocked, n), None)
        if facts is None:  # pragma: no cover - technical_checks FAILs without facts
            raise PublishError(f"part {n}: no video facts")
        exported = report(True)
        try:
            target = self.copy_video(n, video, facts)
            _, disclosure_path, md_path = self.export_files(n)
            _write_text(disclosure_path, exported.disclosure_text)
            _write_text(md_path, exported.to_markdown())
        except PublishError:
            self.save(report(False), n)
            for path in self.export_files(n):  # best effort: the error above is what counts
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            raise
        return PartPublish(n, exported, self.save(exported, n), target)


def build_publish(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    render_dir: Path,
    audio_dir: Path,
    export_dir: Path,
    parts: Sequence[int] | None = None,
    tools: RenderTools | None = None,
    library: SketchLibrary | None = None,
    corpus: QuranCorpus | None = None,
    reciters: ReciterIndex | None = None,
    rules: Rules | None = None,
    prices: PriceTable | None = None,
    now: Callable[[], datetime] | None = None,
) -> PublishResult:
    """Check and export every selected part (`parts`: part indexes; None = all).

    A FAIL blocks only that part's export and is reported in `blocked`; it is never
    raised. A copy failure or an export hash mismatch raises `PublishError` after that
    part's report is saved (as not exported), and the series summary is not written.
    """
    if not series.approved:
        raise PlanningError("approve the series first")
    require_series_channel(series, channel)
    selected = _selected(series, parts)
    _check_dirs(render_dir, export_dir)
    rules = rules if rules is not None else default_rules(settings)
    style_fingerprint(channel.style)  # a ChannelConfigError surfaces before any check
    loaded = [_load_part(p, render_dir=render_dir, audio_dir=audio_dir) for p in selected]
    tools = tools if tools is not None else find_render_tools(settings)
    run = _Run(
        series=series,
        channel=channel,
        settings=settings,
        tools=tools,
        rules=rules,
        prices=prices if prices is not None else default_prices(settings),
        reciters=reciters if reciters is not None else default_reciters(),
        library=library,
        corpus=corpus,
        render_dir=render_dir,
        audio_dir=audio_dir,
        export_dir=export_dir,
        now=(now or _utc_now)(),
    )

    done: list[PartPublish] = []
    for item in loaded:
        published = run.part(item)
        done.append(published)
        logger.info(
            "publish part %d: %s%s",
            published.index,
            published.report.verdict.value,
            "" if published.export_path is not None else " (not exported)",
        )

    reports: dict[int, PublishReport | None] = {p.index: p.report for p in done}
    for part in series.parts:
        if part.index not in reports:
            reports[part.index] = _earlier_report(series, part, render_dir)
    summary = build_summary(
        series, reports, current={p.index for p in selected}, rules=rules, now=run.now
    )
    summary_path = render_dir / f"{SUMMARY_STEM}.json"
    summary.save(summary_path)
    md = summary.to_markdown()
    _write_text(render_dir / f"{SUMMARY_STEM}.md", md)
    if export_dir.is_dir():
        _write_text(export_dir / f"{SUMMARY_STEM}.md", md)
    blocked = [p.index for p in done if p.report.verdict is Severity.FAIL]
    return PublishResult(parts=done, summary_path=summary_path, blocked=blocked)
