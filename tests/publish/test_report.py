"""Per-part production report and series summary."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from fixtures.parts import GENERATED_PROVENANCE, add_sketch, make_series, narr, verse
from fixtures.publish import PACKAGED_RULES, make_inputs

from narratoflow.audio.io import Loudness
from narratoflow.audio.timeline import VoiceUsed
from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import PublishError
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import AspectRatio, RightsBasis, ScriptSource, Series
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.report import (
    PublishReport,
    SeriesSummary,
    build_report,
    build_summary,
    cost_summary,
    md_cell,
)
from narratoflow.publish.rules import Rules, load_rules
from narratoflow.publish.technical import MediaFacts
from narratoflow.render.ffmpeg import MediaProbe

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
PRICES = PriceTable({"image": {"openai/gpt-image-1": (Decimal("0.04"),)}})
FACTS = MediaFacts(
    probe=MediaProbe(
        width=1080,
        height=1920,
        fps=Fraction(30),
        frames=60,
        video_ms=2000,
        audio_ms=2000,
        video_codec="h264",
        pix_fmt="yuv420p",
        audio_codec="aac",
        audio_rate=48000,
    ),
    loudness=Loudness(-14.0, -1.5, 3.0, -24.0, 0.0),
    sha256="ab" * 32,
    size_bytes=10,
)


def _rules() -> Rules:
    return replace(load_rules(PACKAGED_RULES), origin="packaged")


def _check(severity: Severity, reason: str, **kw: object) -> CheckResult:
    return CheckResult(check=CheckId.TECHNICAL, severity=severity, reason=reason, **kw)  # type: ignore[arg-type]


def _report(
    inputs: PartInputs,
    checks: list[CheckResult] | None = None,
    *,
    exported: bool = False,
    prices: PriceTable = PRICES,
    facts: MediaFacts | None = FACTS,
) -> PublishReport:
    return build_report(
        inputs,
        checks=checks if checks is not None else [_check(Severity.PASS, "ok")],
        facts=facts,
        disclosure="AI disclosure:\n- voice\n",
        labels={"youtube": "Altered or synthetic content"},
        rules=_rules(),
        prices=prices,
        exported=exported,
        export_file="part-01.mp4" if exported else None,
        now=NOW,
    )


def _inputs(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> PartInputs:
    return make_inputs(tmp_path, library, channel)


def test_report_roundtrip_and_strict(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    inputs = replace(
        inputs,
        part=inputs.part.model_copy(
            update={
                "spent_usd": Decimal("0.10"),
                "spend": {"image openai/gpt-image-1": Decimal("0.04")},
            }
        ),
    )
    report = _report(inputs, exported=True)
    assert report.outputs == [AspectRatio.PORTRAIT]
    assert report.video_sha256 == FACTS.sha256
    assert (report.width, report.height, report.duration_ms) == (1080, 1920, 2000)
    assert report.rules_version == _rules().file.version
    assert report.assets and report.scenes[0].scene_id == "s1"
    path = tmp_path / "part-01.publish.json"
    report.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["costs"]["spent_usd"] == "0.10"
    assert raw["costs"]["by_provider"] == {"image openai/gpt-image-1": "0.04"}
    assert PublishReport.load(path) == report
    assert not list(tmp_path.glob(".part-01.publish.json.*"))

    raw["extra"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishError, match="invalid"):
        PublishReport.load(path)
    del raw["extra"]
    raw["part_index"] = "1"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishError, match="invalid"):
        PublishReport.load(path)
    raw["part_index"] = 1
    raw["schema_version"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishError, match="schema_version"):
        PublishReport.load(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(PublishError, match="not valid JSON"):
        PublishReport.load(path)
    with pytest.raises(PublishError, match="cannot read"):
        PublishReport.load(tmp_path / "missing.json")
    with pytest.raises(PublishError, match="cannot write"):
        report.save(tmp_path / "no-dir" / "x.json")


def test_exported_fail_report_rejected(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    with pytest.raises(ValueError, match="FAIL"):
        _report(inputs, [_check(Severity.FAIL, "bad")], exported=True)
    blocked = _report(inputs, [_check(Severity.FAIL, "bad")])
    assert blocked.verdict is Severity.FAIL and not blocked.exported
    data = blocked.model_dump()
    data["exported"] = True
    data["export_file"] = "part-01.mp4"
    with pytest.raises(ValueError, match="FAIL"):
        PublishReport.model_validate(data)
    data["exported"] = False
    data["export_file"] = "/Users/x/part-01.mp4"
    with pytest.raises(ValueError, match="bare file name"):
        PublishReport.model_validate(data)


def test_verdict_must_equal_worst(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    report = _report(inputs, [_check(Severity.PASS, "ok"), _check(Severity.WARN, "meh")])
    assert report.verdict is Severity.WARN
    data = report.model_dump()
    data["verdict"] = Severity.PASS
    with pytest.raises(ValueError, match="verdict"):
        PublishReport.model_validate(data)


def test_cost_summary_unattributed_and_savings(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    scenes = [
        inputs.part.scenes[0].model_copy(update={"match": "reused"}),
        inputs.part.scenes[1].model_copy(update={"match": "bundled"}),
    ]
    part = inputs.part.model_copy(
        update={
            "scenes": scenes,
            "est_cost_usd": Decimal("0.50"),
            "spent_usd": Decimal("0.35"),
            "spend": {"image openai/gpt-image-1": Decimal("0.08"), "llm x/y": Decimal("0.07")},
        }
    )
    costs = cost_summary(part, inputs.timeline, channel, PRICES)
    assert costs.estimate_usd == Decimal("0.50")
    assert costs.spent_usd == Decimal("0.35")
    assert costs.unattributed_usd == Decimal("0.20")
    assert costs.savings_usd == Decimal("0.08")
    assert costs.savings_note == "estimated: images not bought at today's price"
    assert costs.cost_estimated is inputs.timeline.cost_estimated
    over = part.model_copy(update={"spent_usd": Decimal("0.01")})
    assert cost_summary(over, inputs.timeline, channel, PRICES).unattributed_usd == 0


def test_savings_none_without_price(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    costs = cost_summary(inputs.part, inputs.timeline, channel, PriceTable({}))
    assert costs.savings_usd is None
    assert costs.savings_note == "no image price"
    report = _report(inputs, prices=PriceTable({}))
    assert "no image price" in report.to_markdown()


def test_markdown_lists_every_fail_and_warn_reason_and_disclosure(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    checks = [
        _check(Severity.PASS, "all good"),
        _check(Severity.WARN, "loud-ish", platform="youtube"),
        _check(Severity.FAIL, "too long", scene_id="s1", platform="tiktok"),
        _check(Severity.FAIL, "second fail"),
    ]
    md = _report(inputs, checks).to_markdown()
    lines = md.splitlines()
    assert lines[0] == "# Scenes - part 1 of 1"
    assert "**Verdict: FAIL** - export blocked: fix every FAIL below" in md
    rows = [ln for ln in lines if ln.startswith("| technical |")]
    assert [r.split(" | ")[1] for r in rows] == ["FAIL", "FAIL", "WARN", "PASS"]
    for reason in ("too long", "second fail", "loud-ish", "all good"):
        assert any(reason in r for r in rows)
    assert "| technical | FAIL | s1 | tiktok | too long |" in md
    rules = _rules()
    assert (
        f"## Publishing check (rules {rules.file.version}, checked "
        f"{rules.file.checked.isoformat()}, packaged, sha256 {rules.sha256[:12]})"
    ) in md
    assert "## AI disclosure\n\n```\nAI disclosure:\n- voice\n```\n" in md
    assert "Platform labels:" in md and "- youtube: Altered or synthetic content" in md
    order = [
        "## Publishing check",
        "## AI disclosure",
        "## Scenes",
        "## Costs",
        "## Models and voices",
        "## Assets",
        "## Script source",
        "## Warnings",
    ]
    assert [md.index(h) for h in order] == sorted(md.index(h) for h in order)
    assert "vision (configured)" in md and "style fingerprint (configured)" in md
    assert "openai-terms" in md  # the sketch's licence in the assets section
    md_ok = _report(inputs, exported=True).to_markdown()
    assert "**Verdict: PASS** - exported as part-01.mp4" in md_ok
    assert str(Path.home()) not in md and str(Path.home()) not in md_ok


def test_markdown_escapes_pipes_in_scene_text(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    assert md_cell("a|b\nc") == "a\\|b c"
    assert md_cell("x\r\ny\u2028z") == "x y z"
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    series = make_series([narr("s|1", "alpha beta", sketch=sid), narr("s2", "delta")])
    first = series.parts[0]
    scenes = [first.scenes[0].model_copy(update={"idea_key": "idea|with\npipe"}), first.scenes[1]]
    series = series.model_copy(
        update={
            "title": "T|itle\n# injected",
            "parts": [first.model_copy(update={"scenes": scenes})],
        }
    )
    inputs = make_inputs(tmp_path, library, channel, series=series)
    md = _report(inputs, [_check(Severity.WARN, "pipe | in\nreason", scene_id="s|1")]).to_markdown()
    assert md.splitlines()[0] == "# T\\|itle \\# injected - part 1 of 1"
    assert "\n# injected" not in md
    assert "idea\\|with pipe" in md
    assert "| technical | WARN | s\\|1 | - | pipe \\| in reason |" in md
    scene_rows = [ln for ln in md.splitlines() if ln.startswith("| s\\|1 |")]
    assert len(scene_rows) == 1


def test_models_used_shows_voice_fallback(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    voice = VoiceUsed(
        provider="edge",
        model="",
        voice_id="ar-SA-HamedNeural",
        terms="edge-tts-terms",
        fallback_from="elevenlabs",
        fallback_reasons=["quota exceeded"],
    )
    inputs = replace(inputs, timeline=inputs.timeline.model_copy(update={"voice": voice}))
    report = _report(inputs)
    models = report.models
    assert models.voice == voice
    assert models.llm == "ollama/qwen2.5"
    assert models.vision is None
    assert models.image_models == ["openai/gpt-image-1"]
    assert len(models.style_fingerprint) == 16
    [record] = [r for r in inputs.sketches.values() if r is not None]
    assert models.style_version_ids == [record.style_version_id]
    md = report.to_markdown()
    assert "fallback from elevenlabs" in md
    assert "quota exceeded" in md


def _summary_series(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> Series:
    base = make_inputs(tmp_path, library, channel).series
    parts = [
        base.parts[0].model_copy(
            update={
                "index": i,
                "count": 3,
                "est_cost_usd": Decimal("0.10"),
                "spent_usd": Decimal(i),
            }
        )
        for i in (1, 2, 3)
    ]
    return base.model_copy(update={"parts": parts, "est_planning_cost_usd": Decimal("0.02")})


def _part_report(
    inputs: PartInputs, index: int, checks: list[CheckResult], prices: PriceTable
) -> PublishReport:
    report = _report(inputs, checks, prices=prices)
    return report.model_copy(
        update={"part_index": index, "created_at": datetime(2026, 1, 1, tzinfo=UTC)}
    )


def test_summary_marks_unchecked_and_earlier_parts(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    series = _summary_series(tmp_path, library, channel)
    r1 = _part_report(inputs, 1, [_check(Severity.FAIL, "x"), _check(Severity.WARN, "y")], PRICES)
    r2 = _part_report(inputs, 2, [_check(Severity.WARN, "y")], PRICES)
    summary = build_summary(series, {1: r1, 2: r2, 3: None}, current={1}, rules=_rules(), now=NOW)
    assert summary.created_at == NOW
    assert summary.rules_version == _rules().file.version
    p1, p2, p3 = summary.parts
    assert (p1.verdict, p1.fails, p1.warns, p1.note) == (Severity.FAIL, 1, 1, "")
    assert (p2.verdict, p2.warns, p2.note) == (Severity.WARN, 1, "from an earlier run")
    assert (p3.verdict, p3.exported, p3.note) == (None, False, "not checked")
    assert p3.duration_ms is None and p3.checked_at is None
    assert p1.checked_at == r1.created_at
    assert summary.spent_usd == Decimal(6)
    assert summary.estimate_usd == Decimal("0.30")
    assert summary.planning_estimate_usd == Decimal("0.02")
    path = tmp_path / "series.publish.json"
    summary.save(path)
    assert SeriesSummary.load(path) == summary
    md = summary.to_markdown()
    assert "from an earlier run" in md and "not checked" in md
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 9
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishError, match="schema_version"):
        SeriesSummary.load(path)


def test_summary_savings_sums_only_priced_reports(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    scenes = [s.model_copy(update={"match": "reused"}) for s in inputs.part.scenes]
    inputs = replace(inputs, part=inputs.part.model_copy(update={"scenes": scenes}))
    series = _summary_series(tmp_path, library, channel)
    ok = [_check(Severity.PASS, "ok")]
    r1 = _part_report(inputs, 1, ok, PRICES)
    r2 = _part_report(inputs, 2, ok, PriceTable({}))
    assert r1.costs.savings_usd == Decimal("0.08") and r2.costs.savings_usd is None
    rules = _rules()
    summary = build_summary(series, {1: r1, 2: r2, 3: None}, current={1, 2}, rules=rules, now=NOW)
    assert summary.savings_usd == Decimal("0.08")
    none = build_summary(series, {1: r2, 2: None, 3: None}, current={1}, rules=rules, now=NOW)
    assert none.savings_usd is None
    assert "savings" in none.to_markdown().lower()


def test_rules_origin_hides_home(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    inputs = _inputs(tmp_path, library, channel)
    home_rules = Rules(
        file=_rules().file, sha256="0" * 64, origin=str(Path.home() / "data" / "rules.toml")
    )
    report = build_report(
        inputs,
        checks=[_check(Severity.PASS, "ok")],
        facts=None,
        disclosure="d\n",
        labels={},
        rules=home_rules,
        prices=PRICES,
        exported=False,
        export_file=None,
        now=NOW,
    )
    assert report.rules_origin == "~/data/rules.toml"
    assert (report.width, report.video_sha256) == (None, None)
    assert report.duration_ms == inputs.render.duration_ms
    assert report.rules_checked == date(2026, 11, 11)


def test_verse_part_report_sections(
    tmp_path: Path, library: SketchLibrary, channel: ChannelConfig
) -> None:
    series = make_series([verse("v1")])
    source = ScriptSource(
        author="a", work="w", edition="2nd", rights=RightsBasis.PERMISSION, permission_note="ok"
    )
    series = series.model_copy(update={"source": source})
    inputs = make_inputs(tmp_path, library, channel, series=series)
    [scene] = inputs.part.scenes
    assert scene.verse is not None
    wide = scene.model_copy(update={"verse": scene.verse.model_copy(update={"ayah_end": 35})})
    ranged = replace(inputs, part=inputs.part.model_copy(update={"scenes": [wide]}))
    assert _report(inputs).scenes[0].verse == "31:33"
    report = build_report(
        ranged,
        checks=[_check(Severity.PASS, "ok")],
        facts=None,
        disclosure="d ``` x\n",
        labels={},
        rules=_rules(),
        prices=PRICES,
        exported=False,
        export_file=None,
        now=NOW,
    )
    assert report.scenes[0].verse == "31:33-35"
    md = report.to_markdown()
    assert "- voice: none" in md
    assert "Platform labels:\n\n- none" in md
    assert "\n````\nd ``` x\n````\n" in md
    assert "- edition: 2nd" in md and "- permission: ok" in md
    data = report.model_dump()
    data["exported"] = True
    with pytest.raises(ValueError, match="export_file"):
        PublishReport.model_validate(data)
