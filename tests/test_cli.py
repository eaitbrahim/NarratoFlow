import json
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from narratoflow import cli
from narratoflow.audio.build import AudioBuildResult, AudioInterruptedError, PartAudio
from narratoflow.audio.timeline import PartTimeline, SceneTiming, TimingSource, VoiceUsed
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
from narratoflow.errors import BudgetExceededError, DialectUnavailableError
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import (
    Language,
    Narration,
    Part,
    Project,
    RightsBasis,
    Scene,
    SceneKind,
    ScriptSource,
    Series,
    VisualSource,
)
from narratoflow.settings import Settings, get_settings
from narratoflow.sketching.generate import GenerationInterruptedError, GenerationResult
from narratoflow.text.dialect import INSTALL_HINT
from narratoflow.visuals.bundled import BundledIcon, BundledIcons

runner = CliRunner()
INPUTS = Path(__file__).parent / "fixtures" / "input"


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Run the real entry point in-process and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["narratoflow", *args])
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_check_input_ok() -> None:
    for name, lang in (("clean_ar.txt", "ar"), ("clean_en.txt", "en")):
        result = runner.invoke(cli.app, ["check-input", str(INPUTS / name), "--language", lang])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output


def test_check_input_corrupt_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_main(monkeypatch, "check-input", str(INPUTS / "corrupt_ar.txt"), "--language", "ar")
    captured = capsys.readouterr()
    assert code == 1
    assert "STRETCHED_LETTERS" in captured.err
    assert "Traceback" not in captured.err


def test_channel_check_missing_key_exit_1(
    channel_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_main(monkeypatch, "channel", "check", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "NARRATOFLOW_IMAGE_API_KEY" in captured.err
    assert "Traceback" not in captured.err


def test_channel_check_never_prints_a_misplaced_secret(
    channel_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    channel_file.write_text(
        channel_file.read_text().replace("[llm]", '[llm]\napi_key = "sk-ant-SECRET999"', 1)
    )
    code = run_main(monkeypatch, "channel", "check", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "llm.api_key" in captured.err
    assert "sk-ant-SECRET999" not in captured.err
    assert "sk-ant-SECRET999" not in captured.out


def test_channel_check_ok(channel_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "secret")
    result = runner.invoke(cli.app, ["channel", "check", str(channel_file)])
    assert result.exit_code == 0, result.output
    channel = load_channel(channel_file)
    assert channel.name in result.output
    assert style_fingerprint(channel.style) in result.output
    assert "image" in result.output
    assert "secret" not in result.output


def test_library_keys_empty(channel_file: Path, data_dir: Path) -> None:
    result = runner.invoke(cli.app, ["library", "keys", str(channel_file)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""
    assert (data_dir / "library.db").exists()


def test_library_keys_and_copy(channel_file: Path, data_dir: Path, tmp_path: Path) -> None:
    channel = load_channel(channel_file)
    library = default_library(get_settings())
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    prov = Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    record = library.add_sketch(
        version,
        "alpha|generic",
        b"A",
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        prov,
    )
    keys = runner.invoke(cli.app, ["library", "keys", str(channel_file)])
    assert keys.exit_code == 0, keys.output
    assert keys.output.splitlines() == ["alpha|generic"]

    other = tmp_path / "other.toml"
    other.write_text(
        channel_file.read_text(encoding="utf-8").replace(channel.slug, "other-channel", 1),
        encoding="utf-8",
    )
    (tmp_path / "ref.png").write_bytes((channel_file.parent / "ref.png").read_bytes())
    copied = runner.invoke(cli.app, ["library", "copy", str(record.id), str(other)])
    assert copied.exit_code == 0, copied.output
    assert int(copied.output.strip()) != record.id


def test_library_copy_unknown_sketch_exit_1(
    channel_file: Path,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_main(monkeypatch, "library", "copy", "999", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "unknown sketch id" in captured.err


def test_validate(tmp_path: Path) -> None:
    path = tmp_path / "storyboard.json"
    Project(title="T", language=Language.ENGLISH).save(path)
    result = runner.invoke(cli.app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "0 scenes" in result.output


def test_validate_refuses_other_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "storyboard.json"
    path.write_text('{"schema_version": 99, "title": "T", "language": "en"}', encoding="utf-8")
    code = run_main(monkeypatch, "validate", str(path))
    assert code == 1
    assert "unsupported project schema_version 99" in capsys.readouterr().err


def test_callback_sets_up_logging(
    channel_file: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    levels: list[str] = []
    monkeypatch.setattr(cli, "setup_logging", levels.append)
    monkeypatch.setenv("NARRATOFLOW_LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()
    result = runner.invoke(cli.app, ["library", "keys", str(channel_file)])
    assert result.exit_code == 0
    assert levels == ["DEBUG"]


def test_check_input_unreadable_file_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_main(monkeypatch, "check-input", str(tmp_path / "missing.txt"))
    assert code == 1
    assert "cannot read" in capsys.readouterr().err


def test_bad_dotenv_value_exit_1_without_leaking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".env").write_text("NARRATOFLOW_LOG_LEVEL=sk-SECRET123\n", encoding="utf-8")
    code = run_main(monkeypatch, "check-input", str(INPUTS / "clean_en.txt"), "--language", "en")
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "sk-SECRET123" not in captured.out + captured.err
    assert "settings" in captured.err.lower()


def test_dotenv_typo_key_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".env").write_text(
        "NARRATOFLOW_IMAGE_APIKEY=sk-SECRET123\nNARRATOFLOW_LOG_LEVEL=debug\n", encoding="utf-8"
    )
    code = run_main(monkeypatch, "check-input", str(INPUTS / "clean_en.txt"), "--language", "en")
    captured = capsys.readouterr()
    assert code == 1
    assert "NARRATOFLOW_IMAGE_APIKEY" in captured.err
    assert "Traceback" not in captured.err
    assert "sk-SECRET123" not in captured.out + captured.err


def test_library_copy_to_quarantined_dedupe_exit_1(
    channel_file: Path,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    channel = load_channel(channel_file)
    library = default_library(get_settings())
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    prov = Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    record = library.add_sketch(
        version,
        "alpha|generic",
        b"A",
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        prov,
    )
    other = tmp_path / "other.toml"
    other.write_text(
        channel_file.read_text(encoding="utf-8").replace(channel.slug, "other-channel", 1),
        encoding="utf-8",
    )
    (tmp_path / "ref.png").write_bytes((channel_file.parent / "ref.png").read_bytes())
    assert run_main(monkeypatch, "library", "copy", str(record.id), str(other)) == 0
    capsys.readouterr()
    target = load_channel(other)
    target_version = library.ensure_style_version(target.slug, style_fingerprint(target.style))
    copy_id = library.copy_to(record.id, target_version).id
    library.quarantine(copy_id, "bad lines")
    code = run_main(monkeypatch, "library", "copy", str(record.id), str(other))
    captured = capsys.readouterr()
    assert code == 1
    assert "quarantined" in captured.err and str(copy_id) in captured.err
    assert "bad lines" in captured.err
    assert "Traceback" not in captured.err


# --- plan / approve -------------------------------------------------------


def _planned_series(*, flags: list[str] | None = None) -> Series:
    return Series(
        title="Scenes [of] Resurrection",
        channel_slug="history",
        language=Language.ENGLISH,
        source=ScriptSource(author="A", work="W", rights=RightsBasis.OWN),
        parts=[
            Part(
                index=1,
                count=1,
                hook_text="h",
                est_cost_usd=Decimal("0.04"),
                flags=flags or [],
                scenes=[
                    Scene(
                        id="s001",
                        narration=Narration(source_text="Hello world."),
                        idea_key="scroll|generic",
                        match="new",
                        est_duration_s=61.2,
                    )
                ],
            )
        ],
        est_planning_cost_usd=Decimal("0.0123"),
    )


def _script(tmp_path: Path, rights: str = "own") -> Path:
    body = (INPUTS / "clean_ar.txt").read_text(encoding="utf-8")
    path = tmp_path / "scripts" / "script.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f'+++\nauthor = "A"\nwork = "W"\nrights = "{rights}"\n+++\n{body}', "utf-8")
    return path


def _patch_plan(
    monkeypatch: pytest.MonkeyPatch, series: Series, seen: list[ChannelConfig] | None = None
) -> None:
    def fake_plan(script: Path, channel: ChannelConfig, settings: Settings) -> Series:
        if seen is not None:
            seen.append(channel)
        return series

    monkeypatch.setattr(cli, "plan_series", fake_plan)


def _set_toml(channel_file: Path, old: str, new: str) -> None:
    text = channel_file.read_text(encoding="utf-8")
    assert old in text
    channel_file.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_cli_plan_writes_series_and_prints_list(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[ChannelConfig] = []
    _patch_plan(monkeypatch, _planned_series(), seen)
    script = _script(tmp_path)
    result = runner.invoke(cli.app, ["plan", str(script), "--channel", str(channel_file)])
    assert result.exit_code == 0, result.output
    [channel] = seen
    assert channel.slug == "history" and channel.language is Language.ARABIC
    target = script.parent / "series.json"
    saved = Series.load(target)
    assert saved.approved is False
    assert "new (paid)" in result.output
    assert "Scenes [of] Resurrection" in result.output
    assert "narratoflow approve" in result.output


def test_cli_plan_auto_approves_with_summary(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_toml(channel_file, 'slug = "history"', 'slug = "history"\napproval_list = false')
    _patch_plan(monkeypatch, _planned_series())
    out = tmp_path / "elsewhere.json"
    result = runner.invoke(
        cli.app, ["plan", str(_script(tmp_path)), "--channel", str(channel_file), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert Series.load(out).approved is True
    assert not (tmp_path / "scripts" / "series.json").exists()
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "1 part" in lines[0] and "61 s" in lines[0]
    assert "$0.04" in lines[0] and "$0.0123" in lines[0]
    assert "approved" in lines[0]


def test_cli_plan_flag_forces_list(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_toml(channel_file, 'slug = "history"', 'slug = "history"\napproval_list = false')
    _patch_plan(monkeypatch, _planned_series(flags=["over_max"]))
    script = _script(tmp_path)
    result = runner.invoke(cli.app, ["plan", str(script), "--channel", str(channel_file)])
    assert result.exit_code == 0, result.output
    assert "over_max" in result.output
    assert Series.load(script.parent / "series.json").approved is False


def test_cli_approve_yes(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    _planned_series().save(path)
    result = runner.invoke(cli.app, ["approve", str(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "new (paid)" in result.output
    assert Series.load(path).approved is True


def test_cli_approve_confirm(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    _planned_series().save(path)
    result = runner.invoke(cli.app, ["approve", str(path)], input="y\n")
    assert result.exit_code == 0, result.output
    assert Series.load(path).approved is True


def test_cli_approve_declined_leaves_unapproved(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    _planned_series().save(path)
    result = runner.invoke(cli.app, ["approve", str(path)], input="n\n")
    assert result.exit_code == 0, result.output
    assert "not approved" in result.output.lower()
    assert Series.load(path).approved is False


def test_cli_approve_declined_on_approved_series_says_it_stays_approved(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    approved = _planned_series()
    approved.approved = True
    approved.save(path)
    result = runner.invoke(cli.app, ["approve", str(path)], input="n\n")
    assert result.exit_code == 0, result.output
    assert "stays approved" in result.output
    assert "not approved" not in result.output.lower()
    assert Series.load(path).approved is True


def test_cli_plan_checks_out_dir_before_planning(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object) -> Series:
        raise AssertionError("planner must not run when --out cannot be written")

    monkeypatch.setattr(cli, "plan_series", boom)
    out = tmp_path / "missing-dir" / "series.json"
    code = run_main(
        monkeypatch,
        "plan",
        str(_script(tmp_path)),
        "--channel",
        str(channel_file),
        "--out",
        str(out),
    )
    assert code == 1


def _existing_series(tmp_path: Path) -> tuple[Path, bytes]:
    script = _script(tmp_path)
    target = script.with_name("series.json")
    _planned_series().save(target)
    return script, target.read_bytes()


def test_plan_refuses_to_overwrite_a_series(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*args: object) -> Series:
        raise AssertionError("planner must not run when the series file exists")

    monkeypatch.setattr(cli, "plan_series", boom)
    script, before = _existing_series(tmp_path)
    code = run_main(monkeypatch, "plan", str(script), "--channel", str(channel_file))
    err = " ".join(capsys.readouterr().err.split())  # stderr wraps at the terminal width
    assert code == 1
    assert "already exists; nothing was spent" in err
    assert "--replace" in err
    assert script.with_name("series.json").read_bytes() == before


def test_plan_replace_overwrites(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, before = _existing_series(tmp_path)
    fresh = _planned_series()
    fresh.est_planning_cost_usd = Decimal("0.5")
    _patch_plan(monkeypatch, fresh)
    code = run_main(monkeypatch, "plan", str(script), "--channel", str(channel_file), "--replace")
    assert code == 0
    target = script.with_name("series.json")
    assert target.read_bytes() != before
    assert Series.load(target).est_planning_cost_usd == Decimal("0.5")


def test_cli_approve_bad_file_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "series.json"
    path.write_text("{broken", encoding="utf-8")
    code = run_main(monkeypatch, "approve", str(path), "--yes")
    captured = capsys.readouterr()
    assert code == 1
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_cli_budget_error_exit_1(
    channel_file: Path,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img")
    monkeypatch.setenv("NARRATOFLOW_LLM_API_KEY", "llm")
    # CI has no assets/quran: provide the fixture corpus where default_corpus looks first
    quran = data_dir / "quran"
    quran.mkdir(parents=True)
    for name in ("quran-simple-clean.txt", "quran-uthmani.txt"):
        shutil.copy(Path(__file__).parent / "fixtures" / "quran" / name, quran / name)
    _set_toml(channel_file, 'slug = "history"', 'slug = "history"\nspending_limit_usd = 0.0001')
    _set_toml(
        channel_file,
        'provider = "ollama"\nmodel = "qwen2.5"',
        'provider = "anthropic"\nmodel = "claude-opus-5-5"',
    )
    script = _script(tmp_path)
    code = run_main(monkeypatch, "plan", str(script), "--channel", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "exceeds the remaining" in captured.err
    assert "Traceback" not in captured.err
    assert not (script.parent / "series.json").exists()


def test_cli_plan_in_copyright_exit_1(
    channel_file: Path,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _script(tmp_path, rights="third_party_in_copyright")
    code = run_main(monkeypatch, "plan", str(script), "--channel", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "copyright" in captured.err
    assert "Traceback" not in captured.err


# --- sketch generate / resolve --------------------------------------------


def _approved_series(tmp_path: Path, *, quarantined: bool = False) -> Path:
    series = _planned_series()
    series.approved = True
    if quarantined:
        scene = series.parts[0].scenes[0]
        series.parts[0].scenes[0] = scene.model_copy(update={"flags": ["sketch_quarantined"]})
    path = tmp_path / "series.json"
    series.save(path)
    return path


def _patch_generate(
    monkeypatch: pytest.MonkeyPatch,
    result: GenerationResult,
    updated: Series | None = None,
    raises: GenerationInterruptedError | None = None,
) -> list[Series]:
    seen: list[Series] = []

    def fake(series: Series, channel: ChannelConfig, settings: Settings) -> tuple[Series, object]:
        seen.append(series)
        if raises is not None:
            raise raises
        return (updated or series), result

    monkeypatch.setattr(cli, "generate_missing", fake)
    return seen


def test_sketch_generate_prints_summary_and_saves(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)
    updated = Series.load(path)
    updated.title = "Saved Title"
    result = GenerationResult(
        generated=2,
        reused=1,
        bundled=3,
        kept=4,
        quarantined=["s001"],
        unchecked=["s002"],
        spent_usd=Decimal("0.0840"),
    )
    _patch_generate(monkeypatch, result, updated)
    out = runner.invoke(cli.app, ["sketch", "generate", str(path), "--channel", str(channel_file)])
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "generated 2" in text and "reused 1" in text and "bundled 3" in text
    assert "kept 4" in text and "quarantined 1" in text and "unchecked 1" in text
    assert "$0.0840" in text
    assert "s001: quarantined - resolve with: narratoflow sketch resolve" in text
    assert f"{path} s001 --regenerate | --swap KEY --description TEXT | --text-card" in text
    assert "vision checking was off" in text
    assert Series.load(path).title == "Saved Title"


def test_sketch_generate_clean_run_has_no_warning(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)
    _patch_generate(monkeypatch, GenerationResult(generated=1))
    out = runner.invoke(cli.app, ["sketch", "generate", str(path), "--channel", str(channel_file)])
    assert out.exit_code == 0, out.output
    assert "vision checking" not in out.output and "quarantined -" not in out.output


def test_sketch_generate_interrupted_saves_partial_and_exits_1(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    partial = Series.load(path)
    partial.title = "Partial"
    cause = BudgetExceededError(Decimal("2"), Decimal("1"))
    exc = GenerationInterruptedError(cause, (partial, GenerationResult(generated=1)))
    _patch_generate(monkeypatch, GenerationResult(), raises=exc)
    code = run_main(monkeypatch, "sketch", "generate", str(path), "--channel", str(channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "exceeds limit" in captured.err and "Traceback" not in captured.err
    assert "generated 1" in captured.out
    assert Series.load(path).title == "Partial"


def test_sketch_generate_unapproved_series_saves_nothing(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "series.json"
    _planned_series().save(path)
    before = path.read_text(encoding="utf-8")
    code = run_main(monkeypatch, "sketch", "generate", str(path), "--channel", str(channel_file))
    assert code == 1
    assert path.read_text(encoding="utf-8") == before


def _resolve(channel_file: Path, path: Path, *args: str, input: str | None = None) -> Result:
    return runner.invoke(
        cli.app,
        ["sketch", "resolve", str(path), "s001", *args, "--channel", str(channel_file)],
        input=input,
    )


def _patch_regenerate(
    monkeypatch: pytest.MonkeyPatch, *, attached: bool = True, key: bool = True
) -> list[str]:
    """Fake `regenerate`: the scene ends attached (or quarantined again), $0.06 spent."""
    calls: list[str] = []
    if key:
        monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img-key")

    def fake(
        series: Series, scene_id: str, channel: ChannelConfig, settings: Settings
    ) -> tuple[Series, GenerationResult]:
        calls.append(scene_id)
        out = series.model_copy(deep=True)
        out.title = "Regenerated"
        result = GenerationResult(spent_usd=Decimal("0.06"))
        sc = out.parts[0].scenes[0]
        if attached:
            out.parts[0].scenes[0] = sc.model_copy(update={"sketch_id": 7, "flags": []})
            result.generated = 1
            result.unchecked.append(scene_id)
        else:
            result.quarantined.append(scene_id)
        return out, result

    monkeypatch.setattr(cli, "regenerate", fake)
    return calls


def _no_confirm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []

    def confirm(text: str, *args: object, **kwargs: object) -> bool:
        asked.append(text)
        return True

    monkeypatch.setattr("typer.confirm", confirm)
    return asked


def test_sketch_resolve_regenerate_reports_attached_and_spend(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    _patch_regenerate(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate", "--yes")
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "s001: regenerated" in text and "$0.06" in text
    assert "vision checking was off" in text


def test_sketch_resolve_regenerate_that_fails_again_says_so(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    _patch_regenerate(monkeypatch, attached=False)
    out = _resolve(channel_file, path, "--regenerate", "--yes")
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "regenerated" not in text
    assert "s001: quarantined again" in text and "$0.06" in text
    assert f"narratoflow sketch resolve {path} s001 --regenerate | --swap KEY" in text


def test_sketch_resolve_regenerate_never_prompts_for_a_scene_it_cannot_regenerate(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)  # s001 is not quarantined
    calls = _patch_regenerate(monkeypatch)
    asked = _no_confirm(monkeypatch)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "s001",
        "--regenerate",
        "--channel",
        str(channel_file),
    )
    assert code == 1 and asked == [] and calls == []


def test_sketch_resolve_regenerate_never_prompts_without_the_image_key(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    calls = _patch_regenerate(monkeypatch, key=False)
    asked = _no_confirm(monkeypatch)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "s001",
        "--regenerate",
        "--channel",
        str(channel_file),
    )
    assert code == 1 and asked == [] and calls == []
    assert "NARRATOFLOW_IMAGE_API_KEY" in capsys.readouterr().err


def test_sketch_resolve_regenerate_never_prompts_for_another_channels_series(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    other = Series.load(path)
    other.channel_slug = "someone-else"
    other.save(path)
    calls = _patch_regenerate(monkeypatch)
    asked = _no_confirm(monkeypatch)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "s001",
        "--regenerate",
        "--channel",
        str(channel_file),
    )
    assert code == 1 and asked == [] and calls == []
    assert "someone-else" in capsys.readouterr().err


def test_sketch_resolve_regenerate_never_prompts_without_the_vision_key(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = channel_file.read_text(encoding="utf-8")
    channel_file.write_text(
        text + '\n[vision]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("NARRATOFLOW_LLM_API_KEY", raising=False)
    path = _approved_series(tmp_path, quarantined=True)
    calls = _patch_regenerate(monkeypatch)
    asked = _no_confirm(monkeypatch)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "s001",
        "--regenerate",
        "--channel",
        str(channel_file),
    )
    assert code == 1 and asked == [] and calls == []
    assert "NARRATOFLOW_LLM_API_KEY" in capsys.readouterr().err


def test_sketch_resolve_regenerate_estimate_has_no_vision_cost_when_off(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the history channel's LLM is ollama with no [vision]: vision is off
    path = _approved_series(tmp_path, quarantined=True)
    _patch_regenerate(monkeypatch)
    asked = _no_confirm(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate")
    assert out.exit_code == 0, out.output
    [prompt] = asked
    assert prompt.endswith("Estimated cost $0.04")


def test_sketch_resolve_regenerate_estimate_includes_the_vision_check_when_on(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.costs.prices import default_prices
    from narratoflow.money import format_usd
    from narratoflow.sketching.vision import estimate_check_cost

    text = channel_file.read_text(encoding="utf-8")
    channel_file.write_text(
        text + '\n[vision]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\n',
        encoding="utf-8",
    )
    channel = load_channel(channel_file)
    monkeypatch.setenv("NARRATOFLOW_LLM_API_KEY", "llm-key")
    path = _approved_series(tmp_path, quarantined=True)
    _patch_regenerate(monkeypatch)
    asked = _no_confirm(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate")
    assert out.exit_code == 0, out.output
    prices = default_prices(get_settings())
    check = estimate_check_cost(
        channel.style,
        "scroll|generic",
        prices,
        "anthropic",
        "claude-haiku-5-5",
        with_reference=True,
    )
    assert check > 0
    [prompt] = asked
    assert prompt.endswith(f"Estimated cost {format_usd(Decimal('0.04') + check)}")


def test_sketch_generate_real_run_without_vision_warns_unchecked(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    from fakes.images import FakeImages

    from narratoflow.sketching import generate as generate_module
    from narratoflow.visuals.bundled import BundledIcons

    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img-key")
    images = FakeImages()
    monkeypatch.setattr(generate_module, "build_image_provider", lambda c, s: images)

    def no_vision(*args: object) -> object:
        raise AssertionError("vision is off: no vision client is built")

    monkeypatch.setattr(generate_module, "build_llm_client", no_vision)
    # the shipped icon pack would cover the series' key; this run must draw it
    monkeypatch.setattr(generate_module, "default_bundled_icons", lambda: BundledIcons({}))
    path = _approved_series(tmp_path)
    out = runner.invoke(cli.app, ["sketch", "generate", str(path), "--channel", str(channel_file)])
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "unchecked 1" in text and "vision checking was off" in text
    assert len(images.requests) == 1


def test_sketch_generate_for_another_channel_exits_1_before_spend(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.sketching import generate as generate_module

    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img-key")

    def no_images(*args: object) -> object:
        raise AssertionError("nothing is drawn for another channel's series")

    monkeypatch.setattr(generate_module, "build_image_provider", no_images)
    _set_toml(channel_file, 'slug = "history"', 'slug = "other-channel"')
    path = _approved_series(tmp_path)
    before = path.read_text(encoding="utf-8")
    code = run_main(monkeypatch, "sketch", "generate", str(path), "--channel", str(channel_file))
    err = capsys.readouterr().err
    assert code == 1
    assert "history" in err and "other-channel" in err
    assert path.read_text(encoding="utf-8") == before


def test_sketch_resolve_regenerate_confirms_with_estimate(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    calls = _patch_regenerate(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate", input="y\n")
    assert out.exit_code == 0, out.output
    assert "Regenerate" in out.output and "$" in out.output
    assert calls == ["s001"]
    assert Series.load(path).title == "Regenerated"


def test_sketch_resolve_regenerate_declined_changes_nothing(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    calls = _patch_regenerate(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate", input="n\n")
    assert out.exit_code == 0, out.output
    assert "Unchanged" in out.output
    assert calls == []
    assert Series.load(path).title != "Regenerated"


def test_sketch_resolve_regenerate_yes_skips_prompt(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    calls = _patch_regenerate(monkeypatch)
    out = _resolve(channel_file, path, "--regenerate", "--yes")
    assert out.exit_code == 0, out.output
    assert calls == ["s001"]
    assert "[y/N]" not in out.output


def test_sketch_resolve_regenerate_interrupted_saves_partial(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    partial = Series.load(path)
    partial.title = "Partial"
    exc = GenerationInterruptedError(
        BudgetExceededError(Decimal("2"), Decimal("1")),
        (partial, GenerationResult(spent_usd=Decimal("0.01"))),
    )

    def boom(*args: object, **kwargs: object) -> Series:
        raise exc

    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img-key")
    monkeypatch.setattr(cli, "regenerate", boom)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "s001",
        "--regenerate",
        "--yes",
        "--channel",
        str(channel_file),
    )
    assert code == 1
    assert "exceeds limit" in capsys.readouterr().err
    assert Series.load(path).title == "Partial"


def test_sketch_resolve_swap(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    out = _resolve(
        channel_file, path, "--swap", "lantern|generic", "--description", "A lit oil lantern"
    )
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "lantern|generic" in text and "narratoflow sketch generate" in text
    scene = Series.load(path).parts[0].scenes[0]
    assert scene.idea_key == "lantern|generic"
    assert "sketch_quarantined" not in scene.flags


def test_sketch_resolve_text_card(channel_file: Path, tmp_path: Path) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    out = _resolve(channel_file, path, "--text-card")
    assert out.exit_code == 0, out.output
    assert "text card" in out.output
    scene = Series.load(path).parts[0].scenes[0]
    assert scene.visual.source is VisualSource.TEXT_CARD
    assert "sketch_quarantined" not in scene.flags


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--regenerate", "--text-card"],
        ["--swap", "a|b", "--description", "d", "--text-card"],
        ["--swap", "lantern|generic"],
        ["--description", "d", "--text-card"],
        ["--text-card", "--yes"],
        ["--swap", "lantern|generic", "--description", "d", "--yes"],
    ],
)
def test_sketch_resolve_bad_option_combinations(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    before = path.read_text(encoding="utf-8")
    calls = _patch_regenerate(monkeypatch)
    out = _resolve(channel_file, path, *args)
    assert out.exit_code == 2, out.output
    assert calls == []
    assert path.read_text(encoding="utf-8") == before


def test_sketch_resolve_requires_channel(tmp_path: Path) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    out = runner.invoke(cli.app, ["sketch", "resolve", str(path), "s001", "--text-card"])
    assert out.exit_code == 2


def test_sketch_resolve_unknown_scene_exit_1(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "nope",
        "--text-card",
        "--channel",
        str(channel_file),
    )
    assert code == 1


def test_sketch_resolve_regenerate_unknown_scene_skips_prompt_and_exits_1(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    code = run_main(
        monkeypatch,
        "sketch",
        "resolve",
        str(path),
        "nope",
        "--regenerate",
        "--channel",
        str(channel_file),
    )
    assert code == 1


# --- audio build ------------------------------------------------------------


def _timeline(
    index: int,
    count: int,
    *,
    voice: VoiceUsed | None = None,
    spent: str = "0",
    estimated: bool = False,
    warnings: list[str] | None = None,
    lufs: float | None = -14.0,
) -> PartTimeline:
    return PartTimeline(
        series_title="S",
        channel_slug="history",
        part_index=index,
        part_count=count,
        audio_file=f"part-{index:02d}.flac",
        duration_ms=61_500,
        integrated_lufs=lufs,
        true_peak_dbtp=-1.5,
        scenes=[
            SceneTiming(
                scene_id="s001",
                kind=SceneKind.NARRATION,
                start_ms=0,
                end_ms=61_500,
                speech_start_ms=0,
                speech_end_ms=61_000,
                words=[],
                timing_source=TimingSource.ESTIMATED,
            )
        ],
        sfx=[],
        voice=voice,
        recitations=[],
        assets=[],
        spent_usd=Decimal(spent),
        cost_estimated=estimated,
        warnings=warnings or [],
    )


def _part_audio(out: Path, timeline: PartTimeline) -> PartAudio:
    stem = f"part-{timeline.part_index:02d}"
    return PartAudio(
        timeline.part_index, out / f"{stem}.flac", out / f"{stem}.timeline.json", timeline
    )


def _patch_audio(
    monkeypatch: pytest.MonkeyPatch,
    result: AudioBuildResult,
    updated: Series | None = None,
    raises: AudioInterruptedError | None = None,
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(
        series: Series, channel: ChannelConfig, settings: Settings, **kw: Any
    ) -> tuple[Series, AudioBuildResult]:
        seen.append(kw)
        if raises is not None:
            raise raises
        return (updated or series), result

    monkeypatch.setattr(cli, "build_audio", fake)
    return seen


def _audio_args(path: Path, channel_file: Path, *extra: str) -> list[str]:
    return ["audio", "build", str(path), "--channel", str(channel_file), *extra]


def test_audio_build_prints_part_lines_and_warnings(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)
    out_dir = tmp_path / "audio"
    paid = VoiceUsed(
        provider="elevenlabs",
        model="m",
        voice_id="v1",
        terms="t",
        fallback_from="edge",
        fallback_reasons=["timeout", "503"],
    )
    verse_only = _timeline(2, 2, spent="0")
    first = _timeline(
        1, 2, voice=paid, spent="0.30", estimated=True, warnings=["loudness off target: -16.0 LUFS"]
    )
    result = AudioBuildResult(
        parts=[_part_audio(out_dir, first), _part_audio(out_dir, verse_only)],
        spent_usd=Decimal("0.10"),
    )
    series = Series.load(path)
    series.parts[0].spent_usd = Decimal("0.20")
    series.save(path)
    _patch_audio(monkeypatch, result)
    out = runner.invoke(cli.app, _audio_args(path, channel_file))
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "part 1/2: 61.5 s, -14.0 LUFS, voice elevenlabs/v1, spent $0.10 (estimated)" in text
    assert "part total $0.30" in text and f"-> {out_dir / 'part-01.flac'}" in text
    assert (
        "part 2/2: 61.5 s, -14.0 LUFS, voice none (verse only), spent $0.00 (part total $0.00) -> "
        in text
    )
    assert "estimated" not in text.split("part 2/2")[1]
    assert f"timeline: {out_dir / 'part-01.timeline.json'}" in text
    assert "warning: voice fell back from edge: timeout; 503" in text
    assert "warning: part 1: loudness off target: -16.0 LUFS" in text
    assert "total spent $0.10" in text


def test_audio_build_saves_series_on_interruption(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    partial = Series.load(path)
    partial.parts[0].spent_usd = Decimal("0.05")
    done = AudioBuildResult(
        parts=[_part_audio(tmp_path, _timeline(1, 1, spent="0.05"))], spent_usd=Decimal("0.05")
    )
    exc = AudioInterruptedError(BudgetExceededError(Decimal("2"), Decimal("1")), (partial, done))
    _patch_audio(monkeypatch, AudioBuildResult(), raises=exc)
    code = run_main(monkeypatch, *_audio_args(path, channel_file))
    captured = capsys.readouterr()
    assert code == 1
    assert "progress so far is kept" in captured.err and "Traceback" not in captured.err
    assert "part 1/1" in captured.out
    assert Series.load(path).parts[0].spent_usd == Decimal("0.05")


def test_audio_build_default_out_dir_and_part_filter(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)
    seen = _patch_audio(monkeypatch, AudioBuildResult())
    out = runner.invoke(cli.app, _audio_args(path, channel_file, "--part", "2", "--part", "3"))
    assert out.exit_code == 0, out.output
    assert seen[0]["out_dir"] == tmp_path / "audio"
    assert seen[0]["cache_root"] == tmp_path / ".cache"
    assert seen[0]["parts"] == [2, 3]
    runner.invoke(cli.app, _audio_args(path, channel_file, "--out", str(tmp_path / "x")))
    assert seen[1]["out_dir"] == tmp_path / "x" and seen[1]["parts"] is None


def test_audio_build_unknown_part_exits_1(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.audio import build as build_module

    monkeypatch.setattr(build_module, "require_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    path = _approved_series(tmp_path)
    before = path.read_text(encoding="utf-8")
    code = run_main(monkeypatch, *_audio_args(path, channel_file, "--part", "9"))
    assert code == 1
    assert "no part 9" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == before
    assert not (tmp_path / "audio").exists()


def test_resolve_text_card_with_card_text(channel_file: Path, tmp_path: Path) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    out = _resolve(channel_file, path, "--text-card", "--card-text", "Patience")
    assert out.exit_code == 0, out.output
    assert Series.load(path).parts[0].scenes[0].card_text == "Patience"


def test_resolve_card_text_again_updates_an_existing_card(
    channel_file: Path, tmp_path: Path
) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    assert _resolve(channel_file, path, "--text-card", "--card-text", "One").exit_code == 0
    out = _resolve(channel_file, path, "--text-card", "--card-text", "Two")
    assert out.exit_code == 0, out.output
    scene = Series.load(path).parts[0].scenes[0]
    assert scene.card_text == "Two" and scene.visual.source is VisualSource.TEXT_CARD


def test_card_text_requires_text_card(channel_file: Path, tmp_path: Path) -> None:
    path = _approved_series(tmp_path, quarantined=True)
    out = _resolve(channel_file, path, "--regenerate", "--card-text", "x")
    assert out.exit_code == 2


def _render_part(part: int, count: int, warnings: list[str]) -> Any:
    from narratoflow.render.build import PartVideo

    report = SimpleNamespace(
        part_index=part,
        part_count=count,
        duration_ms=61500,
        frames=1845,
        width=1080,
        height=1920,
        fps=30,
        warnings=warnings,
    )
    return PartVideo(
        index=part,
        video_path=Path(f"/o/part-{part:02d}.mp4"),
        report_path=Path(f"/o/part-{part:02d}.render.json"),
        report=report,  # type: ignore[arg-type]
    )


def _patch_render(
    monkeypatch: pytest.MonkeyPatch, result: Any = None, raises: BaseException | None = None
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(series: Series, channel: ChannelConfig, settings: Settings, **kw: Any) -> Any:
        seen.append(kw)
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(cli, "build_render", fake)
    return seen


def _render_args(path: Path, channel_file: Path, *extra: str) -> list[str]:
    return ["render", str(path), "--channel", str(channel_file), *extra]


def test_render_prints_part_lines_and_warnings(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.render.build import RenderResult

    path = _approved_series(tmp_path)
    result = RenderResult(parts=[_render_part(1, 2, ["drift 3 ms"]), _render_part(2, 2, [])])
    _patch_render(monkeypatch, result)
    out = runner.invoke(cli.app, _render_args(path, channel_file))
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "part 1/2: 61.5 s, 1845 frames, 1080x1920 @ 30 fps -> /o/part-01.mp4" in text
    assert "report: /o/part-01.render.json" in text
    assert "warning: part 1: drift 3 ms" in text
    assert "part 2/2" in text and "part 2:" not in text


def test_render_force_flag_and_skipped_line(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.render.build import RenderResult

    path = _approved_series(tmp_path)
    result = RenderResult(
        parts=[_render_part(1, 2, ["drift 3 ms"]), _render_part(2, 2, [])], skipped=[1]
    )
    seen = _patch_render(monkeypatch, result)
    out = runner.invoke(cli.app, _render_args(path, channel_file, "--force"))
    assert out.exit_code == 0, out.output
    assert seen[0]["force"] is True
    text = " ".join(out.output.split())
    assert "part 1/2: unchanged, kept /o/part-01.mp4" in text
    assert "1845 frames, 1080x1920 @ 30 fps -> /o/part-02.mp4" in text
    assert "part 1/2: 61.5 s" not in text and "report: /o/part-01.render.json" not in text
    assert "warning: part 1: drift 3 ms" in text
    runner.invoke(cli.app, _render_args(path, channel_file))
    assert seen[1]["force"] is False


def test_render_default_audio_and_out_dirs_and_part_filter(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.render.build import RenderResult

    path = _approved_series(tmp_path)
    seen = _patch_render(monkeypatch, RenderResult())
    out = runner.invoke(
        cli.app, _render_args(path, channel_file, "--part", "2", "--part", "3", "--workers", "2")
    )
    assert out.exit_code == 0, out.output
    assert seen[0]["audio_dir"] == tmp_path / "audio" == seen[0]["out_dir"]
    assert seen[0]["parts"] == [2, 3] and seen[0]["workers"] == 2
    runner.invoke(
        cli.app,
        _render_args(
            path, channel_file, "--audio", str(tmp_path / "a"), "--out", str(tmp_path / "o")
        ),
    )
    assert seen[1]["audio_dir"] == tmp_path / "a" and seen[1]["out_dir"] == tmp_path / "o"
    assert seen[1]["parts"] is None and seen[1]["workers"] is None
    runner.invoke(cli.app, _render_args(path, channel_file, "--audio", str(tmp_path / "a")))
    assert seen[2]["out_dir"] == tmp_path / "a"


def test_render_stale_timeline_exits_1_with_message(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.render import build as build_module

    monkeypatch.setattr(build_module, "find_render_tools", lambda settings: object())
    path = _approved_series(tmp_path)
    code = run_main(monkeypatch, *_render_args(path, channel_file))
    err = capsys.readouterr().err
    assert code == 1
    assert "audio build" in err and "Traceback" not in err


def test_render_workers_below_one_is_a_usage_error(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _approved_series(tmp_path)
    seen = _patch_render(monkeypatch)
    for bad in ("0", "-3"):
        out = runner.invoke(cli.app, _render_args(path, channel_file, "--workers", bad))
        assert out.exit_code == 2 and "Traceback" not in out.output
    assert seen == []


def test_render_interrupt_exits_nonzero_without_traceback(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    _patch_render(monkeypatch, raises=KeyboardInterrupt())
    code = run_main(monkeypatch, *_render_args(path, channel_file))
    err = capsys.readouterr().err
    assert code != 0
    assert "interrupted" in err and "Traceback" not in err


def _patch_render_failing_after_part_one(
    monkeypatch: pytest.MonkeyPatch, raises: BaseException
) -> None:
    def fake(series: Series, channel: ChannelConfig, settings: Settings, **kw: Any) -> Any:
        kw["on_part"](_render_part(1, 3, []))
        raise raises

    monkeypatch.setattr(cli, "build_render", fake)


def test_render_interrupt_lists_finished_parts(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    _patch_render_failing_after_part_one(monkeypatch, KeyboardInterrupt())
    code = run_main(monkeypatch, *_render_args(path, channel_file))
    err = " ".join(capsys.readouterr().err.split())
    assert code == 130
    assert "finished parts: 1" in err and "/o/part-01.mp4" in err


def test_render_failure_lists_finished_parts(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.errors import RenderError

    path = _approved_series(tmp_path)
    _patch_render_failing_after_part_one(monkeypatch, RenderError("scene s9: boom"))
    code = run_main(monkeypatch, *_render_args(path, channel_file))
    err = " ".join(capsys.readouterr().err.split())
    assert code == 1
    assert "finished parts: 1" in err and "/o/part-01.mp4" in err
    assert "scene s9: boom" in err and "Traceback" not in err


def test_render_interrupt_before_any_part(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    _patch_render(monkeypatch, raises=KeyboardInterrupt())
    code = run_main(monkeypatch, *_render_args(path, channel_file))
    assert code == 130
    assert "no part finished" in capsys.readouterr().err


# --- review / publish -------------------------------------------------------


def _publish_part(
    part: int, count: int, checks: list[tuple[str, str, str | None, str | None]], exported: bool
) -> Any:
    from narratoflow.publish.build import PartPublish
    from narratoflow.publish.checks import CheckId, CheckResult, Severity, worst

    results = [
        CheckResult(
            check=CheckId(c), severity=Severity(sev), reason="why", scene_id=scene, platform=plat
        )
        for c, sev, scene, plat in checks
    ]
    report = SimpleNamespace(
        part_index=part, part_count=count, checks=results, verdict=worst(results)
    )
    return PartPublish(
        index=part,
        report=report,  # type: ignore[arg-type]
        report_path=Path(f"/o/part-{part:02d}.publish.md"),
        export_path=Path(f"/e/part-{part:02d}.mp4") if exported else None,
    )


def _patch_publish(monkeypatch: pytest.MonkeyPatch, result: Any = None) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(series: Series, channel: ChannelConfig, settings: Settings, **kw: Any) -> Any:
        seen.append(kw)
        return result

    monkeypatch.setattr(cli, "build_publish", fake)
    return seen


def _publish_args(path: Path, channel_file: Path, *extra: str) -> list[str]:
    return ["publish", str(path), "--channel", str(channel_file), *extra]


def test_publish_prints_verdicts_and_reasons_exit_0_when_none_blocked(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.publish.build import PublishResult

    path = _approved_series(tmp_path)
    before = path.read_bytes()
    parts = [
        _publish_part(1, 2, [("rules_file", "WARN", None, None)], True),
        _publish_part(2, 2, [("render_current", "PASS", None, None)], True),
    ]
    _patch_publish(monkeypatch, PublishResult(parts, Path("/e/series.publish.md"), []))
    out = runner.invoke(cli.app, _publish_args(path, channel_file))
    assert out.exit_code == 0, out.output
    text = " ".join(out.output.split())
    assert "part 1/2: WARN (0 fail, 1 warn) -> /e/part-01.mp4" in text
    assert "WARN rules_file: why" in text and "report: /o/part-01.publish.md" in text
    assert "part 2/2: PASS (0 fail, 0 warn)" in text
    assert "series summary: /e/series.publish.md" in text
    assert path.read_bytes() == before


def test_publish_blocked_exits_1_naming_parts(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.publish.build import PublishResult

    path = _approved_series(tmp_path)
    before = path.read_bytes()
    part = _publish_part(
        1,
        1,
        [("asset_provenance", "FAIL", "s001", "tiktok"), ("rules_file", "WARN", None, None)],
        False,
    )
    _patch_publish(monkeypatch, PublishResult([part], Path("/e/series.publish.md"), [1]))
    code = run_main(monkeypatch, *_publish_args(path, channel_file))
    captured = capsys.readouterr()
    text = " ".join(captured.out.split())
    assert code == 1
    assert "part 1/1: FAIL (1 fail, 1 warn) -> not exported" in text
    assert "FAIL asset_provenance s001 tiktok: why" in text
    assert "export blocked for part(s) 1: fix every FAIL above" in captured.err
    assert "Traceback" not in captured.err
    assert path.read_bytes() == before


def test_publish_default_dirs_and_part_filter(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.publish.build import PublishResult

    path = _approved_series(tmp_path)
    seen = _patch_publish(monkeypatch, PublishResult([], Path("/e/s.md"), []))
    out = runner.invoke(cli.app, _publish_args(path, channel_file, "--part", "2", "--part", "3"))
    assert out.exit_code == 0, out.output
    assert seen[0]["audio_dir"] == tmp_path / "audio" == seen[0]["render_dir"]
    assert seen[0]["export_dir"] == tmp_path / "export" and seen[0]["parts"] == [2, 3]
    runner.invoke(
        cli.app,
        _publish_args(
            path,
            channel_file,
            "--audio",
            str(tmp_path / "a"),
            "--render",
            str(tmp_path / "r"),
            "--export",
            str(tmp_path / "x"),
        ),
    )
    assert seen[1]["audio_dir"] == tmp_path / "a" and seen[1]["render_dir"] == tmp_path / "r"
    assert seen[1]["export_dir"] == tmp_path / "x" and seen[1]["parts"] is None
    runner.invoke(cli.app, _publish_args(path, channel_file, "--audio", str(tmp_path / "a")))
    assert seen[2]["render_dir"] == tmp_path / "a"


def test_publish_missing_render_report_exits_1(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _approved_series(tmp_path)
    before = path.read_bytes()
    code = run_main(monkeypatch, *_publish_args(path, channel_file))
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and err.strip()
    assert path.read_bytes() == before
    code = run_main(monkeypatch, *_publish_args(path, channel_file, "--part", "9"))
    assert code == 1 and "no part 9" in capsys.readouterr().err


def _review_channel(channel_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = channel_file.read_text(encoding="utf-8")
    channel_file.write_text(
        text.replace(
            'provider = "ollama"\nmodel = "qwen2.5"',
            'provider = "anthropic"\nmodel = "claude-haiku-5-5"',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NARRATOFLOW_LLM_API_KEY", "llm")


def _patch_review(
    monkeypatch: pytest.MonkeyPatch, result: Any = None, raises: BaseException | None = None
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    client = object()
    monkeypatch.setattr(cli, "build_llm_client", lambda cfg, settings: client)

    def fake(series: Series, channel: ChannelConfig, settings: Settings, **kw: Any) -> Any:
        seen.append({**kw, "client": client})
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(cli, "review_series", fake)
    return seen


def _review_args(path: Path, channel_file: Path, *extra: str) -> list[str]:
    return ["review", str(path), "--channel", str(channel_file), *extra]


def test_review_asks_with_estimate_and_no_leaves_series_unchanged(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _review_channel(channel_file, monkeypatch)
    path = _approved_series(tmp_path)
    before = path.read_bytes()
    seen = _patch_review(monkeypatch, (Series.load(path), None))
    asked: list[str] = []

    def no(text: str, *a: object, **k: object) -> bool:
        asked.append(text)
        return False

    monkeypatch.setattr("typer.confirm", no)
    out = runner.invoke(cli.app, _review_args(path, channel_file))
    assert out.exit_code == 0, out.output
    assert len(asked) == 1
    assert asked[0].startswith("Review content rules for 1 part(s)? Estimated cost $")
    assert "Unchanged." in out.output and seen == []
    assert path.read_bytes() == before


def test_review_yes_saves_findings_and_spend(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.publish.review import ReviewResult

    _review_channel(channel_file, monkeypatch)
    path = _approved_series(tmp_path)
    updated = Series.load(path)
    updated.parts[0].spent_usd = Decimal("0.02")
    result = ReviewResult(
        reviewed=[1],
        findings=3,
        spent_usd=Decimal("0.02"),
        per_part=[(1, 1, 2, Decimal("0.02"))],
    )
    seen = _patch_review(monkeypatch, (updated, result))
    out = runner.invoke(cli.app, _review_args(path, channel_file, "--yes", "--part", "1"))
    assert out.exit_code == 0, out.output
    assert "part 1: 3 finding(s) (1 fail, 2 warn), spent $0.02" in out.output
    assert "total spent $0.02" in out.output
    assert seen[0]["llm"] is seen[0]["client"] and seen[0]["parts"] == [1]
    assert Series.load(path).parts[0].spent_usd == Decimal("0.02")


def test_review_interrupted_saves_partial(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.publish.review import ReviewInterruptedError, ReviewResult

    _review_channel(channel_file, monkeypatch)
    path = _approved_series(tmp_path)
    partial = Series.load(path)
    partial.parts[0].spent_usd = Decimal("0.05")
    done = ReviewResult(spent_usd=Decimal("0.05"))
    exc = ReviewInterruptedError(BudgetExceededError(Decimal("2"), Decimal("1")), (partial, done))
    _patch_review(monkeypatch, raises=exc)
    code = run_main(monkeypatch, *_review_args(path, channel_file, "--yes"))
    captured = capsys.readouterr()
    assert code == 1
    assert "progress so far is kept" in captured.err and "Traceback" not in captured.err
    assert "total spent $0.05" in captured.out
    assert Series.load(path).parts[0].spent_usd == Decimal("0.05")


def test_review_missing_key_refused_before_prompt(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _review_channel(channel_file, monkeypatch)
    monkeypatch.delenv("NARRATOFLOW_LLM_API_KEY", raising=False)
    path = _approved_series(tmp_path)
    asked = _no_confirm(monkeypatch)
    code = run_main(monkeypatch, *_review_args(path, channel_file))
    assert code == 1 and asked == []
    assert "NARRATOFLOW_LLM_API_KEY" in capsys.readouterr().err


def test_review_unapproved_refused_and_confirmed_run_proceeds(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from narratoflow.publish.review import ReviewResult

    _review_channel(channel_file, monkeypatch)
    plain = tmp_path / "plain.json"
    _planned_series().save(plain)
    asked = _no_confirm(monkeypatch)
    seen = _patch_review(monkeypatch, (Series.load(plain), ReviewResult()))
    assert run_main(monkeypatch, *_review_args(plain, channel_file)) == 1
    assert "approve the series first" in capsys.readouterr().err
    assert asked == [] and seen == []
    path = _approved_series(tmp_path)
    out = runner.invoke(cli.app, _review_args(path, channel_file))
    assert out.exit_code == 0, out.output
    assert len(asked) == 1 and len(seen) == 1


def test_review_ctrl_c_saves_spend_and_exits_130(
    channel_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.publish.review import ReviewCancelledError, ReviewResult

    _review_channel(channel_file, monkeypatch)
    path = _approved_series(tmp_path)
    partial = Series.load(path)
    partial.parts[0].spent_usd = Decimal("0.03")
    done = ReviewResult(spent_usd=Decimal("0.03"))
    _patch_review(monkeypatch, raises=ReviewCancelledError((partial, done)))
    out = runner.invoke(cli.app, _review_args(path, channel_file, "--yes"))
    assert out.exit_code == 130
    assert "interrupted" in out.output and "total spent $0.03" in out.output
    assert "Traceback" not in out.output
    assert Series.load(path).parts[0].spent_usd == Decimal("0.03")


def test_review_unknown_part_refused_before_prompt_and_spend(
    channel_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _review_channel(channel_file, monkeypatch)
    path = _approved_series(tmp_path)
    before = path.read_bytes()
    seen = _patch_review(monkeypatch, None)
    asked = _no_confirm(monkeypatch)
    code = run_main(monkeypatch, *_review_args(path, channel_file, "--part", "9"))
    assert code == 1 and asked == [] and seen == []
    assert "no part 9" in capsys.readouterr().err
    assert path.read_bytes() == before


def _two_icons(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def icon(key: str, description: str) -> BundledIcon:
        return BundledIcon(
            key=key,
            file=tmp_path / "x.png",
            license="MIT",
            source="x",
            description=description,
            sha256="0" * 64,
        )

    pack = BundledIcons(
        {
            "lightbulb|generic": icon("lightbulb|generic", "a bulb for an idea"),
            "eye|generic": icon("eye|generic", "an open eye"),
        }
    )
    monkeypatch.setattr(cli, "default_bundled_icons", lambda: pack)


def test_library_icons_lists_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _two_icons(monkeypatch, tmp_path)
    for name in list(os.environ):
        if name.startswith("NARRATOFLOW_") and name.endswith("_KEY"):
            monkeypatch.delenv(name)
    result = runner.invoke(cli.app, ["library", "icons", "idea"])
    assert result.exit_code == 0, result.output
    assert "lightbulb|generic" in result.output
    assert "eye|generic" not in result.output


def test_library_icons_without_words_lists_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _two_icons(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["library", "icons"])
    assert result.exit_code == 0, result.output
    assert "lightbulb|generic" in result.output and "eye|generic" in result.output


def test_library_icons_no_match_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _two_icons(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["library", "icons", "zzz"])
    assert result.exit_code == 0, result.output
    assert "no bundled icon matches 'zzz'" in result.output
    assert "--text-card" in result.output


class _FakeDialectModel:
    def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]:
        return [("CAI", 0.87) if i == 0 else ("MSA", 0.9) for i, _ in enumerate(sentences)]


def test_check_dialect_prints_flagged_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_camel_model", _FakeDialectModel)
    script = tmp_path / "script.txt"
    first = "\u0643\u062a\u0628 \u0627\u0644\u0637\u0627\u0644\u0628."
    script.write_text(f"Hello.\n{first}\n\u0643\u062a\u0628.\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["check-dialect", str(script)])
    assert result.exit_code == 0, result.output
    assert f"line 2: CAI 0.87 \u00ab{first}\u00bb" in result.output
    report_path = tmp_path / "script.txt.dialect.json"
    assert "1 of 2 Arabic sentences flagged (threshold 0.5); report:" in result.output
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["threshold"] == 0.5
    assert [s["flagged"] for s in data["sentences"]] == [True, False]


def test_check_dialect_without_extra_names_install_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> object:
        raise DialectUnavailableError(f"camel_tools failed to load (ImportError); {INSTALL_HINT}")

    monkeypatch.setattr(cli, "load_camel_model", boom)
    script = tmp_path / "script.txt"
    script.write_text("\u0643\u062a\u0628\n", encoding="utf-8")
    monkeypatch.setenv("COLUMNS", "300")
    code = run_main(monkeypatch, "check-dialect", str(script))
    err = capsys.readouterr().err
    assert code == 1
    assert "uv sync --extra arabic" in err
    assert "camel_data -i dialectid-model26" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


@pytest.mark.parametrize("threshold", ["0", "-1", "2"])
def test_check_dialect_bad_threshold_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: str
) -> None:
    def never() -> object:
        raise AssertionError("model must not be loaded")

    monkeypatch.setattr(cli, "load_camel_model", never)
    script = tmp_path / "script.txt"
    script.write_text("\u0643\u062a\u0628\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["check-dialect", str(script), "--threshold", threshold])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def _never_load() -> object:
    raise AssertionError("model must not be loaded")


def test_check_dialect_refuses_report_equal_to_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_camel_model", _never_load)
    monkeypatch.setenv("COLUMNS", "300")
    script = tmp_path / "script.txt"
    script.write_text("\u0643\u062a\u0628\n", encoding="utf-8")
    link = tmp_path / "alias.json"
    link.symlink_to(script)
    hard = tmp_path / "hard.json"
    hard.hardlink_to(script)
    for report in (script, link, hard):
        code = run_main(monkeypatch, "check-dialect", str(script), "--report", str(report))
        err = capsys.readouterr().err
        assert code == 1
        assert len(err.strip().splitlines()) == 1
    assert script.read_text(encoding="utf-8") == "\u0643\u062a\u0628\n"


def test_check_dialect_unwritable_report_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_camel_model", _never_load)
    monkeypatch.setenv("COLUMNS", "300")
    script = tmp_path / "script.txt"
    script.write_text("\u0643\u062a\u0628\n", encoding="utf-8")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("chmod does not restrict writes here (root or unsupported filesystem)")
        report = locked / "r.json"
        code = run_main(monkeypatch, "check-dialect", str(script), "--report", str(report))
        err = capsys.readouterr().err
        assert code == 1
        assert "--report PATH" in err
        assert len(err.strip().splitlines()) == 1
        missing = tmp_path / "nope" / "r.json"
        code = run_main(monkeypatch, "check-dialect", str(script), "--report", str(missing))
        assert code == 1
        assert "--report PATH" in capsys.readouterr().err
    finally:
        locked.chmod(0o700)


def test_check_dialect_directory_report_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_camel_model", _never_load)
    monkeypatch.setenv("COLUMNS", "300")
    script = tmp_path / "script.txt"
    script.write_text("\u0643\u062a\u0628\n", encoding="utf-8")
    folder = tmp_path / "out"
    folder.mkdir()
    code = run_main(monkeypatch, "check-dialect", str(script), "--report", str(folder))
    assert code == 1
    assert "--report PATH" in capsys.readouterr().err
