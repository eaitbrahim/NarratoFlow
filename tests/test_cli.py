import shutil
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from narratoflow import cli
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
from narratoflow.errors import BudgetExceededError
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import (
    Language,
    Narration,
    Part,
    Project,
    RightsBasis,
    Scene,
    ScriptSource,
    Series,
    VisualSource,
)
from narratoflow.settings import Settings, get_settings
from narratoflow.sketching.generate import GenerationInterruptedError, GenerationResult

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


def test_dotenv_typo_key_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".env").write_text(
        "NARRATOFLOW_IMAGE_APIKEY=sk-SECRET123\nNARRATOFLOW_LOG_LEVEL=debug\n", encoding="utf-8"
    )
    code = run_main(monkeypatch, "check-input", str(INPUTS / "clean_en.txt"), "--language", "en")
    captured = capsys.readouterr()
    assert code == 0, captured.err
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
    from sketching.support import FakeImages

    from narratoflow.sketching import generate as generate_module

    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img-key")
    images = FakeImages()
    monkeypatch.setattr(generate_module, "build_image_provider", lambda c, s: images)

    def no_vision(*args: object) -> object:
        raise AssertionError("vision is off: no vision client is built")

    monkeypatch.setattr(generate_module, "build_llm_client", no_vision)
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
