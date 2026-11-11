import shutil
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from narratoflow import cli
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
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
)
from narratoflow.settings import Settings, get_settings

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
    assert "exceeds limit" in captured.err
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
