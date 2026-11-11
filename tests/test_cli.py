import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from narratoflow import cli
from narratoflow.channels.config import load_channel, style_fingerprint
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import Language, Project
from narratoflow.settings import get_settings

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
        version, "alpha", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov
    )
    keys = runner.invoke(cli.app, ["library", "keys", str(channel_file)])
    assert keys.exit_code == 0, keys.output
    assert keys.output.splitlines() == ["alpha"]

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
