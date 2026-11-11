from pathlib import Path

import pytest

from narratoflow.errors import (
    ChannelConfigError,
    CorruptInputError,
    InvalidIdeaKeyError,
    LibraryError,
    MissingApiKeyError,
    NarratoFlowError,
    SettingsError,
)
from narratoflow.logging_setup import setup_logging
from narratoflow.settings import get_settings


def test_defaults_without_env() -> None:
    s = get_settings()
    assert s.log_level == "INFO"
    assert s.image_api_key is None
    assert s.data_dir == Path("~/.narratoflow").expanduser()


def test_key_is_secret_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "sk-test-123")
    s = get_settings()
    assert s.image_api_key is not None
    assert s.image_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in repr(s)


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_LOG_LEVEL", "LOUD")
    with pytest.raises(SettingsError, match="log_level"):
        get_settings()


def test_log_level_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_LOG_LEVEL", "debug")
    assert get_settings().log_level == "DEBUG"


def test_unknown_narratoflow_key_in_dotenv_is_an_error(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NARRATOFLOW_IMAGE_APIKEY=sk-SECRET123\n", encoding="utf-8")
    with pytest.raises(SettingsError) as exc:
        get_settings()
    assert "narratoflow_image_apikey" in str(exc.value).lower()
    assert "sk-SECRET123" not in str(exc.value)
    assert "sk-SECRET123" not in str(exc.value.__cause__)


def test_unknown_narratoflow_env_var_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_LLM_APIKEY", "sk-SECRET456")
    with pytest.raises(SettingsError) as exc:
        get_settings()
    assert "NARRATOFLOW_LLM_APIKEY" in str(exc.value)
    assert "NARRATOFLOW_LLM_API_KEY" in str(exc.value)
    assert "sk-SECRET456" not in str(exc.value)


def test_foreign_keys_in_dotenv_are_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=x\nNARRATOFLOW_LOG_LEVEL=DEBUG\n", encoding="utf-8"
    )
    assert get_settings().log_level == "DEBUG"


def test_environment_beats_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("NARRATOFLOW_LOG_LEVEL=DEBUG\n", encoding="utf-8")
    monkeypatch.setenv("NARRATOFLOW_LOG_LEVEL", "ERROR")
    assert get_settings().log_level == "ERROR"


def test_invalid_value_error_hides_input(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NARRATOFLOW_LOG_LEVEL=sk-SECRET123\n", encoding="utf-8")
    with pytest.raises(SettingsError) as exc:
        get_settings()
    assert "sk-SECRET123" not in str(exc.value)
    assert exc.value.__cause__ is not None
    assert "sk-SECRET123" not in str(exc.value.__cause__)


def test_data_dir_fixture_overrides(data_dir: Path) -> None:
    assert get_settings().data_dir == data_dir


def test_all_project_errors_share_base() -> None:
    for cls in (
        MissingApiKeyError,
        ChannelConfigError,
        InvalidIdeaKeyError,
        LibraryError,
        CorruptInputError,
        SettingsError,
    ):
        assert issubclass(cls, NarratoFlowError)


def test_setup_logging_runs() -> None:
    setup_logging("INFO")


def test_embed_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings().embed_api_key is None
    monkeypatch.setenv("NARRATOFLOW_EMBED_API_KEY", "pa-test-999")
    get_settings.cache_clear()
    s = get_settings()
    assert s.embed_api_key is not None
    assert s.embed_api_key.get_secret_value() == "pa-test-999"
    assert "pa-test-999" not in repr(s)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_ffmpeg_means_unset(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("NARRATOFLOW_FFMPEG", blank)
    assert get_settings().ffmpeg is None
