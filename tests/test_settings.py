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


def test_unknown_key_in_dotenv_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NARRATOFLOW_IMAGE_APIKEY=sk-SECRET123\n", encoding="utf-8")
    s = get_settings()
    assert s.image_api_key is None
    assert "sk-SECRET123" not in repr(s)


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
