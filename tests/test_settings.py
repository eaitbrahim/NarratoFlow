from pathlib import Path

import pytest
from pydantic import ValidationError

from narratoflow.errors import (
    ChannelConfigError,
    CorruptInputError,
    InvalidIdeaKeyError,
    LibraryError,
    MissingApiKeyError,
    NarratoFlowError,
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
    with pytest.raises(ValidationError):
        get_settings()


def test_data_dir_fixture_overrides(data_dir: Path) -> None:
    assert get_settings().data_dir == data_dir


def test_all_project_errors_share_base() -> None:
    for cls in (
        MissingApiKeyError,
        ChannelConfigError,
        InvalidIdeaKeyError,
        LibraryError,
        CorruptInputError,
    ):
        assert issubclass(cls, NarratoFlowError)


def test_setup_logging_runs() -> None:
    setup_logging("INFO")
