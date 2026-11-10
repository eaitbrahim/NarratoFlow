import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from narratoflow.settings import get_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("NARRATOFLOW_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "data"
    monkeypatch.setenv("NARRATOFLOW_DATA_DIR", str(path))
    return path


@pytest.fixture
def channel_file(tmp_path: Path) -> Path:
    """A copy of the history channel TOML with a generated reference image."""
    from PIL import Image

    work = tmp_path / "channel"
    work.mkdir()
    target = work / "history.toml"
    shutil.copy(FIXTURES / "channels" / "history.toml", target)
    Image.new("RGB", (4, 4), "white").save(work / "ref.png")
    return target
