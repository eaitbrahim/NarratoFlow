import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from narratoflow.errors import RenderError
from narratoflow.library.blobs import BlobStore
from narratoflow.library.db import open_library
from narratoflow.library.repository import SketchLibrary, clear_library_cache
from narratoflow.render.ffmpeg import RenderTools, find_render_tools
from narratoflow.settings import Settings, get_settings

FIXTURES = Path(__file__).parent / "fixtures"

# Read at import, before the autouse _isolate_env fixture strips NARRATOFLOW_*.
_OWNER_FFMPEG = Path(v) if (v := os.environ.get("NARRATOFLOW_FFMPEG")) else None


@pytest.fixture
def render_tools() -> RenderTools:
    """An ffmpeg with libass: skips locally without one, fails under CI."""
    try:
        return find_render_tools(Settings(ffmpeg=_OWNER_FFMPEG))
    except RenderError as exc:
        if os.environ.get("CI"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.fixture
def probe_tools() -> RenderTools:
    """Plain ffmpeg/ffprobe (no libass needed): enough for every check before a frame."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required in CI")
        pytest.skip("ffmpeg not installed")
    return RenderTools(Path(ffmpeg), Path(ffprobe), "ffmpeg version test")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("NARRATOFLOW_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    clear_library_cache()


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


@pytest.fixture
def library(tmp_path: Path) -> Iterator[SketchLibrary]:
    engine = open_library(tmp_path / "library.db")
    try:
        yield SketchLibrary(engine, BlobStore(tmp_path / "blobs"))
    finally:
        engine.dispose()
