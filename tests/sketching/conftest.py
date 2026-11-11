from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import images as img
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.settings import Settings
from narratoflow.sketching import generate as generate_module
from narratoflow.visuals.bundled import BundledIcons
from sketching.support import PAID_VISION


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", image_api_key=SecretStr("img-key"))


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    # a real line drawing as the reference sketch (the fixture writes a 4x4 white PNG)
    ch = load_channel(channel_file)
    ch.style.reference_sketch.write_bytes(img.scroll(size=256))
    return ch.model_copy(update={"vision": PAID_VISION})


@pytest.fixture(autouse=True)
def no_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_module, "default_bundled_icons", lambda: BundledIcons({}))
