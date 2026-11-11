from pathlib import Path

import pytest

from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.settings import Settings


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")
