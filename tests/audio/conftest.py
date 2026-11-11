from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture
def ffmpeg_bin() -> str:
    """Path to ffmpeg; skips locally when missing, fails in CI."""
    found = shutil.which("ffmpeg")
    if found is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required in CI")
        pytest.skip("ffmpeg not installed")
    return found
