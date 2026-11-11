import os
from fractions import Fraction
from pathlib import Path

import pytest

from narratoflow.errors import RenderError
from narratoflow.render.ffmpeg import RenderTools, find_render_tools
from narratoflow.render.layout import Canvas
from narratoflow.settings import Settings

# Read at import, before the autouse _isolate_env fixture strips NARRATOFLOW_*.
_OWNER_FFMPEG = Path(v) if (v := os.environ.get("NARRATOFLOW_FFMPEG")) else None


@pytest.fixture
def render_tools() -> RenderTools:
    try:
        return find_render_tools(Settings(ffmpeg=_OWNER_FFMPEG))
    except RenderError as exc:
        if os.environ.get("CI"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.fixture
def tiny_canvas() -> Canvas:
    return Canvas(Fraction(1, 6))
