from fractions import Fraction

import pytest

from narratoflow.render.layout import Canvas


@pytest.fixture
def tiny_canvas() -> Canvas:
    return Canvas(Fraction(1, 6))
