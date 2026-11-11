import pytest

from narratoflow.planning import pipeline as pipeline_module
from narratoflow.visuals.bundled import BundledIcons


@pytest.fixture(autouse=True)
def no_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shipped icon pack has `scroll|generic`; these tests plan against an empty pack.
    monkeypatch.setattr(pipeline_module, "default_bundled_icons", lambda: BundledIcons({}))
