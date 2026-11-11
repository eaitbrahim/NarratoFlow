import logging
from pathlib import Path

import pytest

from narratoflow.library.keys import parse_idea_key
from narratoflow.visuals import bundled as bundled_module
from narratoflow.visuals.bundled import default_bundled_icons


def test_missing_index_warns_once_and_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(bundled_module, "_dev_index", lambda: tmp_path / "nope" / "index.toml")
    with caplog.at_level(logging.WARNING, logger=bundled_module.logger.name):
        icons = default_bundled_icons()
    assert icons.get(parse_idea_key("eye|generic")) is None
    [record] = caplog.records
    assert record.getMessage() == "bundled icon index not found; bundled icons disabled"
