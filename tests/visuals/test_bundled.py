import logging
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

from narratoflow.errors import ChannelConfigError
from narratoflow.library.keys import parse_idea_key
from narratoflow.visuals import bundled as bundled_module
from narratoflow.visuals.bundled import BundledIcons, default_bundled_icons

ROOT = Path(__file__).parents[2]


def _index(tmp_path: Path, body: str, files: tuple[str, ...] = ("eye.svg",)) -> Path:
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<svg/>", encoding="utf-8")
    index = tmp_path / "index.toml"
    index.write_text(body, encoding="utf-8")
    return index


def _entry(
    key: str = "eye|generic",
    file: str = "eye.svg",
    extra: str = 'license = "CC0"\nsource = "https://example.org"\n',
) -> str:
    return f'[icons."{key}"]\nfile = "{file}"\n{extra}'


def test_valid_index_loads(tmp_path: Path) -> None:
    index = _index(tmp_path, _entry("Eye|generic"))
    icon = BundledIcons.from_index(index).get(parse_idea_key("eye|generic"))
    assert icon is not None
    assert icon.file == tmp_path / "eye.svg"
    assert icon.license == "CC0"


def test_empty_index_loads(tmp_path: Path) -> None:
    index = _index(tmp_path, "[icons]\n")
    assert BundledIcons.from_index(index).get(parse_idea_key("eye|generic")) is None


def test_shipped_index_loads() -> None:
    BundledIcons.from_index(ROOT / "assets" / "icons" / "index.toml")


@pytest.mark.parametrize(
    ("body", "named"),
    [
        (_entry("eye|history"), "eye|history"),
        (_entry("Not A Key"), "Not A Key"),
        (_entry(extra='source = "https://x"\n'), "eye|generic"),
        (_entry(extra='license = "CC0"\n'), "eye|generic"),
        (_entry(extra='license = ""\nsource = "https://x"\n'), "eye|generic"),
        (_entry(extra='license = "CC0"\nsource = "  "\n'), "eye|generic"),
        (_entry(file=""), "eye|generic"),
        (_entry(file="/etc/passwd"), "eye|generic"),
        (_entry(file="a/../eye.svg"), "eye|generic"),
        (_entry(file="../eye.svg"), "eye|generic"),
        (_entry(file="missing.svg"), "eye|generic"),
        (_entry(extra='license = "CC0"\nsource = "s"\nauthor = "me"\n'), "eye|generic"),
        (_entry("Eye|generic") + _entry("eye|generic"), "eye|generic"),
        ('[icons]\n"eye|generic" = "eye.svg"\n', "eye|generic"),
    ],
)
def test_violation_raises_naming_the_entry(tmp_path: Path, body: str, named: str) -> None:
    index = _index(tmp_path, body)
    with pytest.raises(ChannelConfigError, match=re.escape(named)):
        BundledIcons.from_index(index)


def test_symlink_escaping_the_directory_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.svg"
    outside.write_text("<svg/>", encoding="utf-8")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "eye.svg").symlink_to(outside)
    index = pack / "index.toml"
    index.write_text(_entry(), encoding="utf-8")
    with pytest.raises(ChannelConfigError, match=r"eye\|generic"):
        BundledIcons.from_index(index)


def test_duplicate_raw_toml_keys_are_broken(tmp_path: Path) -> None:
    index = _index(tmp_path, _entry() + _entry())
    with pytest.raises(ChannelConfigError):
        BundledIcons.from_index(index)


def test_packaged_index_is_preferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packaged = _index(tmp_path / "pkg", _entry())
    monkeypatch.setattr(bundled_module, "_packaged_index", lambda: packaged)
    monkeypatch.setattr(bundled_module, "_dev_index", lambda: tmp_path / "nope" / "index.toml")
    assert default_bundled_icons().get(parse_idea_key("eye|generic")) is not None


def test_dev_index_is_the_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = _index(tmp_path / "dev", _entry())
    monkeypatch.setattr(bundled_module, "_packaged_index", lambda: tmp_path / "nope" / "i.toml")
    monkeypatch.setattr(bundled_module, "_dev_index", lambda: dev)
    assert default_bundled_icons().get(parse_idea_key("eye|generic")) is not None


def test_no_index_anywhere_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(bundled_module, "_packaged_index", lambda: tmp_path / "a.toml")
    monkeypatch.setattr(bundled_module, "_dev_index", lambda: tmp_path / "b.toml")
    with caplog.at_level(logging.WARNING, logger=bundled_module.logger.name):
        default_bundled_icons()
    assert "bundled icons disabled" in caplog.text


@pytest.mark.slow
def test_wheel_contains_the_icon_index(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(out)],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    [wheel] = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert "narratoflow/assets/icons/index.toml" in archive.namelist()
