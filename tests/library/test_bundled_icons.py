import logging
from pathlib import Path

import pytest

from narratoflow.library.keys import parse_idea_key
from narratoflow.planning.ideas import IdeaOut, validate_idea
from narratoflow.visuals import bundled as bundled_module
from narratoflow.visuals.bundled import BundledIcon, BundledIcons, default_bundled_icons
from narratoflow.visuals.lineart import binarize, check_density
from narratoflow.visuals.strokes import extract_strokes


def test_missing_index_warns_once_and_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(bundled_module, "_dev_index", lambda: tmp_path / "nope" / "index.toml")
    with caplog.at_level(logging.WARNING, logger=bundled_module.logger.name):
        icons = default_bundled_icons()
    assert icons.get(parse_idea_key("eye|generic")) is None
    [record] = caplog.records
    assert record.getMessage() == "bundled icon index not found; bundled icons disabled"


ROOT = Path(__file__).parents[2]
ICONS_DIR = ROOT / "assets" / "icons"
TABLER_URL = "https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz"
CULTURE_DENYLIST = frozenset(
    {
        "church",
        "mosque",
        "synagogue",
        "temple",
        "cathedral",
        "pagoda",
        "cross",
        "crescent",
        "menorah",
        "torii",
        "dollar",
        "euro",
        "pound",
        "yen",
        "rupee",
        "lira",
        "dirham",
        "riyal",
        "bitcoin",
        "brand",
        "zodiac",
        "currency",
    }
)


def _shipped() -> list[BundledIcon]:
    icons = BundledIcons.from_index(ICONS_DIR / "index.toml")
    found = [icons.get(parse_idea_key(key)) for key, _ in icons.catalog()]
    return [icon for icon in found if icon is not None]


def test_shipped_pack_has_at_least_150_mit_icons() -> None:
    icons = _shipped()
    assert len(icons) >= 150
    for icon in icons:
        assert icon.license == "MIT"
        assert icon.source.startswith(TABLER_URL + "#package/icons/outline/")
    assert (ICONS_DIR / "LICENSE-tabler.txt").is_file()
    assert "Tabler Icons 3.49.0" in (ROOT / "assets" / "LICENSES.md").read_text(encoding="utf-8")


def test_no_orphan_png_ships_beside_the_index() -> None:
    listed = {icon.file.name for icon in _shipped()}
    on_disk = {path.name for path in ICONS_DIR.glob("*.png")}
    assert on_disk == listed


def test_every_shipped_icon_obeys_the_drawing_rules() -> None:
    icons = _shipped()
    assert icons
    for icon in icons:
        key = parse_idea_key(icon.key)
        assert key.is_generic
        out = IdeaOut(
            idea_key=icon.key,
            description=icon.description,
            depicts_person=False,
            depicts_logo=False,
        )
        assert validate_idea(out) == [], icon.key
        assert not set(key.object.split("-")) & CULTURE_DENYLIST, icon.key
        member = icon.source.rsplit("/", 1)[-1]
        assert not member.startswith(("brand-", "currency-", "zodiac-")), icon.key


def test_every_shipped_icon_becomes_strokes() -> None:
    icons = _shipped()
    assert icons
    for icon in icons:
        ink = binarize(icon.file.read_bytes())
        report = check_density(ink, line_weight_px=3.0)
        assert report.ok, (icon.key, report.reason)
        assert extract_strokes(ink).strokes, icon.key
