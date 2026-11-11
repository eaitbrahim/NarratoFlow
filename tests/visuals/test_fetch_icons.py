import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from narratoflow.library.keys import parse_idea_key
from narratoflow.visuals.bundled import BundledIcons

ROOT = Path(__file__).parents[2]
FAKE_URL = "https://example.org/fake-icons.tgz"
CIRCLE = "M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0"
# A filled square drawn as many parallel lines, closer together than the stroke is wide.
HATCH = "".join(f"M4 {4 + i * 0.25}h16" for i in range(65))


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tarball() -> bytes:
    nodes = {
        "circle": [["path", {"d": CIRCLE}]],
        "blob": [["path", {"d": CIRCLE, "fill": "currentColor"}]],
        "hatch": [["path", {"d": HATCH}]],
    }
    members = {
        "package/tabler-nodes-outline.json": json.dumps(nodes).encode(),
        "package/LICENSE": b"MIT License\n\nCopyright (c) test\n",
        "package/icons/outline/circle.svg": b"<svg/>",
        "package/icons/outline/blob.svg": b"<svg/>",
        "package/icons/outline/hatch.svg": b"<svg/>",
        "package/icons/outline/ghost.svg": b"<svg/>",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, blob in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


def test_build_pack_pins_sha_and_refuses_bad_icons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = _script("fetch_icons")
    data = _tarball()
    good = fetch.IconSpec("circle", "ring", "a plain ring, wholeness or a cycle")
    monkeypatch.setattr(fetch, "URL", FAKE_URL)
    monkeypatch.setattr(fetch, "SHA256", hashlib.sha256(data).hexdigest())

    monkeypatch.setattr(fetch, "ICONS", (good,))
    wrong = tmp_path / "wrong"
    with pytest.raises(ValueError, match="sha256"):
        fetch.build_pack(data + b"x", wrong)
    assert not wrong.exists() or not any(wrong.iterdir())

    bad = tmp_path / "bad"
    monkeypatch.setattr(
        fetch,
        "ICONS",
        (
            good,
            fetch.IconSpec("ghost", "ghost-shape", "not in the nodes file"),
            fetch.IconSpec("blob", "blob", "has a fill attribute"),
            fetch.IconSpec("hatch", "hatch", "a filled square"),
        ),
    )
    with pytest.raises(ValueError) as exc:
        fetch.build_pack(data, bad)
    message = str(exc.value)
    assert "ghost" in message
    assert "blob" in message
    assert "hatch" in message
    assert "circle" not in message
    assert not bad.exists() or not any(bad.iterdir())

    out = tmp_path / "good"
    monkeypatch.setattr(fetch, "ICONS", (good,))
    text = fetch.build_pack(data, out)
    png = out / "ring.png"
    with Image.open(png) as image:
        assert image.size == (512, 512)
        assert image.mode == "L"
        pixels = np.asarray(image)
    assert pixels[0, 0] == 255
    assert pixels.min() == 0
    assert (out / "LICENSE-tabler.txt").read_bytes().startswith(b"MIT License")
    entry = tomllib.loads(text)["icons"]["ring|generic"]
    assert entry == {
        "file": "ring.png",
        "description": "a plain ring, wholeness or a cycle",
        "license": "MIT",
        "source": f"{FAKE_URL}#package/icons/outline/circle.svg",
        "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
    }
    (out / "index.toml").write_text(text, encoding="utf-8")
    icons = BundledIcons.from_index(out / "index.toml")
    assert len(icons) == 1
    assert icons.get(parse_idea_key("ring|generic")) is not None


def test_write_pack_removes_pngs_the_new_index_no_longer_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = _script("fetch_icons")
    data = _tarball()
    good = fetch.IconSpec("circle", "ring", "a plain ring, wholeness or a cycle")
    monkeypatch.setattr(fetch, "URL", FAKE_URL)
    monkeypatch.setattr(fetch, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(fetch, "ICONS", (good,))
    out = tmp_path / "icons"
    out.mkdir()
    (out / "dropped.png").write_bytes(b"old")
    (out / "notes.txt").write_text("keep", encoding="utf-8")

    fetch.write_pack(data, out)

    assert sorted(p.name for p in out.glob("*.png")) == ["ring.png"]
    assert (out / "notes.txt").exists()
    assert (
        "ring|generic" in tomllib.loads((out / "index.toml").read_text(encoding="utf-8"))["icons"]
    )
