import hashlib
import importlib.util
import io
import pickle
import re
import shutil
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from narratoflow.errors import ChannelConfigError, RenderError
from narratoflow.render import ffmpeg as F
from narratoflow.render.assets import FontPack, HandSprite, default_fonts, default_hand
from narratoflow.render.ffmpeg import RenderTools, check_arabic_shaping

ROOT = Path(__file__).parents[2]
FONTS = ROOT / "assets" / "fonts"
HANDS = ROOT / "assets" / "hands"
SEEN = "\u0633"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fonts_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "fonts"
    shutil.copytree(FONTS, dest)
    return dest / "index.toml"


def _hands_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "hands"
    shutil.copytree(HANDS, dest)
    return dest / "index.toml"


def _edit(index: Path, old: str, new: str) -> None:
    text = index.read_text(encoding="utf-8")
    assert old in text
    index.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_font_index_validation_errors(tmp_path: Path) -> None:
    cases = [
        ('license = "OFL-1.1"', 'license = "MIT"', "license"),
        ('license_file = "OFL-Amiri.txt"', 'license_file = "missing.txt"', "missing.txt"),
        ('license_file = "OFL-Amiri.txt"', 'license_file = "OFL-Amiri.pdf"', ".txt"),
        ('family = "Amiri"', 'family = "Nope"', "family"),
        ('file = "Amiri-Regular.ttf"', 'file = "../x/Amiri-Regular.ttf"', "relative"),
        ('file = "Amiri-Regular.ttf"', 'file = "Amiri-Regular.otf"', ".ttf"),
    ]
    for i, (old, new, needle) in enumerate(cases):
        d = tmp_path / str(i)
        d.mkdir()
        index = _fonts_copy(d)
        _edit(index, old, new)
        with pytest.raises(ChannelConfigError) as exc:
            FontPack.from_index(index)
        assert "'amiri-regular'" in str(exc.value)
        assert needle in str(exc.value)


def test_font_sha_mismatch_names_entry(tmp_path: Path) -> None:
    index = _fonts_copy(tmp_path)
    (index.parent / "Amiri-Regular.ttf").write_bytes(b"not a font")
    with pytest.raises(ChannelConfigError, match=r"amiri-regular.*sha256"):
        FontPack.from_index(index)


def test_font_unreadable_file_with_matching_sha(tmp_path: Path) -> None:
    index = _fonts_copy(tmp_path)
    blob = b"not a font"
    (index.parent / "Amiri-Regular.ttf").write_bytes(blob)
    text = index.read_text(encoding="utf-8")
    old = re.search(r'sha256 = "([0-9a-f]+)"', text)
    assert old is not None
    index.write_text(text.replace(old.group(1), hashlib.sha256(blob).hexdigest(), 1))
    with pytest.raises(ChannelConfigError, match=r"amiri-regular.*cannot read"):
        FontPack.from_index(index)


def test_font_index_structure_errors(tmp_path: Path) -> None:
    bad = tmp_path / "index.toml"
    bad.write_text("[fonts]\nx = 1\n")
    with pytest.raises(ChannelConfigError, match="'x'"):
        FontPack.from_index(bad)
    bad.write_text('[fonts.x]\nfile = "a.ttf"\nextra = "1"\n')
    with pytest.raises(ChannelConfigError, match="unknown fields"):
        FontPack.from_index(bad)
    bad.write_text("not toml [")
    with pytest.raises(ChannelConfigError, match="broken"):
        FontPack.from_index(bad)
    bad.write_text('[fonts.x]\nfile = "a.ttf"\n')
    with pytest.raises(ChannelConfigError, match="family is required"):
        FontPack.from_index(bad)


def test_committed_fonts_load_with_families(tmp_path: Path) -> None:
    pack = default_fonts()
    assert pack.ids() == [
        "amiri-bold",
        "amiri-quran",
        "amiri-regular",
        "notosans-bold",
        "notosans-regular",
    ]
    assert pack.get("amiri-quran").family == "Amiri Quran"
    assert pack.get("notosans-bold").family == "Noto Sans"
    assert pack.get("amiri-regular").license == "OFL-1.1"
    assert pack.get("amiri-regular").license_file.name == "OFL-Amiri.txt"
    pack.copy_to(tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        pack.get(i).file.name for i in pack.ids()
    )


def test_amiri_quran_covers_fixture_corpus() -> None:
    lines = (ROOT / "tests/fixtures/quran/quran-uthmani.txt").read_text(encoding="utf-8")
    pack = default_fonts()
    seen = 0
    for line in lines.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, _, text = line.split("|", 2)
        assert pack.missing_glyphs("amiri-quran", text) == [], line
        seen += 1
    assert seen > 0


def test_missing_glyphs_reports_codepoints() -> None:
    pack = default_fonts()
    text = f"ab {SEEN}{SEEN}\u200c\n"
    assert pack.missing_glyphs("notosans-regular", text) == ["U+0633"]
    assert pack.missing_glyphs("notosans-regular", "plain") == []


def test_hand_default_loads() -> None:
    hand = default_hand()
    assert hand.license == "Apache-2.0"
    assert hand.source == "scripts/make_hand.py"
    assert hand.tip == (24, 24)


def test_hand_index_validation_and_tip_alpha(tmp_path: Path) -> None:
    def broken(i: int, old: str, new: str) -> Path:
        d = tmp_path / str(i)
        d.mkdir()
        index = _hands_copy(d)
        _edit(index, old, new)
        return index

    cases = [
        ('license = "Apache-2.0"', 'license = "GPL-3.0"', "license"),
        ('license = "Apache-2.0"', 'license = "CC-BY-4.0"', "attribution"),
        ("tip_x = 24", "tip_x = 9999", "outside"),
        ("tip_x = 24", 'tip_x = "a"', "integer"),
        ('file = "hand-right.png"', 'file = "hand-right.jpg"', ".png"),
        ('sha256 = "', 'sha256 = "00', "sha256"),
        ("[hands.right]", "[hands.left]", "missing"),
        ("tip_y = 24", "tip_y = 24\nbogus = 1", "unknown"),
    ]
    for i, (old, new, needle) in enumerate(cases):
        with pytest.raises(ChannelConfigError, match=needle):
            HandSprite.from_index(broken(i, old, new))
    (tmp_path / "ok").mkdir()
    ok = _hands_copy(tmp_path / "ok")
    _edit(ok, 'license = "Apache-2.0"', 'license = "CC-BY-4.0"\nattribution = "me"')
    assert HandSprite.from_index(ok).attribution == "me"

    # transparent tip and non-RGBA files, each with a matching sha256
    for name, image in (
        ("clear", Image.new("RGBA", (10, 10), (0, 0, 0, 0))),
        ("rgb", Image.new("RGB", (10, 10), (0, 0, 0))),
    ):
        d = tmp_path / name
        d.mkdir()
        buf = io.BytesIO()
        image.save(buf, "PNG")
        (d / "h.png").write_bytes(buf.getvalue())
        (d / "index.toml").write_text(
            '[hands.right]\nfile = "h.png"\nlicense = "MIT"\nsource = "x"\ntip_x = 1\ntip_y = 1\n'
            f'sha256 = "{hashlib.sha256(buf.getvalue()).hexdigest()}"\n'
        )
        with pytest.raises(ChannelConfigError, match=r"transparent|RGBA"):
            HandSprite.from_index(d / "index.toml")


def test_hand_image_cache_is_module_level_and_sprite_pickles() -> None:
    hand = default_hand()
    assert len(pickle.dumps(hand)) < 1000
    assert pickle.loads(pickle.dumps(hand)) == hand  # noqa: S301
    a, _ = hand.image(Fraction(1, 2))
    b, _ = hand.image(Fraction(1, 2))
    assert a is b
    assert not a.flags.writeable
    assert vars(hand).keys() == {
        "id", "file", "license", "source", "attribution", "tip", "sha256"
    }  # fmt: skip


def test_hand_image_scaled_tip() -> None:
    hand = default_hand()
    full, tip = hand.image(Fraction(1))
    assert full.shape == (440, 320, 4)
    assert tip == (24, 24)
    half, tip = hand.image(Fraction(1, 2))
    assert half.shape == (220, 160, 4)
    assert tip == (12, 12)
    assert half[tip[1], tip[0], 3] > 0
    tiny, tip = hand.image(Fraction(1, 6))
    assert tiny.dtype == np.uint8
    assert tip == (4, 4)


def _fake_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_fetch_fonts_pins_zip_sha_and_members(tmp_path: Path) -> None:
    mod = _script("fetch_fonts")
    files = {"d/A.ttf": b"font-bytes", "d/OFL.txt": b"licence", "d/junk.ttf": b"junk"}
    data = _fake_zip(files)
    font = mod.FontMember("a", "d/A.ttf", "A.ttf", "A")

    def spec(sha: str, member: str = "d/A.ttf") -> object:
        return mod.ZipSpec(
            "u", sha, (mod.FontMember("a", member, "A.ttf", "A"),), "d/OFL.txt", "OFL-A.txt"
        )

    good = spec(hashlib.sha256(data).hexdigest())
    out = tmp_path / "out"
    shas = mod.extract_pack(data, good, out)
    assert shas == {"a": hashlib.sha256(b"font-bytes").hexdigest()}
    assert sorted(p.name for p in out.iterdir()) == ["A.ttf", "OFL-A.txt"]  # junk not extracted
    assert font.id == "a"

    with pytest.raises(ValueError, match="zip sha256"):
        mod.extract_pack(data, spec("0" * 64), tmp_path / "bad1")
    with pytest.raises(ValueError, match="missing zip members"):
        mod.extract_pack(data, spec(hashlib.sha256(data).hexdigest(), "d/B.ttf"), tmp_path / "bad2")
    assert not (tmp_path / "bad1").exists()
    assert not (tmp_path / "bad2").exists()
    text = mod.index_text({f.id: "ab" * 32 for s in mod.SPECS for f in s.fonts})
    assert text.count("[fonts.") == 5
    assert all(s.url in text for s in mod.SPECS)


def test_make_hand_is_deterministic() -> None:
    mod = _script("make_hand")
    a, b = mod.draw(), mod.draw()
    assert a.size == (320, 440)
    assert a.tobytes() == b.tobytes()
    assert a.getpixel(mod.TIP)[3] > 0


def test_check_arabic_shaping_passes_with_bundled_amiri(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    check_arabic_shaping(render_tools, default_fonts(), tmp_path)
    ass = (tmp_path / "shaping.ass").read_text(encoding="utf-8")
    assert "PlayResX: 600" in ass
    assert SEEN * 3 in ass
    assert (tmp_path / "fonts" / "Amiri-Regular.ttf").is_file()


def _blobs(n: int) -> np.ndarray:
    frame = np.full((300, 600, 3), 255, dtype=np.uint8)
    for i in range(n):
        frame[100:140, 50 + 100 * i : 90 + 100 * i] = 0
    return frame


def test_check_arabic_shaping_rejects_unjoined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, tuple[int, int]]] = []

    def fake_still(
        tools: RenderTools, work_dir: Path, ass_name: str, size: tuple[int, int], t_ms: int
    ) -> np.ndarray:
        seen.append((ass_name, size))
        assert (work_dir / ass_name).is_file()
        return _blobs(3)

    monkeypatch.setattr(F, "render_ass_still", fake_still)
    tools = RenderTools(Path("/x/ffmpeg"), Path("/x/ffprobe"), "v")
    with pytest.raises(RenderError, match="did not join Arabic letters"):
        check_arabic_shaping(tools, default_fonts(), tmp_path)
    assert seen == [("shaping.ass", (600, 300))]
    monkeypatch.setattr(F, "render_ass_still", lambda *a, **k: _blobs(1))
    check_arabic_shaping(tools, default_fonts(), tmp_path)
    monkeypatch.setattr(F, "render_ass_still", lambda *a, **k: _blobs(0))
    with pytest.raises(RenderError, match="did not join"):
        check_arabic_shaping(tools, default_fonts(), tmp_path)


def test_default_assets_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib.resources

    from narratoflow.render import assets as A

    monkeypatch.setattr(A, "__file__", str(tmp_path / "a" / "b" / "c" / "assets.py"))
    monkeypatch.setattr(importlib.resources, "files", lambda _: tmp_path / "pkg")
    with pytest.raises(ChannelConfigError, match="bundled fonts not found"):
        A.default_fonts()
    with pytest.raises(ChannelConfigError, match="bundled hands not found"):
        A.default_hand()


@pytest.mark.slow
def test_wheel_contains_fonts_and_hand(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(out)],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    [wheel] = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    for name in (
        "fonts/index.toml",
        "fonts/Amiri-Regular.ttf",
        "fonts/AmiriQuran.ttf",
        "fonts/OFL-Amiri.txt",
        "fonts/OFL-NotoSans.txt",
        "hands/index.toml",
        "hands/hand-right.png",
    ):
        assert f"narratoflow/assets/{name}" in names


def test_hand_index_broken_toml(tmp_path: Path) -> None:
    bad = tmp_path / "index.toml"
    bad.write_text("not toml [")
    with pytest.raises(ChannelConfigError, match="broken"):
        HandSprite.from_index(bad)
