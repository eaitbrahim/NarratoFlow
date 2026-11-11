import subprocess
import zipfile
from datetime import date
from pathlib import Path

import pytest

from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import ChannelConfigError, ReciterError
from narratoflow.quran import reciters as reciters_module
from narratoflow.quran.reciters import Reciter, ReciterIndex, default_reciters, select_reciter

ROOT = Path(__file__).parents[2]
CHECKED = "checked = 2026-11-11"
COMMITTED = ROOT / "assets" / "reciters" / "index.toml"

FILES_ENTRY = """
[[reciters]]
id = "{id}"
name = "Test Reciter"
riwaya = "{riwaya}"
kind = "files"
audio_url = "https://audio.example.org/test/{{surah:03d}}{{ayah:03d}}.mp3"
license = "CC-BY-4.0"
license_url = "https://creativecommons.org/licenses/by/4.0/"
source = "https://audio.example.org/test/"
checked = 2026-11-11
"""

QURANCOM_ENTRY = """
[[reciters]]
id = "qc-hafs"
name = "QC Reciter"
riwaya = "hafs"
kind = "qurancom"
recitation_id = 7
license = "Example licence"
license_url = "https://example.org/terms"
source = "https://quran.com/"
segments_source = "quran.com"
checked = 2026-11-11
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "index.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _files(id_: str = "test-hafs", riwaya: str = "hafs") -> str:
    return FILES_ENTRY.format(id=id_, riwaya=riwaya)


def _channel(**over: object) -> ChannelConfig:
    base: dict[str, object] = {
        "name": "T",
        "slug": "t",
        "language": "ar",
        "style": {
            "reference_sketch": "ref.png",
            "instructions": "ink",
            "image_provider": "openai",
            "image_model": "gpt-image-1",
        },
        "voice": {"provider": "edge", "voice_id": "ar-SA-HamedNeural"},
        "llm": {"provider": "ollama", "model": "qwen2.5"},
    }
    base.update(over)
    return ChannelConfig.model_validate(base)


def test_valid_index_loads(tmp_path: Path) -> None:
    index = ReciterIndex.from_index(_write(tmp_path, _files() + QURANCOM_ENTRY))
    files = index.get("test-hafs")
    assert files is not None
    assert files.kind == "files"
    assert files.checked == date(2026, 11, 11)
    assert files.segments_source is None
    qc = index.get("qc-hafs")
    assert qc is not None
    assert qc.recitation_id == 7
    assert [r.id for r in index.for_riwaya("hafs")] == ["test-hafs", "qc-hafs"]
    assert index.for_riwaya("warsh") == []
    assert index.get("nobody") is None


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (_files().replace('license = "CC-BY-4.0"\n', ""), "license"),
        (_files().replace('license = "CC-BY-4.0"', 'license = "  "'), "license"),
        (_files().replace("https://creativecommons", "http://creativecommons"), "https"),
        (_files().replace('source = "https://', 'source = "http://'), "https"),
        (_files().replace('audio_url = "https://', 'audio_url = "http://'), "https"),
        (_files() + _files(), "duplicate"),
        (QURANCOM_ENTRY.replace("recitation_id = 7\n", ""), "recitation_id"),
        (QURANCOM_ENTRY.replace('segments_source = "quran.com"\n', ""), "segments_source"),
        (QURANCOM_ENTRY.replace('kind = "qurancom"', 'kind = "files"'), "audio_url"),
        (
            _files().replace(CHECKED, CHECKED + "\nrecitation_id = 3"),
            "recitation_id",
        ),
        (
            _files().replace(CHECKED, CHECKED + '\nsegments_source = "x"'),
            "segments_source",
        ),
        (_files().replace("{surah:03d}{ayah:03d}", "{surah}{ayah}"), "audio_url"),
        (_files().replace('kind = "files"', 'kind = "youtube"'), "kind"),
        (_files().replace('id = "test-hafs"', 'id = "Bad_Id"'), "id"),
        (_files().replace(CHECKED, CHECKED + "\nextra = 1"), "extra"),
        (_files().replace("checked = 2026-11-11\n", ""), "checked"),
        ("[[reciters]]\nid = 3\n", "entry 1"),
        ("reciters = 3\n", "reciters"),
        ("reciters = [3]\n", "entry 1"),
        ("not toml [", "broken"),
    ],
)
def test_index_validation_errors(tmp_path: Path, body: str, fragment: str) -> None:
    with pytest.raises(ChannelConfigError) as exc:
        ReciterIndex.from_index(_write(tmp_path, body))
    assert fragment in str(exc.value)


def test_index_errors_name_the_entry(tmp_path: Path) -> None:
    body = _files() + _files("bad-one").replace("https://creativecommons", "http://x")
    with pytest.raises(ChannelConfigError, match="bad-one"):
        ReciterIndex.from_index(_write(tmp_path, body))


def test_missing_index_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ChannelConfigError):
        ReciterIndex.from_index(tmp_path / "absent.toml")


def test_committed_index_loads() -> None:
    index = ReciterIndex.from_index(COMMITTED)
    text = COMMITTED.read_text(encoding="utf-8")
    assert text.isascii()
    entries = list(index)
    for r in entries:
        for value in (r.name, r.riwaya, r.license, r.license_url, r.source):
            assert value.strip()
        for url in (r.license_url, r.source, r.audio_url or "https://"):
            assert url.startswith("https://")
        assert r.checked <= date.today()
    if not entries:
        assert "unlicensed recitation is not allowed" in text
        assert "no reciter is shipped" in text.lower()
    licenses = (ROOT / "assets" / "LICENSES.md").read_text(encoding="utf-8")
    for r in entries:
        assert r.id in licenses


def test_default_reciters_uses_dev_checkout() -> None:
    assert list(default_reciters()) == list(ReciterIndex.from_index(COMMITTED))


def test_default_reciters_missing_warns_and_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(reciters_module, "_packaged_index", lambda: tmp_path / "a.toml")
    monkeypatch.setattr(reciters_module, "_dev_index", lambda: tmp_path / "b.toml")
    index = default_reciters()
    assert list(index) == []
    assert "reciter" in caplog.text
    with pytest.raises(ReciterError, match="no allow-listed reciter for riwaya hafs"):
        select_reciter(_channel(), index)


def test_select_reciter_riwaya_mismatch_and_not_listed_and_none(tmp_path: Path) -> None:
    index = ReciterIndex.from_index(
        _write(tmp_path, _files("warsh-one", "warsh") + _files("hafs-one", "hafs"))
    )
    # Default: the first allow-listed reciter of the channel's riwaya.
    assert select_reciter(_channel(), index).id == "hafs-one"
    assert select_reciter(_channel(riwaya="warsh"), index).id == "warsh-one"
    assert select_reciter(_channel(reciter="hafs-one"), index).id == "hafs-one"
    with pytest.raises(ReciterError, match="reciter ghost is not on the allow-list"):
        select_reciter(_channel(reciter="ghost"), index)
    with pytest.raises(ReciterError, match="warsh-one") as exc:
        select_reciter(_channel(reciter="warsh-one"), index)
    assert "hafs" in str(exc.value)
    with pytest.raises(
        ReciterError,
        match="no allow-listed reciter for riwaya qalun; unlicensed recitation is not allowed",
    ):
        select_reciter(_channel(riwaya="qalun"), index)


def test_reciter_model_forbids_extra() -> None:
    with pytest.raises(ValueError, match="extra"):
        Reciter.model_validate(
            {
                "id": "a",
                "name": "n",
                "riwaya": "hafs",
                "kind": "files",
                "audio_url": "https://x.org/{surah:03d}{ayah:03d}.mp3",
                "license": "l",
                "license_url": "https://x.org",
                "source": "https://x.org",
                "checked": "2026-11-11",
                "bogus": 1,
            }
        )


@pytest.mark.slow
def test_wheel_contains_the_reciter_index(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(out)],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    [wheel] = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert "narratoflow/assets/reciters/index.toml" in archive.namelist()
