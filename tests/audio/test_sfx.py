from __future__ import annotations

import itertools
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fixtures.audio import tone

from narratoflow.audio import io as aio
from narratoflow.audio import sfx as sfx_module
from narratoflow.audio.sfx import (
    MIN_SFX_STROKE_MS,
    SFX_GAIN_DB,
    SFX_MIN_SPACING_MS,
    SfxAsset,
    SfxPack,
    default_sfx,
    place_sfx,
)
from narratoflow.errors import ChannelConfigError
from narratoflow.visuals.schedule import StrokeEvent

ROOT = Path(__file__).parents[2]


def _index(tmp_path: Path, body: str, files: tuple[str, ...] = ("a.flac",)) -> Path:
    for name in files:
        (tmp_path / name).write_bytes(b"fLaC")
    index = tmp_path / "index.toml"
    index.write_text(body, encoding="utf-8")
    return index


_OK = 'file = "a.flac"\nlicense = "Apache-2.0"\nsource = "s"\n'


def test_valid_index_loads(tmp_path: Path) -> None:
    pack = SfxPack.from_index(_index(tmp_path, f"[sfx.b]\n{_OK}\n[sfx.a]\n{_OK}"))
    assert pack.ids() == ["a", "b"]


@pytest.mark.parametrize(
    ("entry", "needle"),
    [
        ('file = "a.flac"\nlicense = "GPL-3.0"\nsource = "s"\n', "license"),
        ('file = "a.flac"\nlicense = "CC-BY-4.0"\nsource = "s"\n', "attribution"),
        ('file = "../a.flac"\nlicense = "MIT"\nsource = "s"\n', "relative path"),
        ('file = "a.wav"\nlicense = "MIT"\nsource = "s"\n', ".flac"),
        ('file = "missing.flac"\nlicense = "MIT"\nsource = "s"\n', "does not exist"),
        ('file = "a.flac"\nlicense = "MIT"\nsource = "s"\nextra = 1\n', "unknown"),
        ('file = "a.flac"\nlicense = "MIT"\n', "source"),
    ],
)
def test_sfx_index_validation_errors(tmp_path: Path, entry: str, needle: str) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    index = _index(tmp_path, f"[sfx.bad]\n{entry}")
    with pytest.raises(ChannelConfigError) as exc:
        SfxPack.from_index(index)
    assert "bad" in str(exc.value)
    assert needle in str(exc.value)


def test_cc_by_with_attribution_is_accepted(tmp_path: Path) -> None:
    body = _OK.replace("Apache-2.0", "CC-BY-4.0") + 'attribution = "by X"\n'
    pack = SfxPack.from_index(_index(tmp_path, f"[sfx.a]\n{body}"))
    assert pack.ids() == ["a"]


def test_broken_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ChannelConfigError):
        SfxPack.from_index(_index(tmp_path, "[sfx.a\n"))


def test_committed_pack_loads_and_each_sample_under_400ms(ffmpeg_bin: str) -> None:
    pack = SfxPack.from_index(ROOT / "assets" / "sfx" / "index.toml")
    assert len(pack.ids()) == 4
    for sid in pack.ids():
        samples = pack.samples(sid)
        assert 0 < len(samples) <= aio.ms_to_samples(400)
        assert float(np.max(np.abs(samples))) <= 0.6


def test_default_sfx_warns_without_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sfx_module, "_packaged_index", lambda: tmp_path / "a.toml")
    monkeypatch.setattr(sfx_module, "_dev_index", lambda: tmp_path / "b.toml")
    pack = default_sfx()
    assert pack.ids() == []
    assert "SFX" in caplog.text


def _fake_pack(ids: tuple[str, ...] = ("m1", "m2", "m3", "m4"), ms: int = 150) -> SfxPack:
    pack = SfxPack({i: SfxAsset(id=i, file=Path("x.flac"), license="MIT", source="s") for i in ids})
    for i in ids:
        pack._cache[i] = tone(ms)
    return pack


def _events(durations: list[int]) -> list[StrokeEvent]:
    out, t = [], 0
    for i, d in enumerate(durations):
        out.append(StrokeEvent(i, t, t + d))
        t += d
    return out


def test_place_sfx_spacing_and_min_duration() -> None:
    durations = [60, 200] * 20  # 40 strokes
    events = _events(durations)
    hits = place_sfx(events, scene_id="s", seed=1, pack=_fake_pack(), blocked=[])
    assert hits
    for h in hits:
        assert durations[h.stroke_index] >= MIN_SFX_STROKE_MS
        assert h.at_ms == events[h.stroke_index].start_ms
        assert h.gain_db == SFX_GAIN_DB
        assert h.duration_ms <= durations[h.stroke_index]
    assert all(b.at_ms - a.at_ms >= SFX_MIN_SPACING_MS for a, b in itertools.pairwise(hits))


def test_place_sfx_duration_is_min_of_sample_and_stroke() -> None:
    hits = place_sfx(_events([130, 300, 500]), scene_id="s", seed=0, pack=_fake_pack(), blocked=[])
    assert [h.duration_ms for h in hits] == [130, 150]


def test_place_sfx_respects_blocked_window() -> None:
    events = _events([200] * 10)
    hits = place_sfx(events, scene_id="s", seed=1, pack=_fake_pack(), blocked=[(500, 1100)])
    assert hits
    for h in hits:
        end = h.at_ms + 200
        assert end <= 500 or h.at_ms >= 1100


def test_place_sfx_deterministic_per_seed() -> None:
    events = _events([200] * 30)
    a = place_sfx(events, scene_id="s", seed=7, pack=_fake_pack(), blocked=[])
    b = place_sfx(events, scene_id="s", seed=7, pack=_fake_pack(), blocked=[])
    c = place_sfx(events, scene_id="s", seed=8, pack=_fake_pack(), blocked=[])
    assert a == b
    assert [h.asset_id for h in a] != [h.asset_id for h in c]


def test_place_sfx_empty_pack() -> None:
    assert place_sfx(_events([200] * 3), scene_id="s", seed=1, pack=SfxPack({}), blocked=[]) == []


@pytest.mark.slow
def test_wheel_contains_sfx_and_reciters(tmp_path: Path) -> None:
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
    assert "narratoflow/assets/sfx/index.toml" in names
    assert "narratoflow/assets/sfx/marker-1.flac" in names
    assert "narratoflow/assets/reciters/index.toml" in names
