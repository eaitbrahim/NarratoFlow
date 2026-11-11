"""RenderReport: atomic save, strict load."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from narratoflow.audio.timeline import TimingSource
from narratoflow.errors import RenderError
from narratoflow.models import SceneKind
from narratoflow.render.report import RenderAsset, RenderReport, SceneRender, VerseRender
from narratoflow.render.variation import Variation


def _report(**kw: Any) -> RenderReport:
    fields: dict[str, Any] = {
        "series_title": "Scenes",
        "channel_slug": "history",
        "part_index": 1,
        "part_count": 2,
        "video_file": "part-01.mp4",
        "timeline_file": "part-01.timeline.json",
        "audio_file": "part-01.flac",
        "captions_file": "part-01.ass",
        "width": 1080,
        "height": 1920,
        "frames": 90,
        "duration_ms": 3000,
        "crf": 18,
        "ffmpeg_version": "ffmpeg version 8.0",
        "integrated_lufs": -14.0,
        "true_peak_dbtp": None,
        "scenes": [
            SceneRender(
                scene_id="s1",
                kind=SceneKind.NARRATION,
                visual="sketch",
                sketch_id=3,
                frames=90,
                variation=Variation(
                    mirrored=False,
                    rotation_deg=1.5,
                    scale=0.95,
                    dx=0.01,
                    dy=-0.01,
                    line_weight_factor=1.0,
                    reversed_strokes=[0, 2],
                    wobble_seed=7,
                ),
            )
        ],
        "verses": [
            VerseRender(
                scene_id="v1",
                ayat=["31:33"],
                shown_sha256="ab" * 32,
                font_id="amiri-quran",
                timing_source=TimingSource.AYAH,
            )
        ],
        "assets": [
            RenderAsset(kind="hand", asset_id="right", source="s", license="MIT", sha256="cd")
        ],
        "warnings": ["16:9 is not rendered in v1"],
        **kw,
    }
    return RenderReport(**fields)


def test_report_roundtrip_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "part-01.render.json"
    report = _report()
    report.save(path)
    assert RenderReport.load(path) == report
    assert path.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp file left

    raw = json.loads(path.read_text(encoding="utf-8"))
    mutations: list[tuple[Callable[[dict[str, Any]], None], str]] = [
        (lambda d: d.update(extra=1), "extra"),
        (lambda d: d.update(frames="90"), "frames"),
        (lambda d: d.update(fps=25), "fps"),
        (lambda d: d.update(video_codec="hevc"), "video_codec"),
        (lambda d: d.update(video_file="../x.mp4"), "bare"),
        (lambda d: d["scenes"][0].update(visual="photo"), "visual"),
        (lambda d: d["assets"][0].update(kind="sfx"), "kind"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(schema_version=True), "schema_version"),
    ]
    for mutate, match in mutations:
        bad = json.loads(json.dumps(raw))
        mutate(bad)
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(RenderError, match=match):
            RenderReport.load(path)


def test_report_refuses_nan_and_bad_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integrated_lufs"):
        _report(integrated_lufs=float("nan"))
    with pytest.raises(ValueError, match="bare"):
        _report(captions_file="")
    path = tmp_path / "r.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RenderError, match="not valid JSON"):
        RenderReport.load(path)
    path.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    with pytest.raises(RenderError):
        RenderReport.load(path)
    with pytest.raises(RenderError, match="cannot read"):
        RenderReport.load(tmp_path / "missing.json")
    with pytest.raises(RenderError, match="cannot write"):
        _report().save(tmp_path / "no-such-dir" / "r.json")


def test_save_failure_is_render_error_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(src: object, dst: object) -> None:
        raise OSError("cross-device")

    monkeypatch.setattr("narratoflow.render.report.os.replace", boom)
    with pytest.raises(RenderError, match="cannot write render report"):
        _report().save(tmp_path / "part-01.render.json")
    assert list(tmp_path.iterdir()) == []
