"""The Gradio layer: wiring, error translation and the CLI command, against a fake gradio."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fakes.gradio import Button, FakeError, FakeGradio
from fixtures import images as img
from fixtures.parts import GENERATED_PROVENANCE, _strokes, make_series
from pydantic import SecretStr
from test_cli import run_main
from visuals.lineart_samples import sample_png

from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import SketchOrigin, SketchStatus
from narratoflow.models import AspectRatio, Narration, Scene, Series, VisualSource
from narratoflow.settings import Settings
from narratoflow.sketching.generate import GenerationResult
from narratoflow.ui import app as app_module
from narratoflow.ui import station
from narratoflow.ui.app import build_app
from narratoflow.ui.station import RebuildReport, StationPaths, series_sha256

BUTTONS = {
    "Save text": (
        ["Scene", "Vocalized text (tts_text)", "Caption text", "series.json sha256"],
        11,
    ),
    "Save timing": (["Scene", "Minimum duration (ms)", "Draw fraction", "series.json sha256"], 11),
    "Upload": (
        ["Scene", "Your own image", "Licence (own-work for your own)",
         "Where the image came from", "Attribution", "series.json sha256"],
        11,
    ),
    "Regenerate": (["Scene", "Confirm the estimated cost", "series.json sha256"], 11),
    "Text card": (["Scene", "Text card text", "series.json sha256"], 11),
    "Rebuild scene": (["Scene", "Confirm the estimated cost", "series.json sha256"], 11),
    "Export 9:16": ([], 3),
    "Export 16:9": ([], 3),
}  # fmt: skip


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    ch = load_channel(channel_file)
    ch.style.reference_sketch.write_bytes(img.scroll(size=256))
    return ch


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", image_api_key=SecretStr("img-key"))


@pytest.fixture
def paths(
    channel: ChannelConfig, channel_file: Path, settings: Settings, tmp_path: Path
) -> StationPaths:
    library = default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    record = library.add_sketch(
        version, "scroll|roman", img.line_drawing_scroll(), ".png", SketchOrigin.GENERATED,
        SketchStatus.PASSED, True, GENERATED_PROVENANCE,
    )  # fmt: skip
    library.put_strokes(record.id, _strokes())
    scenes = [
        Scene(
            id=sid,
            narration=Narration(source_text=text),
            sketch_id=record.id,
            idea_key="scroll|roman",
            idea_description="a scroll",
            match="new",
        )
        for sid, text in (("s1", "alpha beta"), ("s2", "gamma"))
    ]
    work = tmp_path / "work"
    work.mkdir()
    make_series(scenes).save(work / "series.json")
    return StationPaths.derive(work / "series.json", channel_file)


def build(paths: StationPaths, settings: Settings) -> tuple[FakeGradio, Any]:
    gr = FakeGradio()
    return gr, build_app(paths, settings, gr)


def click(gr: FakeGradio, label: str, *values: Any) -> Any:
    [wired] = gr.buttons()[label].wired
    return wired.call(*values)


def labels(components: list[Any]) -> list[str]:
    return [c.label for c in components]


def test_app_wires_every_button(paths: StationPaths, settings: Settings) -> None:
    gr, demo = build(paths, settings)
    assert set(gr.buttons()) == set(BUTTONS) and len(gr.buttons()) == 8
    for label, (inputs, outputs) in BUTTONS.items():
        [wired] = gr.buttons()[label].wired
        assert labels(wired.inputs) == inputs, label
        assert len(wired.outputs) == outputs, label
    assert isinstance(gr.buttons()["Save text"], Button)
    [load] = demo.loads
    assert labels(load.inputs) == ["Scene"] and len(load.outputs) == 11


def test_page_load_shows_the_selected_scene(paths: StationPaths, settings: Settings) -> None:
    gr, demo = build(paths, settings)
    [dropdown] = [c for c in gr.components if c.label == "Scene"]
    assert dropdown.kwargs["choices"] == ["s1", "s2"] and dropdown.kwargs["value"] == "s1"
    [load] = demo.loads
    source, tts, caption, audio, image, ms, fraction, sha, estimate, status, _confirm = load.call(
        "s1"
    )
    assert (source, tts, caption, audio) == ("alpha beta", "", "", None)
    assert image is not None and (ms, fraction) == (3000, 0.65)
    assert sha == series_sha256(paths.series) and status == ""
    assert "$0.04" in estimate  # the history channel has no vision model: the image price


def test_page_load_shows_the_rebuild_estimate_too(paths: StationPaths, settings: Settings) -> None:
    series = Series.load(paths.series)
    series.parts[0].scenes[1] = series.parts[0].scenes[1].model_copy(update={"sketch_id": None})
    series.save(paths.series)
    _, demo = build(paths, settings)
    estimate = demo.loads[0].call("s2")[8]
    assert "Regenerating a sketch is estimated at $0.04" in estimate
    assert "Rebuilding the scene (voice and sketch) is estimated at $0.04" in estimate
    free = demo.loads[0].call("s1")[8]
    assert "Rebuilding the scene is free." in free


def test_every_handler_unticks_the_confirm_box(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gr, demo = build(paths, settings)
    confirm = next(c for c in gr.components if c.label == "Confirm the estimated cost")
    every = [*demo.loads, *(w for b in gr.buttons().values() for w in b.wired)]
    assert every and all(w.outputs[-1] is confirm for w in every)
    monkeypatch.setattr(station, "export", lambda *a, **k: [])
    assert click(gr, "Export 9:16")[-1] is False
    sha = series_sha256(paths.series)
    assert click(gr, "Save timing", "s1", 4000.0, 0.5, sha)[-1] is False
    assert demo.loads[0].call("s1")[-1] is False


def test_unestimatable_scene_still_loads(paths: StationPaths, settings: Settings) -> None:
    series = Series.load(paths.series)
    series.parts[0].scenes[1].visual.source = VisualSource.TEXT_CARD
    series.save(paths.series)
    _, demo = build(paths, settings)
    estimate = demo.loads[0].call("s2")[8]
    assert estimate.startswith("No estimate:")


def test_save_text_and_timing_write_series_json(paths: StationPaths, settings: Settings) -> None:
    gr, _ = build(paths, settings)
    sha = series_sha256(paths.series)
    out = click(gr, "Save text", "s1", "alpha beta", "alpha beta", sha)
    assert out[7] == series_sha256(paths.series) != sha and "Text saved" in out[9]
    scene = Series.load(paths.series).parts[0].scenes[0]
    assert scene.narration.tts_text == "alpha beta"
    out = click(gr, "Save timing", "s1", 4000.0, 0.5, out[7])
    scene = Series.load(paths.series).parts[0].scenes[0]
    assert (scene.min_duration_ms, scene.draw_fraction) == (4000, 0.5)
    assert out[5] == 4000 and out[6] == 0.5


def test_upload_and_text_card_buttons(
    paths: StationPaths, settings: Settings, tmp_path: Path
) -> None:
    gr, _ = build(paths, settings)
    photo = tmp_path / "photo.png"
    photo.write_bytes(sample_png("camera"))
    sha = series_sha256(paths.series)
    out = click(gr, "Upload", "s1", str(photo), "own-work", "my photo", "", sha)
    scenes = Series.load(paths.series).parts[0].scenes
    assert scenes[0].visual.source is VisualSource.IMAGE_TO_SKETCH and "line art" in out[9]
    out = click(gr, "Text card", "s2", "Hello", out[7])
    assert Series.load(paths.series).parts[0].scenes[1].card_text == "Hello"
    with pytest.raises(FakeError, match="choose an image"):
        click(gr, "Upload", "s1", None, "own-work", "x", "", out[7])
    with pytest.raises(FakeError, match="cannot read image"):
        click(gr, "Upload", "s1", str(tmp_path / "gone.png"), "own-work", "x", "", out[7])


def test_regenerate_button_asks_for_confirmation(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bool, str]] = []

    def fake(
        p: StationPaths, scene_id: str, *, confirm: bool, expected_sha256: str, **kw: Any
    ) -> station.RegenerateReport:
        calls.append((confirm, scene_id))
        return station.RegenerateReport(expected_sha256, GenerationResult(), Decimal("0.04"))

    monkeypatch.setattr(station, "regenerate_visual", fake)
    gr, _ = build(paths, settings)
    sha = series_sha256(paths.series)
    out = click(gr, "Regenerate", "s1", True, sha)
    assert calls == [(True, "s1")] and "spent $0.04" in out[9]


def test_rebuild_button_reports_the_rendered_parts(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = iter([[1], []])

    def fake(p: StationPaths, scene_id: str, **kw: Any) -> RebuildReport:
        return RebuildReport(
            scene_id,
            1,
            GenerationResult(),
            [1],
            next(reports),
            Decimal(0),
            kw["expected_sha256"],
        )

    monkeypatch.setattr(station, "rebuild_scene", fake)
    gr, _ = build(paths, settings)
    sha = series_sha256(paths.series)
    assert "rendered part(s): 1;" in click(gr, "Rebuild scene", "s1", False, sha)[9]
    assert "none (unchanged)" in click(gr, "Rebuild scene", "s1", False, sha)[9]


def test_export_buttons_call_both_ratios(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ratios: list[AspectRatio] = []

    def fake(p: StationPaths, ratio: AspectRatio, **kw: Any) -> list[Path]:
        ratios.append(ratio)
        return [tmp_path / f"part-01-{ratio.name}.mp4"] if ratio is AspectRatio.PORTRAIT else []

    monkeypatch.setattr(station, "export", fake)
    gr, _ = build(paths, settings)
    files, status, _ = click(gr, "Export 9:16")
    assert files == [str(tmp_path / "part-01-PORTRAIT.mp4")] and "part-01-PORTRAIT.mp4" in status
    files, status, _ = click(gr, "Export 16:9")
    assert ratios == [AspectRatio.PORTRAIT, AspectRatio.LANDSCAPE]
    assert files == [] and "nothing" in status and app_module.PILLARBOX_NOTE in status


def test_callback_errors_become_gr_error(paths: StationPaths, settings: Settings) -> None:
    gr, demo = build(paths, settings)
    with pytest.raises(FakeError, match="reload") as caught:
        click(gr, "Save text", "s1", "alpha beta", "", "0" * 64)
    assert caught.value.__cause__ is not None
    with pytest.raises(FakeError, match="unknown scene"):
        demo.loads[0].call("nope")


def test_page_with_no_scenes_still_builds(paths: StationPaths, settings: Settings) -> None:
    Series.load(paths.series).model_copy(update={"parts": []}).save(paths.series)
    gr, _ = build(paths, settings)
    [dropdown] = [c for c in gr.components if c.label == "Scene"]
    assert dropdown.kwargs["choices"] == [] and dropdown.kwargs["value"] is None


def test_cli_ui_without_extra(
    paths: StationPaths,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "gradio", None)  # `import gradio` now raises ImportError
    code = run_main(monkeypatch, "ui", str(paths.series), "--channel", str(paths.channel))
    assert code == 1
    assert "uv sync --extra ui" in capsys.readouterr().err


def test_cli_ui_launches_on_host_and_port(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGradio()
    monkeypatch.setitem(sys.modules, "gradio", fake)
    monkeypatch.setenv("NARRATOFLOW_DATA_DIR", str(settings.data_dir))
    code = run_main(
        monkeypatch, "ui", str(paths.series), "--channel", str(paths.channel),
        "--host", "192.0.2.1", "--port", "7861",
    )  # fmt: skip
    assert code == 0 and fake.blocks is not None
    served = [str(paths.audio_dir.resolve()), str(paths.export_dir.resolve())]
    assert fake.blocks.launched == {
        "server_name": "192.0.2.1",
        "server_port": 7861,
        "allowed_paths": served,
    }


def test_cli_ui_serves_audio_and_export_outside_the_launch_directory(
    paths: StationPaths, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gradio serves only cwd, its temp dir and allowed_paths: the stage dirs must be listed."""
    audio, export = tmp_path / "far" / "wav", tmp_path / "far" / "out"
    fake = FakeGradio()
    monkeypatch.setitem(sys.modules, "gradio", fake)
    monkeypatch.setenv("NARRATOFLOW_DATA_DIR", str(settings.data_dir))
    code = run_main(
        monkeypatch, "ui", str(paths.series), "--channel", str(paths.channel),
        "--audio", str(audio), "--export", str(export),
    )  # fmt: skip
    assert code == 0 and fake.blocks is not None and fake.blocks.launched is not None
    assert fake.blocks.launched["allowed_paths"] == [str(audio.resolve()), str(export.resolve())]


def test_real_gradio_builds(paths: StationPaths, settings: Settings) -> None:
    # a bare leftover `gradio/` directory imports as a namespace package: ask for a real module
    pytest.importorskip("gradio.blocks")
    import gradio

    demo = build_app(paths, settings, gradio)
    assert isinstance(demo, gradio.Blocks)


def test_free_regeneration_says_so() -> None:
    assert app_module._estimate_text(Decimal(0)) == "Regenerating a sketch is free."
    assert "$0.04" in app_module._estimate_text(Decimal("0.04"))
