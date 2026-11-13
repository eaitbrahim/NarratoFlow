"""One CLI run from Arabic text to verified MP4s and a disclosure, with only factory seams faked."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fakes.planning import ScriptedLLM
from fakes.voice import FakeVoice
from typer.testing import CliRunner

from narratoflow.align.ctc import CtcAligner
from narratoflow.cli import app
from narratoflow.quran.corpus import SIMPLE_NAME, UTHMANI_NAME
from narratoflow.render.encoder import verify_mp4
from narratoflow.render.ffmpeg import RenderTools, probe
from narratoflow.render.report import RenderReport
from narratoflow.tts import VoiceChoice

REPO = Path(__file__).resolve().parent.parent
QURAN_FIXTURES = Path(__file__).parent / "fixtures" / "quran"


def test_text_to_verified_mp4_and_disclosure(
    channel_file: Path,
    data_dir: Path,
    render_tools: RenderTools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "sample_ar.md"
    shutil.copy(REPO / "examples" / "sample_ar.md", script)
    (data_dir / "quran").mkdir(parents=True)
    for name in (SIMPLE_NAME, UTHMANI_NAME):
        shutil.copy(QURAN_FIXTURES / name, data_dir / "quran" / name)
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img")
    # Use the same ffmpeg as the render_tools fixture (the autouse isolation cleared the env).
    monkeypatch.setenv("NARRATOFLOW_FFMPEG", str(render_tools.ffmpeg))

    voice = FakeVoice("edge", "edge-tts", spans=True)
    image_builds: list[object] = []

    def no_image_provider(*args: object, **kwargs: object) -> object:
        image_builds.append(args)
        raise AssertionError("the bundled sketch must cover every scene; no image is drawn")

    from narratoflow.audio import build as audio_build
    from narratoflow.planning import pipeline
    from narratoflow.sketching import generate

    monkeypatch.setattr(
        pipeline, "make_llm", lambda channel, settings, tracker, prices: ScriptedLLM()
    )
    monkeypatch.setattr(generate, "build_image_provider", no_image_provider)
    monkeypatch.setattr(
        audio_build,
        "build_voice_chain",
        lambda channel, settings: [VoiceChoice(voice, "ar-SA-HamedNeural")],
    )
    monkeypatch.setattr(CtcAligner, "available", staticmethod(lambda: False))

    series = tmp_path / "series.json"
    channel = str(channel_file)
    runner = CliRunner()
    stages = [
        ["plan", str(script), "--channel", channel],
        ["approve", str(series), "--yes"],
        ["sketch", "generate", str(series), "--channel", channel],
        ["audio", "build", str(series), "--channel", channel],
        ["render", str(series), "--channel", channel, "--workers", "2"],
        ["publish", str(series), "--channel", channel],
    ]
    for args in stages:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{args[0]} failed:\n{result.output}"

    reports = sorted((tmp_path / "audio").glob("part-*.render.json"))
    assert reports
    for report_path in reports:
        report = RenderReport.load(report_path)
        mp4 = report_path.with_name(report.video_file)
        verify_mp4(
            render_tools,
            mp4,
            size=(1080, 1920),
            frames=report.frames,
            duration_ms=report.duration_ms,
        )
        got = probe(render_tools, mp4)
        assert (got.video_codec, got.audio_codec) == ("h264", "aac")
        disclosure = (
            tmp_path / "export" / f"{report_path.name.removesuffix('.render.json')}.disclosure.txt"
        )
        assert disclosure.read_text(encoding="utf-8").strip()
    assert image_builds == []
    assert voice.requests
