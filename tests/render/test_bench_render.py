"""The CPU benchmark script: a short series renders, and a run is never a skipped part."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

from narratoflow.audio.timeline import PartTimeline
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.render.layout import Canvas

ROOT = Path(__file__).parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _script("bench_render")


def test_bench_renders_a_short_series(render_tools: RenderTools, tmp_path: Path) -> None:
    inputs = bench.prepare(tmp_path, duration_ms=3000, scene_ms=1500)
    result = bench.run(inputs, workers=1, canvas=Canvas(Fraction(1, 6)), tools=render_tools)
    assert result.frames == 90
    assert result.duration_ms == 3000
    assert result.wall_s > 0
    assert result.workers == 1
    assert result.video.is_file()
    assert bench.row(result, render_tools).startswith("| 20")


def test_every_run_renders_into_a_fresh_directory(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    inputs = bench.prepare(tmp_path, duration_ms=1500, scene_ms=1500)
    canvas = Canvas(Fraction(1, 6))
    first = bench.run(inputs, workers=1, canvas=canvas, tools=render_tools)
    second = bench.run(inputs, workers=1, canvas=canvas, tools=render_tools)
    assert first.video != second.video
    assert first.video.is_file() and second.video.is_file()
    assert second.frames == 45


def test_prepare_covers_duration(tmp_path: Path) -> None:
    inputs = bench.prepare(tmp_path)
    timeline = PartTimeline.load(inputs.audio_dir / "part-01.timeline.json")
    scenes = timeline.scenes
    assert len(scenes) == 18
    assert scenes[0].start_ms == 0 and scenes[-1].end_ms == 180_000
    assert all(a.end_ms == b.start_ms for a, b in itertools.pairwise(scenes))
    assert all(s.draw_end_ms == s.start_ms + 6000 for s in scenes)
    assert len({s.sketch_id for s in scenes}) > 1
