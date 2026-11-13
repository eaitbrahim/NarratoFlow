"""The CPU benchmark script: a short series renders, and a run is never a skipped part."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from narratoflow.audio import io as aio
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


TOOLS = RenderTools(Path("ffmpeg"), Path("ffprobe"), "ffmpeg version 7.1.2 Copyright")


def _result(wall_s: float, workers: int | None = 4) -> object:
    return bench.BenchResult(
        wall_s=wall_s, frames=5400, duration_ms=180_000, workers=workers, video=Path("v.mp4")
    )


def test_row_columns() -> None:
    cells = [c.strip() for c in bench.row(_result(123.46), TOOLS).strip("|").split("|")]
    assert len(cells) == 8
    assert cells[2:] == ["7.1.2", "4", "180", "5400", "123.5", "pass"]


def test_row_default_workers_and_short_version() -> None:
    short = RenderTools(Path("f"), Path("p"), "n-7")
    cells = [c.strip() for c in bench.row(_result(1.0, None), short).strip("|").split("|")]
    assert cells[2] == "n-7"
    assert cells[3] == "default"


@pytest.mark.parametrize(("wall", "verdict"), [(599.9, "pass"), (600.0, "fail"), (900.0, "fail")])
def test_nfr_verdict_boundary(wall: float, verdict: str) -> None:
    assert bench.row(_result(wall), TOOLS).endswith(f"| {verdict} |")


@pytest.fixture
def fake_bench(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    def prepare(root: Path, *, duration_ms: int) -> str:
        seen["root"], seen["duration_ms"] = root, duration_ms
        return "inputs"

    def run(inputs: object, *, workers: int | None, tools: RenderTools) -> object:
        seen["workers"], seen["tools"] = workers, tools
        return _result(10.0, workers)

    monkeypatch.setattr(bench, "prepare", prepare)
    monkeypatch.setattr(bench, "run", run)
    monkeypatch.setattr(bench, "find_render_tools", lambda settings: TOOLS)
    return seen


def test_main_prints_header_and_row_and_cleans_up(
    fake_bench: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert bench.main(["--workers", "2", "--minutes", "1.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "\n".join(lines[:2]) == bench.HEADER
    assert lines[2].endswith("| 2 | 180 | 5400 | 10.0 | pass |")
    assert fake_bench["duration_ms"] == 90_000
    assert fake_bench["workers"] == 2
    assert fake_bench["tools"] is TOOLS
    assert not Path(str(fake_bench["root"])).exists()


def test_main_keep_leaves_the_directory(
    fake_bench: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keep = tmp_path / "keep"
    assert bench.main(["--keep", str(keep)]) == 0
    assert fake_bench["root"] == keep.resolve()
    assert keep.is_dir()
    assert fake_bench["duration_ms"] == 180_000
    assert fake_bench["workers"] is None
    assert capsys.readouterr().out.count("\n") == 3


def test_main_rejects_too_few_minutes(fake_bench: dict[str, object]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--minutes", "0.01"])
    assert excinfo.value.code == 2
    assert "root" not in fake_bench


def test_prepare_numbers_scenes_from_s01(tmp_path: Path) -> None:
    inputs = bench.prepare(tmp_path, duration_ms=30_000, scene_ms=10_000)
    timeline = PartTimeline.load(inputs.audio_dir / "part-01.timeline.json")
    assert [s.scene_id for s in timeline.scenes] == ["s01", "s02", "s03"]
    assert [s.id for s in inputs.series.parts[0].scenes] == ["s01", "s02", "s03"]


@pytest.mark.parametrize(
    ("duration_ms", "scene_ms"), [(20_000, 0), (20_000, -10_000), (25_000, 10_000)]
)
def test_prepare_refuses_bad_scene_lengths(tmp_path: Path, duration_ms: int, scene_ms: int) -> None:
    with pytest.raises(ValueError, match="positive multiple"):
        bench.prepare(tmp_path, duration_ms=duration_ms, scene_ms=scene_ms)


def test_prepare_twice_into_the_same_root(tmp_path: Path) -> None:
    bench.prepare(tmp_path, duration_ms=10_000, scene_ms=10_000)
    again = bench.prepare(tmp_path, duration_ms=10_000, scene_ms=10_000)
    assert (again.audio_dir / "part-01.timeline.json").is_file()


def test_prepare_writes_flac_of_exactly_the_duration(ffmpeg_bin: str, tmp_path: Path) -> None:
    inputs = bench.prepare(tmp_path, duration_ms=3000, scene_ms=1500)
    samples = aio.decode((inputs.audio_dir / "part-01.flac").read_bytes())
    assert abs(samples.size - aio.SAMPLE_RATE * 3) <= 1
    peak_hz = np.argmax(np.abs(np.fft.rfft(samples))) * aio.SAMPLE_RATE / samples.size
    assert peak_hz == pytest.approx(440, abs=2)


def test_prepare_creates_missing_parent_directories(tmp_path: Path) -> None:
    inputs = bench.prepare(tmp_path / "a" / "b", duration_ms=10_000, scene_ms=10_000)
    assert (inputs.audio_dir / "part-01.flac").is_file()
    assert bench._mkdir(tmp_path / "x" / "y").is_dir()
    assert bench._mkdir(tmp_path / "x" / "y").is_dir()


def test_row_version_is_the_third_token_only_when_there_is_one() -> None:
    three = RenderTools(Path("f"), Path("p"), "ffmpeg version 9.0.2")
    assert bench.row(_result(1.0), three).split("|")[3].strip() == "9.0.2"
    two = RenderTools(Path("f"), Path("p"), "ffmpeg 9.0")
    assert bench.row(_result(1.0), two).split("|")[3].strip() == "ffmpeg 9.0"


@pytest.mark.parametrize("minutes", ["0", "-1"])
def test_main_rejects_non_positive_minutes(fake_bench: dict[str, object], minutes: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        bench.main(["--minutes", minutes])
    assert excinfo.value.code == 2
