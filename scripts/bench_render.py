"""CPU render benchmark: one Arabic part of 3 minutes, every scene drawing a bundled icon.

No TTS and no network: the timeline is written by hand and the audio is a sine FLAC.
It renders at the full 1080x1920 size and prints one Markdown row for
docs/benchmarks/2026-11-13-render-cpu.md. The NFR passes when the wall time is under 600 s.
Run: uv run python scripts/bench_render.py [--workers N] [--minutes M] [--keep DIR]
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from PIL import Image

from narratoflow.audio import io as aio
from narratoflow.audio.timeline import PartTimeline, SceneTiming, TimelineWord, TimingSource
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import (
    Language,
    Narration,
    Part,
    RightsBasis,
    Scene,
    SceneKind,
    ScriptSource,
    Series,
    Visual,
)
from narratoflow.render.build import build_render
from narratoflow.render.ffmpeg import RenderTools, find_render_tools
from narratoflow.render.layout import Canvas
from narratoflow.settings import Settings
from narratoflow.visuals.bundled import default_bundled_icons
from narratoflow.visuals.lineart import binarize
from narratoflow.visuals.strokes import extract_strokes

NFR_SECONDS = 600
WORD_MS = 500  # 2 words per second
DRAW_SHARE = 60  # percent of each scene spent drawing
NARRATION = "\u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631"
HEADER = (
    "| date | cpu / cores | ffmpeg | workers | duration s | frames | wall s | NFR (< 600 s) |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |"
)
CHANNEL_TOML = """name = "Bench"
slug = "bench"
language = "ar"

[style]
reference_sketch = "ref.png"
instructions = "Black ink whiteboard sketch, minimal, no shading."
image_provider = "openai"
image_model = "gpt-image-1"

[voice]
provider = "edge"
voice_id = "ar-SA-HamedNeural"

[llm]
provider = "ollama"
model = "qwen2.5"
"""


@dataclass(frozen=True)
class BenchInputs:
    series: Series
    channel: ChannelConfig
    settings: Settings
    audio_dir: Path
    out_dir: Path


@dataclass(frozen=True)
class BenchResult:
    wall_s: float
    frames: int
    duration_ms: int
    workers: int | None
    video: Path


def _channel(root: Path) -> ChannelConfig:
    work = root / "channel"
    work.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "white").save(work / "ref.png")
    (work / "bench.toml").write_text(CHANNEL_TOML, encoding="utf-8")
    return load_channel(work / "bench.toml")


def _sketch_ids(settings: Settings, channel: ChannelConfig, count: int) -> list[int]:
    """One library sketch per scene, cycling the bundled icons in catalog order."""
    icons = default_bundled_icons()
    catalog = icons.catalog()
    if not catalog:
        raise RuntimeError("no bundled icons: run scripts/fetch_icons.py")
    library = default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    by_key: dict[str, int] = {}
    for key in {catalog[i % len(catalog)][0] for i in range(count)}:
        icon = icons.get(parse_idea_key(key))
        if icon is None:
            raise RuntimeError(f"bundled icon {key} vanished")
        data = icon.file.read_bytes()
        record = library.add_sketch(
            version,
            key,
            data,
            ".png",
            SketchOrigin.BUNDLED,
            SketchStatus.PASSED,
            True,
            Provenance(source=icon.source, license=icon.license, created_at=datetime.now(UTC)),
        )
        library.put_strokes(record.id, extract_strokes(binarize(data), rtl=True))
        by_key[key] = record.id
    return [by_key[catalog[i % len(catalog)][0]] for i in range(count)]


def _timing(index: int, sketch_id: int, stroke_count: int, scene_ms: int) -> SceneTiming:
    start, end = index * scene_ms, (index + 1) * scene_ms
    words = [
        TimelineWord(
            text=NARRATION.split()[n % 2],
            start_ms=start + n * WORD_MS,
            end_ms=start + (n + 1) * WORD_MS,
        )
        for n in range(scene_ms // WORD_MS)
    ]
    return SceneTiming(
        scene_id=f"s{index + 1:02d}",
        kind=SceneKind.NARRATION,
        start_ms=start,
        end_ms=end,
        speech_start_ms=start,
        speech_end_ms=end,
        words=words,
        timing_source=TimingSource.ESTIMATED,
        draw_start_ms=start,
        draw_end_ms=start + scene_ms * DRAW_SHARE // 100,
        sketch_id=sketch_id,
        stroke_count=stroke_count,
    )


def prepare(root: Path, *, duration_ms: int = 180_000, scene_ms: int = 10_000) -> BenchInputs:
    """Write the channel, library, timeline and FLAC for one part under `root`."""
    if scene_ms <= 0 or duration_ms % scene_ms != 0:
        raise ValueError("duration_ms must be a positive multiple of scene_ms")
    count = duration_ms // scene_ms
    settings = Settings(data_dir=root / "data")
    channel = _channel(root)
    library = default_library(settings)
    sketch_ids = _sketch_ids(settings, channel, count)
    scenes = [
        Scene(
            id=f"s{i + 1:02d}",
            narration=Narration(source_text=" ".join([NARRATION] * (scene_ms // WORD_MS // 2))),
            visual=Visual(),
            sketch_id=sketch_ids[i],
            min_duration_ms=0,
        )
        for i in range(count)
    ]
    series = Series(
        title="Bench",
        channel_slug=channel.slug,
        language=Language.ARABIC,
        source=ScriptSource(author="bench", work="Bench", rights=RightsBasis.OWN),
        parts=[Part(index=1, count=1, hook_text="Hook", scenes=scenes)],
        approved=True,
    )
    timings: list[SceneTiming] = []
    for i, sketch_id in enumerate(sketch_ids):
        found = library.get_strokes(sketch_id)
        if found is None:
            raise RuntimeError(f"sketch {sketch_id} has no strokes")
        timings.append(_timing(i, sketch_id, len(found.strokes), scene_ms))
    audio_dir = root / "audio"
    PartTimeline(
        series_title=series.title,
        channel_slug=series.channel_slug,
        part_index=1,
        part_count=1,
        audio_file="part-01.flac",
        duration_ms=duration_ms,
        integrated_lufs=None,
        true_peak_dbtp=None,
        scenes=timings,
        sfx=[],
        voice=None,
        recitations=[],
        assets=[],
        spent_usd=Decimal(0),
        cost_estimated=False,
        warnings=[],
    ).save(_mkdir(audio_dir) / "part-01.timeline.json")
    t = np.arange(aio.SAMPLE_RATE * duration_ms // 1000) / aio.SAMPLE_RATE
    aio.encode_flac(
        (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), audio_dir / "part-01.flac"
    )
    return BenchInputs(series, channel, settings, audio_dir, root / "video")


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(
    inputs: BenchInputs,
    *,
    workers: int | None,
    canvas: Canvas | None = None,
    tools: RenderTools | None = None,
) -> BenchResult:
    """Render the part once, timed. Always forced into a fresh directory, never skipped."""
    out_dir = Path(tempfile.mkdtemp(prefix="run-", dir=_mkdir(inputs.out_dir)))
    started = time.perf_counter()
    result = build_render(
        inputs.series,
        inputs.channel,
        inputs.settings,
        audio_dir=inputs.audio_dir,
        out_dir=out_dir,
        workers=workers,
        canvas=canvas,
        tools=tools,
        library=default_library(inputs.settings),
        force=True,
    )
    wall = time.perf_counter() - started
    [part] = result.parts
    return BenchResult(
        wall_s=wall,
        frames=part.report.frames,
        duration_ms=part.report.duration_ms,
        workers=workers,
        video=part.video_path,
    )


def row(result: BenchResult, tools: RenderTools) -> str:
    version = tools.version.split()[2] if len(tools.version.split()) > 2 else tools.version
    verdict = "pass" if result.wall_s < NFR_SECONDS else "fail"
    workers = "default" if result.workers is None else str(result.workers)
    cpu = f"{platform.processor() or platform.machine()} / {os.cpu_count()}"
    return (
        f"| {date.today().isoformat()} | {cpu} | {version} | {workers} "
        f"| {result.duration_ms / 1000:.0f} | {result.frames} | {result.wall_s:.1f} | {verdict} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--keep", type=Path, default=None, help="work directory to keep")
    args = parser.parse_args(argv)
    duration_ms = round(args.minutes * 60_000 / 10_000) * 10_000
    if duration_ms <= 0:
        parser.error("--minutes is too small for one 10 s scene")
    tools = find_render_tools(Settings())
    if args.keep is not None:
        work = _mkdir(args.keep.resolve())
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="bench-render-"))
        cleanup = True
    try:
        inputs = prepare(work, duration_ms=duration_ms)
        print(HEADER)
        print(row(run(inputs, workers=args.workers, tools=tools), tools))
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
