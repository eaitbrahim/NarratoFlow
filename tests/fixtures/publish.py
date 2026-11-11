"""`PartInputs` and platform-rules builders for publish tests."""

from __future__ import annotations

import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

from fixtures.parts import (
    CORPUS,
    GENERATED_PROVENANCE,
    add_sketch,
    fake_render_report,
    make_series,
    narr,
    write_timelines,
)
from narratoflow.audio.timeline import PartTimeline, VoiceUsed
from narratoflow.captions.part import build_part_ass
from narratoflow.channels.config import ChannelConfig
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchRecord
from narratoflow.models import SceneKind, Series
from narratoflow.publish.checks import PartInputs
from narratoflow.render.assets import default_fonts
from narratoflow.render.report import RenderReport

PACKAGED_RULES = Path(__file__).parents[2] / "src" / "narratoflow" / "data" / "platform_rules.toml"

VOICE = VoiceUsed(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    voice_id="v1",
    terms="elevenlabs-terms",
)


def make_inputs(
    tmp_path: Path,
    library: SketchLibrary,
    channel: ChannelConfig,
    *,
    series: Series | None = None,
    render: RenderReport | None = None,
    timeline: PartTimeline | None = None,
    **part_overrides: Any,
) -> PartInputs:
    """Inputs for part 1: by default a narration-only Arabic part with one generated sketch.

    `tmp_path` is the render directory; it gets a placeholder ``part-01.mp4`` and the
    real ``part-01.ass`` (and the render report), so `render_current` can pass."""
    if series is None:
        sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
        series = make_series(
            [narr("s1", "alpha beta gamma", sketch=sid, title="One"), narr("s2", "delta epsilon")]
        )
    if part_overrides:
        first = series.parts[0].model_copy(update=part_overrides)
        series = series.model_copy(update={"parts": [first, *series.parts[1:]]})
    part = series.parts[0]
    if timeline is None:
        narrated = any(s.kind is SceneKind.NARRATION for s in part.scenes)
        timeline = write_timelines(
            series, tmp_path / "audio", library, flac=False, voice=VOICE if narrated else None
        )[0]
    if render is None:
        render = fake_render_report(series, part, timeline)
    doc, _ = build_part_ass(
        series, part, timeline, corpus=CORPUS, fonts=default_fonts(), ink=channel.style.ink
    )
    doc.write(tmp_path / render.captions_file)
    (tmp_path / render.video_file).write_bytes(b"placeholder mp4")
    render.save(tmp_path / f"part-{part.index:02d}.render.json")
    sketches: dict[int, SketchRecord | None] = {
        s.sketch_id: library.get_sketch(s.sketch_id)
        for s in render.scenes
        if s.sketch_id is not None
    }
    return PartInputs(
        series=series,
        part=part,
        channel=channel,
        render=render,
        timeline=timeline,
        sketches=sketches,
        render_dir=tmp_path,
    )


# -- a minimal TOML emitter (tests only; enough for the rules file's shapes) ----------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _table(table: dict[str, Any], path: list[str], out: list[str]) -> None:
    scalars = [(k, v) for k, v in table.items() if not isinstance(v, dict)]
    tables = [(k, v) for k, v in table.items() if isinstance(v, dict)]
    if path and (scalars or not tables):
        out.append("[" + ".".join(_key(p) for p in path) + "]")
    out.extend(f"{_key(k)} = {_value(v)}" for k, v in scalars)
    if scalars:
        out.append("")
    for k, v in tables:
        _table(v, [*path, k], out)


def dumps_toml(data: dict[str, Any]) -> str:
    out: list[str] = []
    _table(data, [], out)
    return "\n".join(out) + "\n"


def _set(data: dict[str, Any], dotted: str, value: object) -> None:
    """Set `dotted` in `data`; a part may itself contain dots ("CC-BY-4.0"), so the
    longest existing key wins at each level and the remainder names a new key."""
    parts = dotted.split(".")
    table = data
    while True:
        for n in range(len(parts), 0, -1):
            name = ".".join(parts[:n])
            if name in table:
                break
        else:
            name, n = ".".join(parts), len(parts)
        rest = parts[n:]
        if not rest:
            if value is None:
                del table[name]
            else:
                table[name] = value
            return
        table = table.setdefault(name, {})
        parts = rest


def write_rules(tmp_path: Path, **changes: object) -> Path:
    """The packaged rules with dotted-key `changes` (None deletes a key), written to
    ``tmp_path/platform_rules.toml``."""
    data = tomllib.loads(PACKAGED_RULES.read_text(encoding="utf-8"))
    for dotted, value in changes.items():
        _set(data, dotted, value)
    path = tmp_path / "platform_rules.toml"
    path.write_text(dumps_toml(data), encoding="utf-8")
    return path
