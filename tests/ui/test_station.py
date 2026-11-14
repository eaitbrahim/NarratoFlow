"""The review station's functions: edit series.json, rebuild one scene, export both ratios."""

from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from fakes.images import FakeImages
from fakes.voice import FakeVoice
from fixtures import images as img
from fixtures.parts import (
    CORPUS,
    GENERATED_PROVENANCE,
    _strokes,
    add_sketch,
    make_series,
    narr,
    write_timelines,
)
from fixtures.publish import write_rules
from pydantic import SecretStr
from visuals.lineart_samples import sample_png

from narratoflow.audio.build import build_audio
from narratoflow.audio.timeline import AssetUse, PartTimeline, VoiceUsed
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import PlanningError, ProviderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchOrigin, SketchStatus
from narratoflow.models import (
    AspectRatio,
    Narration,
    Scene,
    SceneKind,
    Series,
    VisualSource,
)
from narratoflow.money import format_usd
from narratoflow.publish.landscape import landscape_name
from narratoflow.publish.rules import Rules, load_rules
from narratoflow.render.build import build_render
from narratoflow.render.ffmpeg import RenderTools, probe
from narratoflow.render.layout import Canvas
from narratoflow.settings import Settings
from narratoflow.sketching.generate import QUARANTINED_FLAG
from narratoflow.sketching.resolve import regenerate
from narratoflow.sketching.upload import attach_upload
from narratoflow.tts import VoiceChoice
from narratoflow.ui import station
from narratoflow.ui.station import (
    StationPaths,
    edit_text,
    edit_timing,
    export,
    list_scenes,
    rebuild_scene,
    regenerate_cost,
    regenerate_visual,
    scene_view,
    series_sha256,
    upload_visual,
    use_text_card,
)

PER_KCHAR = Decimal("0.30")
PRICES = PriceTable._from_text(
    f"""
[llm."anthropic/claude-opus-5-5"]
input_per_mtok = "0"
output_per_mtok = "0"
[image."openai/gpt-image-1"]
per_image = "0.04"
[vision."anthropic/claude-opus-5-5"]
per_image_check = "0.01"
[voice."paid/m1"]
per_kchar = "{PER_KCHAR}"
""",
    "test",
)
IMAGE_PRICE = Decimal("0.04")
EDGE = VoiceUsed(provider="edge", model="", voice_id="ar-SA-HamedNeural", terms="edge-tts-terms")
MARKER = AssetUse(
    kind="sfx", asset_id="marker-1", source="scripts/make_sfx.py", license="Apache-2.0"
)
TINY = Canvas(Fraction(1, 6))


def cost(text: str) -> Decimal:
    return Decimal(len(text)) / 1000 * PER_KCHAR


class BrokenImages(FakeImages):
    def generate(self, req: Any) -> Any:
        self.requests.append(req)
        raise ProviderError("image service down")


@dataclass
class World:
    paths: StationPaths
    channel: ChannelConfig
    settings: Settings
    library: SketchLibrary
    voice: FakeVoice
    images: FakeImages
    tools: RenderTools

    def kit(self, *names: str) -> dict[str, Any]:
        everything: dict[str, Any] = {
            "settings": self.settings,
            "library": self.library,
            "images": self.images,
            "vision": None,
            "voices": [VoiceChoice(self.voice, "v1")],
            "aligner": None,
            "tools": self.tools,
            "prices": PRICES,
            "canvas": TINY,
            "workers": 1,
            "corpus": CORPUS,
        }
        return {k: everything[k] for k in names}

    def rebuild_kit(self) -> dict[str, Any]:
        return self.kit(
            "settings", "images", "vision", "voices", "aligner", "tools", "library", "prices",
            "canvas", "workers", "corpus",
        )  # fmt: skip

    def series(self) -> Series:
        return Series.load(self.paths.series)

    def sha(self) -> str:
        return series_sha256(self.paths.series)


def _stored_sketch(library: SketchLibrary, channel: ChannelConfig) -> int:
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    record = library.add_sketch(
        version,
        "scroll|roman",
        img.line_drawing_scroll(),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        GENERATED_PROVENANCE,
    )
    library.put_strokes(record.id, _strokes())
    return record.id


def _scene(sid: str, text: str, sketch: int | None) -> Scene:
    return Scene(
        id=sid,
        narration=Narration(source_text=text),
        sketch_id=sketch,
        idea_key="scroll|roman",
        idea_description="a scroll",
        match="new",
        min_duration_ms=1000,
    )


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    ch = load_channel(channel_file)
    ch.style.reference_sketch.write_bytes(img.scroll(size=256))
    return ch


@pytest.fixture
def world(
    channel: ChannelConfig,
    channel_file: Path,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> World:
    """Two parts, built end to end the way the CLI would: audio, then render."""
    work = tmp_path / "work"
    work.mkdir()
    sid = _stored_sketch(library, channel)
    series = make_series(
        [_scene("p1-a", "alpha beta", sid), _scene("p1-b", "gamma delta", sid)],
        [_scene("p2-a", "epsilon zeta", sid)],
    )
    paths = StationPaths(work / "series.json", channel_file, work / "audio", work / "export")
    settings = Settings(data_dir=tmp_path / "data", image_api_key=SecretStr("img-key"))
    voice = FakeVoice()
    built, _ = build_audio(
        series,
        channel,
        settings,
        out_dir=paths.audio_dir,
        cache_root=paths.cache_root,
        voices=[VoiceChoice(voice, "v1")],
        aligner=None,
        library=library,
        corpus=CORPUS,
        prices=PRICES,
    )
    built.save(paths.series)
    build_render(
        built,
        channel,
        settings,
        audio_dir=paths.audio_dir,
        out_dir=paths.render_dir,
        workers=1,
        canvas=TINY,
        tools=render_tools,
        library=library,
        corpus=CORPUS,
    )
    voice.requests.clear()
    return World(paths, channel, settings, library, voice, FakeImages(), render_tools)


def _fresh_idea(world: World, scene_id: str, *, drop_sketch: bool = False) -> None:
    """Give the scene an idea no library sketch matches, so drawing it calls the image model."""
    series = world.series()
    for part in series.parts:
        for i, scene in enumerate(part.scenes):
            if scene.id == scene_id:
                update: dict[str, Any] = {"idea_key": "lamp|roman"}
                if drop_sketch:
                    update["sketch_id"] = None
                part.scenes[i] = scene.model_copy(update=update)
    series.save(world.paths.series)


def _scenes(series: Series) -> dict[str, Scene]:
    return {s.id: s for p in series.parts for s in p.scenes}


def _bytes_and_mtime(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), os.stat(path).st_mtime_ns


# -- read ----------------------------------------------------------------------


def test_paths_derive_render_dir_and_cache_root_like_the_cli(tmp_path: Path) -> None:
    paths = StationPaths(
        tmp_path / "w" / "series.json", tmp_path / "c.toml", tmp_path / "a", tmp_path / "e"
    )
    assert paths.render_dir == tmp_path / "a"
    assert paths.cache_root == tmp_path / "w" / ".cache"
    derived = StationPaths.derive(tmp_path / "w" / "series.json", tmp_path / "c.toml")
    assert derived.audio_dir == tmp_path / "w" / "audio"
    assert derived.export_dir == tmp_path / "w" / "export"
    custom = StationPaths.derive(
        tmp_path / "w" / "series.json",
        tmp_path / "c.toml",
        audio_dir=tmp_path / "x",
        export_dir=tmp_path / "y",
    )
    assert (custom.audio_dir, custom.export_dir) == (tmp_path / "x", tmp_path / "y")


def test_list_scenes_names_every_scene_with_its_part(world: World) -> None:
    rows = list_scenes(world.paths)
    assert [(r.scene_id, r.part) for r in rows] == [("p1-a", 1), ("p1-b", 1), ("p2-a", 2)]
    assert rows[0].kind == "narration" and rows[0].visual == "icon" and rows[0].flags == []


def test_scene_view_shows_texts_sketch_and_the_view_sha(world: World) -> None:
    view = scene_view(world.paths, "p1-b", **world.kit("library", "settings"))
    assert view.sha256 == world.sha() and view.part == 1
    assert (view.source_text, view.tts_text, view.caption_text) == ("gamma delta", "", "")
    assert (view.min_duration_ms, view.draw_fraction) == (1000, 0.65)
    assert view.sketch_png is not None and view.sketch_png.startswith(b"\x89PNG")
    assert view.audio_clip is not None and view.audio_clip.is_file()
    with pytest.raises(PlanningError, match="unknown scene"):
        scene_view(world.paths, "nope", **world.kit("library", "settings"))


def test_scene_view_without_audio_or_sketch(world: World) -> None:
    series = world.series()
    series.parts[0].scenes[0].sketch_id = None
    series.save(world.paths.series)
    fresh = StationPaths(
        world.paths.series,
        world.paths.channel,
        world.paths.audio_dir / "none",
        world.paths.export_dir,
    )
    view = scene_view(fresh, "p1-a", **world.kit("library", "settings"))
    assert view.sketch_png is None and view.audio_clip is None


def test_scene_view_of_a_scene_whose_sketch_record_is_gone(world: World) -> None:
    series = world.series()
    series.parts[0].scenes[0].sketch_id = 9999
    series.save(world.paths.series)
    assert scene_view(world.paths, "p1-a", **world.kit("library", "settings")).sketch_png is None


def test_scene_view_of_a_scene_the_timeline_does_not_know(world: World) -> None:
    series = world.series()
    series.parts[0].scenes.append(_scene("p1-new", "new words", None))
    series.save(world.paths.series)
    view = scene_view(world.paths, "p1-new", **world.kit("library", "settings"))
    assert view.audio_clip is None


def test_scene_view_audio_clip_cut_to_slot(world: World) -> None:
    kit = world.kit("library", "settings")
    view = scene_view(world.paths, "p1-b", **kit)
    timeline = PartTimeline.load(world.paths.audio_dir / "part-01.timeline.json")
    slot = next(s for s in timeline.scenes if s.scene_id == "p1-b")
    assert view.audio_clip is not None
    assert view.audio_clip.parent == world.paths.audio_dir / ".ui"
    with wave.open(str(view.audio_clip)) as w:
        got_ms = w.getnframes() * 1000 / w.getframerate()
    assert abs(got_ms - (slot.end_ms - slot.start_ms)) <= 30
    before = _bytes_and_mtime(view.audio_clip)
    again = scene_view(world.paths, "p1-b", **kit)
    assert again.audio_clip == view.audio_clip
    assert _bytes_and_mtime(again.audio_clip) == before  # reused, not rewritten
    other = scene_view(world.paths, "p1-a", **kit)
    assert other.audio_clip != view.audio_clip


# -- edit ----------------------------------------------------------------------


def test_edit_text_persists_to_series_json(world: World) -> None:
    before = _scenes(world.series())
    sha = edit_text(
        world.paths,
        "p1-a",
        tts_text="alpha beta",
        caption_text="alpha beta",
        expected_sha256=world.sha(),
    )
    after = Series.load(world.paths.series)
    assert sha == series_sha256(world.paths.series)
    edited = _scenes(after)["p1-a"].narration
    assert edited.tts_text == "alpha beta" and edited.caption_text == "alpha beta"
    for sid in ("p1-b", "p2-a"):
        assert _scenes(after)[sid].model_dump() == before[sid].model_dump()


def test_edit_text_blank_clears_the_override(world: World) -> None:
    sha = edit_text(
        world.paths, "p1-a", tts_text="alpha beta", caption_text="", expected_sha256=world.sha()
    )
    narration = _scenes(world.series())["p1-a"].narration
    assert narration.tts_text == "alpha beta" and narration.caption_text is None
    edit_text(world.paths, "p1-a", tts_text="  ", caption_text="", expected_sha256=sha)
    assert _scenes(world.series())["p1-a"].narration.tts_text is None


def test_edit_text_rejects_changed_words(world: World) -> None:
    sha = world.sha()
    with pytest.raises(PlanningError, match="p1-a"):
        edit_text(world.paths, "p1-a", tts_text="alpha gamma", caption_text="", expected_sha256=sha)
    with pytest.raises(PlanningError, match="caption has 1 words"):
        edit_text(world.paths, "p1-a", tts_text="", caption_text="alpha", expected_sha256=sha)
    assert world.sha() == sha


def test_edit_refuses_stale_view(world: World) -> None:
    stale = world.sha()
    external = world.series()
    external.title = "Changed elsewhere"
    external.save(world.paths.series)
    changed = world.sha()
    with pytest.raises(PlanningError, match="reload"):
        edit_text(
            world.paths, "p1-a", tts_text="alpha beta", caption_text="", expected_sha256=stale
        )
    with pytest.raises(PlanningError, match="reload"):
        edit_timing(
            world.paths, "p1-a", min_duration_ms=2000, draw_fraction=0.5, expected_sha256=stale
        )
    assert world.sha() == changed and world.series().title == "Changed elsewhere"


def test_edit_timing_bounds(world: World) -> None:
    sha = world.sha()
    for ms, fraction in ((999, 0.5), (60001, 0.5), (2000, 0.05), (2000, 1.01)):
        with pytest.raises(PlanningError):
            edit_timing(
                world.paths, "p1-a", min_duration_ms=ms, draw_fraction=fraction, expected_sha256=sha
            )
    assert world.sha() == sha
    new = edit_timing(
        world.paths, "p1-a", min_duration_ms=1000, draw_fraction=0.1, expected_sha256=sha
    )
    scene = _scenes(world.series())["p1-a"]
    assert (scene.min_duration_ms, scene.draw_fraction) == (1000, 0.1)
    assert new == world.sha()
    edit_timing(world.paths, "p1-a", min_duration_ms=60000, draw_fraction=1.0, expected_sha256=new)
    assert _scenes(world.series())["p1-a"].min_duration_ms == 60000


@pytest.mark.parametrize("blank", [None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("which", ["min_duration_ms", "draw_fraction"])
def test_edit_timing_refuses_a_cleared_or_non_finite_number(
    world: World, which: str, blank: float | None
) -> None:
    """A cleared gr.Number arrives as None: a plain PlanningError, not a TypeError."""
    sha = world.sha()
    numbers: dict[str, Any] = {"min_duration_ms": 2000, "draw_fraction": 0.5, which: blank}
    with pytest.raises(PlanningError, match="number"):
        edit_timing(world.paths, "p1-a", expected_sha256=sha, **numbers)
    assert world.sha() == sha


def test_edits_refuse_unknown_and_verse_scenes(world: World) -> None:
    sha = world.sha()
    with pytest.raises(PlanningError, match="unknown scene"):
        edit_text(world.paths, "nope", tts_text="", caption_text="", expected_sha256=sha)
    series = world.series()
    series.parts[0].scenes[0] = (
        series.parts[0].scenes[0].model_copy(update={"kind": SceneKind.VERSE})
    )
    series.save(world.paths.series)
    with pytest.raises(PlanningError, match="verse"):
        edit_text(world.paths, "p1-a", tts_text="", caption_text="", expected_sha256=world.sha())
    with pytest.raises(PlanningError, match="verse"):
        edit_timing(
            world.paths,
            "p1-a",
            min_duration_ms=2000,
            draw_fraction=0.5,
            expected_sha256=world.sha(),
        )
    with pytest.raises(PlanningError, match="verse"):
        use_text_card(world.paths, "p1-a", "x", expected_sha256=world.sha())


def test_unreadable_series_is_a_planning_error(tmp_path: Path) -> None:
    with pytest.raises(PlanningError, match="cannot read"):
        series_sha256(tmp_path / "missing.json")


def test_station_holds_no_state(world: World) -> None:
    for name, value in vars(station).items():
        if name.startswith("__"):
            continue
        assert not isinstance(value, list | dict | set | bytearray), name
    first = scene_view(world.paths, "p1-a", **world.kit("library", "settings"))
    external = world.series()
    external.parts[0].scenes[0].narration.source_text = "alpha beta"
    external.parts[0].scenes[0].title = "Seen"
    external.save(world.paths.series)
    second = scene_view(world.paths, "p1-a", **world.kit("library", "settings"))
    assert second.sha256 != first.sha256
    assert _scenes(world.series())["p1-a"].title == "Seen"
    assert next(r.scene_id for r in list_scenes(world.paths)) == "p1-a"


# -- visual --------------------------------------------------------------------


def test_upload_visual_attaches_lineart_and_returns_the_new_sha(world: World) -> None:
    sha = world.sha()
    done = upload_visual(
        world.paths,
        "p1-b",
        sample_png("camera"),
        licence="own-work",
        source="my photo",
        expected_sha256=sha,
        **world.kit("settings", "library"),
    )
    scene = _scenes(world.series())["p1-b"]
    assert scene.visual.source is VisualSource.IMAGE_TO_SKETCH and scene.sketch_id == done.sketch_id
    assert done.sha256 == world.sha() != sha


def test_upload_visual_refuses_a_licence_the_rules_do_not_allow(world: World) -> None:
    sha = world.sha()
    attached = _scenes(world.series())["p1-a"].sketch_id
    assert attached is not None
    first_free = attached + 1
    assert world.library.get_sketch(first_free) is None
    with pytest.raises(PlanningError, match="not allowed"):
        upload_visual(
            world.paths,
            "p1-b",
            sample_png("camera"),
            licence="all-rights-reserved",
            source="my photo",
            expected_sha256=sha,
            **world.kit("settings", "library"),
        )
    with pytest.raises(PlanningError, match="needs an attribution"):
        upload_visual(
            world.paths,
            "p1-b",
            sample_png("camera"),
            licence="CC-BY-4.0",
            source="my photo",
            expected_sha256=sha,
            **world.kit("settings", "library"),
        )
    assert _scenes(world.series())["p1-b"].visual.source is not VisualSource.IMAGE_TO_SKETCH
    assert world.library.get_sketch(first_free) is None  # refused and stored nothing
    stale = "0" * 64
    with pytest.raises(PlanningError, match="reload"):
        upload_visual(
            world.paths,
            "p1-b",
            sample_png("camera"),
            licence="own-work",
            source="my photo",
            expected_sha256=stale,
            **world.kit("settings", "library"),
        )
    assert world.sha() == sha


def test_use_text_card_persists_and_returns_the_new_sha(world: World) -> None:
    new = use_text_card(world.paths, "p1-b", "Hello", expected_sha256=world.sha())
    scene = _scenes(world.series())["p1-b"]
    assert scene.visual.source is VisualSource.TEXT_CARD and scene.card_text == "Hello"
    assert scene.sketch_id is None and new == world.sha()


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_use_text_card_refuses_blank_text(world: World, blank: str) -> None:
    sha = world.sha()
    with pytest.raises(PlanningError, match="card needs text"):
        use_text_card(world.paths, "p1-b", blank, expected_sha256=sha)
    assert world.sha() == sha


def test_a_text_card_scene_can_be_redrawn_by_regenerate(world: World) -> None:
    """Text card was one click with no way back; regenerate is the way back (ICON)."""
    _fresh_idea(world, "p1-a")
    sha = use_text_card(world.paths, "p1-a", "Hello", expected_sha256=world.sha())
    kit = world.kit("settings", "library", "images", "vision", "prices")
    kit["vision"] = None
    done = regenerate_visual(world.paths, "p1-a", confirm=True, expected_sha256=sha, **kit)
    scene = _scenes(world.series())["p1-a"]
    assert scene.visual.source is VisualSource.ICON and scene.sketch_id is not None
    assert done.generated.generated == 1


def test_regenerate_visual_needs_confirm_then_redraws(world: World) -> None:
    _fresh_idea(world, "p1-a")
    sha = world.sha()
    kit = world.kit("settings", "library", "images", "vision", "prices")
    kit["vision"] = None
    with pytest.raises(PlanningError, match="estimated cost") as refused:
        regenerate_visual(world.paths, "p1-a", confirm=False, expected_sha256=sha, **kit)
    # the channel has no vision model, so the estimate is the image price
    assert format_usd(IMAGE_PRICE) in str(refused.value)
    assert world.images.requests == [] and world.sha() == sha
    done = regenerate_visual(world.paths, "p1-a", confirm=True, expected_sha256=sha, **kit)
    scene = _scenes(world.series())["p1-a"]
    assert scene.sketch_attempt == 1 and scene.sketch_id is not None
    assert len(world.images.requests) == 1
    assert done.generated.generated == 1 and done.sha256 == world.sha()
    assert done.spent_usd == IMAGE_PRICE


def test_regenerate_cost_is_the_estimate_without_spending(world: World) -> None:
    kit = world.kit("settings", "prices")
    assert regenerate_cost(world.paths, "p1-a", **kit) == IMAGE_PRICE
    with pytest.raises(PlanningError, match="unknown scene"):
        regenerate_cost(world.paths, "nope", **kit)
    assert world.images.requests == []


def test_regenerate_visual_saves_the_partial_when_interrupted(world: World) -> None:
    _fresh_idea(world, "p1-a")
    sha = world.sha()
    broken = BrokenImages()
    kit = world.kit("settings", "library", "vision", "prices")
    from narratoflow.sketching.generate import GenerationInterruptedError

    with pytest.raises(GenerationInterruptedError):
        regenerate_visual(
            world.paths, "p1-a", confirm=True, expected_sha256=sha, images=broken, **kit
        )
    scene = _scenes(world.series())["p1-a"]
    assert QUARANTINED_FLAG in scene.flags and scene.sketch_id is None


def test_regenerate_visual_refuses_a_stale_view_before_any_estimate(world: World) -> None:
    with pytest.raises(PlanningError, match="reload"):
        regenerate_visual(
            world.paths,
            "p1-a",
            confirm=True,
            expected_sha256="0" * 64,
            **world.kit("settings", "library", "images", "vision", "prices"),
        )
    assert world.images.requests == []


# -- pipeline ------------------------------------------------------------------


def test_rebuild_one_scene_only(world: World) -> None:
    other_video = world.paths.render_dir / "part-02.mp4"
    other_files = [
        other_video,
        world.paths.render_dir / "part-02.render.json",
        world.paths.audio_dir / "part-02.flac",
        world.paths.audio_dir / "part-02.timeline.json",
    ]
    before = {p: _bytes_and_mtime(p) for p in other_files}
    first_video = world.paths.render_dir / "part-01.mp4"
    first_before = first_video.read_bytes()
    uploaded = upload_visual(
        world.paths,
        "p1-b",
        sample_png("camera"),
        licence="own-work",
        source="my photo",
        expected_sha256=world.sha(),
        **world.kit("settings", "library"),
    )
    report = rebuild_scene(
        world.paths,
        "p1-b",
        confirm=False,
        expected_sha256=uploaded.sha256,
        **world.rebuild_kit(),
    )
    assert report.scene_id == "p1-b" and report.part == 1
    assert report.rendered_parts == [1] and report.audio_parts == [1]
    assert world.voice.requests == []  # the TTS cache serves every scene
    assert world.images.requests == []
    assert report.spent_usd == 0
    assert first_video.read_bytes() != first_before  # the edited part was re-rendered
    assert {p: _bytes_and_mtime(p) for p in other_files} == before
    assert report.sha256 == world.sha()


def test_rebuild_with_nothing_changed_renders_nothing(world: World) -> None:
    report = rebuild_scene(
        world.paths,
        "p1-b",
        confirm=False,
        expected_sha256=world.sha(),
        **world.rebuild_kit(),
    )
    assert report.rendered_parts == [] and report.audio_parts == [1]


def test_paid_rebuild_needs_confirm(world: World) -> None:
    voiced = "gamma\u064e delta"  # same skeleton, a new voice text: the TTS cache misses
    sha = edit_text(
        world.paths, "p1-b", tts_text=voiced, caption_text="", expected_sha256=world.sha()
    )
    with pytest.raises(PlanningError, match="estimated cost") as refused:
        rebuild_scene(
            world.paths, "p1-b", confirm=False, expected_sha256=sha, **world.rebuild_kit()
        )
    assert format_usd(cost(voiced)) in str(refused.value)
    assert world.voice.requests == [] and world.images.requests == []
    assert world.sha() == sha
    report = rebuild_scene(
        world.paths, "p1-b", confirm=True, expected_sha256=sha, **world.rebuild_kit()
    )
    assert [r.text for r in world.voice.requests] == [voiced]
    assert report.spent_usd == cost(voiced) and report.rendered_parts == [1]


def test_rebuild_draws_a_scene_that_has_no_sketch_after_confirm(world: World) -> None:
    _fresh_idea(world, "p1-a", drop_sketch=True)
    sha = world.sha()
    kit = world.rebuild_kit()
    with pytest.raises(PlanningError, match="estimated cost"):
        rebuild_scene(world.paths, "p1-a", confirm=False, expected_sha256=sha, **kit)
    assert world.images.requests == []
    report = rebuild_scene(world.paths, "p1-a", confirm=True, expected_sha256=sha, **kit)
    assert len(world.images.requests) == 1
    assert report.generated.generated == 1 and report.spent_usd == IMAGE_PRICE
    assert _scenes(world.series())["p1-a"].sketch_id is not None


def test_rebuild_estimates_a_redraw_of_a_stale_attached_sketch(world: World) -> None:
    _fresh_idea(world, "p1-a")  # the attached record no longer matches the idea key
    sha = world.sha()
    kit = world.rebuild_kit()
    with pytest.raises(PlanningError, match="estimated cost") as refused:
        rebuild_scene(world.paths, "p1-a", confirm=False, expected_sha256=sha, **kit)
    assert format_usd(IMAGE_PRICE) in str(refused.value)
    assert world.images.requests == [] and world.sha() == sha
    report = rebuild_scene(world.paths, "p1-a", confirm=True, expected_sha256=sha, **kit)
    assert len(world.images.requests) == 1 and report.spent_usd == IMAGE_PRICE


def test_rebuild_saves_progress_when_the_image_stage_is_interrupted(world: World) -> None:
    from narratoflow.sketching.generate import GenerationInterruptedError

    _fresh_idea(world, "p1-a", drop_sketch=True)
    kit = world.rebuild_kit()
    del kit["images"]
    with pytest.raises(GenerationInterruptedError):
        rebuild_scene(
            world.paths,
            "p1-a",
            confirm=True,
            expected_sha256=world.sha(),
            images=BrokenImages(),
            **kit,
        )
    assert _scenes(world.series())["p1-a"].sketch_id is None  # saved as it stood


def test_rebuild_saves_progress_when_the_voice_stage_is_interrupted(world: World) -> None:
    from narratoflow.audio.build import AudioInterruptedError

    series = world.series()
    narration = series.parts[0].scenes[1].narration
    series.parts[0].scenes[1].narration = narration.model_copy(
        update={"source_text": "gamma  delta!"}
    )
    series.save(world.paths.series)
    world.voice.fail_on = "gamma"
    kit = world.rebuild_kit()
    before = world.sha()
    with pytest.raises(AudioInterruptedError):
        rebuild_scene(world.paths, "p1-b", confirm=True, expected_sha256=before, **kit)
    assert len(world.voice.requests) == 1  # tried once; the series file is still valid
    assert world.series().title == "Scenes"


def test_rebuild_refuses_a_stale_view_and_an_unknown_scene(world: World) -> None:
    kit = world.rebuild_kit()
    with pytest.raises(PlanningError, match="reload"):
        rebuild_scene(world.paths, "p1-b", confirm=True, expected_sha256="0" * 64, **kit)
    with pytest.raises(PlanningError, match="unknown scene"):
        rebuild_scene(world.paths, "nope", confirm=True, expected_sha256=world.sha(), **kit)


def _concurrent_caption(world: World, scene_id: str, caption: str) -> None:
    """What a second tab saves while a slow stage runs: a caption edit on the file."""
    edit_text(
        world.paths,
        scene_id,
        tts_text="",
        caption_text=caption,
        expected_sha256=world.sha(),
    )


def _external_caption(world: World, scene_id: str, caption: str) -> None:
    """What the CLI (not a station handler, so no lock to honour) saves meanwhile."""
    series = world.series()
    for part in series.parts:
        for i, scene in enumerate(part.scenes):
            if scene.id == scene_id:
                narration = scene.narration.model_copy(update={"caption_text": caption})
                part.scenes[i] = scene.model_copy(update={"narration": narration})
    series.save(world.paths.series)


def test_rebuild_refuses_to_overwrite_an_edit_saved_during_the_audio_stage(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_audio(*args: Any, **kw: Any) -> Any:
        _external_caption(world, "p1-a", "alpha beta")
        return build_audio(*args, **kw)

    monkeypatch.setattr(station, "build_audio", slow_audio)
    with pytest.raises(PlanningError, match="reload"):
        rebuild_scene(
            world.paths,
            "p1-b",
            confirm=True,
            expected_sha256=world.sha(),
            **world.rebuild_kit(),
        )
    assert _scenes(world.series())["p1-a"].narration.caption_text == "alpha beta"


def test_regenerate_refuses_to_overwrite_an_edit_saved_during_the_draw(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fresh_idea(world, "p1-b")
    real = regenerate

    def slow_draw(*args: Any, **kw: Any) -> Any:
        _external_caption(world, "p1-a", "alpha beta")
        return real(*args, **kw)

    monkeypatch.setattr(station, "regenerate", slow_draw)
    with pytest.raises(PlanningError, match="reload"):
        regenerate_visual(
            world.paths,
            "p1-b",
            confirm=True,
            expected_sha256=world.sha(),
            **world.kit("settings", "library", "images", "vision", "prices"),
        )
    assert _scenes(world.series())["p1-a"].narration.caption_text == "alpha beta"


def test_upload_refuses_to_overwrite_an_edit_saved_during_the_conversion(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = attach_upload

    def slow_upload(*args: Any, **kw: Any) -> Any:
        _concurrent_caption(world, "p1-a", "alpha beta")
        return real(*args, **kw)

    monkeypatch.setattr(station, "attach_upload", slow_upload)
    with pytest.raises(PlanningError, match="reload"):
        upload_visual(
            world.paths,
            "p1-b",
            sample_png("camera"),
            licence="own-work",
            source="my photo",
            expected_sha256=world.sha(),
            **world.kit("settings", "library"),
        )
    assert _scenes(world.series())["p1-a"].narration.caption_text == "alpha beta"


# -- one paid run at a time ----------------------------------------------------


def _lock(world: World) -> Path:
    return world.paths.series.with_name("series.json.lock")


class SpyImages(FakeImages):
    """An image model that, while it draws, tries what a second browser tab would."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world
        self.lock_seen = False
        self.refusals: list[str] = []

    def generate(self, req: Any) -> Any:
        self.lock_seen = _lock(self.world).is_file()
        with pytest.raises(PlanningError) as refused:
            _concurrent_caption(self.world, "p1-a", "alpha beta")
        self.refusals.append(str(refused.value))
        return super().generate(req)


def test_a_save_during_a_paid_rebuild_is_refused_and_the_spend_is_kept(world: World) -> None:
    _fresh_idea(world, "p1-b", drop_sketch=True)
    spy = SpyImages(world)
    kit = world.rebuild_kit()
    kit["images"] = spy
    report = rebuild_scene(world.paths, "p1-b", confirm=True, expected_sha256=world.sha(), **kit)
    assert spy.lock_seen and len(spy.refusals) == 1 and "paid action" in spy.refusals[0]
    assert report.spent_usd == IMAGE_PRICE and report.sha256 == world.sha()
    assert _scenes(world.series())["p1-a"].narration.caption_text is None
    assert _scenes(world.series())["p1-b"].sketch_id is not None
    assert not _lock(world).exists()


def test_a_save_during_a_paid_regenerate_is_refused(world: World) -> None:
    _fresh_idea(world, "p1-b")
    spy = SpyImages(world)
    kit = world.kit("settings", "library", "vision", "prices")
    done = regenerate_visual(
        world.paths, "p1-b", confirm=True, expected_sha256=world.sha(), images=spy, **kit
    )
    assert spy.lock_seen and "paid action" in spy.refusals[0]
    assert done.sha256 == world.sha() and not _lock(world).exists()


class CrashingImages(FakeImages):
    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world
        self.lock_seen = False

    def generate(self, req: Any) -> Any:
        self.lock_seen = _lock(self.world).is_file()
        raise RuntimeError("boom")


def test_the_lock_is_removed_when_a_paid_run_crashes(world: World) -> None:
    _fresh_idea(world, "p1-b", drop_sketch=True)
    kit = world.rebuild_kit()
    crashing = CrashingImages(world)
    kit["images"] = crashing
    with pytest.raises(RuntimeError, match="boom"):
        rebuild_scene(world.paths, "p1-b", confirm=True, expected_sha256=world.sha(), **kit)
    assert crashing.lock_seen and not _lock(world).exists()
    _fresh_idea(world, "p1-b")
    kit2 = world.kit("settings", "library", "vision", "prices")
    crashing.lock_seen = False
    with pytest.raises(RuntimeError, match="boom"):
        regenerate_visual(
            world.paths,
            "p1-b",
            confirm=True,
            expected_sha256=world.sha(),
            images=crashing,
            **kit2,
        )
    assert crashing.lock_seen and not _lock(world).exists()


def test_every_series_writer_refuses_while_the_lock_is_held(world: World) -> None:
    lock = _lock(world)
    lock.write_text("someone else's run")
    sha = world.sha()
    refusals: list[Any] = [
        lambda: edit_text(world.paths, "p1-a", tts_text="", caption_text="x", expected_sha256=sha),
        lambda: edit_timing(
            world.paths, "p1-a", min_duration_ms=2000, draw_fraction=0.5, expected_sha256=sha
        ),
        lambda: use_text_card(world.paths, "p1-a", "Hi", expected_sha256=sha),
        lambda: upload_visual(
            world.paths,
            "p1-a",
            sample_png("camera"),
            licence="own-work",
            source="photo",
            expected_sha256=sha,
            **world.kit("settings", "library"),
        ),
        lambda: regenerate_visual(
            world.paths,
            "p1-a",
            confirm=True,
            expected_sha256=sha,
            **world.kit("settings", "library", "images", "vision", "prices"),
        ),
        lambda: rebuild_scene(
            world.paths, "p1-a", confirm=True, expected_sha256=sha, **world.rebuild_kit()
        ),
    ]
    for attempt in refusals:
        with pytest.raises(PlanningError, match=r"paid action.*series\.json\.lock"):
            attempt()
    assert lock.read_text() == "someone else's run"  # a refusal never removes another's lock
    assert world.sha() == sha and world.images.requests == []


class ChangedElsewhereImages(BrokenImages):
    """Fails like BrokenImages after the CLI saved another edit to the file."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world

    def generate(self, req: Any) -> Any:
        _external_caption(self.world, "p1-a", "alpha beta")
        return super().generate(req)


def test_an_interrupted_regenerate_keeps_its_own_error_when_the_file_changed(
    world: World,
) -> None:
    from narratoflow.sketching.generate import GenerationInterruptedError

    _fresh_idea(world, "p1-b")
    kit = world.kit("settings", "library", "vision", "prices")
    with pytest.raises(GenerationInterruptedError, match="image service down") as caught:
        regenerate_visual(
            world.paths,
            "p1-b",
            confirm=True,
            expected_sha256=world.sha(),
            images=ChangedElsewhereImages(world),
            **kit,
        )
    assert any("not saved" in note for note in caught.value.__notes__)
    assert _scenes(world.series())["p1-a"].narration.caption_text == "alpha beta"


def test_an_interrupted_rebuild_keeps_its_own_error_when_the_file_changed(world: World) -> None:
    from narratoflow.sketching.generate import GenerationInterruptedError

    _fresh_idea(world, "p1-b", drop_sketch=True)
    kit = world.rebuild_kit()
    kit["images"] = ChangedElsewhereImages(world)
    with pytest.raises(GenerationInterruptedError) as caught:
        rebuild_scene(world.paths, "p1-b", confirm=True, expected_sha256=world.sha(), **kit)
    assert any("not saved" in note for note in caught.value.__notes__)
    assert not _lock(world).exists()


def test_an_interrupted_voice_stage_keeps_its_own_error_when_the_file_changed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from narratoflow.audio.build import AudioInterruptedError

    series = world.series()
    narration = series.parts[0].scenes[1].narration
    series.parts[0].scenes[1].narration = narration.model_copy(
        update={"source_text": "gamma  delta!"}
    )
    series.save(world.paths.series)
    world.voice.fail_on = "gamma"
    real = build_audio

    def changed_then_run(*args: Any, **kw: Any) -> Any:
        _external_caption(world, "p1-a", "alpha beta")
        return real(*args, **kw)

    monkeypatch.setattr(station, "build_audio", changed_then_run)
    with pytest.raises(AudioInterruptedError) as caught:
        rebuild_scene(
            world.paths, "p1-b", confirm=True, expected_sha256=world.sha(), **world.rebuild_kit()
        )
    assert any("not saved" in note for note in caught.value.__notes__)


# -- export --------------------------------------------------------------------


def _tiny_rules(tmp_path: Path) -> Rules:
    changes: dict[str, object] = {
        "platforms.youtube-shorts.width": 180,
        "platforms.youtube-shorts.height": 320,
        "platforms.tiktok.width": 180,
        "platforms.tiktok.height": 320,
    }
    return load_rules(write_rules(tmp_path, **changes))


@pytest.fixture
def publishable(
    channel: ChannelConfig,
    channel_file: Path,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> tuple[StationPaths, dict[str, Any]]:
    """A one-part series with mastered timelines, ready to render and publish."""
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    series = make_series(
        [narr("s1", "alpha beta gamma", sketch=sid, title="One"), narr("s2", "delta")]
    )
    work = tmp_path / "work"
    work.mkdir()
    paths = StationPaths(work / "series.json", channel_file, work / "audio", work / "export")
    series.save(paths.series)
    write_timelines(series, paths.audio_dir, library, master=True, voice=EDGE, assets=[MARKER])
    kit: dict[str, Any] = {
        "settings": Settings(data_dir=tmp_path / "data"),
        "tools": render_tools,
        "library": library,
        "canvas": TINY,
        "workers": 1,
        "corpus": CORPUS,
        "rules": _tiny_rules(tmp_path),
    }
    return paths, kit


def test_export_portrait_and_landscape(publishable: tuple[StationPaths, dict[str, Any]]) -> None:
    paths, kit = publishable
    portrait = export(paths, AspectRatio.PORTRAIT, **kit)
    assert portrait == [paths.export_dir / "part-01.mp4"]
    landscape = export(paths, AspectRatio.LANDSCAPE, **kit)
    assert landscape == [paths.export_dir / landscape_name(1)]
    found = probe(kit["tools"], landscape[0])
    assert (found.width, found.height) == (1920, 1080)
    assert (paths.export_dir / "part-01.mp4").is_file()  # the 9:16 file is kept


def test_export_of_a_failing_part_produces_neither_file(
    publishable: tuple[StationPaths, dict[str, Any]],
) -> None:
    paths, kit = publishable
    assert export(paths, AspectRatio.LANDSCAPE, **kit) == [paths.export_dir / landscape_name(1)]
    undisclosed = Series.load(paths.series).model_copy(update={"ai_disclosure": False})
    undisclosed.save(paths.series)
    assert export(paths, AspectRatio.PORTRAIT, **kit) == []
    assert export(paths, AspectRatio.LANDSCAPE, **kit) == []
    assert not (paths.export_dir / landscape_name(1)).exists()  # the older copy is removed
    assert not (paths.export_dir / "part-01.mp4").exists()


def test_portrait_export_removes_the_now_stale_landscape_copy(
    publishable: tuple[StationPaths, dict[str, Any]],
) -> None:
    paths, kit = publishable
    [old] = export(paths, AspectRatio.LANDSCAPE, **kit)
    assert old.is_file()
    assert export(paths, AspectRatio.PORTRAIT, **kit) == [paths.export_dir / "part-01.mp4"]
    assert not old.exists()  # it would show the previous cut next to the new 9:16 file


def test_portrait_export_of_a_failing_part_removes_the_landscape_copy(
    publishable: tuple[StationPaths, dict[str, Any]],
) -> None:
    paths, kit = publishable
    [old] = export(paths, AspectRatio.LANDSCAPE, **kit)
    Series.load(paths.series).model_copy(update={"ai_disclosure": False}).save(paths.series)
    assert export(paths, AspectRatio.PORTRAIT, **kit) == []
    assert not old.exists()


def test_series_sha256_is_the_files_sha256(world: World) -> None:
    assert world.sha() == hashlib.sha256(world.paths.series.read_bytes()).hexdigest()
