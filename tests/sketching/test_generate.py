from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from fixtures import images as img
from llm.fakes import FakeLLM
from PIL import Image
from pydantic import BaseModel, SecretStr

from narratoflow.channels.config import ChannelConfig, VisionConfig, load_channel
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import BudgetExceededError, ChannelConfigError, PlanningError
from narratoflow.images import GeneratedImage, ImageRequest
from narratoflow.images.prompt import build_prompt
from narratoflow.library.repository import SketchLibrary
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
    VerseRef,
)
from narratoflow.settings import Settings
from narratoflow.sketching import generate as generate_module
from narratoflow.sketching.generate import (
    QUARANTINED_FLAG,
    GenerationInterruptedError,
    generate_missing,
    load_reference,
)
from narratoflow.sketching.vision import VisionVerdict
from narratoflow.visuals.bundled import BundledIcon, BundledIcons
from narratoflow.visuals.marker import marker_seed

PRICES = PriceTable._from_text(
    """
[llm."anthropic/claude-opus-5-5"]
input_per_mtok = "0"
output_per_mtok = "0"
[image."openai/gpt-image-1"]
per_image = "0.04"
[vision."anthropic/claude-opus-5-5"]
per_image_check = "0.01"
""",
    "test",
)
PAID_VISION = VisionConfig(provider="anthropic", model="claude-opus-5-5")
PER_SKETCH = Decimal("0.04") + 2 * Decimal("0.01")  # image + candidate and reference checks


class FakeImages:
    name = "openai"
    model = "gpt-image-1"

    def __init__(self, pngs: list[bytes] | None = None) -> None:
        self.pngs = list(pngs or [])
        self.requests: list[ImageRequest] = []

    def generate(self, req: ImageRequest) -> GeneratedImage:
        self.requests.append(req)
        png = self.pngs.pop(0) if self.pngs else img.line_drawing_scroll()
        return GeneratedImage(png, "openai", "gpt-image-1", None)


def verdict(**over: object) -> VisionVerdict:
    base: dict[str, object] = {
        "human_figure": False,
        "face": False,
        "logo_or_text": False,
        "style_outlier": False,
        "mirror_safe": True,
        "reason": "clean",
    }
    return VisionVerdict.model_validate({**base, **over})


def clean(n: int) -> list[BaseModel | Exception]:
    return [verdict() for _ in range(n)]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", image_api_key=SecretStr("img-key"))


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    # a real line drawing as the reference sketch (the fixture writes a 4x4 white PNG)
    ch = load_channel(channel_file)
    ch.style.reference_sketch.write_bytes(img.scroll(size=256))
    return ch.model_copy(update={"vision": PAID_VISION})


@pytest.fixture(autouse=True)
def no_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_module, "default_bundled_icons", lambda: BundledIcons({}))


def scene(i: int, key: str = "", **over: object) -> Scene:
    data: dict[str, object] = {
        "id": f"s{i:03d}",
        "narration": Narration(source_text=f"text {i}"),
        "idea_key": key or f"thing-{i}|roman",
        "idea_description": f"thing number {i}",
        "match": "new",
    }
    return Scene.model_validate({**data, **over})


def series(*parts: list[Scene], approved: bool = True) -> Series:
    return Series(
        title="Scenes",
        channel_slug="history",
        language=Language.ARABIC,
        source=ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN),
        parts=[
            Part(index=i + 1, count=len(parts), hook_text="h", scenes=scenes)
            for i, scenes in enumerate(parts)
        ],
        approved=approved,
    )


def run(
    s: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    images: FakeImages,
    vision: FakeLLM | None,
) -> tuple[Series, generate_module.GenerationResult]:
    return generate_missing(
        s, channel, settings, images=images, vision=vision, library=library, prices=PRICES
    )


def scenes_of(s: Series) -> list[Scene]:
    return [sc for p in s.parts for sc in p.scenes]


def test_happy_path_three_new_scenes_pass_with_strokes(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    original = series([scene(1), scene(2), scene(3)])
    images, vision = FakeImages(), FakeLLM(clean(3))
    out, result = run(original, channel, settings, library, images, vision)

    assert (result.generated, result.reused, result.bundled) == (3, 0, 0)
    assert result.quarantined == [] and result.unchecked == []
    assert result.spent_usd == 3 * PER_SKETCH
    reference = channel.style.reference_sketch.read_bytes()
    for sc, req in zip(scenes_of(out), images.requests, strict=True):
        assert sc.sketch_id is not None and sc.match == "new" and sc.flags == []
        [record] = library.find_reusable(1, sc.idea_key or "")
        assert record.id == sc.sketch_id
        assert record.status is SketchStatus.PASSED and record.origin is SketchOrigin.GENERATED
        assert record.mirror_safe is True
        assert record.provenance.source == "generated"
        assert record.provenance.license == "openai-terms"
        assert (record.provenance.provider, record.provenance.model) == ("openai", "gpt-image-1")
        assert record.provenance.usage_terms
        strokes = library.get_strokes(record.id)
        assert strokes is not None and strokes.strokes
        assert req.reference_png == reference
        assert req.seed == marker_seed("history/Scenes", sc.id)
        assert req.prompt == build_prompt(channel.style, sc.idea_key or "", sc.idea_description)
    assert [imgs[1] for imgs in vision.images] == [reference] * 3
    # the input series is not mutated
    assert all(sc.sketch_id is None for sc in scenes_of(original))


def test_seeds_are_stable_across_runs(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary, tmp_path: Path
) -> None:
    first, second = FakeImages(), FakeImages()
    run(series([scene(1)]), channel, settings, library, first, FakeLLM(clean(1)))
    from narratoflow.library.blobs import BlobStore
    from narratoflow.library.db import open_library

    engine = open_library(tmp_path / "other.db")
    try:
        other = SketchLibrary(engine, BlobStore(tmp_path / "other-blobs"))
        run(series([scene(1)]), channel, settings, other, second, FakeLLM(clean(1)))
    finally:
        engine.dispose()
    assert first.requests[0].seed == second.requests[0].seed


def test_silhouette_fails_density_without_a_vision_call(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    images, vision = FakeImages([img.filled_silhouette()]), FakeLLM([])
    out, result = run(series([scene(1)]), channel, settings, library, images, vision)

    assert vision.calls == []
    [sc] = scenes_of(out)
    assert sc.sketch_id is None and QUARANTINED_FLAG in sc.flags
    assert result.quarantined == ["s001"] and result.generated == 0
    assert result.spent_usd == Decimal("0.04")
    assert library.find_reusable(1, "thing-1|roman") == []
    record = library.add_sketch(  # dedupe returns the stored FAILED record
        1,
        "thing-1|roman",
        img.filled_silhouette(),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        Provenance.model_validate(
            {"source": "t", "license": "t", "created_at": "2026-01-01T00:00:00Z"}
        ),
    )
    assert record.status is SketchStatus.FAILED
    assert (record.status_reason or "").startswith("density: stroke_width")
    assert library.get_strokes(record.id) is None


def test_blank_image_is_stored_failed_as_density_empty(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    images, vision = FakeImages([img.blank()]), FakeLLM([])
    out, result = run(series([scene(1)]), channel, settings, library, images, vision)
    assert vision.calls == [] and result.quarantined == ["s001"]
    [sc] = scenes_of(out)
    assert sc.sketch_id is None and QUARANTINED_FLAG in sc.flags
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from narratoflow.library.orm import Sketch

    with Session(library._engine) as s:
        [row] = s.scalars(select(Sketch)).all()
    assert (row.status, row.status_reason) == (SketchStatus.FAILED, "density: empty")


def test_face_verdict_quarantines_the_scene(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    vision = FakeLLM([verdict(face=True, reason="a face"), verdict()])
    out, result = run(
        series([scene(1), scene(2)]), channel, settings, library, FakeImages(), vision
    )
    first, second = scenes_of(out)
    assert first.sketch_id is None and first.flags == [QUARANTINED_FLAG]
    assert second.sketch_id is not None and second.flags == []
    assert result.quarantined == ["s001"] and result.generated == 1
    assert library.find_reusable(1, "thing-1|roman") == []


def test_budget_for_exactly_two_stops_the_third_and_rerun_reuses(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    limited = channel.model_copy(update={"spending_limit_usd": 2 * PER_SKETCH})
    original = series([scene(1), scene(2), scene(3)])
    images = FakeImages()
    with pytest.raises(GenerationInterruptedError) as info:
        run(original, limited, settings, library, images, FakeLLM(clean(3)))

    assert isinstance(info.value, BudgetExceededError)
    assert info.value.limit_usd == 2 * PER_SKETCH
    assert len(images.requests) == 2
    partial, so_far = info.value.partial
    done = scenes_of(partial)
    assert [sc.sketch_id is not None for sc in done] == [True, True, False]
    assert so_far.generated == 2 and so_far.spent_usd == 2 * PER_SKETCH
    assert all(sc.sketch_id is None for sc in scenes_of(original))
    assert len(library.idea_keys(1)) == 2

    again = FakeImages()
    out, result = run(partial, limited, settings, library, again, FakeLLM(clean(1)))
    assert (result.reused, result.generated) == (2, 1)
    assert len(again.requests) == 1
    assert [sc.sketch_id for sc in scenes_of(out)][:2] == [sc.sketch_id for sc in done][:2]


def test_limit_is_per_part(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    limited = channel.model_copy(update={"spending_limit_usd": PER_SKETCH})
    s = series([scene(1)], [scene(2)])
    _, result = run(s, limited, settings, library, FakeImages(), FakeLLM(clean(2)))
    assert result.generated == 2 and result.spent_usd == 2 * PER_SKETCH


def _bundled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, png: bytes) -> None:
    icon_file = tmp_path / "eye.png"
    icon_file.write_bytes(png)
    icon = BundledIcon(key="eye|generic", file=icon_file, license="CC0-1.0", source="https://x")
    icons = BundledIcons({"eye|generic": icon})
    monkeypatch.setattr(generate_module, "default_bundled_icons", lambda: icons)


def test_bundled_icon_is_copied_on_first_use_and_reused_after(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bundled(monkeypatch, tmp_path, img.scroll())
    s = series([scene(1, "eye|generic", match="bundled"), scene(2, "eye|generic")])
    images, vision = FakeImages(), FakeLLM([])
    out, result = run(s, channel, settings, library, images, vision)

    assert images.requests == [] and vision.calls == []
    assert (result.bundled, result.reused, result.generated) == (1, 1, 0)
    assert result.spent_usd == 0
    first, second = scenes_of(out)
    assert first.sketch_id is not None and first.sketch_id == second.sketch_id
    assert first.match == second.match == "reused"
    [record] = library.find_reusable(1, "eye|generic")
    assert record.origin is SketchOrigin.COPIED
    assert (record.provenance.license, record.provenance.source) == ("CC0-1.0", "https://x")
    assert record.status is SketchStatus.UNCHECKED
    assert library.get_strokes(record.id) is not None


def test_bundled_icon_failing_density_is_treated_as_new(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bundled(monkeypatch, tmp_path, img.filled_silhouette())
    images = FakeImages()
    out, result = run(
        series([scene(1, "eye|generic")]), channel, settings, library, images, FakeLLM(clean(1))
    )
    assert len(images.requests) == 1
    assert (result.bundled, result.generated) == (0, 1)
    [sc] = scenes_of(out)
    [record] = library.find_reusable(1, "eye|generic")
    assert sc.sketch_id == record.id and record.origin is SketchOrigin.GENERATED


def test_unapproved_series_is_refused_before_anything(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    images = FakeImages()
    with pytest.raises(PlanningError, match="approve the series first"):
        run(series([scene(1)], approved=False), channel, settings, library, images, None)
    assert images.requests == []


def test_verse_scenes_are_untouched(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    verse = Scene(
        id="s002",
        narration=Narration(source_text="v"),
        kind=SceneKind.VERSE,
        verse=VerseRef(surah=1, ayah_start=1, ayah_end=1, certain=True, uthmani_text="v"),
        idea_key="book|generic",
    )
    no_idea = Scene(id="s003", narration=Narration(source_text="n"))
    images = FakeImages()
    out, result = run(
        series([scene(1), verse, no_idea]), channel, settings, library, images, FakeLLM(clean(1))
    )
    assert scenes_of(out)[1:] == [verse, no_idea]
    assert len(images.requests) == 1 and result.generated == 1


def test_unchecked_mode_without_vision(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    out, result = run(series([scene(1)]), channel, settings, library, FakeImages(), None)
    [sc] = scenes_of(out)
    assert sc.sketch_id is not None and sc.flags == []
    assert result.unchecked == ["s001"] and result.generated == 1
    assert result.spent_usd == Decimal("0.04")
    [record] = library.find_reusable(1, "thing-1|roman")
    assert (record.status, record.status_reason) == (SketchStatus.UNCHECKED, "vision: not checked")
    assert record.mirror_safe is False


def test_reused_sketch_gets_missing_strokes_extracted(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    from narratoflow.channels.config import style_fingerprint

    version = library.ensure_style_version("history", style_fingerprint(channel.style))
    prov = Provenance.model_validate(
        {"source": "t", "license": "t", "created_at": "2026-01-01T00:00:00Z"}
    )
    unchecked = library.add_sketch(
        version,
        "lamp|roman",
        img.scroll(width=4),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.UNCHECKED,
        False,
        prov,
    )
    passed = library.add_sketch(
        version,
        "lamp|roman",
        img.line_drawing_scroll(),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        prov,
    )
    assert library.get_strokes(passed.id) is None
    s = series([scene(1, "lamp|roman", flags=[QUARANTINED_FLAG])])
    images = FakeImages()
    out, result = run(s, channel, settings, library, images, FakeLLM([]))
    [sc] = scenes_of(out)
    assert sc.sketch_id == passed.id != unchecked.id
    assert sc.match == "reused" and sc.flags == []
    assert result.reused == 1 and images.requests == []
    assert library.get_strokes(passed.id) is not None


def test_dedupe_onto_a_failed_record_quarantines(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    run(series([scene(1)]), channel, settings, library, FakeImages(), FakeLLM(clean(1)))
    [record] = library.find_reusable(1, "thing-1|roman")
    library.quarantine(record.id, "creator rejected")
    out, result = run(
        series([scene(1)]), channel, settings, library, FakeImages(), FakeLLM(clean(1))
    )
    [sc] = scenes_of(out)
    assert sc.sketch_id is None and QUARANTINED_FLAG in sc.flags
    assert result.quarantined == ["s001"] and result.generated == 0


def test_quarantined_scene_is_not_regenerated_automatically(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG])])
    images = FakeImages()
    out, result = run(s, channel, settings, library, images, FakeLLM([]))
    assert images.requests == []
    assert scenes_of(out) == scenes_of(s) and result.quarantined == ["s001"]


def test_missing_reference_fails_before_spend(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    channel.style.reference_sketch.unlink()
    images = FakeImages()
    with pytest.raises(ChannelConfigError, match="reference sketch"):
        run(series([scene(1)]), channel, settings, library, images, None)
    assert images.requests == []


def test_missing_price_fails_before_spend(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    unpriced = channel.model_copy(update={"vision": VisionConfig(provider="anthropic", model="x")})
    images = FakeImages()
    with pytest.raises(PlanningError, match="no price for anthropic/x"):
        run(series([scene(1)]), unpriced, settings, library, images, FakeLLM([]))
    assert images.requests == []
    # without vision only the image price is needed
    run(series([scene(1)]), unpriced, settings, library, images, None)


def test_missing_image_key_fails_fast(
    channel: ChannelConfig, tmp_path: Path, library: SketchLibrary
) -> None:
    from narratoflow.errors import MissingApiKeyError

    with pytest.raises(MissingApiKeyError):
        run(series([scene(1)]), channel, Settings(data_dir=tmp_path), library, FakeImages(), None)


def _encode(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("size", "fmt", "long_side"),
    [((300, 200), "JPEG", 300), ((3000, 1500), "PNG", 1024), ((1500, 2600), "JPEG", 1024)],
)
def test_reference_is_normalized_to_png(
    channel: ChannelConfig,
    size: tuple[int, int],
    fmt: Literal["JPEG", "PNG"],
    long_side: int,
) -> None:
    channel.style.reference_sketch.write_bytes(_encode(Image.new("RGB", size, "white"), fmt))
    data = load_reference(channel.style)
    with Image.open(io.BytesIO(data)) as out:
        assert out.format == "PNG" and max(out.size) == long_side
        ratio = size[0] / size[1]
        assert abs(out.size[0] / out.size[1] - ratio) < 0.01


def test_small_png_reference_is_used_as_is(channel: ChannelConfig) -> None:
    raw = channel.style.reference_sketch.read_bytes()
    assert load_reference(channel.style) == raw


def test_unreadable_reference_image_is_a_config_error(channel: ChannelConfig) -> None:
    channel.style.reference_sketch.write_bytes(b"not an image")
    with pytest.raises(ChannelConfigError, match="reference sketch"):
        load_reference(channel.style)


def test_auto_vision_uses_the_configured_vision_model(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, str]] = []
    fake = FakeLLM(clean(1))

    def build(cfg: object, _settings: Settings) -> FakeLLM:
        built.append((getattr(cfg, "provider", ""), getattr(cfg, "model", "")))
        return fake

    monkeypatch.setattr(generate_module, "build_llm_client", build)
    _, result = generate_missing(
        series([scene(1)]),
        channel,
        settings,
        images=FakeImages(),
        library=library,
        prices=PRICES,
    )
    assert built == [("anthropic", "claude-opus-5-5")]
    assert len(fake.calls) == 1 and result.spent_usd == PER_SKETCH


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow"
    for path in (root / "sketching").glob("*.py"):
        path.read_bytes().decode("ascii")


def test_blank_bundled_icon_is_treated_as_new(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bundled(monkeypatch, tmp_path, img.blank())
    images = FakeImages()
    _, result = run(
        series([scene(1, "eye|generic")]), channel, settings, library, images, FakeLLM(clean(1))
    )
    assert len(images.requests) == 1 and result.generated == 1


def test_unreadable_bundled_icon_is_a_config_error(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bundled(monkeypatch, tmp_path, img.scroll())
    (tmp_path / "eye.png").unlink()
    with pytest.raises(ChannelConfigError, match="bundled icon"):
        run(series([scene(1, "eye|generic")]), channel, settings, library, FakeImages(), None)


def test_image_provider_is_built_only_when_a_sketch_is_new(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    built: list[FakeImages] = []

    def build(_channel: ChannelConfig, _settings: Settings) -> FakeImages:
        built.append(FakeImages())
        return built[-1]

    monkeypatch.setattr(generate_module, "build_image_provider", build)
    _bundled(monkeypatch, tmp_path, img.scroll())
    generate_missing(
        series([scene(1, "eye|generic")]),
        channel,
        settings,
        vision=None,
        library=library,
        prices=PRICES,
    )
    assert built == []
    generate_missing(
        series([scene(1), scene(2)]),
        channel,
        settings,
        vision=None,
        library=library,
        prices=PRICES,
    )
    [provider] = built
    assert len(provider.requests) == 2


def test_generate_one_needs_an_idea_key(channel: ChannelConfig, library: SketchLibrary) -> None:
    from narratoflow.costs.tracker import SpendTracker

    with pytest.raises(PlanningError, match="no idea key"):
        generate_module.generate_one(
            Scene(id="s1", narration=Narration(source_text="x")),
            channel,
            library=library,
            style_version_id=1,
            images=FakeImages(),
            vision=None,
            tracker=SpendTracker(None),
            reference_png=b"",
            prices=PRICES,
            video_id="v",
        )
