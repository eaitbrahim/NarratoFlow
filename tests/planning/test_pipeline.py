from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from conftest import FIXTURES
from llm.fakes import FakeLLM
from pydantic import SecretStr
from rich.console import Console

from narratoflow.channels.config import ChannelConfig, LlmConfig, load_channel, style_fingerprint
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import SpendTracker, TrackedLLM
from narratoflow.errors import (
    BudgetExceededError,
    ChannelConfigError,
    MissingApiKeyError,
    PlanningError,
    ScriptBlockedError,
)
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.llm.base import LLMClient
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
from narratoflow.planning import pipeline as pipeline_module
from narratoflow.planning.approval import needs_approval, render_approval_list
from narratoflow.planning.ideas import IdeaOut
from narratoflow.planning.pipeline import plan_series
from narratoflow.planning.segment import SegmentOut, collapse_ws
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.script.source import load_script
from narratoflow.settings import Settings
from narratoflow.text.tashkeel.engines import Vocalized
from narratoflow.visuals.bundled import BundledIcon, BundledIcons

from .fakes import FATHA, SCRIPTED_IDEA, ConstantEmbedder, ScriptedLLM, add_fatha

INPUT = FIXTURES / "input"
PAID_LLM = LlmConfig(provider="anthropic", model="claude-opus-5-5")


def read(name: str) -> str:
    return (INPUT / name).read_text(encoding="utf-8")


def write_script(tmp_path: Path, body: str, rights: str = "own") -> Path:
    path = tmp_path / "script.txt"
    path.write_text(
        f'+++\nauthor = "Sayyid"\nwork = "Scenes of Resurrection"\nrights = "{rights}"\n+++\n'
        f"{body}",
        encoding="utf-8",
    )
    return path


def fixture_corpus() -> QuranCorpus:
    q = FIXTURES / "quran"
    return QuranCorpus.from_files(q / "quran-simple-clean.txt", q / "quran-uthmani.txt")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        image_api_key=SecretStr("img-key"),
        llm_api_key=SecretStr("llm-key"),
    )


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


@pytest.fixture
def prices(settings: Settings) -> PriceTable:
    return default_prices(settings)


@pytest.fixture(autouse=True)
def no_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shipped icon pack has `scroll|generic`; these tests plan against an empty pack.
    monkeypatch.setattr(pipeline_module, "default_bundled_icons", lambda: BundledIcons({}))


def _all_scenes(series: Series) -> list[Scene]:
    return [s for p in series.parts for s in p.scenes]


def _render(series: Series) -> str:
    console = Console(width=200, record=True, color_system=None)
    console.print(render_approval_list(series))
    return console.export_text()


def test_plan_reference_script_1_end_to_end(
    tmp_path: Path, channel: ChannelConfig, settings: Settings, prices: PriceTable
) -> None:
    body = read("clean_ar.txt")
    fake = ScriptedLLM()
    series = plan_series(
        write_script(tmp_path, body),
        channel,
        settings,
        llm=fake,
        embedder=None,
        corpus=fixture_corpus(),
        prices=prices,
    )

    scenes = _all_scenes(series)
    verse_scenes = [s for s in scenes if s.kind is SceneKind.VERSE]
    assert verse_scenes and verse_scenes[0].verse is not None
    verse = verse_scenes[0].verse
    assert (verse.surah, verse.ayah_start) == (31, 33)
    assert verse.uthmani_text == fixture_corpus().text_range(31, 33, verse.ayah_end)
    # narration is verbatim across all scenes and parts
    assert collapse_ws("".join(s.narration.source_text for s in scenes)) == collapse_ws(body)
    narration = [s for s in scenes if s.kind is SceneKind.NARRATION]
    assert narration
    assert all(s.idea_key for s in narration)
    assert all(str(parse_idea_key(s.idea_key or "")) == s.idea_key for s in narration)
    assert all(s.match == "new" for s in narration)
    assert all(s.idea_key is None and s.match is None for s in verse_scenes)
    # vocalized for TTS, the author's text for captions, verses untouched
    for s in narration:
        assert s.narration.tts_text == add_fatha(s.narration.source_text)
        assert s.narration.caption_text == s.narration.source_text
        assert "tashkeel_rejected" not in s.flags
    assert all(FATHA not in s.narration.source_text for s in verse_scenes)
    assert verse_scenes[0].narration.tts_text is None
    # series metadata
    assert series.title == "Scenes of Resurrection"
    assert series.channel_slug == "history"
    assert series.language is Language.ARABIC
    assert series.source.rights is RightsBasis.OWN
    assert series.approved is False
    assert series.est_planning_cost_usd == 0  # unwrapped fake, free ollama
    for part in series.parts:
        new = sum(1 for s in part.scenes if s.match == "new")
        assert part.est_cost_usd == new * Decimal("0.04")
    # calls: one segmentation per stretch, one idea and one vocalization per narration
    kinds = [c[2] for c in fake.calls]
    assert kinds.count(IdeaOut) == len(narration)
    assert kinds.count(Vocalized) == len(narration)
    assert kinds.count(SegmentOut) >= 1
    # the approval list shows paid sketches and the verse
    text = _render(series)
    assert "new (paid)" in text
    assert "31:33" in text
    assert "total" in text


def test_budget_below_planning_cost_blocks_before_llm(
    tmp_path: Path, channel: ChannelConfig, settings: Settings, prices: PriceTable
) -> None:
    paid = channel.model_copy(update={"llm": PAID_LLM, "spending_limit_usd": Decimal("0.0001")})
    fake = FakeLLM([])
    with pytest.raises(BudgetExceededError) as info:
        plan_series(
            write_script(tmp_path, read("clean_ar.txt")),
            paid,
            settings,
            llm=fake,
            corpus=fixture_corpus(),
            prices=prices,
        )
    assert fake.calls == []
    assert info.value.limit_usd == Decimal("0.0001")
    assert info.value.estimate_usd > info.value.limit_usd


def test_in_copyright_blocked_before_llm(
    tmp_path: Path, channel: ChannelConfig, settings: Settings, prices: PriceTable
) -> None:
    fake = FakeLLM([])
    script = write_script(tmp_path, read("clean_ar.txt"), rights="third_party_in_copyright")
    with pytest.raises(ScriptBlockedError, match="copyright"):
        plan_series(script, channel, settings, llm=fake, corpus=fixture_corpus(), prices=prices)
    assert fake.calls == []


def test_missing_key_blocked_before_llm(
    tmp_path: Path, channel: ChannelConfig, prices: PriceTable
) -> None:
    fake = FakeLLM([])
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_IMAGE_API_KEY"):
        plan_series(
            write_script(tmp_path, read("clean_ar.txt")),
            channel,
            Settings(data_dir=tmp_path / "data"),
            llm=fake,
            corpus=fixture_corpus(),
            prices=prices,
        )
    assert fake.calls == []


def test_planning_cost_is_tracker_spend_and_llm_is_built(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = channel.model_copy(update={"llm": PAID_LLM, "spending_limit_usd": Decimal(50)})
    built: list[TrackedLLM] = []

    def fake_make_llm(
        ch: ChannelConfig, st: Settings, tracker: SpendTracker, pr: PriceTable
    ) -> LLMClient:
        assert ch is paid and st is settings and pr is prices
        assert tracker.limit_usd == Decimal(50)
        wrapped = TrackedLLM(ScriptedLLM(), tracker, "anthropic", "claude-opus-5-5", pr)
        built.append(wrapped)
        return wrapped

    monkeypatch.setattr(pipeline_module, "make_llm", fake_make_llm)
    series = plan_series(
        write_script(tmp_path, read("clean_ar.txt")),
        paid,
        settings,
        corpus=fixture_corpus(),
        prices=prices,
    )
    [wrapped] = built
    assert series.est_planning_cost_usd == wrapped.tracker.spent > 0
    # the pre-call estimate covers every call the run really made
    _, body = load_script(write_script(tmp_path, read("clean_ar.txt")), Language.ARABIC)
    estimate = pipeline_module.estimate_planning(body, paid, prices, catalog=[], segment_calls=2)
    assert estimate >= wrapped.tracker.spent
    assert estimate < 4 * wrapped.tracker.spent  # and is not wildly inflated (long scenes)


def test_large_catalog_builds_embedder(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(settings, channel, "lamp|generic")
    embedder = ConstantEmbedder()
    monkeypatch.setattr(pipeline_module, "CATALOG_EMBED_THRESHOLD", 0)

    def fake_make_embedder(
        ch: ChannelConfig, st: Settings, tracker: SpendTracker, pr: PriceTable
    ) -> ConstantEmbedder:
        return embedder

    monkeypatch.setattr(pipeline_module, "make_embedder", fake_make_embedder)
    fake = ScriptedLLM()
    plan_series(
        write_script(tmp_path, read("clean_en.txt")),
        channel.model_copy(update={"language": Language.ENGLISH}),
        settings,
        llm=fake,
        prices=prices,
    )
    assert embedder.calls  # catalog and scene texts were embedded
    ideas = [user for _, user, out in fake.calls if out is IdeaOut]
    assert ideas and all("lamp|generic" in user for user in ideas)


def test_small_catalog_needs_no_embedder(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object) -> None:
        raise AssertionError("embedder must not be built for a small catalog")

    monkeypatch.setattr(pipeline_module, "make_embedder", boom)
    plan_series(
        write_script(tmp_path, read("clean_en.txt")),
        channel.model_copy(update={"language": Language.ENGLISH}),
        settings,
        llm=ScriptedLLM(),
        prices=prices,
    )


def _seed(settings: Settings, channel: ChannelConfig, key: str) -> None:
    library = default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    prov = Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    library.add_sketch(
        version,
        key,
        b"png",
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        prov,
        description="a drawing",
    )


def test_english_skips_corpus_and_tashkeel(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object) -> None:
        raise AssertionError("no corpus for English")

    monkeypatch.setattr(pipeline_module, "default_corpus", boom)
    _seed(settings, channel, SCRIPTED_IDEA)
    fake = ScriptedLLM()
    body = read("clean_en.txt")
    english = channel.model_copy(update={"language": Language.ENGLISH})
    series = plan_series(write_script(tmp_path, body), english, settings, llm=fake, prices=prices)
    scenes = _all_scenes(series)
    assert all(s.kind is SceneKind.NARRATION for s in scenes)
    assert all(s.match == "reused" for s in scenes)
    assert all(s.narration.tts_text is None for s in scenes)
    assert Vocalized not in [c[2] for c in fake.calls]
    assert all(p.est_cost_usd == 0 for p in series.parts)  # reused + free voice
    assert "reused" in _render(series)


def test_arabic_default_corpus_and_bundled_icon(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[Settings] = []

    def corpus_for(st: Settings) -> QuranCorpus:
        loaded.append(st)
        return fixture_corpus()

    icon = BundledIcon(
        key=SCRIPTED_IDEA,
        file=tmp_path / "scroll.svg",
        license="CC0",
        source="x",
        description="a scroll",
        sha256=hashlib.sha256(b"").hexdigest(),
    )
    monkeypatch.setattr(pipeline_module, "default_corpus", corpus_for)
    monkeypatch.setattr(
        pipeline_module, "default_bundled_icons", lambda: BundledIcons({SCRIPTED_IDEA: icon})
    )
    series = plan_series(
        write_script(tmp_path, read("clean_ar.txt")),
        channel,
        settings,
        llm=ScriptedLLM(),
        prices=prices,
    )
    assert loaded == [settings]
    narration = [s for s in _all_scenes(series) if s.kind is SceneKind.NARRATION]
    assert narration and all(s.match == "bundled" for s in narration)
    assert all(p.est_cost_usd == 0 for p in series.parts)


class _FakeMishkal:
    name = "mishkal"

    def diacritize(self, text: str) -> str:
        return text + "x"  # changes the skeleton: must be rejected


def test_mishkal_diacritizer_selected_and_rejection_flagged(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "MishkalDiacritizer", _FakeMishkal)
    fake = ScriptedLLM()
    series = plan_series(
        write_script(tmp_path, read("clean_ar.txt")),
        channel.model_copy(update={"diacritizer": "mishkal", "approval_list": False}),
        settings,
        llm=fake,
        corpus=fixture_corpus(),
        prices=prices,
    )
    narration = [s for s in _all_scenes(series) if s.kind is SceneKind.NARRATION]
    assert all("tashkeel_rejected" in s.flags for s in narration)
    assert all(s.narration.tts_text == s.narration.source_text for s in narration)
    assert Vocalized not in [c[2] for c in fake.calls]
    assert needs_approval(series, channel.model_copy(update={"approval_list": False}))


# --- needs_approval -------------------------------------------------------


def _scene(
    sid: str = "s001",
    text: str = "Hello world.",
    *,
    match: str | None = "new",
    flags: list[str] | None = None,
    verse: VerseRef | None = None,
) -> Scene:
    return Scene.model_validate(
        {
            "id": sid,
            "narration": Narration(source_text=text),
            "kind": SceneKind.VERSE if verse else SceneKind.NARRATION,
            "verse": verse,
            "idea_key": None if verse else "scroll|generic",
            "match": None if verse else match,
            "est_duration_s": 5.0,
            "flags": flags or [],
        }
    )


def _series(*parts: Part, approved: bool = False) -> Series:
    return Series(
        title="W",
        channel_slug="history",
        language=Language.ENGLISH,
        source=ScriptSource(author="A", work="W", rights=RightsBasis.OWN),
        parts=list(parts),
        approved=approved,
        est_planning_cost_usd=Decimal("0.0123"),
    )


def _part(*scenes: Scene, cost: str = "0.04", flags: list[str] | None = None) -> Part:
    return Part(
        index=1,
        count=1,
        hook_text="h",
        scenes=list(scenes),
        est_cost_usd=Decimal(cost),
        flags=flags or [],
    )


def _quiet(channel: ChannelConfig, limit: str | None = "1") -> ChannelConfig:
    value = Decimal(limit) if limit is not None else None
    return channel.model_copy(update={"approval_list": False, "spending_limit_usd": value})


def test_needs_approval_forced_by_limit(channel: ChannelConfig) -> None:
    quiet = _quiet(channel)
    assert not needs_approval(_series(_part(_scene(), cost="1")), quiet)
    assert needs_approval(_series(_part(_scene(), cost="1.01")), quiet)
    assert not needs_approval(_series(_part(_scene(), cost="99")), _quiet(channel, None))


def test_needs_approval_by_channel_setting(channel: ChannelConfig) -> None:
    series = _series(_part(_scene()))
    assert needs_approval(series, channel.model_copy(update={"approval_list": True}))
    assert not needs_approval(series, _quiet(channel))


@pytest.mark.parametrize(
    "flag", ["tashkeel_rejected", "unfinished_ending", "over_max", "unmatched_verse_brackets"]
)
def test_needs_approval_by_flag(channel: ChannelConfig, flag: str) -> None:
    quiet = _quiet(channel)
    assert needs_approval(_series(_part(_scene(flags=[flag]))), quiet)
    assert needs_approval(_series(_part(_scene(), flags=[flag])), quiet)
    assert not needs_approval(_series(_part(_scene(flags=["other"]))), quiet)


def test_needs_approval_by_uncertain_verse(channel: ChannelConfig) -> None:
    quiet = _quiet(channel)
    sure = VerseRef(surah=2, ayah_start=255, ayah_end=255, certain=True, uthmani_text="u")
    unsure = sure.model_copy(update={"certain": False})
    assert not needs_approval(_series(_part(_scene(verse=sure))), quiet)
    assert needs_approval(_series(_part(_scene(verse=unsure))), quiet)


# --- render_approval_list -------------------------------------------------


def test_render_approval_list_rows_and_totals() -> None:
    long_text = "word " * 30
    sure = VerseRef(surah=2, ayah_start=1, ayah_end=3, certain=True, uthmani_text="UTHMANI")
    unsure = VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=False, uthmani_text="U2")
    first = _part(
        _scene("s001", long_text, match="reused"),
        _scene("s002", "[bold]markup[/bold] stays literal", match="bundled", flags=["over_max"]),
        cost="0.08",
    )
    second = Part(
        index=2,
        count=2,
        hook_text="h2",
        scenes=[
            _scene("s003", match="new"),
            _scene("s004", "quoted", verse=sure),
            _scene("s005", "quoted", verse=unsure),
        ],
        est_cost_usd=Decimal("0.04"),
        flags=["unfinished_ending"],
    )
    first = first.model_copy(update={"count": 2})
    table = render_approval_list(_series(first, second))
    assert [c.header for c in table.columns] == [
        "part",
        "scene",
        "narration",
        "idea",
        "match",
        "verse",
        "flags",
    ]
    text = _render(_series(first, second))
    assert " ".join(long_text.split())[:60] + "\u2026" in text
    assert "[bold]markup[/bold] stays literal" in text
    assert "reused" in text and "bundled" in text and "new (paid)" in text
    assert "2:1-3" in text and "31:33 (uncertain)" in text
    assert "UTHMANI" in text  # verse scenes show the verified Uthmani text
    assert "over_max" in text and "unfinished_ending" in text
    assert "$0.08" in text and "$0.04" in text
    assert "total" in text and "$0.12" in text and "$0.0123" in text


@pytest.mark.parametrize("split_words", [None, 12])
@pytest.mark.parametrize("catalog_size", [0, 20])
def test_estimate_covers_actual_spend_with_seeded_catalog(
    split_words: int | None,
    catalog_size: int,
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    english = channel.model_copy(
        update={"language": Language.ENGLISH, "llm": PAID_LLM, "spending_limit_usd": Decimal(50)}
    )
    library = default_library(settings)
    version = library.ensure_style_version(english.slug, style_fingerprint(english.style))
    prov = Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    for i in range(catalog_size):
        library.add_sketch(
            version,
            f"object-{i}|generic",
            b"png",
            ".png",
            SketchOrigin.GENERATED,
            SketchStatus.PASSED,
            True,
            prov,
            description=f"a hand drawn picture of object number {i} on an old wooden table",
        )
    catalog = library.idea_catalog(version)
    assert len(catalog) == catalog_size
    built: list[TrackedLLM] = []
    scripted = ScriptedLLM(split_words=split_words)

    def fake_make_llm(
        ch: ChannelConfig, st: Settings, tracker: SpendTracker, pr: PriceTable
    ) -> LLMClient:
        wrapped = TrackedLLM(scripted, tracker, "anthropic", "claude-opus-5-5", pr)
        built.append(wrapped)
        return wrapped

    monkeypatch.setattr(pipeline_module, "make_llm", fake_make_llm)
    script = write_script(tmp_path, read("clean_en.txt"))
    plan_series(script, english, settings, prices=prices)
    [wrapped] = built
    _, body = load_script(script, Language.ENGLISH)
    estimate = pipeline_module.estimate_planning(body, english, prices, catalog=catalog)
    actual = wrapped.tracker.spent
    print(f"estimate={estimate} actual={actual}")
    assert 0 < actual <= estimate
    assert estimate < 3 * actual
    # the prompts really carried the whole catalog
    ideas = [u for _, u, o in scripted.calls if o is IdeaOut]
    assert ideas
    if catalog_size:
        assert all("object-19|generic" in u for u in ideas)


def test_estimate_counts_one_segment_system_per_call(
    channel: ChannelConfig, prices: PriceTable
) -> None:
    paid = channel.model_copy(update={"llm": PAID_LLM})
    one = pipeline_module.estimate_planning("a b c", paid, prices, catalog=[])
    three = pipeline_module.estimate_planning("a b c", paid, prices, catalog=[], segment_calls=3)
    assert three > one


def test_large_catalog_estimate_uses_five_longest_lines(
    channel: ChannelConfig, prices: PriceTable
) -> None:
    paid = channel.model_copy(update={"llm": PAID_LLM})
    small = [(f"k{i}|generic", "x") for i in range(301)]
    longer = small[:296] + [(f"k{i}|generic", "y" * 400) for i in range(5)]
    low = pipeline_module.estimate_planning("a b c", paid, prices, catalog=small)
    high = pipeline_module.estimate_planning("a b c", paid, prices, catalog=longer)
    assert high > low


def test_bundled_icons_load_before_any_spend(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    prices: PriceTable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> BundledIcons:
        raise ChannelConfigError("bundled icon index is broken")

    monkeypatch.setattr(pipeline_module, "default_bundled_icons", broken)
    fake = ScriptedLLM()
    with pytest.raises(ChannelConfigError, match="bundled icon"):
        plan_series(
            write_script(tmp_path, read("clean_ar.txt")),
            channel,
            settings,
            llm=fake,
            corpus=fixture_corpus(),
            prices=prices,
        )
    assert fake.calls == []


def test_missing_vision_price_blocks_before_llm(
    tmp_path: Path, channel: ChannelConfig, settings: Settings, prices: PriceTable
) -> None:
    unpriced = channel.model_copy(
        update={"llm": LlmConfig(provider="anthropic", model="claude-unpriced-vision")}
    )
    table = PriceTable(
        {
            **prices._tables,
            "llm": {
                **prices._tables["llm"],
                "anthropic/claude-unpriced-vision": (Decimal(1), Decimal(1)),
            },
        }
    )
    fake = FakeLLM([])
    with pytest.raises(PlanningError, match="no price for anthropic/claude-unpriced-vision"):
        plan_series(
            write_script(tmp_path, read("clean_ar.txt")),
            unpriced,
            settings,
            llm=fake,
            corpus=fixture_corpus(),
            prices=table,
        )
    assert fake.calls == []
