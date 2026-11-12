import logging
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from llm.fakes import FakeLLM
from pydantic import BaseModel

from narratoflow.channels.config import (
    ChannelConfig,
    EmbedConfig,
    VisionConfig,
    load_channel,
    vision_model,
)
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import (
    SpendTracker,
    TrackedLLM,
    add_spend,
    estimate_llm_call,
    estimate_part_cost,
    estimate_planning_cost,
    estimate_tokens,
    require_generation_prices,
    require_prices,
    voice_chars,
)
from narratoflow.errors import (
    BilledProviderError,
    BudgetExceededError,
    PlanningError,
    ProviderError,
    ProviderRefusalError,
)
from narratoflow.llm.base import Usage
from narratoflow.models import Language, Narration, Part, Scene, SceneKind
from narratoflow.settings import Settings
from narratoflow.sketching.vision import estimate_check_cost
from narratoflow.text.spoken import Lexicon


class Out(BaseModel):
    x: int = 1


@pytest.fixture
def prices() -> PriceTable:
    return default_prices(Settings(data_dir=Path("/nonexistent-narratoflow")))


def test_tracker_blocks_before_spend() -> None:
    t = SpendTracker(Decimal("0.05"))
    with pytest.raises(BudgetExceededError):
        t.ensure_room(Decimal("0.06"), "planning")
    assert t.spent == 0


def test_ensure_room_accounts_for_spent_and_boundary() -> None:
    t = SpendTracker(Decimal("1"))
    t.charge(Decimal("0.5"), "a")
    t.ensure_room(Decimal("0.5"), "b")
    with pytest.raises(BudgetExceededError):
        t.ensure_room(Decimal("0.51"), "b")


def test_budget_refusal_names_what_spent_and_remaining() -> None:
    t = SpendTracker(Decimal("0.08"), spent=Decimal("0.08"))
    with pytest.raises(BudgetExceededError) as info:
        t.ensure_room(Decimal("0.04"), "voice elevenlabs/eleven_multilingual_v2 part 2")
    assert str(info.value) == (
        "voice elevenlabs/eleven_multilingual_v2 part 2: estimated cost $0.04 exceeds the "
        "remaining $0.00 (spent $0.08 of the $0.08 per-part limit; "
        "raise spending_limit_usd in the channel file)"
    )
    assert info.value.spent_usd == Decimal("0.08")
    assert info.value.what == "voice elevenlabs/eleven_multilingual_v2 part 2"


def test_no_limit_never_blocks() -> None:
    t = SpendTracker(None)
    t.ensure_room(Decimal("1000000"), "x")


def test_charge_llm_uses_prices(prices: PriceTable) -> None:
    t = SpendTracker(None)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = t.charge_llm("anthropic", "claude-opus-5-5", usage, prices)
    assert cost == Decimal("24.00")
    assert t.spent == Decimal("24.00")
    assert t.entries[0].amount_usd == Decimal("24.00")
    assert t.entries[0].at.utcoffset() is not None
    assert "anthropic/claude-opus-5-5" in t.entries[0].what


def test_estimate_tokens_arabic_vs_english() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefghi") == 3  # ceil(9/4)
    assert estimate_tokens("ب" * 9) == 3  # ceil(9/3)
    assert estimate_tokens("ب" * 10) == 4
    assert estimate_tokens("a" * 8) == 2


def _paid(channel_file: Path) -> ChannelConfig:
    base = load_channel(channel_file)
    return base.model_copy(
        update={
            "llm": base.llm.model_copy(
                update={"provider": "anthropic", "model": "claude-opus-5-5"}
            ),
            "embed": EmbedConfig(provider="voyage", model="voyage-multilingual-2"),
        }
    )


_SYS = {"segment_system_tokens": 100, "ideas_system_tokens": 600, "tashkeel_system_tokens": 50}


def _usd(input_tokens: int, output_tokens: int) -> Decimal:
    return (Decimal(input_tokens) * 4 + Decimal(output_tokens) * 20) / 1_000_000


def test_estimate_planning_cost_english(prices: PriceTable, channel_file: Path) -> None:
    ch = _paid(channel_file).model_copy(update={"language": Language.ENGLISH})
    text = " ".join(["abc"] * 1000)  # 3999 chars -> 1000 tokens, 1000 words
    calls = 100  # ceil(1000 / 10)
    segment = _usd(100 + 1000, 1200)
    ideas = _usd(calls * 600 + 1000, calls * 80)
    expected = (segment + ideas) * Decimal("1.2")
    assert estimate_planning_cost(text, ch, prices, **_SYS) == expected
    # the caller passes the real token count of the candidate block each idea call carries
    with_block = (segment + _usd(calls * (600 + 300) + 1000, calls * 80)) * Decimal("1.2")
    assert (
        estimate_planning_cost(text, ch, prices, **_SYS, catalog_size=300, candidate_tokens=300)
        == with_block
    )
    # segmentation makes one call per stretch, each carrying the segment system prompt
    many = (_usd(3 * 100 + 1000, 1200) + ideas) * Decimal("1.2")
    assert estimate_planning_cost(text, ch, prices, **_SYS, segment_calls=3) == many
    # large catalogs are embedded
    big = estimate_planning_cost(text, ch, prices, **_SYS, catalog_size=301, candidate_tokens=75)
    narrowed = segment + _usd(calls * (600 + 75) + 1000, calls * 80)
    embed = Decimal(301 * 20 + 1000) * Decimal("0.12") / 1_000_000
    assert big == (narrowed + embed) * Decimal("1.2")
    free = ch.model_copy(update={"embed": EmbedConfig(provider="ollama", model="x")})
    assert estimate_planning_cost(
        text, free, prices, **_SYS, catalog_size=301, candidate_tokens=75
    ) == (narrowed * Decimal("1.2"))


def test_estimate_planning_cost_arabic_tashkeel(prices: PriceTable, channel_file: Path) -> None:
    ch = _paid(channel_file)
    assert ch.language is Language.ARABIC and ch.diacritizer == "llm"
    text = "\u0643\u062a\u0628 " * 10  # 40 chars, all-Arabic -> 14 tokens, 10 words
    segment = _usd(100 + 14, 17)
    ideas = _usd(600 + 14, 80)
    tashkeel = _usd(50 + 14, 35)
    assert estimate_planning_cost(text, ch, prices, **_SYS) == (
        (segment + ideas + tashkeel) * Decimal("1.2")
    )
    mishkal = ch.model_copy(update={"diacritizer": "mishkal"})
    assert estimate_planning_cost(text, mishkal, prices, **_SYS) == (
        (segment + ideas) * Decimal("1.2")
    )


def test_estimate_planning_cost_free_llm_is_zero(prices: PriceTable, channel_file: Path) -> None:
    assert (
        estimate_planning_cost("some words here", load_channel(channel_file), prices, **_SYS) == 0
    )


def test_estimate_part_cost(prices: PriceTable, channel_file: Path) -> None:
    base = load_channel(channel_file)
    ch = base.model_copy(
        update={
            "llm": base.llm.model_copy(
                update={"provider": "anthropic", "model": "claude-opus-5-5"}
            ),
            "style": base.style.model_copy(
                update={"image_provider": "openai", "image_model": "gpt-image-1"}
            ),
            "voice": base.voice.model_copy(
                update={"provider": "elevenlabs", "model": "eleven_multilingual_v2"}
            ),
        }
    )

    def scene(i: int, match: Literal["reused", "bundled", "new"], key: str = "") -> Scene:
        return Scene(
            id=f"s{i}",
            narration=Narration(source_text="x"),
            match=match,
            idea_key=key or f"thing-{i}|roman",
        )

    verse = Scene(
        id="s9",
        narration=Narration(source_text="v"),
        kind=SceneKind.VERSE,
        match="new",
        idea_key="book|generic",
    )
    no_key = Scene(id="s8", narration=Narration(source_text="n"), match="new")
    part = Part(
        index=1,
        count=1,
        hook_text="h",
        scenes=[
            scene(1, "new"),
            scene(2, "new"),
            scene(3, "reused"),
            scene(4, "bundled", "eye|generic"),
            scene(5, "bundled", "eye|generic"),  # second use: reused, no check
            verse,
            no_key,
        ],
    )
    speech = Decimal("0.60")

    def check(channel: ChannelConfig, key: str) -> Decimal:
        # exactly what generate_one / the bundled first copy check before the call
        ident = vision_model(channel)
        assert ident is not None
        provider, model = ident
        return estimate_check_cost(channel.style, key, prices, provider, model, with_reference=True)

    for vision in (None, VisionConfig(provider="anthropic", model="claude-haiku-5-5")):
        c = ch.model_copy(update={"vision": vision or ch.vision})
        cost = estimate_part_cost(part, c, prices, voice_chars=2000)
        image = Decimal("0.04")
        expected = (
            image
            + check(c, "thing-1|roman")
            + image
            + check(c, "thing-2|roman")
            + check(c, "eye|generic")
            + speech
        )
        assert cost == expected
    # each check sends the candidate and the reference sketch: two vision images
    # (opus 0.01 per image) on top of the prompt tokens
    tokens_only = check(ch, "thing-1|roman") - 2 * Decimal("0.01")
    assert tokens_only > 0
    # vision off: images and voice only, no check is priced
    off = ch.model_copy(update={"vision": VisionConfig(provider="none")})
    assert estimate_part_cost(part, off, prices, voice_chars=2000) == 2 * image + speech


def test_require_generation_prices_skips_vision_when_off(
    prices: PriceTable, channel_file: Path
) -> None:
    base = load_channel(channel_file)
    unpriced = base.model_copy(
        update={"llm": base.llm.model_copy(update={"provider": "anthropic", "model": "unpriced"})}
    )
    with pytest.raises(PlanningError, match="anthropic/unpriced"):
        require_generation_prices(unpriced, prices, vision=True)
    off = unpriced.model_copy(update={"vision": VisionConfig(provider="none")})
    require_generation_prices(off, prices, vision=True)


def test_estimate_llm_call_matches_tracked_llm_check(prices: PriceTable) -> None:
    text = "x" * 400  # 100 tokens in, 120 out
    plain = estimate_llm_call(prices, "anthropic", "claude-opus-5-5", text)
    assert plain == (100 * Decimal("4.00") + 120 * Decimal("20.00")) / 1_000_000
    with_images = estimate_llm_call(prices, "anthropic", "claude-opus-5-5", text, images=2)
    assert with_images == plain + 2 * Decimal("0.01")
    assert estimate_llm_call(prices, "ollama", "llava", text, images=2) == 0


def test_require_generation_prices_uses_vision_config(
    prices: PriceTable, channel_file: Path
) -> None:
    base = load_channel(channel_file)  # ollama LLM, openai/gpt-image-1
    require_generation_prices(base, prices, vision=True)
    paid = base.model_copy(update={"vision": VisionConfig(provider="anthropic", model="unpriced")})
    with pytest.raises(PlanningError, match="anthropic/unpriced"):
        require_generation_prices(paid, prices, vision=True)
    require_generation_prices(paid, prices, vision=False)
    with pytest.raises(PlanningError, match="anthropic/unpriced"):
        require_prices(paid, prices)
    no_image = base.model_copy(
        update={"style": base.style.model_copy(update={"image_model": "unpriced"})}
    )
    with pytest.raises(PlanningError, match="openai/unpriced"):
        require_generation_prices(no_image, prices, vision=False)


def _opus_channel(channel_file: Path) -> ChannelConfig:
    base = load_channel(channel_file)
    return base.model_copy(
        update={
            "llm": base.llm.model_copy(update={"provider": "anthropic", "model": "claude-opus-5-5"})
        }
    )


def test_tracked_llm_blocks_before_call(prices: PriceTable) -> None:
    fake = FakeLLM([Out()])
    t = SpendTracker(Decimal("0.0000001"))
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with pytest.raises(BudgetExceededError):
        llm.complete_structured(system="s" * 400, user="u" * 400, output=Out)
    assert fake.calls == []
    assert t.spent == 0


def test_tracked_llm_charges_usage(prices: PriceTable) -> None:
    fake = FakeLLM([Out()])
    t = SpendTracker(Decimal("10"))
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    res = llm.complete_structured(system="s", user="u", output=Out)
    assert res.value == Out()
    assert llm.name == "fake"
    assert t.spent == (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000
    assert len(fake.calls) == 1


def test_tracked_llm_provider_error_logs_str_only(
    prices: PriceTable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLM([ProviderError("boom")])
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with caplog.at_level(logging.WARNING), pytest.raises(ProviderError):
        llm.complete_structured(system="secret-prompt", user="u", output=Out)
    assert "boom" in caplog.text
    assert "secret-prompt" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
    assert t.spent == 0


def test_tracked_llm_output_estimate_is_1_2x(prices: PriceTable) -> None:
    tokens = 1000
    system, user = "a" * 4000, ""
    one_x = (Decimal(tokens) * 4 + Decimal(tokens) * 20) / 1_000_000
    one_two = (Decimal(tokens) * 4 + Decimal(1200) * 20) / 1_000_000
    limit = (one_x + one_two) / 2
    fake = FakeLLM([Out()])
    llm = TrackedLLM(fake, SpendTracker(limit), "anthropic", "claude-opus-5-5", prices)
    with pytest.raises(BudgetExceededError):
        llm.complete_structured(system=system, user=user, output=Out)
    assert fake.calls == []


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow"
    for path in [*(root / "costs").glob("*.py"), root / "llm" / "factory.py"]:
        path.read_bytes().decode("ascii")


def test_tracked_llm_charges_fallback_model_and_logs_once(
    prices: PriceTable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLM([Out(), Out()], model="claude-haiku-5-5")
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with caplog.at_level(logging.INFO):
        llm.complete_structured(system="s", user="u", output=Out)
        llm.complete_structured(system="s", user="u", output=Out)
    assert (
        t.spent == 2 * (Decimal(100) * Decimal("0.10") + Decimal(50) * Decimal("0.50")) / 1_000_000
    )
    assert caplog.text.count("served by fallback model claude-haiku-5-5") == 1


def test_tracked_llm_unpriced_served_model_charges_configured(prices: PriceTable) -> None:
    fake = FakeLLM([Out()], model="mystery-model")
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    llm.complete_structured(system="s", user="u", output=Out)
    assert t.spent == (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000


def test_tracked_llm_same_model_charges_configured_without_log(
    prices: PriceTable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLM([Out()], model="claude-opus-5-5")
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with caplog.at_level(logging.INFO):
        llm.complete_structured(system="s", user="u", output=Out)
    assert t.spent == (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000
    assert "fallback" not in caplog.text


def test_tracked_llm_forwards_images_and_charges_vision(prices: PriceTable) -> None:
    fake = FakeLLM([Out()])
    t = SpendTracker(Decimal("10"))
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    llm.complete_structured(system="s", user="u", output=Out, images=[b"a", b"b"])
    assert fake.images == [(b"a", b"b")]
    tokens = (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000
    assert t.spent == tokens + 2 * Decimal("0.01")


def test_tracked_llm_estimate_includes_vision_before_call(prices: PriceTable) -> None:
    fake = FakeLLM([Out()])
    t = SpendTracker(Decimal("0.015"))  # room for one image check, not two
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with pytest.raises(BudgetExceededError):
        llm.complete_structured(system="s", user="u", output=Out, images=[b"a", b"b"])
    assert fake.calls == []


def test_tracked_llm_without_images_charges_no_vision(prices: PriceTable) -> None:
    fake = FakeLLM([Out()])
    t = SpendTracker(Decimal("10"))
    TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices).complete_structured(
        system="s", user="u", output=Out
    )
    assert fake.images == [()]
    assert t.spent == (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000


def test_tracked_llm_falls_back_to_configured_vision_price(prices: PriceTable) -> None:
    fake = FakeLLM([Out()], model="claude-served-x")
    served_llm = prices.llm

    class P:
        def llm(self, provider: str, model: str) -> tuple[Decimal, Decimal]:
            return served_llm("anthropic", "claude-opus-5-5")

        def vision(self, provider: str, model: str) -> Decimal:
            if model == "claude-served-x":
                raise PlanningError("no vision price")
            return prices.vision(provider, model)

    t = SpendTracker(Decimal("10"))
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", P())  # type: ignore[arg-type]
    llm.complete_structured(system="s", user="u", output=Out, images=[b"a"])
    assert t.spent == (Decimal(100) * 4 + Decimal(50) * 20) / 1_000_000 + Decimal("0.01")


def test_voice_chars_excludes_verses() -> None:
    part = Part(
        index=1,
        count=1,
        hook_text="HOOK never spoken",
        recap_text="RECAP",
        teaser_text="TEASER",
        scenes=[
            Scene(id="n1", narration=Narration(source_text="abcd")),
            Scene(id="n2", narration=Narration(source_text="abcd", tts_text="abcdefg")),
            Scene(
                id="v1",
                narration=Narration(source_text="verse text never voiced"),
                kind=SceneKind.VERSE,
            ),
        ],
    )
    assert voice_chars(part, Language.ENGLISH, Lexicon.EMPTY) == 4 + 7


def test_tracked_llm_billed_failure_charges_the_estimate_and_reraises(prices: PriceTable) -> None:
    error = BilledProviderError("Anthropic output did not match Out")
    fake = FakeLLM([error])
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    assert llm.cost_estimated is False
    with pytest.raises(BilledProviderError) as info:
        llm.complete_structured(system="s" * 40, user="u" * 40, output=Out, images=[b"a"])
    assert info.value is error
    estimate = estimate_llm_call(
        prices, "anthropic", "claude-opus-5-5", "s" * 40 + "u" * 40, images=1
    )
    assert t.spent == estimate > 0
    assert [e.what for e in t.entries] == ["llm anthropic/claude-opus-5-5"]
    assert llm.cost_estimated is True


def test_tracked_llm_refusal_still_charges_nothing(prices: PriceTable) -> None:
    fake = FakeLLM([ProviderRefusalError("cyber")])
    t = SpendTracker(None)
    llm = TrackedLLM(fake, t, "anthropic", "claude-opus-5-5", prices)
    with pytest.raises(ProviderRefusalError):
        llm.complete_structured(system="s", user="u", output=Out)
    assert t.spent == 0 and llm.cost_estimated is False


def test_by_what_sums_entries_in_order() -> None:
    tracker = SpendTracker(None, spent=Decimal("5"))  # the seed is not in the entries
    tracker.charge(Decimal("0.1"), "image a/b")
    tracker.charge(Decimal("0.2"), "voice c/d")
    tracker.charge(Decimal("0.3"), "image a/b")
    assert list(tracker.by_what().items()) == [
        ("image a/b", Decimal("0.4")),
        ("voice c/d", Decimal("0.2")),
    ]
    assert SpendTracker(None).by_what() == {}


def test_add_spend_accumulates_across_stages() -> None:
    ledger: dict[str, Decimal] = {"image a/b": Decimal("0.1")}
    first = SpendTracker(None)
    first.charge(Decimal("0.2"), "image a/b")
    first.charge(Decimal("0.5"), "voice c/d")
    add_spend(ledger, first)
    second = SpendTracker(None)
    second.charge(Decimal("0.05"), "voice c/d")
    add_spend(ledger, second)
    assert ledger == {"image a/b": Decimal("0.3"), "voice c/d": Decimal("0.55")}


def test_voice_chars_counts_spoken_form() -> None:
    part = Part(
        index=1,
        count=1,
        hook_text="h",
        recap_text="r",
        teaser_text="t",
        scenes=[Scene(id="n1", narration=Narration(source_text="1453"))],
    )
    got = voice_chars(part, Language.ENGLISH, Lexicon.EMPTY)
    assert got == len("fourteen fifty-three")
