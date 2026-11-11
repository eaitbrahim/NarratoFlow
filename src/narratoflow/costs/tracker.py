"""Spend tracking, cost estimates and the budget-checked LLM wrapper.

Estimates are deterministic heuristics, not quotes. Prompts are never logged.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel

from narratoflow.channels.config import ChannelConfig, vision_model
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import BudgetExceededError, PlanningError, ProviderError
from narratoflow.library.keys import parse_idea_key
from narratoflow.llm.base import LLMClient, StructuredResult, Usage
from narratoflow.models import Language, Part, SceneKind

logger = logging.getLogger(__name__)

_ARABIC = re.compile(r"[\u0600-\u06ff]")
_MILLION = Decimal(1_000_000)
_OUTPUT_FACTOR = Decimal("1.2")
CATALOG_EMBED_THRESHOLD = 300
CATALOG_DESCRIPTION_TOKENS = 20
WORDS_PER_SCENE = 10  # SEGMENT_SYSTEM asks for 5-10 s scenes
CANDIDATE_TOP_K = 5
IDEA_OUTPUT_TOKENS = 80
_TASHKEEL_OUTPUT_FACTOR = Decimal("2.5")
PLANNING_MARGIN = Decimal("1.2")
VISION_IMAGES_PER_CHECK = 2  # the candidate sketch plus the channel's reference sketch


@dataclass(frozen=True)
class SpendEntry:
    what: str
    amount_usd: Decimal
    at: datetime


@dataclass
class SpendTracker:
    limit_usd: Decimal | None
    spent: Decimal = Decimal(0)
    entries: list[SpendEntry] = field(default_factory=list)

    def ensure_room(self, estimate: Decimal, what: str) -> None:
        """Raise `BudgetExceededError` if `spent + estimate` would pass the limit."""
        if self.limit_usd is not None and self.spent + estimate > self.limit_usd:
            raise BudgetExceededError(self.spent + estimate, self.limit_usd)
        logger.debug("budget ok for %s", what)

    def charge(self, amount: Decimal, what: str) -> None:
        self.spent += amount
        self.entries.append(SpendEntry(what, amount, datetime.now(UTC)))

    def charge_llm(self, provider: str, model: str, usage: Usage, prices: PriceTable) -> Decimal:
        """Record the actual cost of a call; the limit is checked beforehand."""
        price_in, price_out = prices.llm(provider, model)
        cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / _MILLION
        self.charge(cost, f"llm {provider}/{model}")
        return cost


def estimate_tokens(text: str) -> int:
    """`ceil(len/3)` if Arabic letters exceed 30% of non-space chars, else `ceil(len/4)`."""
    if not text:
        return 0
    non_space = sum(1 for c in text if not c.isspace())
    arabic = len(_ARABIC.findall(text))
    divisor = 3 if non_space and arabic / non_space > 0.3 else 4
    return math.ceil(len(text) / divisor)


def _llm_cost(
    prices: PriceTable, provider: str, model: str, input_tokens: int, output_tokens: int
) -> Decimal:
    price_in, price_out = prices.llm(provider, model)
    return (input_tokens * price_in + output_tokens * price_out) / _MILLION


def estimate_llm_call(
    prices: PriceTable, provider: str, model: str, text: str, *, images: int = 0
) -> Decimal:
    """Pre-call estimate of one LLM call over `text` (system + user), as `TrackedLLM` checks it.

    Output is 1.2x the input tokens (at least 1); each image adds the vision price.
    """
    tokens = estimate_tokens(text)
    estimate = _llm_cost(
        prices, provider, model, tokens, max(1, math.ceil(tokens * _OUTPUT_FACTOR))
    )
    if images:
        estimate += images * prices.vision(provider, model)
    return estimate


def estimate_planning_cost(
    text: str,
    channel: ChannelConfig,
    prices: PriceTable,
    *,
    segment_system_tokens: int,
    ideas_system_tokens: int,
    tashkeel_system_tokens: int,
    catalog_size: int = 0,
    candidate_tokens: int = 0,
    segment_calls: int = 1,
) -> Decimal:
    """Estimate every planning call, times `PLANNING_MARGIN`.

    With T text tokens and n = max(1, ceil(words / WORDS_PER_SCENE)) scenes:
    - segmentation: `segment_calls` calls (one per non-verse stretch), each
      carrying the segment system prompt, plus T in and 1.2 x T out in all;
    - ideas: n calls, each ideas system + `candidate_tokens` (the real token
      count of the rendered candidate block one call carries) + T / n in, 80 out;
    - tashkeel (Arabic with the LLM diacritizer only): n calls, n x tashkeel
      system + T in, 2.5 x T out;
    - embeddings when `catalog_size` exceeds 300: the catalog descriptions
      (~20 tokens each) plus the text, priced for `channel.embed`.

    The system-prompt token counts are passed in because the prompts live in
    `planning`, which imports this module.
    """
    text_tokens = estimate_tokens(text)
    calls = max(1, math.ceil(len(text.split()) / WORDS_PER_SCENE))
    llm = channel.llm

    def cost(input_tokens: int, output_tokens: int) -> Decimal:
        return _llm_cost(prices, llm.provider, llm.model, input_tokens, output_tokens)

    total = cost(
        segment_calls * segment_system_tokens + text_tokens, math.ceil(text_tokens * _OUTPUT_FACTOR)
    )
    total += cost(
        calls * (ideas_system_tokens + candidate_tokens) + text_tokens,
        calls * IDEA_OUTPUT_TOKENS,
    )
    if channel.language is Language.ARABIC and channel.diacritizer == "llm":
        total += cost(
            calls * tashkeel_system_tokens + text_tokens,
            math.ceil(text_tokens * _TASHKEEL_OUTPUT_FACTOR),
        )
    if catalog_size > CATALOG_EMBED_THRESHOLD:
        embed_tokens = catalog_size * CATALOG_DESCRIPTION_TOKENS + text_tokens
        embed = channel.embed
        total += embed_tokens * prices.embed(embed.provider, embed.model) / _MILLION
    return total * PLANNING_MARGIN


def require_prices(channel: ChannelConfig, prices: PriceTable, *, embed: bool = False) -> None:
    """Raise `PlanningError` unless every price the pipeline needs is known.

    Called before any paid call so a missing entry cannot surface after spend.
    """
    llm, voice = channel.llm, channel.voice
    prices.llm(llm.provider, llm.model)
    require_generation_prices(channel, prices, vision=True)
    prices.voice(voice.provider, voice.model)
    if embed:
        prices.embed(channel.embed.provider, channel.embed.model)


def require_generation_prices(channel: ChannelConfig, prices: PriceTable, *, vision: bool) -> None:
    """Raise `PlanningError` unless the image price and, with `vision`, the vision
    model's LLM and per-image prices are known (`vision_model`; none when it is off).
    """
    style = channel.style
    prices.image(style.image_provider, style.image_model)
    ident = vision_model(channel) if vision else None
    if ident is not None:
        provider, model = ident
        prices.llm(provider, model)
        prices.vision(provider, model)


def voice_chars(part: Part) -> int:
    """Characters a voice speaks for `part`: `tts_text or source_text` of NARRATION scenes.

    Verse scenes are recited, never voiced, so they count zero; hook, recap and
    teaser texts are never spoken.
    """
    return sum(
        len(s.narration.tts_text or s.narration.source_text)
        for s in part.scenes
        if s.kind is SceneKind.NARRATION
    )


def estimate_part_cost(
    part: Part, channel: ChannelConfig, prices: PriceTable, *, voice_chars: int
) -> Decimal:
    """What sketch generation may spend on `part`, plus voice characters x per-kchar price.

    `voice_chars` is the part's spoken characters (`voice_chars(part)`: NARRATION
    scenes only, never verses).

    Mirrors the pre-call budget checks of sketch generation, so the approval
    estimate is never below what the spending limit enforces:
    - a "new" scene: image price + the vision check estimate (`estimate_check_cost`:
      prompt tokens plus `VISION_IMAGES_PER_CHECK` images, candidate and reference);
    - a "bundled" scene, on the first use of its key in the part: the vision check
      of the copied icon (later uses reuse that copy);
    - verse scenes and scenes without an idea key are never generated: zero.
    The vision model is `vision_model(channel)`; when vision is off a check costs zero.
    """
    # imported here: sketching.vision imports this module for `estimate_llm_call`
    from narratoflow.sketching.vision import estimate_check_cost

    style, voice = channel.style, channel.voice
    ident = vision_model(channel)
    image = prices.image(style.image_provider, style.image_model)

    def check(key: str) -> Decimal:
        if ident is None:
            return Decimal(0)
        return estimate_check_cost(style, key, prices, *ident, with_reference=True)

    total = Decimal(0)
    bundled_seen: set[str] = set()
    for scene in part.scenes:
        if scene.kind is SceneKind.VERSE or scene.idea_key is None:
            continue
        if scene.match == "new":
            total += image + check(scene.idea_key)
        elif scene.match == "bundled":
            key = str(parse_idea_key(scene.idea_key))
            if key not in bundled_seen:
                bundled_seen.add(key)
                total += check(key)
    return total + estimate_speech_cost(prices, voice.provider, voice.model, voice_chars)


def estimate_speech_cost(prices: PriceTable, provider: str, model: str, chars: int) -> Decimal:
    """USD for `chars` characters of speech: `chars / 1000 x per_kchar`.

    Free providers (`FREE_PROVIDERS`, e.g. edge and piper) cost 0 without a price
    entry. Both the approval estimate (`estimate_part_cost`) and the limit check of
    `TrackedSpeechProvider` use this one formula.
    """
    return Decimal(chars) / 1000 * prices.voice(provider, model)


class TrackedLLM:
    """Wraps an `LLMClient`: checks the budget before each call, charges after."""

    def __init__(
        self,
        inner: LLMClient,
        tracker: SpendTracker,
        provider: str,
        model: str,
        prices: PriceTable,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.tracker = tracker
        self.provider = provider
        self.model = model
        self.prices = prices
        self._fallback_logged = False

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output: type[T],
        images: Sequence[bytes] = (),
    ) -> StructuredResult[T]:
        estimate = estimate_llm_call(
            self.prices, self.provider, self.model, system + user, images=len(images)
        )
        self.tracker.ensure_room(estimate, f"llm {self.provider}/{self.model}")
        try:
            result = self.inner.complete_structured(
                system=system, user=user, output=output, images=images
            )
        except ProviderError as err:
            # A failed call is charged nothing: no usage was returned to bill.
            logger.warning("LLM call failed (%s/%s): %s", self.provider, self.model, err)
            raise
        billed = self._billed_model(result.model)
        self.tracker.charge_llm(self.provider, billed, result.usage, self.prices)
        if images:
            self.tracker.charge(
                len(images) * self._vision_price(billed),
                f"vision {self.provider}/{billed}",
            )
        return result

    def _vision_price(self, billed: str) -> Decimal:
        """Vision price of the billed model, else the configured model's."""
        try:
            return self.prices.vision(self.provider, billed)
        except PlanningError:
            return self.prices.vision(self.provider, self.model)

    def _billed_model(self, served: str) -> str:
        """The served model when it differs and is priced, else the configured one."""
        if served == self.model:
            return served
        try:
            self.prices.llm(self.provider, served)
        except PlanningError:
            return self.model
        if not self._fallback_logged:
            self._fallback_logged = True
            logger.info("served by fallback model %s", served)
        return served
