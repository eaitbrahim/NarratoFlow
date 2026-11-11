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

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import BudgetExceededError, PlanningError, ProviderError
from narratoflow.llm.base import LLMClient, StructuredResult, Usage
from narratoflow.models import Language, Part

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
    llm, style, voice = channel.llm, channel.style, channel.voice
    prices.llm(llm.provider, llm.model)
    prices.vision(llm.provider, llm.model)
    prices.image(style.image_provider, style.image_model)
    prices.voice(voice.provider, voice.model)
    if embed:
        prices.embed(channel.embed.provider, channel.embed.model)


def estimate_part_cost(
    part: Part, channel: ChannelConfig, prices: PriceTable, *, voice_chars: int
) -> Decimal:
    """New sketches x (image + vision check) plus voice characters x per-kchar price."""
    new = sum(1 for scene in part.scenes if scene.match == "new")
    style, llm, voice = channel.style, channel.llm, channel.voice
    per_sketch = prices.image(style.image_provider, style.image_model) + prices.vision(
        llm.provider, llm.model
    )
    speech = Decimal(voice_chars) / 1000 * prices.voice(voice.provider, voice.model)
    return new * per_sketch + speech


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
        tokens = estimate_tokens(system + user)
        estimate = _llm_cost(
            self.prices,
            self.provider,
            self.model,
            tokens,
            max(1, math.ceil(tokens * _OUTPUT_FACTOR)),
        )
        if images:
            estimate += len(images) * self.prices.vision(self.provider, self.model)
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
                len(images) * self.prices.vision(self.provider, billed),
                f"vision {self.provider}/{billed}",
            )
        return result

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
