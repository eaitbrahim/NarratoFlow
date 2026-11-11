"""`narratoflow review`: the optional, cost-gated content-rules review of a series.

The only publish module that builds an LLM client. Every check that can refuse
runs before any call; each part is budget-checked against its own limit, and
each call goes through `TrackedLLM` on a tracker seeded with `Part.spent_usd`,
so `Part.spent_usd` and `Part.spend` record what was really billed. A
`NarratoFlowError` mid-run is re-raised as `ReviewInterruptedError` with the
progress so far.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import SpendTracker, TrackedLLM, add_spend, estimate_llm_call
from narratoflow.errors import NarratoFlowError, PlanningError
from narratoflow.llm.base import LLMClient
from narratoflow.llm.factory import build_llm_client
from narratoflow.models import ContentFinding, ContentReview, Part, Series
from narratoflow.publish.checks import Severity
from narratoflow.publish.content import content_checks, review_prompt, review_sha256
from narratoflow.publish.rules import Rules, default_rules
from narratoflow.settings import Settings
from narratoflow.sketching.generate import require_series_channel

logger = logging.getLogger(__name__)


class _ReviewOutput(BaseModel):
    """The review model's structured output."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ContentFinding]


@dataclass
class ReviewResult:
    reviewed: list[int] = field(default_factory=list)
    findings: int = 0
    spent_usd: Decimal = Decimal(0)
    """This run's spend."""
    per_part: list[tuple[int, int, int, Decimal]] = field(default_factory=list)
    """`(index, fails, warns, spent)` per reviewed part, from `content_checks`."""


class ReviewInterruptedError(NarratoFlowError):
    """A `NarratoFlowError` stopped the review mid-run; finished parts are kept.

    Raised `from` the cause; the message is the cause's plus " (progress so far is
    kept)". `partial` is the series with each part's review and spend so far, and
    the result for the parts already reviewed. The caller saves `partial[0]`.
    """

    def __init__(self, cause: NarratoFlowError, partial: tuple[Series, ReviewResult]) -> None:
        super().__init__(f"{cause} (progress so far is kept)")
        self.partial = partial


def _ident(channel: ChannelConfig) -> tuple[str, str]:
    """Provider and model exactly as `make_llm` uses them."""
    return channel.llm.provider.strip().lower(), channel.llm.model


def _selected(series: Series, parts: Sequence[int] | None) -> list[Part]:
    if parts is None:
        return list(series.parts)
    known = {p.index for p in series.parts}
    for n in parts:
        if n not in known:
            raise PlanningError(f"no part {n}; the series has parts {sorted(known)}")
    wanted = set(parts)
    return [p for p in series.parts if p.index in wanted]


def _part_estimate(
    part: Part, prices: PriceTable, rules: Rules, provider: str, model: str
) -> Decimal:
    system, user = review_prompt(part, rules)
    return estimate_llm_call(prices, provider, model, system + user)


def estimate_review(
    series: Series,
    channel: ChannelConfig,
    prices: PriceTable,
    rules: Rules,
    parts: Sequence[int] | None = None,
) -> Decimal:
    """The pre-call estimate of reviewing the selected parts: exactly what
    `TrackedLLM` checks before each call."""
    provider, model = _ident(channel)
    return sum(
        (_part_estimate(p, prices, rules, provider, model) for p in _selected(series, parts)),
        Decimal(0),
    )


def review_part(
    part: Part, rules: Rules, llm: LLMClient, *, provider: str, model: str, now: datetime
) -> ContentReview:
    """One structured call; the findings are stored as returned (clamping is in
    `content_checks`). `llm` is the tracked client."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise PlanningError("the review clock must be timezone-aware")
    system, user = review_prompt(part, rules)
    result = llm.complete_structured(system=system, user=user, output=_ReviewOutput)
    return ContentReview(
        rules_version=rules.file.version,
        provider=provider,
        model=model,
        reviewed_at=now,
        text_sha256=review_sha256(part),
        findings=list(result.value.findings),
    )


def _counts(part: Part, rules: Rules) -> tuple[int, int]:
    severities = [r.severity for r in content_checks(part, rules)]
    return severities.count(Severity.FAIL), severities.count(Severity.WARN)


def review_series(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    parts: Sequence[int] | None = None,
    llm: LLMClient | None = None,
    prices: PriceTable | None = None,
    rules: Rules | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[Series, ReviewResult]:
    """Review the selected parts of an approved series; the input is not mutated.

    `llm` must be an untracked client (tests inject fakes); it is wrapped on each
    part's tracker here. Preflight errors (approval, channel, part indices, price,
    key, budget) are raised as they are, before any call.
    """
    if not series.approved:
        raise PlanningError("approve the series first")
    require_series_channel(series, channel)
    selected = _selected(series, parts)
    provider, model = _ident(channel)
    prices = prices if prices is not None else default_prices(settings)
    prices.llm(provider, model)
    if isinstance(llm, TrackedLLM):
        raise TypeError("pass an untracked LLM client; it is tracked here")
    client = llm if llm is not None else build_llm_client(channel.llm, settings)
    rules = rules if rules is not None else default_rules(settings)
    clock = now if now is not None else (lambda: datetime.now(UTC))
    what = f"llm {provider}/{model}"
    for part in selected:
        estimate = _part_estimate(part, prices, rules, provider, model)
        SpendTracker(channel.spending_limit_usd, spent=part.spent_usd).ensure_room(estimate, what)

    out = series.model_copy(deep=True)
    by_index = {p.index: p for p in out.parts}
    result = ReviewResult()
    for chosen in selected:
        part = by_index[chosen.index]
        start = part.spent_usd
        tracker = SpendTracker(channel.spending_limit_usd, spent=start)
        tracked = TrackedLLM(client, tracker, provider, model, prices)
        try:
            review = review_part(part, rules, tracked, provider=provider, model=model, now=clock())
        except NarratoFlowError as exc:
            part.spent_usd = tracker.spent
            add_spend(part.spend, tracker)
            result.spent_usd += tracker.spent - start
            partial = ReviewResult(
                reviewed=list(result.reviewed),
                findings=result.findings,
                spent_usd=result.spent_usd,
                per_part=list(result.per_part),
            )
            raise ReviewInterruptedError(exc, (out.model_copy(deep=True), partial)) from exc
        part.content_review = review
        part.spent_usd = tracker.spent
        add_spend(part.spend, tracker)
        spent = tracker.spent - start
        fails, warns = _counts(part, rules)
        result.reviewed.append(part.index)
        result.findings += len(review.findings)
        result.spent_usd += spent
        result.per_part.append((part.index, fails, warns, spent))
        logger.info("content review part %d: %d fail, %d warn", part.index, fails, warns)
    return out, result
