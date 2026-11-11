"""The planning pipeline: script file in, unapproved `Series` out.

Every gate that can stop a run without spending comes first: the rights and
clean-text checks, the provider keys, and the planning-cost estimate against
the channel's spending limit. Only then is any LLM or embedding call made.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Literal

from narratoflow.channels.config import ChannelConfig, style_fingerprint
from narratoflow.channels.keys import check_provider_keys
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import (
    CANDIDATE_TOP_K,
    CATALOG_EMBED_THRESHOLD,
    SpendTracker,
    estimate_part_cost,
    estimate_planning_cost,
    estimate_tokens,
)
from narratoflow.embed import Embedder, make_embedder
from narratoflow.interfaces import Diacritizer
from narratoflow.library.candidates import candidate_keys
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.matching import MatchKind, match_idea
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.llm.base import LLMClient
from narratoflow.llm.factory import make_llm
from narratoflow.models import Language, Part, Scene, SceneKind, Series
from narratoflow.planning.ideas import assign_ideas
from narratoflow.planning.prompts import IDEAS_SYSTEM, SEGMENT_SYSTEM, TASHKEEL_SYSTEM
from narratoflow.planning.segment import segment_script
from narratoflow.planning.series import split_into_parts
from narratoflow.quran.corpus import QuranCorpus, default_corpus
from narratoflow.quran.detect import VerseHit, detect_verses
from narratoflow.script.source import load_script
from narratoflow.settings import Settings
from narratoflow.text.tashkeel.engines import LlmDiacritizer, MishkalDiacritizer, vocalize_scene
from narratoflow.visuals.bundled import BundledIcons, default_bundled_icons

logger = logging.getLogger(__name__)

_MATCH: dict[MatchKind, Literal["reused", "bundled", "new"]] = {
    MatchKind.REUSED: "reused",
    MatchKind.BUNDLED: "bundled",
    MatchKind.NEW: "new",
}


def estimate_planning(
    text: str, channel: ChannelConfig, prices: PriceTable, *, catalog_size: int
) -> Decimal:
    """The pre-call planning estimate for `text`, using the real system prompts."""
    return estimate_planning_cost(
        text,
        channel,
        prices,
        segment_system_tokens=estimate_tokens(SEGMENT_SYSTEM),
        ideas_system_tokens=estimate_tokens(IDEAS_SYSTEM),
        tashkeel_system_tokens=estimate_tokens(TASHKEEL_SYSTEM),
        catalog_size=catalog_size,
    )


def _diacritizer(channel: ChannelConfig, llm: LLMClient) -> Diacritizer:
    if channel.diacritizer == "mishkal":
        return MishkalDiacritizer()
    return LlmDiacritizer(llm)


def _candidates(
    library: SketchLibrary, version: int, scenes: list[Scene], embedder: Embedder | None
) -> list[list[tuple[str, str]]]:
    """Candidate lists parallel to `scenes`; verse scenes get none and are not embedded."""
    narration = [i for i, s in enumerate(scenes) if s.kind is SceneKind.NARRATION]
    found = candidate_keys(
        library,
        version,
        [scenes[i].narration.source_text for i in narration],
        embedder,
        full_limit=CATALOG_EMBED_THRESHOLD,
        top_k=CANDIDATE_TOP_K,
    )
    result: list[list[tuple[str, str]]] = [[] for _ in scenes]
    for i, cands in zip(narration, found, strict=True):
        result[i] = cands
    return result


def _match(library: SketchLibrary, version: int, scene: Scene, bundled: BundledIcons) -> Scene:
    if scene.kind is SceneKind.VERSE or scene.idea_key is None:
        return scene
    found = match_idea(library, version, parse_idea_key(scene.idea_key), bundled)
    return scene.model_copy(update={"match": _MATCH[found.kind]})


def _voice_chars(part: Part) -> int:
    return sum(len(s.narration.tts_text or s.narration.source_text) for s in part.scenes)


def plan_series(
    script_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    corpus: QuranCorpus | None = None,
    prices: PriceTable | None = None,
) -> Series:
    """Plan a script into an unapproved series of parts.

    `llm`, `embedder`, `corpus` and `prices` are injectable for tests; an injected
    client is used as given. Raises `ScriptBlockedError`, `MissingApiKeyError` or
    `BudgetExceededError` before any LLM or embedding call.
    """
    source, text = load_script(script_path, channel.language)
    check_provider_keys(channel, settings)
    prices = prices if prices is not None else default_prices(settings)
    tracker = SpendTracker(channel.spending_limit_usd)
    library = default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    catalog_size = len(library.idea_catalog(version))
    estimate = estimate_planning(text, channel, prices, catalog_size=catalog_size)
    tracker.ensure_room(estimate, "planning")

    if llm is None:
        llm = make_llm(channel, settings, tracker, prices)
    if embedder is None and catalog_size > CATALOG_EMBED_THRESHOLD:
        embedder = make_embedder(channel, settings, tracker, prices)
    arabic = channel.language is Language.ARABIC
    diacritizer = _diacritizer(channel, llm) if arabic else None
    verses: list[VerseHit] = []
    if arabic:
        corpus = corpus if corpus is not None else default_corpus(settings)
        verses = detect_verses(text, corpus)

    scenes = segment_script(text, verses, llm)
    scenes = assign_ideas(scenes, _candidates(library, version, scenes, embedder), llm)
    bundled = default_bundled_icons()
    scenes = [_match(library, version, s, bundled) for s in scenes]
    if diacritizer is not None:
        scenes = [vocalize_scene(s, diacritizer) for s in scenes]

    parts = split_into_parts(
        scenes, channel.language, min_s=channel.part_min_s, max_s=channel.part_max_s
    )
    parts = [
        p.model_copy(
            update={
                "est_cost_usd": estimate_part_cost(p, channel, prices, voice_chars=_voice_chars(p))
            }
        )
        for p in parts
    ]
    logger.info("planned %d scenes into %d parts", len(scenes), len(parts))
    return Series(
        title=source.work,
        channel_slug=channel.slug,
        language=channel.language,
        source=source,
        parts=parts,
        est_planning_cost_usd=tracker.spent,
    )
