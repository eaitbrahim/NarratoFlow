"""Approval gate and the approval list shown before any paid production step."""

from __future__ import annotations

from decimal import Decimal

from rich.markup import escape
from rich.table import Table

from narratoflow.channels.config import ChannelConfig
from narratoflow.models import Scene, SceneKind, Series, VerseRef
from narratoflow.planning.segment import UNMATCHED_BRACKETS_FLAG
from narratoflow.text.tashkeel.engines import REJECTED_FLAG

NARRATION_PREVIEW = 60
ELLIPSIS = "\u2026"
APPROVAL_FLAGS = frozenset(
    {REJECTED_FLAG, "unfinished_ending", "over_max", UNMATCHED_BRACKETS_FLAG}
)
_MATCH_LABELS = {"reused": "reused", "bundled": "bundled", "new": "new (paid)"}
_CENT = Decimal("0.01")


def needs_approval(series: Series, channel: ChannelConfig) -> bool:
    """True when the creator must review the plan before production starts."""
    if channel.approval_list:
        return True
    limit = channel.spending_limit_usd
    for part in series.parts:
        if limit is not None and part.est_cost_usd > limit:
            return True
        if APPROVAL_FLAGS.intersection(part.flags):
            return True
        for scene in part.scenes:
            if APPROVAL_FLAGS.intersection(scene.flags):
                return True
            if scene.verse is not None and not scene.verse.certain:
                return True
    return False


def format_usd(amount: Decimal) -> str:
    """Whole cents as `$0.12`; finer amounts to four decimals, e.g. `$0.0123`."""
    if amount == amount.quantize(_CENT):
        return f"${amount:.2f}"
    return f"${amount:.4f}"


def verse_label(ref: VerseRef) -> str:
    span = f"{ref.surah}:{ref.ayah_start}"
    if ref.ayah_end != ref.ayah_start:
        span += f"-{ref.ayah_end}"
    return span if ref.certain else f"{span} (uncertain)"


def _preview(scene: Scene) -> str:
    if scene.kind is SceneKind.VERSE and scene.verse is not None:
        text = scene.verse.uthmani_text
    else:
        text = scene.narration.source_text
    text = " ".join(text.split())
    if len(text) > NARRATION_PREVIEW:
        text = text[:NARRATION_PREVIEW] + ELLIPSIS
    return text


def render_approval_list(series: Series) -> Table:
    """One row per scene, then one estimate row per part and a series total. No I/O."""
    table = Table(title=escape(series.title), show_lines=False)
    table.add_column("part", no_wrap=True)
    table.add_column("scene", no_wrap=True)
    table.add_column("narration")
    table.add_column("idea", no_wrap=True)
    table.add_column("match", no_wrap=True)
    table.add_column("verse", no_wrap=True)
    table.add_column("flags")
    for part in series.parts:
        label = f"{part.index}/{part.count}"
        for scene in part.scenes:
            table.add_row(
                label,
                escape(scene.id),
                escape(_preview(scene)),
                escape(scene.idea_key or ""),
                _MATCH_LABELS.get(scene.match or "", ""),
                verse_label(scene.verse) if scene.verse is not None else "",
                escape(", ".join(scene.flags)),
            )
    table.add_section()
    for part in series.parts:
        new = sum(1 for s in part.scenes if s.match == "new")
        table.add_row(
            f"{part.index}/{part.count}",
            "",
            f"estimate: {part.est_duration_s:.0f} s, {new} new sketch(es), "
            f"{format_usd(part.est_cost_usd)}",
            "",
            "",
            "",
            escape(", ".join(part.flags)),
        )
    parts_cost = sum((p.est_cost_usd for p in series.parts), Decimal(0))
    duration = sum(p.est_duration_s for p in series.parts)
    table.add_section()
    table.add_row(
        "total",
        "",
        f"{len(series.parts)} part(s), {duration:.0f} s, {format_usd(parts_cost)} production"
        f" + {format_usd(series.est_planning_cost_usd)} planning",
        "",
        "",
        "",
        "",
    )
    return table
