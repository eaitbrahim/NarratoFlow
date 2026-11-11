"""Command-line entry point: `narratoflow --help`."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from narratoflow.audio.build import AudioBuildResult, AudioInterruptedError, build_audio
from narratoflow.channels.config import ChannelConfig, load_channel, style_fingerprint, vision_model
from narratoflow.channels.keys import check_provider_keys, required_key_env_vars
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import require_generation_prices
from narratoflow.errors import CorruptInputError, LibraryError, NarratoFlowError, PlanningError
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import SketchStatus
from narratoflow.llm.factory import build_llm_client, vision_llm_config
from narratoflow.logging_setup import setup_logging
from narratoflow.models import Language, Project, Series
from narratoflow.money import format_usd
from narratoflow.planning.approval import needs_approval, render_approval_list
from narratoflow.planning.pipeline import plan_series
from narratoflow.render.build import RenderResult, build_render
from narratoflow.settings import get_settings
from narratoflow.sketching.generate import (
    GenerationInterruptedError,
    GenerationResult,
    generate_missing,
    require_series_channel,
)
from narratoflow.sketching.resolve import (
    check_regenerable,
    regenerate,
    swap_idea,
    use_text_card,
)
from narratoflow.sketching.vision import estimate_check_cost
from narratoflow.text.quality import require_clean

app = typer.Typer(no_args_is_help=True, help="NarratoFlow whiteboard video engine.")
channel_app = typer.Typer(no_args_is_help=True, help="Channel configuration commands.")
library_app = typer.Typer(no_args_is_help=True, help="Sketch library commands.")
sketch_app = typer.Typer(no_args_is_help=True, help="Sketch generation commands.")
app.add_typer(channel_app, name="channel")
app.add_typer(library_app, name="library")
app.add_typer(sketch_app, name="sketch")
audio_app = typer.Typer(no_args_is_help=True, help="Audio build commands.")
app.add_typer(audio_app, name="audio")

_out = Console()
_err = Console(stderr=True)


@app.callback()
def _configure() -> None:
    setup_logging(get_settings().log_level)


@app.command()
def validate(storyboard: Path) -> None:
    """Validate a storyboard file against the schema."""
    project = Project.load(storyboard)
    _out.print(f"[green]OK[/green] '{project.title}': {len(project.scenes)} scenes")


@app.command("check-input")
def check_input(file: Path, language: Language = Language.ARABIC) -> None:
    """Reject corrupt (e.g. PDF copy-paste) input text before any spend."""
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CorruptInputError(f"cannot read {file}: {exc}") from exc
    require_clean(text, language)
    _out.print("[green]OK[/green]")


@channel_app.command("check")
def channel_check(channel_toml: Path) -> None:
    """Load a channel and verify its paid providers have API keys."""
    channel = load_channel(channel_toml)
    check_provider_keys(channel, get_settings())
    paid = ", ".join(sorted(k.value for k in required_key_env_vars(channel))) or "none"
    _out.print(f"[green]OK[/green] {channel.name}", highlight=False)
    _out.print(f"style fingerprint: {style_fingerprint(channel.style)}", highlight=False)
    _out.print(f"paid providers: {paid}", highlight=False)


@library_app.command("keys")
def library_keys(channel_toml: Path) -> None:
    """List reusable idea keys for the channel's current style version."""
    channel = load_channel(channel_toml)
    library = default_library(get_settings())
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    for key in library.idea_keys(version):
        _out.print(key, markup=False, highlight=False)


@library_app.command("copy")
def library_copy(sketch_id: int, target_channel_toml: Path) -> None:
    """Copy a sketch into the target channel's current style version."""
    target = load_channel(target_channel_toml)
    library = default_library(get_settings())
    version = library.ensure_style_version(target.slug, style_fingerprint(target.style))
    record = library.copy_to(sketch_id, version)
    if record.status is SketchStatus.FAILED:
        raise LibraryError(
            f"sketch {sketch_id} is quarantined in channel {target.slug} "
            f"(copy id {record.id}): {record.status_reason}"
        )
    _out.print(record.id, highlight=False)


@app.command()
def plan(
    script: Path,
    channel_toml: Annotated[Path, typer.Option("--channel", help="Channel TOML file.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Series file (default: series.json next to SCRIPT)."),
    ] = None,
) -> None:
    """Plan a script into a series of parts; checks rights, keys and budget before any spend."""
    channel = load_channel(channel_toml)
    target = out if out is not None else script.with_name("series.json")
    if not target.parent.is_dir():
        raise PlanningError(f"output directory {target.parent} does not exist; nothing was spent")
    series = plan_series(script, channel, get_settings())
    if needs_approval(series, channel):
        series.save(target)
        _out.print(render_approval_list(series))
        _out.print(
            f"Review the plan, then run: narratoflow approve {escape(str(target))}",
            highlight=False,
        )
        return
    series.approved = True
    series.save(target)
    parts_cost = sum((p.est_cost_usd for p in series.parts), Decimal(0))
    duration = sum(p.est_duration_s for p in series.parts)
    _out.print(
        f"[green]auto-approved[/green] {len(series.parts)} part(s), {duration:.0f} s, "
        f"{format_usd(parts_cost)} production + {format_usd(series.est_planning_cost_usd)} "
        f"planning -> {escape(str(target))}",
        highlight=False,
        soft_wrap=True,
    )


@app.command()
def approve(
    series_json: Path,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Approve without asking.")] = False,
) -> None:
    """Show a planned series and approve it for production."""
    series = Series.load(series_json)
    _out.print(render_approval_list(series))
    if not yes and not typer.confirm("Approve this series for production?"):
        state = "stays approved" if series.approved else "is still not approved"
        _out.print(f"Unchanged; the series {state}.")
        return
    series.approved = True
    series.save(series_json)
    _out.print(f"[green]Approved[/green] {escape(str(series_json))}", highlight=False)


def _print_resolve_hint(scene_id: str, state: str, series_json: Path, channel_toml: Path) -> None:
    _out.print(
        f"{escape(scene_id)}: {state} - resolve with: narratoflow sketch resolve "
        f"{escape(str(series_json))} {escape(scene_id)} --regenerate | "
        f"--swap KEY --description TEXT | --text-card --channel {escape(str(channel_toml))}",
        highlight=False,
        soft_wrap=True,
    )


def _print_generation(result: GenerationResult, series_json: Path, channel_toml: Path) -> None:
    _out.print(
        f"generated {result.generated}, reused {result.reused}, bundled {result.bundled}, "
        f"kept {result.kept}, quarantined {len(result.quarantined)}, "
        f"unchecked {len(result.unchecked)}, spent {format_usd(result.spent_usd)}",
        highlight=False,
        soft_wrap=True,
    )
    for scene_id in result.quarantined:
        _print_resolve_hint(scene_id, "quarantined", series_json, channel_toml)
    _print_unchecked(result)


def _print_unchecked(result: GenerationResult) -> None:
    if result.unchecked:
        _out.print(
            f"[yellow]warning:[/yellow] {len(result.unchecked)} sketch(es) are unchecked: "
            "vision checking was off for them",
            highlight=False,
            soft_wrap=True,
        )


@sketch_app.command("generate")
def sketch_generate(
    series_json: Path,
    channel_toml: Annotated[Path, typer.Option("--channel", help="Channel TOML file.")],
) -> None:
    """Generate (or reuse) a sketch for every scene of an approved series."""
    channel = load_channel(channel_toml)
    series = Series.load(series_json)
    try:
        updated, result = generate_missing(series, channel, get_settings())
    except GenerationInterruptedError as exc:
        exc.partial[0].save(series_json)
        _print_generation(exc.partial[1], series_json, channel_toml)
        raise
    updated.save(series_json)
    _print_generation(result, series_json, channel_toml)


def _regenerate_estimate(idea_key: str, channel: ChannelConfig, prices: PriceTable) -> Decimal:
    """Image price plus the vision check (none when vision is off) for one sketch."""
    style = channel.style
    estimate = prices.image(style.image_provider, style.image_model)
    ident = vision_model(channel)
    if ident is not None:
        estimate += estimate_check_cost(style, idea_key, prices, *ident, with_reference=True)
    return estimate


@sketch_app.command("resolve")
def sketch_resolve(
    series_json: Path,
    scene_id: str,
    channel_toml: Annotated[Path, typer.Option("--channel", help="Channel TOML file.")],
    do_regenerate: Annotated[
        bool, typer.Option("--regenerate", help="Draw a new sketch (paid, asks first).")
    ] = False,
    swap: Annotated[str | None, typer.Option("--swap", help="New idea key.")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Description for --swap.")
    ] = None,
    text_card: Annotated[bool, typer.Option("--text-card", help="Show a text card.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Regenerate without asking.")] = False,
    card_text: Annotated[
        str | None, typer.Option("--card-text", help="Text for --text-card.")
    ] = None,
) -> None:
    """Settle a quarantined scene: regenerate it, swap its idea, or use a text card."""
    if (do_regenerate, swap is not None, text_card).count(True) != 1:
        raise typer.BadParameter("give exactly one of --regenerate, --swap KEY, --text-card")
    if swap is not None and description is None:
        raise typer.BadParameter("--swap needs --description")
    if description is not None and swap is None:
        raise typer.BadParameter("--description only goes with --swap")
    if yes and not do_regenerate:
        raise typer.BadParameter("--yes only goes with --regenerate")
    if card_text is not None and not text_card:
        raise typer.BadParameter("--card-text only goes with --text-card")
    channel = load_channel(channel_toml)
    series = Series.load(series_json)
    if swap is not None and description is not None:
        swapped = swap_idea(series, scene_id, swap, description)
        swapped.save(series_json)
        _out.print(
            f"{escape(scene_id)}: idea is now {escape(swap)}; run narratoflow sketch generate "
            f"{escape(str(series_json))} --channel {escape(str(channel_toml))} to draw it",
            highlight=False,
            soft_wrap=True,
        )
        return
    if text_card:
        use_text_card(series, scene_id, card_text or "").save(series_json)
        _out.print(f"{escape(scene_id)}: now uses a text card", highlight=False)
        return
    # everything that can refuse without spend is checked before asking for money
    settings = get_settings()
    idea_key = check_regenerable(series, scene_id)
    require_series_channel(series, channel)
    check_provider_keys(channel, settings)
    vision_cfg = vision_llm_config(channel)
    if vision_cfg is not None:
        build_llm_client(vision_cfg, settings)  # refuses on a missing key; no network call
    prices = default_prices(settings)
    require_generation_prices(channel, prices, vision=True)
    if not yes:
        estimate = _regenerate_estimate(idea_key, channel, prices)
        if not typer.confirm(
            f"Regenerate the sketch for {scene_id}? Estimated cost {format_usd(estimate)}"
        ):
            _out.print("Unchanged.")
            return
    try:
        updated, result = regenerate(series, scene_id, channel, settings)
    except GenerationInterruptedError as exc:
        exc.partial[0].save(series_json)
        _print_generation(exc.partial[1], series_json, channel_toml)
        raise
    updated.save(series_json)
    attached = any(
        s.id == scene_id and s.sketch_id is not None for p in updated.parts for s in p.scenes
    )
    if attached:
        _out.print(f"{escape(scene_id)}: regenerated", highlight=False)
    else:
        _print_resolve_hint(scene_id, "quarantined again", series_json, channel_toml)
    _out.print(f"spent {format_usd(result.spent_usd)}", highlight=False)
    _print_unchecked(result)


def _print_audio(result: AudioBuildResult, before: Series) -> None:
    seeds = {p.index: p.spent_usd for p in before.parts}
    for built in result.parts:
        tl = built.timeline
        lufs = "n/a" if tl.integrated_lufs is None else f"{tl.integrated_lufs:.1f}"
        voice = tl.voice
        used = "none (verse only)" if voice is None else f"{voice.provider}/{voice.voice_id}"
        run_spend = tl.spent_usd - seeds.get(built.index, Decimal(0))
        estimated = " (estimated)" if tl.cost_estimated and run_spend > 0 else ""
        _out.print(
            f"part {tl.part_index}/{tl.part_count}: {tl.duration_ms / 1000:.1f} s, {lufs} LUFS, "
            f"voice {escape(used)}, spent {format_usd(run_spend)}{estimated} "
            f"(part total {format_usd(tl.spent_usd)}) -> {escape(str(built.audio_path))}",
            highlight=False,
            soft_wrap=True,
        )
        _out.print(
            f"  timeline: {escape(str(built.timeline_path))}", highlight=False, soft_wrap=True
        )
        if voice is not None and voice.fallback_from is not None:
            reasons = "; ".join(voice.fallback_reasons)
            _out.print(
                f"[yellow]warning:[/yellow] voice fell back from {escape(voice.fallback_from)}: "
                f"{escape(reasons)}",
                highlight=False,
                soft_wrap=True,
            )
        for warning in tl.warnings:
            _out.print(
                f"[yellow]warning:[/yellow] part {tl.part_index}: {escape(warning)}",
                highlight=False,
                soft_wrap=True,
            )
    _out.print(f"total spent {format_usd(result.spent_usd)}", highlight=False)


@audio_app.command("build")
def audio_build(
    series_json: Path,
    channel_toml: Annotated[Path, typer.Option("--channel", help="Channel TOML file.")],
    part: Annotated[
        list[int] | None,
        typer.Option("--part", help="Part number to build; repeatable (default: all parts)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory (default: audio/ next to SERIES_JSON)."),
    ] = None,
) -> None:
    """Narrate, mix and master every part of an approved series to a FLAC plus a timeline."""
    channel = load_channel(channel_toml)
    series = Series.load(series_json)
    target = out if out is not None else series_json.parent / "audio"
    try:
        updated, result = build_audio(
            series,
            channel,
            get_settings(),
            out_dir=target,
            cache_root=series_json.parent / ".cache",
            parts=part or None,
        )
    except AudioInterruptedError as exc:
        exc.partial[0].save(series_json)
        _print_audio(exc.partial[1], series)
        raise
    updated.save(series_json)
    _print_audio(result, series)


def _print_render(result: RenderResult) -> None:
    for built in result.parts:
        rep = built.report
        _out.print(
            f"part {rep.part_index}/{rep.part_count}: {rep.duration_ms / 1000:.1f} s, "
            f"{rep.frames} frames, {rep.width}x{rep.height} @ {rep.fps} fps -> "
            f"{escape(str(built.video_path))}",
            highlight=False,
            soft_wrap=True,
        )
        _out.print(f"  report: {escape(str(built.report_path))}", highlight=False, soft_wrap=True)
        for warning in rep.warnings:
            _out.print(
                f"[yellow]warning:[/yellow] part {rep.part_index}: {escape(warning)}",
                highlight=False,
                soft_wrap=True,
            )


@app.command()
def render(
    series_json: Path,
    channel_toml: Annotated[Path, typer.Option("--channel", help="Channel TOML file.")],
    part: Annotated[
        list[int] | None,
        typer.Option("--part", help="Part number to render; repeatable (default: all parts)."),
    ] = None,
    audio: Annotated[
        Path | None,
        typer.Option("--audio", help="Audio directory (default: audio/ next to SERIES_JSON)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory (default: the audio directory)."),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Render processes (default: one per CPU core)."),
    ] = None,
) -> None:
    """Render every part of an approved series to a 1080x1920 MP4 plus a render report."""
    channel = load_channel(channel_toml)
    series = Series.load(series_json)
    audio_dir = audio if audio is not None else series_json.parent / "audio"
    try:
        result = build_render(
            series,
            channel,
            get_settings(),
            audio_dir=audio_dir,
            out_dir=out if out is not None else audio_dir,
            parts=part or None,
            workers=workers,
        )
    except KeyboardInterrupt:
        _err.print("[red]interrupted: render stopped; finished parts are kept[/red]")
        raise typer.Exit(130) from None
    _print_render(result)


def main() -> None:
    """Console-script entry point: the only place NarratoFlowError is caught."""
    try:
        app()
    except NarratoFlowError as exc:
        _err.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
