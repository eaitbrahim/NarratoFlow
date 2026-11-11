"""Command-line entry point: `narratoflow --help`."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from narratoflow.channels.config import load_channel, style_fingerprint
from narratoflow.channels.keys import check_provider_keys, required_key_env_vars
from narratoflow.errors import CorruptInputError, LibraryError, NarratoFlowError, PlanningError
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import SketchStatus
from narratoflow.logging_setup import setup_logging
from narratoflow.models import Language, Project, Series
from narratoflow.money import format_usd
from narratoflow.planning.approval import needs_approval, render_approval_list
from narratoflow.planning.pipeline import plan_series
from narratoflow.settings import get_settings
from narratoflow.text.quality import require_clean

app = typer.Typer(no_args_is_help=True, help="NarratoFlow whiteboard video engine.")
channel_app = typer.Typer(no_args_is_help=True, help="Channel configuration commands.")
library_app = typer.Typer(no_args_is_help=True, help="Sketch library commands.")
app.add_typer(channel_app, name="channel")
app.add_typer(library_app, name="library")

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


def main() -> None:
    """Console-script entry point: the only place NarratoFlowError is caught."""
    try:
        app()
    except NarratoFlowError as exc:
        _err.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
