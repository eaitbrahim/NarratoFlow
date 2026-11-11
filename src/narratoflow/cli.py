"""Command-line entry point: `narratoflow --help`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from narratoflow.channels.config import load_channel, style_fingerprint
from narratoflow.channels.keys import check_provider_keys, required_key_env_vars
from narratoflow.errors import CorruptInputError, LibraryError, NarratoFlowError
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import SketchStatus
from narratoflow.logging_setup import setup_logging
from narratoflow.models import Language, Project
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


def main() -> None:
    """Console-script entry point: the only place NarratoFlowError is caught."""
    try:
        app()
    except NarratoFlowError as exc:
        _err.print(f"[red]{escape(str(exc))}[/red]", highlight=False)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
