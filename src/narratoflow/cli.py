"""Command-line entry point: `narratoflow --help`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from narratoflow.models import Language, Project
from narratoflow.text.tashkeel.verify import strip_diacritics

app = typer.Typer(no_args_is_help=True, help="NarratoFlow whiteboard video engine.")


@app.command()
def init(
    title: str,
    language: Language = Language.ARABIC,
    out: Path = Path("storyboard.json"),
) -> None:
    """Create an empty storyboard."""
    Project(title=title, language=language).save(out)
    print(f"[green]Created[/green] {out}")


@app.command()
def validate(storyboard: Path) -> None:
    """Validate a storyboard file against the schema."""
    project = Project.load(storyboard)
    print(f"[green]OK[/green] '{project.title}': {len(project.scenes)} scenes")


@app.command()
def strip(text: str) -> None:
    """Print Arabic text with diacritics removed (debug helper)."""
    print(strip_diacritics(text))


@app.command()
def render(storyboard: Path, out_dir: Path = Path("out")) -> None:
    """Run the full pipeline (not yet implemented, see M1 issues)."""
    raise typer.Exit(code=_not_implemented("render"))


def _not_implemented(cmd: str) -> int:
    print(f"[yellow]'{cmd}' is not implemented yet.[/yellow]")
    return 2


if __name__ == "__main__":
    app()
