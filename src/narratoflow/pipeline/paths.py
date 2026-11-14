"""Where a series' stage files live by default: the CLI and the review station agree here."""

from __future__ import annotations

from pathlib import Path


def default_audio_dir(series_json: Path) -> Path:
    """`audio/` next to the series file: FLACs and timelines."""
    return series_json.parent / "audio"


def default_render_dir(audio_dir: Path) -> Path:
    """Renders (MP4, captions, reports) go next to the audio unless told otherwise."""
    return audio_dir


def default_export_dir(series_json: Path) -> Path:
    """`export/` next to the series file: the published MP4s."""
    return series_json.parent / "export"


def default_cache_root(series_json: Path) -> Path:
    """The stage cache (TTS clips) of a series: `.cache/` next to its file."""
    return series_json.parent / ".cache"
