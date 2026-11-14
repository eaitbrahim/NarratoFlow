"""One home for where the CLI and the review station keep a series' files."""

from pathlib import Path

from narratoflow.pipeline.paths import (
    default_audio_dir,
    default_cache_root,
    default_export_dir,
    default_render_dir,
)


def test_defaults_sit_next_to_the_series_json() -> None:
    series = Path("work/series.json")
    assert default_audio_dir(series) == Path("work/audio")
    assert default_export_dir(series) == Path("work/export")
    assert default_cache_root(series) == Path("work/.cache")


def test_render_dir_defaults_to_the_audio_dir() -> None:
    assert default_render_dir(Path("somewhere/audio")) == Path("somewhere/audio")
