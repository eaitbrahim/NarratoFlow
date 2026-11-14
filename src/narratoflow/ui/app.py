"""The Gradio layer of the review station: wiring only, no logic of its own.

`build_app` takes the `gradio` module as an argument so the layer is tested with a fake
and the package imports without the `ui` extra. Every callback calls one function of
`narratoflow.ui.station` and turns a `NarratoFlowError` into `gr.Error`; this is the one
place outside `cli.main` that catches it, because it re-raises as the UI's own error type.
The page keeps nothing but `series.json`'s sha256 (a hidden textbox, D33).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image

from narratoflow.errors import CorruptInputError, NarratoFlowError
from narratoflow.models import AspectRatio
from narratoflow.money import format_usd
from narratoflow.settings import Settings
from narratoflow.ui import station
from narratoflow.ui.station import StationPaths

SAVE_TEXT = "Save text"
SAVE_TIMING = "Save timing"
UPLOAD = "Upload"
REGENERATE = "Regenerate"
TEXT_CARD = "Text card"
REBUILD = "Rebuild scene"
EXPORT_PORTRAIT = "Export 9:16"
EXPORT_LANDSCAPE = "Export 16:9"
PILLARBOX_NOTE = "16:9 is a pillarboxed copy of the 9:16 video"


def _estimate_text(cost: Decimal) -> str:
    if cost == 0:
        return "Regenerating a sketch is free."
    return f"Regenerating a sketch is estimated at {format_usd(cost)}; tick confirm to spend it."


def _rebuild_estimate_text(cost: Decimal) -> str:
    if cost == 0:
        return "Rebuilding the scene is free."
    return (
        f"Rebuilding the scene (voice and sketch) is estimated at {format_usd(cost)}; "
        "tick confirm to spend it."
    )


def _sketch_image(png: bytes | None) -> Image.Image | None:
    return None if png is None else Image.open(io.BytesIO(png))


def _read(path: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise CorruptInputError(f"cannot read image {path}: {exc}") from exc


def _estimate(paths: StationPaths, settings: Settings, scene_id: str) -> str:
    """Both prices the shared confirm box covers: Regenerate and Rebuild scene."""
    lines: list[str] = []
    for label, text, cost in (
        ("estimate", _estimate_text, station.regenerate_cost),
        ("rebuild estimate", _rebuild_estimate_text, station.rebuild_cost),
    ):
        try:
            lines.append(text(cost(paths, scene_id, settings=settings)))
        except NarratoFlowError as exc:
            lines.append(f"No {label}: {exc}")
    return "\n\n".join(lines)


def _show(
    paths: StationPaths, settings: Settings, scene_id: str, status: str = ""
) -> tuple[Any, ...]:
    """The values of every view component for one scene, the status line, then the
    confirm box reset to unticked: a confirmation covers one action, never the next."""
    view = station.scene_view(paths, scene_id, settings=settings)
    return (
        view.source_text,
        view.tts_text,
        view.caption_text,
        None if view.audio_clip is None else str(view.audio_clip),
        _sketch_image(view.sketch_png),
        view.min_duration_ms,
        view.draw_fraction,
        view.sha256,
        _estimate(paths, settings, scene_id),
        status,
        False,
    )


def _on_select(paths: StationPaths, settings: Settings, scene_id: str) -> tuple[Any, ...]:
    return _show(paths, settings, scene_id)


def _on_save_text(
    paths: StationPaths, settings: Settings, scene_id: str, tts: str, caption: str, sha: str
) -> tuple[Any, ...]:
    station.edit_text(
        paths, scene_id, tts_text=tts or "", caption_text=caption or "", expected_sha256=sha
    )
    return _show(paths, settings, scene_id, "Text saved. Rebuild the scene to hear and see it.")


def _on_save_timing(
    paths: StationPaths,
    settings: Settings,
    scene_id: str,
    ms: float | None,
    fraction: float | None,
    sha: str,
) -> tuple[Any, ...]:
    station.edit_timing(
        paths, scene_id, min_duration_ms=ms, draw_fraction=fraction, expected_sha256=sha
    )
    return _show(paths, settings, scene_id, "Timing saved. Rebuild the scene to apply it.")


def _on_upload(
    paths: StationPaths,
    settings: Settings,
    scene_id: str,
    file: str | None,
    licence: str,
    source: str,
    attribution: str,
    sha: str,
) -> tuple[Any, ...]:
    if not file:
        raise CorruptInputError("choose an image to upload")
    station.upload_visual(
        paths,
        scene_id,
        _read(file),
        licence=licence or "",
        source=source or "",
        attribution=attribution or "",
        expected_sha256=sha,
        settings=settings,
    )
    return _show(
        paths, settings, scene_id, "Image attached as line art. Rebuild the scene to render it."
    )


def _on_regenerate(
    paths: StationPaths, settings: Settings, scene_id: str, confirm: bool, sha: str
) -> tuple[Any, ...]:
    done = station.regenerate_visual(
        paths, scene_id, confirm=bool(confirm), expected_sha256=sha, settings=settings
    )
    return _show(paths, settings, scene_id, f"Sketch redrawn, spent {format_usd(done.spent_usd)}.")


def _on_text_card(
    paths: StationPaths, settings: Settings, scene_id: str, card_text: str, sha: str
) -> tuple[Any, ...]:
    station.use_text_card(paths, scene_id, card_text or "", expected_sha256=sha)
    return _show(paths, settings, scene_id, "The scene now shows a text card.")


def _on_rebuild(
    paths: StationPaths, settings: Settings, scene_id: str, confirm: bool, sha: str
) -> tuple[Any, ...]:
    done = station.rebuild_scene(
        paths, scene_id, confirm=bool(confirm), expected_sha256=sha, settings=settings
    )
    parts = ", ".join(str(n) for n in done.rendered_parts) or "none (unchanged)"
    status = f"Rebuilt; rendered part(s): {parts}; spent {format_usd(done.spent_usd)}."
    return _show(paths, settings, scene_id, status)


def _on_export(
    paths: StationPaths, settings: Settings, ratio: AspectRatio, note: str
) -> tuple[list[str], str, bool]:
    files = station.export(paths, ratio, settings=settings)
    names = ", ".join(f.name for f in files) or "nothing (a part failed its checks)"
    return [str(f) for f in files], f"Exported {names}. {note}".strip(), False


def build_app(paths: StationPaths, settings: Settings, gr: Any) -> Any:
    """The Blocks page for one series; the caller launches it."""

    def bind(fn: Callable[..., Any], *fixed: Any) -> Callable[..., Any]:
        """`fn` with the paths, settings and `fixed` already given, errors as `gr.Error`."""

        def run(*values: Any) -> Any:
            try:
                return fn(paths, settings, *fixed, *values)
            except NarratoFlowError as exc:
                raise gr.Error(str(exc)) from exc

        return run

    ids = [row.scene_id for row in station.list_scenes(paths)]
    with gr.Blocks(title="NarratoFlow review station") as demo:
        gr.Markdown("# NarratoFlow review station\nEdits are saved to series.json.")
        scene = gr.Dropdown(choices=ids, value=ids[0] if ids else None, label="Scene")
        source = gr.Textbox(label="Source text (read only)", interactive=False, lines=2)
        tts = gr.Textbox(label="Vocalized text (tts_text)", lines=3)
        caption = gr.Textbox(label="Caption text", lines=2)
        audio = gr.Audio(label="Scene audio", type="filepath", interactive=False)
        image = gr.Image(label="Sketch", type="pil", interactive=False)
        minimum = gr.Number(label="Minimum duration (ms)", precision=0)
        fraction = gr.Number(label="Draw fraction")
        sha = gr.Textbox(label="series.json sha256", visible=False)
        estimate_md = gr.Markdown("", elem_id="estimate")
        confirm = gr.Checkbox(label="Confirm the estimated cost", value=False)
        status = gr.Markdown("", elem_id="status")
        upload_file = gr.File(label="Your own image", type="filepath")
        licence = gr.Textbox(label="Licence (own-work for your own)")
        origin = gr.Textbox(label="Where the image came from")
        attribution = gr.Textbox(label="Attribution")
        card_text = gr.Textbox(label="Text card text")
        exports = gr.File(label="Exported files", file_count="multiple", interactive=False)
        view = [source, tts, caption, audio, image, minimum, fraction, sha, estimate_md]
        view += [status, confirm]
        gr.Button(SAVE_TEXT).click(bind(_on_save_text), [scene, tts, caption, sha], view)
        gr.Button(SAVE_TIMING).click(bind(_on_save_timing), [scene, minimum, fraction, sha], view)
        gr.Button(UPLOAD).click(
            bind(_on_upload), [scene, upload_file, licence, origin, attribution, sha], view
        )
        gr.Button(REGENERATE).click(bind(_on_regenerate), [scene, confirm, sha], view)
        gr.Button(TEXT_CARD).click(bind(_on_text_card), [scene, card_text, sha], view)
        gr.Button(REBUILD).click(bind(_on_rebuild), [scene, confirm, sha], view)
        with gr.Row():
            gr.Button(EXPORT_PORTRAIT).click(
                bind(_on_export, AspectRatio.PORTRAIT, ""), [], [exports, status, confirm]
            )
            gr.Button(EXPORT_LANDSCAPE).click(
                bind(_on_export, AspectRatio.LANDSCAPE, PILLARBOX_NOTE),
                [],
                [exports, status, confirm],
            )
        scene.change(bind(_on_select), [scene], view)
        demo.load(bind(_on_select), [scene], view)
    return demo
