from __future__ import annotations

import pytest
from fixtures import images as img
from visuals.lineart_samples import sample_png

from narratoflow.channels.config import ChannelConfig, style_fingerprint
from narratoflow.errors import PlanningError, SketchRejectedError
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import SketchOrigin, SketchStatus
from narratoflow.models import Narration, Scene, SceneKind, Series, VerseRef, VisualSource
from narratoflow.publish.rules import default_rules
from narratoflow.settings import Settings
from narratoflow.sketching import upload as upload_module
from narratoflow.sketching.generate import QUARANTINED_FLAG
from narratoflow.sketching.upload import attach_upload, check_upload_licence
from narratoflow.visuals.lineart import binarize, ink_to_png, xdog_lineart
from narratoflow.visuals.strokes import extract_strokes
from sketching.support import scene, scenes_of, series


def _version(library: SketchLibrary, channel: ChannelConfig) -> int:
    return library.ensure_style_version(channel.slug, style_fingerprint(channel.style))


def _attach(
    s: Series, scene_id: str, channel: ChannelConfig, library: SketchLibrary, image: bytes
) -> tuple[Series, int]:
    out, record = attach_upload(
        s, scene_id, channel, image, licence="own-work", source="my photo", library=library
    )
    return out, record.id


def test_upload_attaches_lineart(channel: ChannelConfig, library: SketchLibrary) -> None:
    raw = sample_png("camera")
    s = series([scene(1, flags=[QUARANTINED_FLAG, "other"])])
    before = s.model_copy(deep=True)
    out, record = attach_upload(
        s,
        "s001",
        channel,
        raw,
        licence="CC0-1.0",
        source="my photo",
        attribution="me",
        library=library,
    )
    assert s == before
    assert record.origin is SketchOrigin.UPLOADED
    assert record.status is SketchStatus.UNCHECKED
    assert record.status_reason == "uploaded: vision not run"
    assert record.mirror_safe is False
    assert record.provenance.license == "CC0-1.0"
    assert record.provenance.source == "my photo"
    assert record.provenance.attribution == "me"
    assert record.provenance.provider is None
    expected = ink_to_png(xdog_lineart(raw))
    assert library.sketch_bytes(record) == expected
    strokes = library.get_strokes(record.id)
    assert strokes is not None and strokes.strokes
    [sc] = scenes_of(out)
    assert sc.visual.source is VisualSource.IMAGE_TO_SKETCH
    assert sc.sketch_id == record.id and sc.match == "new"
    assert sc.flags == ["other"]


def test_upload_strokes_follow_the_channel_language(
    channel: ChannelConfig, library: SketchLibrary
) -> None:
    out, sketch_id = _attach(series([scene(1)]), "s001", channel, library, sample_png("camera"))
    ink = xdog_lineart(sample_png("camera"))
    assert extract_strokes(ink, rtl=True) != extract_strokes(ink, rtl=False)
    assert library.get_strokes(sketch_id) == extract_strokes(ink, rtl=True)  # Arabic channel
    assert scenes_of(out)[0].sketch_id == sketch_id


def _density_failure(monkeypatch: pytest.MonkeyPatch) -> bytes:
    ink = binarize(img.crosshatched_box())
    monkeypatch.setattr(upload_module, "xdog_lineart", lambda raster: ink)
    return b"any bytes"


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("density", SketchRejectedError),
        ("undecodable", SketchRejectedError),
        ("verse", PlanningError),
        ("text_card", PlanningError),
        ("no_key", PlanningError),
        ("unknown", PlanningError),
        ("other_channel", PlanningError),
    ],
)
def test_upload_refusals_store_nothing(
    case: str,
    error: type[Exception],
    channel: ChannelConfig,
    library: SketchLibrary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = sample_png("camera")
    verse = Scene(
        id="s003",
        narration=Narration(source_text="v"),
        kind=SceneKind.VERSE,
        verse=VerseRef(surah=1, ayah_start=1, ayah_end=1, certain=True, uthmani_text="v"),
        idea_key="book|generic",
    )
    s = series([scene(1), scene(2, idea_key=None), verse])
    scene_id = {"verse": "s003", "no_key": "s002", "unknown": "s999"}.get(case, "s001")
    if case == "text_card":
        s = series([scene(1, visual={"source": "text_card"})])
    if case == "density":
        image = _density_failure(monkeypatch)
    if case == "undecodable":
        image = b"not an image"
    if case == "other_channel":
        s = s.model_copy(update={"channel_slug": "other"})
    version = _version(library, channel)
    before = (library.idea_keys(version), s.model_copy(deep=True))
    with pytest.raises(error):
        attach_upload(
            s, scene_id, channel, image, licence="own-work", source="photo", library=library
        )
    assert library.idea_keys(version) == before[0]
    assert library.get_sketch(1) is None  # idea_keys and find_reusable skip uploads
    assert s == before[1]


def test_upload_density_refusal_names_the_reason(
    channel: ChannelConfig, library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _density_failure(monkeypatch)
    with pytest.raises(SketchRejectedError, match=r"^density: dense_tiles"):
        _attach(series([scene(1)]), "s001", channel, library, image)


def test_upload_does_not_need_an_approved_series(
    channel: ChannelConfig, library: SketchLibrary
) -> None:
    out, _ = _attach(
        series([scene(1)], approved=False), "s001", channel, library, sample_png("camera")
    )
    assert scenes_of(out)[0].sketch_id is not None


def test_upload_onto_a_quarantined_earlier_upload_is_refused(
    channel: ChannelConfig, library: SketchLibrary
) -> None:
    raw = sample_png("camera")
    _, sketch_id = _attach(series([scene(1)]), "s001", channel, library, raw)
    library.quarantine(sketch_id, "creator rejected it")
    with pytest.raises(SketchRejectedError, match="quarantined: creator rejected it"):
        _attach(series([scene(1)]), "s001", channel, library, raw)


def test_upload_blank_source_is_refused(channel: ChannelConfig, library: SketchLibrary) -> None:
    with pytest.raises(PlanningError, match="source"):
        attach_upload(
            series([scene(1)]),
            "s001",
            channel,
            sample_png("camera"),
            licence="own-work",
            source="  ",
            library=library,
        )


def test_check_upload_licence(settings: Settings) -> None:
    rules = default_rules(settings)
    check_upload_licence(rules, "own-work", "")
    check_upload_licence(rules, "CC-BY-4.0", "Ada")
    with pytest.raises(PlanningError, match=r"CC-BY-4\.0 needs an attribution"):
        check_upload_licence(rules, "CC-BY-4.0", " ")
    with pytest.raises(PlanningError, match="mystery-terms is not allowed"):
        check_upload_licence(rules, "mystery-terms", "")
