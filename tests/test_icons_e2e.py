"""Real shipped icon pack across plan, sketch generate and the Plan 6 provenance check."""

from dataclasses import replace
from pathlib import Path

from llm.fakes import FakeLLM
from planning.fakes import ScriptedLLM
from pydantic import SecretStr
from sketching.support import PAID_VISION, PRICES, FakeImages, clean

from narratoflow.channels.config import load_channel, style_fingerprint
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.repository import default_library
from narratoflow.library.schemas import SketchOrigin, SketchStatus
from narratoflow.models import Language
from narratoflow.planning.ideas import IdeaOut
from narratoflow.planning.pipeline import plan_series
from narratoflow.publish.provenance import AssetRecord, _record_findings
from narratoflow.publish.rules import default_rules
from narratoflow.settings import Settings
from narratoflow.sketching.generate import generate_missing
from narratoflow.visuals.bundled import default_bundled_icons

BODY = (
    "An idea came to him at night. He wrote it down before dawn. "
    "By morning the whole town knew of it."
)
KEY = "lightbulb|generic"


class AnswersWith(ScriptedLLM):
    """The scripted planner, but every idea answer is `KEY`."""

    def complete_structured(self, *, system, user, output, images=()):  # type: ignore[no-untyped-def]
        result = super().complete_structured(system=system, user=user, output=output, images=images)
        if output is IdeaOut:
            return replace(result, value=result.value.model_copy(update={"idea_key": KEY}))
        return result


def test_shipped_icon_is_planned_copied_and_passes_provenance(
    channel_file: Path, tmp_path: Path
) -> None:
    pack = default_bundled_icons()
    icon = pack.get(parse_idea_key(KEY))
    assert icon is not None
    settings = Settings(
        data_dir=tmp_path / "data", image_api_key=SecretStr("img"), llm_api_key=SecretStr("llm")
    )
    channel = load_channel(channel_file).model_copy(update={"language": Language.ENGLISH})
    script = tmp_path / "script.txt"
    script.write_text(
        f'+++\nauthor = "A"\nwork = "W"\nrights = "own"\n+++\n{BODY}', encoding="utf-8"
    )
    library = default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    assert library.idea_catalog(version) == []  # the channel library is empty

    llm = AnswersWith()
    series = plan_series(script, channel, settings, llm=llm)
    prompts = [user for _, user, out in llm.calls if out is IdeaOut]
    assert prompts
    assert all("lightbulb|generic — a lightbulb, an idea or insight" in p for p in prompts)
    assert all(p.count("|generic — ") == len(pack.catalog()) for p in prompts)
    scenes = [s for part in series.parts for s in part.scenes]
    assert scenes
    assert all(s.match == "bundled" for s in scenes)

    series.approved = True
    images = FakeImages()
    out, result = generate_missing(
        series,
        channel.model_copy(update={"vision": PAID_VISION}),
        settings,
        images=images,
        vision=FakeLLM(clean(20)),
        library=library,
        prices=PRICES,
    )
    assert images.requests == []
    assert (result.bundled, result.generated) == (1, 0)

    attached = {sc.sketch_id for part in out.parts for sc in part.scenes}
    assert len(attached) == 1
    [sketch_id] = attached
    record = library.get_sketch(sketch_id or 0)
    assert record is not None
    assert record.origin is SketchOrigin.BUNDLED
    assert record.status is SketchStatus.PASSED
    assert record.provenance.license == "MIT"
    assert record.provenance.source == icon.source
    assert record.blob_hash == icon.sha256

    asset = AssetRecord(
        kind="icon",
        asset_id=str(record.id),
        source=record.provenance.source,
        license=record.provenance.license,
        provider=record.provenance.provider,
        scene_ids=["s1"],
    )
    assert _record_findings(asset, default_rules(settings)) == []
