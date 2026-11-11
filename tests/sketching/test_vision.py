from pathlib import Path

import pytest
from llm.fakes import FakeLLM

from narratoflow.channels.config import ChannelConfig, StyleConfig, VisionConfig
from narratoflow.errors import ProviderError
from narratoflow.library.schemas import SketchStatus
from narratoflow.planning.prompts import VISION_SYSTEM
from narratoflow.sketching.vision import VisionVerdict, check_sketch

STYLE = StyleConfig(
    reference_sketch=Path("ref.png"),
    instructions="thin black marker line art",
    image_provider="openai",
    image_model="gpt-image-1",
    line_weight_px=3.0,
    ink="#111111",
)


def _verdict(**over: object) -> VisionVerdict:
    base: dict[str, object] = {
        "human_figure": False,
        "face": False,
        "logo_or_text": False,
        "style_outlier": False,
        "mirror_safe": True,
        "reason": "clean",
    }
    return VisionVerdict.model_validate({**base, **over})


def test_no_llm_is_unchecked_and_not_mirror_safe() -> None:
    assert check_sketch(b"png", STYLE, "eye|generic", None) == (
        SketchStatus.UNCHECKED,
        "vision: not checked",
        False,
        None,
    )


def test_clean_verdict_passes_with_mirror_safe() -> None:
    v = _verdict()
    llm = FakeLLM([v])
    assert check_sketch(b"png", STYLE, "eye|generic", llm) == (
        SketchStatus.PASSED,
        None,
        True,
        v,
    )


def test_clean_verdict_keeps_mirror_unsafe() -> None:
    v = _verdict(mirror_safe=False)
    status, reason, mirror, _ = check_sketch(b"png", STYLE, "map|x", FakeLLM([v]))
    assert (status, reason, mirror) == (SketchStatus.PASSED, None, False)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("human_figure", "human_figure"),
        ("face", "face"),
        ("logo_or_text", "logo_or_text"),
    ],
)
def test_blocking_fields_fail(field: str, expected: str) -> None:
    v = _verdict(**{field: True, "reason": "bad one"})
    status, reason, mirror, verdict = check_sketch(b"png", STYLE, "k|x", FakeLLM([v]))
    assert status is SketchStatus.FAILED
    assert reason == f"vision: {expected}; bad one"
    assert mirror is True
    assert verdict == v


def test_several_blocking_fields_are_all_listed() -> None:
    v = _verdict(face=True, human_figure=True, mirror_safe=False, reason="r")
    _, reason, mirror, _ = check_sketch(b"png", STYLE, "k|x", FakeLLM([v]))
    assert reason == "vision: human_figure, face; r"
    assert mirror is False


def test_style_outlier_alone_warns_but_passes() -> None:
    v = _verdict(style_outlier=True, reason="shaded")
    status, reason, mirror, verdict = check_sketch(b"png", STYLE, "k|x", FakeLLM([v]))
    assert (status, reason, mirror, verdict) == (
        SketchStatus.PASSED,
        "warn: style outlier; shaded",
        True,
        v,
    )


def test_outlier_with_blocking_field_fails() -> None:
    v = _verdict(style_outlier=True, face=True, reason="r")
    status, reason, _, _ = check_sketch(b"png", STYLE, "k|x", FakeLLM([v]))
    assert status is SketchStatus.FAILED
    assert reason == "vision: face; r"


def test_reference_is_sent_as_second_image_and_named_in_text() -> None:
    llm = FakeLLM([_verdict()])
    check_sketch(b"cand", STYLE, "eye|generic", llm, reference_png=b"ref")
    assert llm.images == [(b"cand", b"ref")]
    system, user, output = llm.calls[0]
    assert system == VISION_SYSTEM
    assert output is VisionVerdict
    assert "eye|generic" in user
    assert "first image" in user and "second image" in user
    assert STYLE.instructions in user


def test_without_reference_only_candidate_is_sent() -> None:
    llm = FakeLLM([_verdict()])
    check_sketch(b"cand", STYLE, "k|x", llm)
    assert llm.images == [(b"cand",)]
    assert "second image" not in llm.calls[0][1]


def test_llm_errors_propagate() -> None:
    with pytest.raises(ProviderError):
        check_sketch(b"png", STYLE, "k|x", FakeLLM([ProviderError("down")]))


def test_reason_is_capped_at_200_chars() -> None:
    with pytest.raises(ValueError):
        _verdict(reason="x" * 201)


def test_vision_system_carries_the_calibration() -> None:
    text = VISION_SYSTEM.lower()
    for allowed in ("eyes", "hands", "footprints", "paths", "holding"):
        assert allowed in text
    for rejected in (
        "faces",
        "heads",
        "torsos",
        "stick figures",
        "silhouettes",
        "likeness",
        "logos",
        "trademarks",
        "copyrighted characters",
        "written text",
    ):
        assert rejected in text
    for outlier in ("shading", "fills", "colour", "texture", "detail"):
        assert outlier in text
    assert "mirror_safe" in text and "maps" in text and "direction" in text


def test_channel_vision_config_defaults_to_channel_llm() -> None:
    assert VisionConfig() == VisionConfig(provider=None, model=None)
    assert "vision" in ChannelConfig.model_fields


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow" / "sketching"
    for path in root.glob("*.py"):
        path.read_text(encoding="ascii")
