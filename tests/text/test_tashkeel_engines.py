import builtins
import logging
import re
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES
from llm.fakes import FakeLLM

import narratoflow.planning.prompts as prompts_module
import narratoflow.text.tashkeel.engines as engines_module
from narratoflow.errors import ChannelConfigError, ProviderError
from narratoflow.models import Narration, Scene, SceneKind
from narratoflow.text.tashkeel.engines import (
    LlmDiacritizer,
    MishkalDiacritizer,
    Vocalized,
    vocalize_scene,
)
from narratoflow.text.tashkeel.verify import assert_skeleton_preserved


def scene(text: str, **kw: Any) -> Scene:
    protected = bool(kw.pop("protected", False))
    return Scene(id="s1", narration=Narration(source_text=text, protected=protected), **kw)


def test_llm_diacritics_accepted() -> None:
    d = LlmDiacritizer(FakeLLM([Vocalized(text="اَلْعِلْمُ نُورٌ")]))
    s = vocalize_scene(scene("العلم نور"), d)
    assert s.narration.tts_text == "اَلْعِلْمُ نُورٌ"
    assert s.narration.caption_text == "العلم نور"
    assert not s.flags


def test_word_change_rejected_and_flagged(caplog: pytest.LogCaptureFixture) -> None:
    d = LlmDiacritizer(FakeLLM([Vocalized(text="اَلْعِلْمُ ضِيَاءٌ")]))
    original = scene("العلم نور")
    with caplog.at_level(logging.WARNING):
        s = vocalize_scene(original, d)
    assert s.narration.tts_text == "العلم نور"
    assert s.narration.caption_text == "العلم نور"
    assert s.flags == ["tashkeel_rejected"]
    assert original.flags == [] and original.narration.tts_text is None
    assert "s1" in caplog.text and "العلم" not in caplog.text


def test_existing_flag_not_duplicated() -> None:
    d = LlmDiacritizer(FakeLLM([Vocalized(text="x")]))
    s = vocalize_scene(scene("العلم نور", flags=["tashkeel_rejected"]), d)
    assert s.flags == ["tashkeel_rejected"]


def test_verse_scene_untouched() -> None:
    llm = FakeLLM([])
    s0 = scene("نص", kind=SceneKind.VERSE)
    assert vocalize_scene(s0, LlmDiacritizer(llm)) == s0
    assert llm.calls == []


def test_protected_scene_untouched() -> None:
    llm = FakeLLM([])
    s0 = scene("نص", protected=True)
    assert vocalize_scene(s0, LlmDiacritizer(llm)) == s0
    assert llm.calls == []


def test_provider_error_flagged_not_raised() -> None:
    d = LlmDiacritizer(FakeLLM([ProviderError("boom")]))
    s = vocalize_scene(scene("العلم نور"), d)
    assert s.narration.tts_text == "العلم نور" and "tashkeel_rejected" in s.flags


def test_other_errors_propagate() -> None:
    d = LlmDiacritizer(FakeLLM([RuntimeError("bug")]))
    with pytest.raises(RuntimeError):
        vocalize_scene(scene("العلم نور"), d)


def test_llm_sends_frozen_system_prompt() -> None:
    llm = FakeLLM([Vocalized(text="نُورٌ")])
    LlmDiacritizer(llm).diacritize("نور")
    assert llm.calls[0][0] == prompts_module.TASHKEEL_SYSTEM
    assert llm.calls[0][1] == "نور"


def test_mishkal_missing_extra_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("mishkal"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ChannelConfigError, match="'arabic' extra"):
        MishkalDiacritizer()


def test_mishkal_runs_and_passes_guard() -> None:
    pytest.importorskip("mishkal")
    text = "العلم نور"
    assert_skeleton_preserved(text, MishkalDiacritizer().diacritize(text))
    padded = "  العلم نور \n"
    assert_skeleton_preserved(padded, MishkalDiacritizer().diacritize(padded))


def test_mishkal_outer_whitespace_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class FakeTashkeel:
        def tashkeel(self, text: str) -> str:
            return " " + add_fatha(text.strip()) + "  "

    pkg = types.ModuleType("mishkal")
    sub = types.ModuleType("mishkal.tashkeel")
    sub.TashkeelClass = FakeTashkeel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mishkal", pkg)
    monkeypatch.setitem(sys.modules, "mishkal.tashkeel", sub)
    d = MishkalDiacritizer()
    out = d.diacritize("العلم نور")
    assert out == add_fatha("العلم نور")
    assert d.diacritize(" نور\n") == " " + add_fatha("نور") + "\n"
    assert not vocalize_scene(scene("العلم نور"), d).flags


def add_fatha(text: str) -> str:
    return re.sub(r"([\u0621-\u064a])", "\\1\u064e", text)


def test_first_sentence_of_fixture_accepted() -> None:
    raw = (FIXTURES / "input" / "clean_ar.txt").read_text(encoding="utf-8")
    sentence = raw.split("\n\n")[1].split("؛")[0]
    d = LlmDiacritizer(FakeLLM([Vocalized(text=add_fatha(sentence))]))
    s = vocalize_scene(scene(sentence), d)
    assert s.narration.tts_text == add_fatha(sentence)
    assert not s.flags


def test_case_ending_added_accepted_dropped_word_rejected() -> None:
    src = "العلم نور في القلب"
    ok = LlmDiacritizer(FakeLLM([Vocalized(text="العِلْمُ نُورٌ فِي القَلْبِ")]))
    assert not vocalize_scene(scene(src), ok).flags
    bad = LlmDiacritizer(FakeLLM([Vocalized(text="العِلْمُ نُورٌ القَلْبِ")]))
    assert "tashkeel_rejected" in vocalize_scene(scene(src), bad).flags


def test_sources_are_ascii() -> None:
    for module in (engines_module, prompts_module):
        assert Path(module.__file__ or "").read_bytes().isascii()


def test_mishkal_is_free_and_diacritizer_defaults_to_llm() -> None:
    from narratoflow.channels.config import ChannelConfig
    from narratoflow.channels.keys import FREE_PROVIDERS

    assert "mishkal" in FREE_PROVIDERS
    assert ChannelConfig.model_fields["diacritizer"].default == "llm"
