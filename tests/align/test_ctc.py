"""CtcAligner against fake `ctc_forced_aligner`, `onnxruntime` and `unidecode` modules.

The fakes copy the call signatures of the versions pinned in uv.lock, read from
their published sources (never installed, never run):

- ctc-forced-aligner 1.0.2 (PyPI sdist, deskpai/ctc_forced_aligner,
  ``ctc_forced_aligner/__init__.py``): ``MODEL_URL``, ``Tokenizer()``,
  ``ensure_onnx_model(model_path, url)``,
  ``generate_emissions(session, audio_waveform, window_length=30, context_length=2,
  batch_size=4) -> (emissions, stride_ms)``,
  ``preprocess_text(text, romanize, language, split_size="word",
  star_frequency="segment") -> (tokens_starred, text_starred)``,
  ``get_alignments(emissions, tokens, tokenizer) -> (segments, scores, blank)``,
  ``get_spans(tokens, segments, blank)`` and
  ``postprocess_results(text_starred, spans, stride, scores, merge_threshold=0.0)``
  returning ``[{"start", "end", "text", "score"}]`` in seconds;
- onnxruntime 1.31.1 (wheel, ``capi/onnxruntime_inference_collection.py``):
  ``InferenceSession(path_or_bytes, sess_options=None, providers=None,
  provider_options=None, **kwargs)``;
- unidecode: imported by ``preprocess_text(romanize=True)``; never called here.

No model is downloaded and nothing touches the network.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from narratoflow.align import Aligner, ctc
from narratoflow.align.ctc import EXTRA_HINT, CtcAligner
from narratoflow.audio.words import TimedSpan, timings_from_spans, tokens
from narratoflow.errors import AlignerUnavailableError, ProviderError
from narratoflow.models import Language
from narratoflow.text.tashkeel.verify import strip_diacritics

# "bismi allahi" vocalized: bi-s-mi a-l-l-a-hi (with harakat and shadda).
AR_VOCALIZED = "\u0628\u0650\u0633\u0652\u0645\u0650 \u0627\u0644\u0644\u0651\u064e\u0647\u0650"
AR_BARE = "\u0628\u0633\u0645 \u0627\u0644\u0644\u0647"
SAMPLES = np.zeros(48_000, dtype=np.float32)
MODULES = ("ctc_forced_aligner", "onnxruntime", "unidecode")


class Calls:
    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []

    def add(self, name: str, **kwargs: Any) -> None:
        self.log.append((name, kwargs))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.log if n == name]


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    results: list[dict[str, Any]] | None = None,
    *,
    fail_in: str | None = None,
) -> Calls:
    calls = Calls()

    def maybe_fail(where: str) -> None:
        if fail_in == where:
            raise AssertionError("SECRET library detail")

    class Tokenizer:
        def __init__(self) -> None:
            calls.add("Tokenizer")

    def ensure_onnx_model(model_path: str, url: str) -> None:
        calls.add("ensure_onnx_model", model_path=model_path, url=url)
        maybe_fail("ensure_onnx_model")

    def generate_emissions(
        session: Any,
        audio_waveform: Any,
        window_length: int = 30,
        context_length: int = 2,
        batch_size: int = 4,
    ) -> tuple[Any, int]:
        calls.add("generate_emissions", session=session, audio_waveform=audio_waveform)
        maybe_fail("generate_emissions")
        return "EMISSIONS", 20

    def preprocess_text(
        text: str,
        romanize: bool,
        language: str,
        split_size: str = "word",
        star_frequency: str = "segment",
    ) -> tuple[list[str], list[str]]:
        calls.add("preprocess_text", text=text, romanize=romanize, language=language)
        words = text.split()
        starred: list[str] = []
        for w in words:
            starred.extend(["<star>", w])
        return starred, list(starred)

    def get_alignments(emissions: Any, tokens: list[str], tokenizer: Any) -> tuple[Any, Any, str]:
        calls.add("get_alignments", emissions=emissions, tokens=tokens, tokenizer=tokenizer)
        maybe_fail("get_alignments")
        return "SEGMENTS", "SCORES", "<blank>"

    def get_spans(tokens: list[str], segments: Any, blank: str) -> Any:
        calls.add("get_spans", tokens=tokens, segments=segments, blank=blank)
        return "SPANS"

    def postprocess_results(
        text_starred: list[str],
        spans: Any,
        stride: float,
        scores: Any,
        merge_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        calls.add(
            "postprocess_results",
            text_starred=text_starred,
            spans=spans,
            stride=stride,
            scores=scores,
        )
        if results is not None:
            return results
        words = [t for t in text_starred if t != "<star>"]
        return [
            {"start": 0.5 * i, "end": 0.5 * i + 0.4, "text": w, "score": -1.0}
            for i, w in enumerate(words)
        ]

    cfa = _module("ctc_forced_aligner")
    cfa.MODEL_URL = "https://example.invalid/model.onnx"  # type: ignore[attr-defined]
    for fn in (
        Tokenizer,
        ensure_onnx_model,
        generate_emissions,
        preprocess_text,
        get_alignments,
        get_spans,
        postprocess_results,
    ):
        setattr(cfa, fn.__name__, fn)

    class InferenceSession:
        def __init__(
            self,
            path_or_bytes: str,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
            **kwargs: Any,
        ) -> None:
            calls.add("InferenceSession", path_or_bytes=path_or_bytes, providers=providers)

    ort = _module("onnxruntime")
    ort.InferenceSession = InferenceSession  # type: ignore[attr-defined]
    uni = _module("unidecode")
    uni.unidecode = lambda s: s  # type: ignore[attr-defined]
    for name, module in (("ctc_forced_aligner", cfa), ("onnxruntime", ort), ("unidecode", uni)):
        monkeypatch.setitem(sys.modules, name, module)

    def identity(samples: Any, rate: int) -> Any:
        calls.add("resample", rate=rate)
        return samples

    monkeypatch.setattr(ctc, "resample", identity)
    return calls


@pytest.fixture(autouse=True)
def _fresh_model_cache() -> Iterator[None]:
    ctc._load_model.cache_clear()
    yield
    ctc._load_model.cache_clear()


def _hide(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, name, None)


def test_available_false_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(monkeypatch)
    assert CtcAligner.available() is True
    for name in MODULES:
        with monkeypatch.context() as m:
            _hide(m, name)
            assert CtcAligner.available() is False


@pytest.mark.parametrize("name", MODULES)
def test_missing_extra_raises_aligner_unavailable(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    install_fakes(monkeypatch)
    _hide(monkeypatch, name)
    aligner = CtcAligner()  # the constructor imports nothing
    with pytest.raises(AlignerUnavailableError) as exc:
        aligner.align(SAMPLES, 48_000, "hello", Language.ENGLISH)
    assert str(exc.value) == "install the 'align' extra: uv sync --extra align"
    assert str(exc.value) == EXTRA_HINT
    assert not isinstance(exc.value, ImportError)


def test_constructor_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODULES:
        _hide(monkeypatch, name)
    assert isinstance(CtcAligner(), Aligner)
    with pytest.raises(ValueError, match="device"):
        CtcAligner(device="tpu")


def test_text_sent_undiacritized_with_iso_language(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fakes(monkeypatch)
    CtcAligner().align(SAMPLES, 48_000, AR_VOCALIZED, Language.ARABIC)
    CtcAligner().align(SAMPLES, 48_000, "Hello, world.", Language.ENGLISH)
    sent = calls.named("preprocess_text")
    assert sent[0] == {"text": AR_BARE, "romanize": True, "language": "ara"}
    assert sent[0]["text"] == strip_diacritics(AR_VOCALIZED)
    assert sent[1] == {"text": "Hello, world.", "romanize": True, "language": "eng"}
    assert calls.named("resample") == [{"rate": 16_000}, {"rate": 16_000}]
    flow = [n for n, _ in calls.log if n not in {"InferenceSession", "Tokenizer"}]
    assert flow[:7] == [
        "resample",
        "ensure_onnx_model",
        "generate_emissions",
        "preprocess_text",
        "get_alignments",
        "get_spans",
        "postprocess_results",
    ]
    tokens_starred = ["<star>", "Hello,", "<star>", "world."]
    assert calls.named("get_spans")[1] == {
        "tokens": tokens_starred,
        "segments": "SEGMENTS",
        "blank": "<blank>",
    }
    assert calls.named("postprocess_results")[1] == {
        "text_starred": tokens_starred,
        "spans": "SPANS",
        "stride": 20,
        "scores": "SCORES",
    }


def test_seconds_to_ms_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(
        monkeypatch,
        [
            {"start": 0.0, "end": 0.4204, "text": AR_BARE.split()[0], "score": -0.5},
            {"start": 0.5, "end": 1.0006, "text": AR_BARE.split()[1], "score": -0.2},
        ],
    )
    spans = CtcAligner().align(SAMPLES, 48_000, AR_VOCALIZED, Language.ARABIC)
    assert spans == [
        TimedSpan(AR_BARE.split()[0], 0, 420),
        TimedSpan(AR_BARE.split()[1], 500, 1001),
    ]
    timed = timings_from_spans(tokens(AR_VOCALIZED), spans)
    assert timed is not None
    assert [w.text for w in timed] == AR_VOCALIZED.split()
    assert [(w.start_ms, w.end_ms) for w in timed] == [(0, 420), (500, 1001)]


@pytest.mark.parametrize(
    "bad",
    [
        {"start": -0.1, "end": 0.2, "text": "a"},
        {"start": 0.3, "end": 0.2, "text": "a"},
        {"start": float("nan"), "end": 0.2, "text": "a"},
        {"start": True, "end": 0.2, "text": "a"},
        {"start": 0.0, "end": 0.2, "text": 3},
        {"start": 0.0, "text": "a"},
        "not a dict",
    ],
)
def test_malformed_result_is_provider_error(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    install_fakes(monkeypatch, [bad])
    with pytest.raises(ProviderError, match="malformed"):
        CtcAligner().align(SAMPLES, 48_000, "a", Language.ENGLISH)


def test_non_list_result_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(monkeypatch)
    cfa = sys.modules["ctc_forced_aligner"]
    monkeypatch.setattr(cfa, "postprocess_results", lambda *_a, **_k: {"start": 0})
    with pytest.raises(ProviderError, match="malformed result"):
        CtcAligner().align(SAMPLES, 48_000, "a", Language.ENGLISH)


def test_model_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fakes(monkeypatch)
    first, second = CtcAligner(), CtcAligner()
    first.align(SAMPLES, 48_000, "one", Language.ENGLISH)
    second.align(SAMPLES, 48_000, "two", Language.ENGLISH)
    first.align(SAMPLES, 48_000, "three", Language.ENGLISH)
    sessions = calls.named("InferenceSession")
    assert len(sessions) == 1
    assert sessions[0]["providers"] == ["CPUExecutionProvider"]
    assert len(calls.named("ensure_onnx_model")) == 1
    assert len(calls.named("Tokenizer")) == 1
    assert calls.named("ensure_onnx_model")[0]["url"] == "https://example.invalid/model.onnx"
    assert sessions[0]["path_or_bytes"] == calls.named("ensure_onnx_model")[0]["model_path"]
    assert len(calls.named("generate_emissions")) == 3


def test_cuda_device_and_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls = install_fakes(monkeypatch)
    path = tmp_path / "m.onnx"
    CtcAligner(device="cuda", model_path=path).align(SAMPLES, 48_000, "x", Language.ENGLISH)
    (session,) = calls.named("InferenceSession")
    assert session == {
        "path_or_bytes": str(path),
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }


@pytest.mark.parametrize("where", ["ensure_onnx_model", "generate_emissions", "get_alignments"])
def test_library_error_maps_to_provider_error(monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    calls = install_fakes(monkeypatch, fail_in=where)
    with pytest.raises(ProviderError) as exc:
        CtcAligner().align(SAMPLES, 48_000, "hello", Language.ENGLISH)
    assert str(exc.value) == "alignment failed: AssertionError"
    assert "SECRET" not in str(exc.value)
    assert not isinstance(exc.value, AlignerUnavailableError)
    with pytest.raises(ProviderError):
        CtcAligner().align(SAMPLES, 48_000, "hello", Language.ENGLISH)
    # a failed model load is not cached: the next call tries again
    assert len(calls.named("ensure_onnx_model")) == (2 if where == "ensure_onnx_model" else 1)


def test_empty_text_aligns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fakes(monkeypatch)
    assert CtcAligner().align(SAMPLES, 48_000, " \u064e ", Language.ARABIC) == []
    assert calls.log == []


def test_wrong_input_rate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(monkeypatch)
    with pytest.raises(ValueError, match="48000"):
        CtcAligner().align(SAMPLES, 16_000, "x", Language.ENGLISH)
