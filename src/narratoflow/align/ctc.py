"""CTC forced aligner (extra: align); aligns on undiacritized text.

API verified 2026-11-11 against the sources pinned in uv.lock (read, never installed):
- ctc-forced-aligner 1.0.2, the PyPI sdist (deskpai/ctc_forced_aligner, ONNX based,
  no torch): ``ensure_onnx_model(model_path, url)`` downloads ``MODEL_URL`` once;
  ``Tokenizer()``; ``generate_emissions(session, audio_waveform, ...)`` takes a 1-D
  16 kHz float32 array and returns ``(emissions, stride_ms)``;
  ``preprocess_text(text, romanize, language, ...)`` returns
  ``(tokens_starred, text_starred)`` and imports ``unidecode`` when romanizing;
  ``get_alignments(emissions, tokens, tokenizer)`` returns
  ``(segments, scores, blank)``; ``get_spans(tokens, segments, blank)``;
  ``postprocess_results(text_starred, spans, stride, scores)`` returns
  ``[{"start", "end", "text", "score"}]`` in seconds. This release has no
  ``load_alignment_model`` and does not use torch, so the model is an
  ``onnxruntime.InferenceSession`` (onnxruntime 1.31.1:
  ``InferenceSession(path_or_bytes, sess_options=None, providers=None, ...)``).

The aligner sees ``strip_diacritics(text)``. It returns the words of that text, and
``timings_from_spans`` maps them back onto the author's vocalized tokens unchanged.

A missing extra (``ctc_forced_aligner``, ``onnxruntime`` or ``unidecode``) raises
``AlignerUnavailableError``: it is the one type callers catch for "aligner not
installed". Every other library failure is a ``ProviderError`` naming only the
exception type.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

from narratoflow.audio.io import SAMPLE_RATE, Samples, resample
from narratoflow.audio.words import TimedSpan
from narratoflow.errors import AlignerUnavailableError, ProviderError
from narratoflow.models import Language
from narratoflow.text.tashkeel.verify import strip_diacritics

EXTRA_HINT = "install the 'align' extra: uv sync --extra align"
ALIGN_RATE = 16_000
ISO_LANGUAGE: dict[Language, str] = {Language.ARABIC: "ara", Language.ENGLISH: "eng"}
REQUIRED_MODULES = ("ctc_forced_aligner", "onnxruntime", "unidecode")
PROVIDERS: dict[str, tuple[str, ...]] = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
}


def default_model_path() -> Path:
    """Where ctc-forced-aligner 1.0.2 keeps its ONNX model by default."""
    return Path.home() / "ctc_forced_aligner" / "model.onnx"


def _modules() -> tuple[ModuleType, ModuleType]:
    try:
        loaded = [importlib.import_module(name) for name in REQUIRED_MODULES]
    except ImportError as err:
        raise AlignerUnavailableError(EXTRA_HINT) from err
    return loaded[0], loaded[1]


@functools.cache
def _load_model(model_path: str, device: str) -> tuple[Any, Any]:
    """The ONNX session and tokenizer, once per process (failures are not cached)."""
    cfa, ort = _modules()
    cfa.ensure_onnx_model(model_path, cfa.MODEL_URL)
    session = ort.InferenceSession(model_path, providers=list(PROVIDERS[device]))
    return session, cfa.Tokenizer()


def _seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProviderError("aligner returned a malformed word")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ProviderError("aligner returned a malformed word")
    return seconds


def _span(item: object) -> TimedSpan:
    if not isinstance(item, dict):
        raise ProviderError("aligner returned a malformed word")
    text = item.get("text")
    if not isinstance(text, str):
        raise ProviderError("aligner returned a malformed word")
    start = round(_seconds(item.get("start")) * 1000)
    end = round(_seconds(item.get("end")) * 1000)
    if end < start:
        raise ProviderError("aligner returned a malformed word")
    return TimedSpan(text=text, start_ms=start, end_ms=end)


class CtcAligner:
    name: str = "ctc"

    def __init__(self, device: str = "cpu", model_path: Path | None = None) -> None:
        if device not in PROVIDERS:
            raise ValueError(f"unknown aligner device {device!r}; use one of {sorted(PROVIDERS)}")
        self.device = device
        self.model_path = model_path

    @staticmethod
    def available() -> bool:
        return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)

    def align(
        self, samples: Samples, sample_rate: int, text: str, language: Language
    ) -> list[TimedSpan]:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"aligner input must be {SAMPLE_RATE} Hz, got {sample_rate}")
        cfa, _ = _modules()
        bare = strip_diacritics(text)
        if not bare.split():
            return []
        wave = resample(samples, ALIGN_RATE)
        path = str(self.model_path or default_model_path())
        try:
            session, tokenizer = _load_model(path, self.device)
            emissions, stride = cfa.generate_emissions(session, wave)
            tokens_starred, text_starred = cfa.preprocess_text(
                bare, romanize=True, language=ISO_LANGUAGE[language]
            )
            segments, scores, blank = cfa.get_alignments(emissions, tokens_starred, tokenizer)
            spans = cfa.get_spans(tokens_starred, segments, blank)
            results = cfa.postprocess_results(text_starred, spans, stride, scores)
        except Exception as err:
            raise ProviderError(f"alignment failed: {type(err).__name__}") from err
        if not isinstance(results, list):
            raise ProviderError("aligner returned a malformed result")
        return [_span(item) for item in results]
