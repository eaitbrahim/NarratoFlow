"""Dialect flagging with CAMeL Tools DID (extra: arabic); flags, never rejects.

API verified 2026-11-12 against the camel-tools 1.6.0 wheel pinned in uv.lock
(sha256 17e51cf9..., read, never installed):
- ``camel_tools.dialectid.DialectIdentifier`` is ``DIDModel26``;
- ``DIDModel26.pretrained()`` loads ``did_pretrained_{major}{minor}.dill`` from the
  catalogue path ``dialectid/model26`` and raises ``PretrainedModelError`` when the
  data is absent (``camel_data -i dialectid-model26``, 274 MB; never downloaded here);
- ``predict(sentences, output='label') -> list[DIDPred(top: str, scores: dict[str, float])]``;
- the MSA label is ``"MSA"``;
- importing ``camel_tools.data`` loads the catalogue at import time, so the import
  is lazy.

A failure to import or load becomes ``DialectUnavailableError``; a failure in
``predict`` becomes ``ProviderError``. Both name only the exception type.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from narratoflow.errors import DialectUnavailableError, ProviderError
from narratoflow.text.quality import normalize_input

MSA_LABEL = "MSA"
INSTALL_HINT = (
    "install with: uv sync --extra arabic (plus your other extras) "
    "&& uv run camel_data -i dialectid-model26"
)
_SENTENCE_END = re.compile("(?<=[.!?\\u061f\\u06d4])(?=\\s|$)")
_VERSE = re.compile("\\ufd3e[^\\ufd3f]*\\ufd3f")
_ARABIC_LETTER = re.compile("[\u0621-\u064a]")


class DialectModel(Protocol):
    def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]:
        """One (top_label, top_probability) per sentence, in order."""
        ...


class SentenceDialect(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: int
    text: str
    label: str
    confidence: float
    flagged: bool


class DialectReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    model: str
    sentences: tuple[SentenceDialect, ...]

    @property
    def flagged(self) -> tuple[SentenceDialect, ...]:
        return tuple(s for s in self.sentences if s.flagged)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """(1-based line of the sentence start, sentence) split on newlines and . ! ? U+061F U+06D4
    when followed by whitespace or the end of the line (so 3.5 stays whole)."""
    out: list[tuple[int, str]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for piece in _SENTENCE_END.split(line):
            piece = piece.strip()
            if piece:
                out.append((number, piece))
    return out


def _strip_verses(text: str) -> str:
    """Remove Qur'anic verses (ornate brackets included), keeping their newlines."""
    return _VERSE.sub(lambda m: "\n" * m.group().count("\n"), text)


def check_dialect(
    text: str, model: DialectModel, *, threshold: float, model_name: str
) -> DialectReport:
    """Flag Arabic sentences whose top label is not MSA with probability >= threshold."""
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must satisfy 0 < threshold <= 1, got {threshold}")
    found = [
        (line, sentence)
        for line, sentence in split_sentences(_strip_verses(normalize_input(text)))
        if _ARABIC_LETTER.search(sentence)
    ]
    predictions = model.predict([sentence for _, sentence in found]) if found else []
    if len(predictions) != len(found):
        raise ProviderError("dialect model returned the wrong number of predictions")
    sentences = tuple(
        SentenceDialect(
            line=line,
            text=sentence,
            label=label,
            confidence=confidence,
            flagged=label != MSA_LABEL and confidence >= threshold,
        )
        for (line, sentence), (label, confidence) in zip(found, predictions, strict=True)
    )
    return DialectReport(threshold=threshold, model=model_name, sentences=sentences)


class _CamelAdapter:
    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]:
        try:
            preds = self._model.predict(list(sentences), output="label")
            return [(str(p.top), float(p.scores[p.top])) for p in preds]
        except Exception as err:
            raise ProviderError(f"dialect model failed ({type(err).__name__})") from err


def load_camel_model() -> DialectModel:
    """The pretrained CAMeL dialect identifier, or DialectUnavailableError."""
    try:
        module = importlib.import_module("camel_tools.dialectid")
        model = module.DialectIdentifier.pretrained()
    except Exception as err:
        raise DialectUnavailableError(
            f"camel_tools dialect model failed to load ({type(err).__name__}); {INSTALL_HINT}"
        ) from err
    return _CamelAdapter(model)
