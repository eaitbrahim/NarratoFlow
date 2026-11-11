"""Plugin contracts. Concrete engines are selected from config; tests use fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from narratoflow.align import Aligner
from narratoflow.models import Scene, Visual
from narratoflow.tts import SpeechProvider


@runtime_checkable
class Diacritizer(Protocol):
    name: str

    def diacritize(self, text: str) -> str: ...


@runtime_checkable
class VisualProvider(Protocol):
    def produce(self, scene: Scene, out_dir: Path) -> Visual: ...


__all__ = ["Aligner", "Diacritizer", "SpeechProvider", "VisualProvider"]
