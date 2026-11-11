"""Plugin contracts. Concrete engines are selected from config; tests use fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from narratoflow.align import Aligner
from narratoflow.models import AspectRatio, Project, Scene, Visual
from narratoflow.tts import SpeechProvider


@runtime_checkable
class Diacritizer(Protocol):
    name: str

    def diacritize(self, text: str) -> str: ...


@runtime_checkable
class VisualProvider(Protocol):
    def produce(self, scene: Scene, out_dir: Path) -> Visual: ...


@runtime_checkable
class Renderer(Protocol):
    def render(self, project: Project, aspect: AspectRatio, out_path: Path) -> Path: ...


__all__ = ["Aligner", "Diacritizer", "Renderer", "SpeechProvider", "VisualProvider"]
