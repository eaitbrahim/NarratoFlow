"""Plugin contracts. Concrete engines are selected from config; tests use fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from narratoflow.models import AspectRatio, Project, Scene, Visual, Voice, WordTiming


@runtime_checkable
class Diacritizer(Protocol):
    name: str

    def diacritize(self, text: str) -> str: ...


@runtime_checkable
class TTSEngine(Protocol):
    name: str

    def synthesize(self, text: str, voice: Voice, out_path: Path) -> list[WordTiming]:
        """Write audio to `out_path`; return word timings (may be empty if unsupported)."""
        ...


@runtime_checkable
class Aligner(Protocol):
    def align(self, audio_path: Path, text: str) -> list[WordTiming]: ...


@runtime_checkable
class VisualProvider(Protocol):
    def produce(self, scene: Scene, out_dir: Path) -> Visual: ...


@runtime_checkable
class Renderer(Protocol):
    def render(self, project: Project, aspect: AspectRatio, out_path: Path) -> Path: ...
