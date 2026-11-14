"""Fakes for the diffusion provider: a pure-function pipeline and fake torch/diffusers."""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
from PIL import Image

from narratoflow.visuals.diffusion import DiffusionSpec


class FakePipeline:
    """An image that is a pure function of the prompt and the seed; records every call."""

    def __init__(self, spec: DiffusionSpec | None = None) -> None:
        self.spec = spec
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int,
    ) -> Image.Image:
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
            }
        )
        digest = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
        image = Image.new("RGB", (16, 16), "white")
        image.putpixel((digest[0] % 16, digest[1] % 16), (0, 0, 0))
        return image


@dataclass
class Recorder:
    """What the fake torch/diffusers saw, in order."""

    events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def names(self) -> list[str]:
        return [name for name, _, _ in self.events]

    def kwargs_of(self, name: str) -> dict[str, Any]:
        return next(kw for n, _, kw in self.events if n == name)

    def args_of(self, name: str) -> tuple[Any, ...]:
        return next(a for n, a, _ in self.events if n == name)


class _Generator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def fake_torch_diffusers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    negative_prompt: bool = True,
    cuda: bool = False,
    mps: bool = False,
    fail_load: bool = False,
) -> Recorder:
    """Install fake `torch` and `diffusers` modules; the pipeline records its calls."""
    rec = Recorder()

    class Output:
        def __init__(self) -> None:
            self.images = [Image.new("RGB", (8, 8), "white")]

    class Base:
        def to(self, device: str) -> Base:
            rec.events.append(("to", (device,), {}))
            return self

        def load_lora_weights(self, name: str, **kwargs: Any) -> None:
            rec.events.append(("load_lora_weights", (name,), kwargs))

        def fuse_lora(self, **kwargs: Any) -> None:
            rec.events.append(("fuse_lora", (), kwargs))

    class WithNegative(Base):
        def __call__(
            self,
            prompt: str,
            negative_prompt: str,
            generator: Any,
            num_inference_steps: int,
            guidance_scale: float,
            width: int,
            height: int,
        ) -> Output:
            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            rec.events.append(("call", (), kwargs))
            return Output()

    class WithoutNegative(Base):
        def __call__(
            self,
            prompt: str,
            generator: Any,
            num_inference_steps: int,
            guidance_scale: float,
            width: int,
            height: int,
        ) -> Output:
            kwargs = dict(
                prompt=prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            rec.events.append(("call", (), kwargs))
            return Output()

    class AutoPipelineForText2Image:
        @staticmethod
        def from_pretrained(model: str, **kwargs: Any) -> Base:
            rec.events.append(("from_pretrained", (model,), kwargs))
            if fail_load:
                raise OSError("gated repo")
            return WithNegative() if negative_prompt else WithoutNegative()

    torch = types.ModuleType("torch")
    torch.__dict__.update(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        bfloat16="bfloat16",
        float32="float32",
        Generator=_Generator,
    )
    diffusers = types.ModuleType("diffusers")
    diffusers.__dict__.update(AutoPipelineForText2Image=AutoPipelineForText2Image)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return rec
