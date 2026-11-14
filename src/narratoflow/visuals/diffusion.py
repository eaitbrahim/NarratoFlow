"""Text-to-sketch via a diffusion model plus an optional line-art LoRA (extra: diffusion).

`LineArtDiffusionProvider` is an `images.ImageProvider` that ignores the reference
image: the look comes from the pinned model, LoRA and seed (`DiffusionSpec`), not from
ControlNet. The pipeline is built on the first `generate` call, so a run that never
draws a new sketch never imports torch or diffusers.
"""

from __future__ import annotations

import hashlib
import inspect
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from PIL import Image

from narratoflow.channels.config import StyleConfig
from narratoflow.errors import ChannelConfigError, ExtraMissingError, ProviderError
from narratoflow.images import GeneratedImage, ImageRequest
from narratoflow.images.local_diffusion import NEGATIVE_PROMPT, pick_device

EXTRA_HINT = "install the 'diffusion' extra: uv sync --extra diffusion"
PROVIDER_NAME = "lineart-diffusion"


@dataclass(frozen=True)
class DiffusionSpec:
    """Everything that decides what the pipeline draws (the channel's pins)."""

    model: str
    revision: str | None = None
    lora: str | None = None
    lora_weight_name: str | None = None
    lora_revision: str | None = None
    lora_scale: float = 1.0
    seed: int = 0
    steps: int = 4
    guidance_scale: float = 0.0
    size: int = 1024
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    @classmethod
    def from_style(cls, style: StyleConfig) -> DiffusionSpec:
        pins = style.diffusion
        if pins is None:
            raise ChannelConfigError("image_provider lineart-diffusion needs a [style.diffusion]")
        return cls(
            model=style.image_model,
            revision=pins.model_revision,
            lora=pins.lora,
            lora_weight_name=pins.lora_weight_name,
            lora_revision=pins.lora_revision,
            lora_scale=pins.lora_scale,
            seed=pins.seed,
            steps=pins.steps,
            guidance_scale=pins.guidance_scale,
            size=pins.size,
            device=pins.device,
        )


class TextToImagePipeline(Protocol):
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
    ) -> Image.Image: ...


PipelineFactory = Callable[[DiffusionSpec], TextToImagePipeline]


def effective_seed(base: int, request_seed: int | None) -> int:
    """The 32-bit seed for one drawing: the channel pin mixed with the request's seed.

    `request_seed` already encodes video, scene and attempt (`marker_seed`), so a
    regenerate changes the drawing while a rerun of the same request does not.
    """
    digest = hashlib.sha256(f"{base}:{request_seed or 0}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class _DiffusersPipeline:
    """Adapts a diffusers pipeline to `TextToImagePipeline` with a CPU generator."""

    def __init__(self, pipe: Any, torch: Any) -> None:
        self._pipe = pipe
        self._torch = torch
        self._takes_negative = "negative_prompt" in inspect.signature(pipe.__call__).parameters

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
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            # a CPU generator reproduces across devices (diffusers reproducibility guide)
            "generator": self._torch.Generator("cpu").manual_seed(seed),
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }
        if self._takes_negative:
            kwargs["negative_prompt"] = negative_prompt
        image: Image.Image = self._pipe(**kwargs).images[0]
        return image


def default_pipeline_factory(spec: DiffusionSpec) -> TextToImagePipeline:
    """Load the pinned model (and LoRA) with diffusers; needs the `diffusion` extra."""
    try:
        import diffusers
        import torch
    except ImportError as err:
        raise ExtraMissingError(EXTRA_HINT) from err
    device = pick_device(torch, spec.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            spec.model, revision=spec.revision, torch_dtype=dtype
        ).to(device)
        if spec.lora:
            pipe.load_lora_weights(
                spec.lora, weight_name=spec.lora_weight_name, revision=spec.lora_revision
            )
            pipe.fuse_lora(lora_scale=spec.lora_scale)
    except Exception as err:
        raise ChannelConfigError(
            f"cannot load diffusion model {spec.model!r} ({type(err).__name__})"
        ) from err
    return _DiffusersPipeline(pipe, torch)


class LineArtDiffusionProvider:
    name: str = PROVIDER_NAME

    def __init__(self, spec: DiffusionSpec, factory: PipelineFactory = default_pipeline_factory):
        self.spec = spec
        self.model = spec.model
        self._factory = factory
        self._pipeline: TextToImagePipeline | None = None

    def generate(self, req: ImageRequest) -> GeneratedImage:
        if self._pipeline is None:
            self._pipeline = self._factory(self.spec)
        try:
            image = self._pipeline(
                prompt=req.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                steps=self.spec.steps,
                guidance_scale=self.spec.guidance_scale,
                width=self.spec.size,
                height=self.spec.size,
                seed=effective_seed(self.spec.seed, req.seed),
            )
            buf = io.BytesIO()
            image.save(buf, format="PNG", pnginfo=None)  # no metadata: equal pixels, equal bytes
        except Exception as err:
            raise ProviderError(f"lineart diffusion failed ({type(err).__name__})") from err
        return GeneratedImage(buf.getvalue(), self.name, self.model, None)
