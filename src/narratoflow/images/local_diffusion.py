"""Local diffusion image provider (optional 'diffusion' extra; free of charge)."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, UnidentifiedImageError

from narratoflow.errors import ChannelConfigError, ProviderError
from narratoflow.images import GeneratedImage, ImageRequest
from narratoflow.images.prompt import NEGATIVE_TERMS

IMG2IMG_STRENGTH = 0.6
NEGATIVE_PROMPT = ", ".join(NEGATIVE_TERMS) + ", people, faces, human figures, text, letters, logos"
_EXTRA_HINT = "install the 'diffusion' extra: uv sync --extra diffusion"


class LocalDiffusionProvider:
    name: str = "local-diffusion"

    def __init__(self, model: str, device: str = "auto") -> None:
        try:
            import diffusers
            import torch
        except ImportError as err:
            raise ChannelConfigError(_EXTRA_HINT) from err
        self.model = model
        self._torch: Any = torch
        self._diffusers: Any = diffusers
        self._device = self._pick_device(device)
        self._t2i: Any = diffusers.AutoPipelineForText2Image.from_pretrained(model).to(self._device)
        self._i2i: Any = None

    def _pick_device(self, device: str) -> str:
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(self, req: ImageRequest) -> GeneratedImage:
        kwargs: dict[str, Any] = {"prompt": req.prompt, "negative_prompt": NEGATIVE_PROMPT}
        kwargs["generator"] = (
            None
            if req.seed is None
            else self._torch.Generator(device=self._device).manual_seed(req.seed)
        )
        if req.reference_png is None:
            pipe = self._t2i
        else:
            try:
                ref = Image.open(io.BytesIO(req.reference_png)).convert("RGB")
            except (UnidentifiedImageError, OSError) as err:
                raise ProviderError("reference image is not a valid image") from err
            if self._i2i is None:
                self._i2i = self._diffusers.AutoPipelineForImage2Image.from_pretrained(
                    self.model
                ).to(self._device)
            pipe = self._i2i
            kwargs["image"] = ref
            kwargs["strength"] = IMG2IMG_STRENGTH
        image = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return GeneratedImage(buf.getvalue(), self.name, self.model, None)
