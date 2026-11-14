from __future__ import annotations

import io
import struct
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fakes.diffusion import FakePipeline, fake_torch_diffusers
from PIL import Image

from narratoflow.channels.config import load_channel
from narratoflow.costs.prices import default_prices
from narratoflow.costs.tracker import SpendTracker
from narratoflow.errors import ChannelConfigError, ExtraMissingError, ProviderError
from narratoflow.images import ImageRequest
from narratoflow.images.factory import build_image_provider, make_image_provider
from narratoflow.settings import get_settings
from narratoflow.visuals.diffusion import (
    DiffusionSpec,
    LineArtDiffusionProvider,
    default_pipeline_factory,
    effective_seed,
)

MODEL = "black-forest-labs/FLUX.1-schnell"


def _spec(**changes: Any) -> DiffusionSpec:
    return replace(DiffusionSpec(model=MODEL, revision="rev1"), **changes)


def _provider(spec: DiffusionSpec | None = None) -> tuple[LineArtDiffusionProvider, FakePipeline]:
    pipe = FakePipeline()
    return LineArtDiffusionProvider(spec or _spec(), factory=lambda _spec: pipe), pipe


def test_same_request_same_bytes() -> None:
    first, _ = _provider()
    second, _ = _provider()
    req = ImageRequest("a balance", None, seed=7)
    a, b = first.generate(req), second.generate(req)
    assert a.png == b.png
    assert a.png.startswith(b"\x89PNG")
    assert (a.provider, a.model) == ("lineart-diffusion", MODEL)


def test_seed_varies_by_scene_and_base() -> None:
    provider, pipe = _provider()
    other_base, other_pipe = _provider(_spec(seed=1))
    for p in (provider, other_base):
        p.generate(ImageRequest("a balance", None, seed=7))
    provider.generate(ImageRequest("a balance", None, seed=8))
    seeds = [c["seed"] for c in pipe.calls]
    assert seeds == [effective_seed(0, 7), effective_seed(0, 8)]
    assert other_pipe.calls[0]["seed"] == effective_seed(1, 7)
    assert len({seeds[0], seeds[1], other_pipe.calls[0]["seed"]}) == 3


def test_effective_seed_pinned() -> None:
    # frozen formula: changing it silently re-rolls every channel's drawings
    assert effective_seed(0, 7) == 737120251
    assert effective_seed(0, None) == effective_seed(0, 0)
    assert 0 <= effective_seed(2**32 - 1, 2**31) < 2**32


def test_reference_png_ignored() -> None:
    provider, pipe = _provider()
    with_ref = provider.generate(ImageRequest("a balance", b"not even a png", seed=3))
    without = provider.generate(ImageRequest("a balance", None, seed=3))
    assert with_ref.png == without.png
    assert pipe.calls[0] == pipe.calls[1]


def test_sampler_settings_reach_the_pipeline() -> None:
    provider, pipe = _provider(_spec(steps=6, guidance_scale=1.5))
    provider.generate(ImageRequest("a balance", None, seed=1))
    [call] = pipe.calls
    assert (call["steps"], call["guidance_scale"]) == (6, 1.5)
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["prompt"] == "a balance"
    assert "shading" in call["negative_prompt"]


def test_pipeline_built_once_lazily() -> None:
    built: list[DiffusionSpec] = []

    def factory(spec: DiffusionSpec) -> FakePipeline:
        built.append(spec)
        return FakePipeline(spec)

    provider = LineArtDiffusionProvider(_spec(), factory=factory)
    assert built == []
    provider.generate(ImageRequest("a", None, seed=1))
    provider.generate(ImageRequest("b", None, seed=2))
    assert built == [_spec()]


def test_default_factory_uses_cpu_generator_and_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = fake_torch_diffusers(monkeypatch)
    spec = _spec(
        lora="someone/lineart",
        lora_weight_name="lineart.safetensors",
        lora_revision="lrev",
        lora_scale=0.8,
    )
    provider = LineArtDiffusionProvider(spec)
    provider.generate(ImageRequest("a balance", None, seed=7))
    assert rec.names() == ["from_pretrained", "to", "load_lora_weights", "fuse_lora", "call"]
    assert rec.args_of("from_pretrained") == (MODEL,)
    assert rec.kwargs_of("from_pretrained")["revision"] == "rev1"
    assert rec.args_of("load_lora_weights") == ("someone/lineart",)
    assert rec.kwargs_of("load_lora_weights") == {
        "weight_name": "lineart.safetensors",
        "revision": "lrev",
    }
    assert rec.kwargs_of("fuse_lora") == {"lora_scale": 0.8}
    call = rec.kwargs_of("call")
    assert (call["num_inference_steps"], call["guidance_scale"]) == (4, 0.0)
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["generator"].device == "cpu"
    assert call["generator"].seed == effective_seed(0, 7)
    assert "shading" in call["negative_prompt"]


def test_default_factory_without_lora_loads_none(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = fake_torch_diffusers(monkeypatch)
    default_pipeline_factory(_spec())
    assert rec.names() == ["from_pretrained", "to"]


def test_negative_prompt_only_when_pipeline_accepts_it(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = fake_torch_diffusers(monkeypatch, negative_prompt=False)
    LineArtDiffusionProvider(_spec()).generate(ImageRequest("a", None, seed=1))
    assert "negative_prompt" not in rec.kwargs_of("call")


@pytest.mark.parametrize(
    ("device", "cuda", "mps", "expected", "dtype"),
    [
        ("auto", True, True, "cuda", "bfloat16"),
        ("auto", False, True, "mps", "float32"),
        ("auto", False, False, "cpu", "float32"),
        ("cpu", True, True, "cpu", "float32"),
        ("cuda", False, False, "cuda", "bfloat16"),
    ],
)
def test_device_choice(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    cuda: bool,
    mps: bool,
    expected: str,
    dtype: str,
) -> None:
    rec = fake_torch_diffusers(monkeypatch, cuda=cuda, mps=mps)
    default_pipeline_factory(_spec(device=device))
    assert rec.args_of("to") == (expected,)
    assert rec.kwargs_of("from_pretrained")["torch_dtype"] == dtype


def test_load_failure_is_channel_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch_diffusers(monkeypatch, fail_load=True)
    with pytest.raises(ChannelConfigError, match=r"cannot load diffusion model.*OSError"):
        default_pipeline_factory(_spec())


def test_extra_missing_names_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)
    provider = LineArtDiffusionProvider(_spec())  # building is lazy: no error yet
    with pytest.raises(ExtraMissingError, match="uv sync --extra diffusion"):
        provider.generate(ImageRequest("a", None, seed=1))


class _Boom:
    def __call__(self, **_: object) -> Image.Image:
        raise RuntimeError("cuda oom")


def test_pipeline_error_is_provider_error() -> None:
    provider = LineArtDiffusionProvider(_spec(), factory=lambda _spec: _Boom())
    with pytest.raises(ProviderError, match=r"lineart diffusion failed \(RuntimeError\)"):
        provider.generate(ImageRequest("a", None, seed=1))


def _png_with_text_chunk() -> Image.Image:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    png = buf.getvalue()
    marked = png[:33] + chunk(b"tEXt", b"parameters\x00secret prompt") + png[33:]
    image = Image.open(io.BytesIO(marked))
    assert "parameters" in image.info
    return image


def test_png_has_no_metadata() -> None:
    marked = _png_with_text_chunk()
    provider = LineArtDiffusionProvider(_spec(), factory=lambda _spec: lambda **_: marked)
    png = provider.generate(ImageRequest("a", None, seed=1)).png
    assert b"tEXt" not in png and b"secret" not in png
    assert Image.open(io.BytesIO(png)).size == (4, 4)


def test_spec_from_style(channel_file: Path) -> None:
    text = channel_file.read_text(encoding="utf-8").replace(
        'image_provider = "openai"', 'image_provider = "lineart-diffusion"'
    )
    channel_file.write_text(
        text
        + '\n[style.diffusion]\nmodel_license = "Apache-2.0"\nmodel_source = "s"\n'
        + 'model_revision = "r"\nseed = 9\nsteps = 5\ndevice = "cpu"\n'
        + 'lora = "a/b"\nlora_license = "MIT"\nlora_source = "t"\nlora_scale = 0.5\n',
        encoding="utf-8",
    )
    style = load_channel(channel_file).style
    spec = DiffusionSpec.from_style(style)
    assert spec.model == "gpt-image-1"  # the channel's image_model
    assert (spec.revision, spec.seed, spec.steps, spec.device) == ("r", 9, 5, "cpu")
    assert (spec.lora, spec.lora_scale, spec.size) == ("a/b", 0.5, 1024)


def test_spec_from_style_needs_diffusion_table(channel_file: Path) -> None:
    with pytest.raises(ChannelConfigError, match=r"style\.diffusion"):
        DiffusionSpec.from_style(load_channel(channel_file).style)


def test_factory_builds_lineart_provider(channel_file: Path) -> None:
    text = channel_file.read_text(encoding="utf-8").replace(
        'image_provider = "openai"', 'image_provider = "lineart-diffusion"'
    )
    channel_file.write_text(
        text + '\n[style.diffusion]\nmodel_license = "Apache-2.0"\nmodel_source = "s"\n',
        encoding="utf-8",
    )
    channel = load_channel(channel_file)
    provider = build_image_provider(channel, get_settings())
    assert isinstance(provider, LineArtDiffusionProvider)
    assert (provider.name, provider.model) == ("lineart-diffusion", "gpt-image-1")
    assert "torch" not in sys.modules and "diffusers" not in sys.modules
    tracked = make_image_provider(
        channel, get_settings(), SpendTracker(None), default_prices(get_settings())
    )
    assert tracked.name == "lineart-diffusion"
