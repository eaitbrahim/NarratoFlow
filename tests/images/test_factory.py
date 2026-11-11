import io
import sys
import types
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import SecretStr

import narratoflow.images.factory as factory_module
import narratoflow.images.local_diffusion as local_module
from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import SpendTracker
from narratoflow.errors import (
    BudgetExceededError,
    ChannelConfigError,
    MissingApiKeyError,
    ProviderError,
)
from narratoflow.images import GeneratedImage, ImageProvider, ImageRequest
from narratoflow.images.factory import TrackedImageProvider, make_image_provider
from narratoflow.images.local_diffusion import LocalDiffusionProvider
from narratoflow.images.openai_images import OpenAIImageProvider
from narratoflow.images.prompt import NEGATIVE_TERMS
from narratoflow.settings import Settings


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeImage:
    name = "fake"
    model = "m1"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def generate(self, req: ImageRequest) -> GeneratedImage:
        self.calls += 1
        if self.fail:
            raise ProviderError("boom")
        return GeneratedImage(png_bytes(), "fake", "m1", None)


class FakePrices:
    def image(self, provider: str, model: str) -> Decimal:
        return Decimal("0.04")


def tracked(inner: FakeImage, tracker: SpendTracker) -> TrackedImageProvider:
    prices: Any = FakePrices()
    return TrackedImageProvider(inner, tracker, prices, "fake", "m1")


def test_tracked_charges_after_success() -> None:
    tracker = SpendTracker(Decimal("1"))
    inner = FakeImage()
    t = tracked(inner, tracker)
    out = t.generate(ImageRequest("secret prompt", None))
    assert out.provider == "fake"
    assert (t.name, t.model) == ("fake", "m1")
    assert tracker.spent == Decimal("0.04")
    assert tracker.entries[0].what == "image fake/m1"


def test_tracked_blocks_before_call() -> None:
    tracker = SpendTracker(Decimal("0.01"))
    inner = FakeImage()
    with pytest.raises(BudgetExceededError):
        tracked(inner, tracker).generate(ImageRequest("p", None))
    assert inner.calls == 0
    assert tracker.spent == 0


def test_tracked_failure_charges_nothing() -> None:
    tracker = SpendTracker(None)
    with pytest.raises(ProviderError):
        tracked(FakeImage(fail=True), tracker).generate(ImageRequest("p", None))
    assert tracker.spent == 0


def _channel(channel_file: Path, provider: str, model: str = "gpt-image-1") -> ChannelConfig:
    base = load_channel(channel_file)
    style = base.style.model_copy(update={"image_provider": provider, "image_model": model})
    return base.model_copy(update={"style": style})


def _make(channel: ChannelConfig, settings: Settings) -> ImageProvider:
    prices: PriceTable = default_prices(settings)
    return make_image_provider(channel, settings, SpendTracker(None), prices)


def test_make_openai_requires_key(channel_file: Path) -> None:
    for key in (None, SecretStr("")):
        with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_IMAGE_API_KEY"):
            _make(_channel(channel_file, "openai"), Settings(image_api_key=key))


def test_make_openai_case_insensitive(channel_file: Path) -> None:
    p = _make(_channel(channel_file, " OpenAI "), Settings(image_api_key=SecretStr("k")))
    assert isinstance(p, TrackedImageProvider)
    assert isinstance(p.inner, OpenAIImageProvider)
    assert (p.name, p.model) == ("openai", "gpt-image-1")


def test_make_unknown_provider(channel_file: Path) -> None:
    with pytest.raises(ChannelConfigError, match="nope"):
        _make(_channel(channel_file, "nope"), Settings())


class FakePipe:
    def __init__(self, cls: str) -> None:
        self.cls = cls
        self.moved_to: str | None = None
        self.kwargs: dict[str, Any] = {}

    def to(self, device: str) -> "FakePipe":
        self.moved_to = device
        return self

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "FakeGenerator":
        self.seed = seed
        return self


def install_fakes(
    monkeypatch: pytest.MonkeyPatch, cuda: bool = False, mps: bool = False
) -> dict[str, list[FakePipe]]:
    made: dict[str, list[FakePipe]] = {"t2i": [], "i2i": []}

    def pipeline_class(tag: str) -> type:
        class Pipeline:
            @staticmethod
            def from_pretrained(model: str, **_: Any) -> FakePipe:
                pipe = FakePipe(f"{tag}:{model}")
                made[tag].append(pipe)
                return pipe

        return Pipeline

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    torch.Generator = FakeGenerator  # type: ignore[attr-defined]
    diffusers = types.ModuleType("diffusers")
    diffusers.AutoPipelineForText2Image = pipeline_class("t2i")  # type: ignore[attr-defined]
    diffusers.AutoPipelineForImage2Image = pipeline_class("i2i")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return made


def test_local_missing_extra_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(ChannelConfigError, match="uv sync --extra diffusion"):
        LocalDiffusionProvider("stabilityai/sdxl-turbo")


def test_local_text_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    made = install_fakes(monkeypatch)
    p = LocalDiffusionProvider("stabilityai/sdxl-turbo")
    out = p.generate(ImageRequest("a balance", None, seed=7))
    assert out.png.startswith(b"\x89PNG")
    assert (out.provider, out.model) == ("local-diffusion", "stabilityai/sdxl-turbo")
    assert len(made["t2i"]) == 1 and made["i2i"] == []
    pipe = made["t2i"][0]
    assert pipe.moved_to == "cpu"
    assert pipe.kwargs["prompt"] == "a balance"
    expected = ", ".join(NEGATIVE_TERMS) + ", people, faces, human figures, text, letters, logos"
    assert pipe.kwargs["negative_prompt"] == expected
    gen = pipe.kwargs["generator"]
    assert gen.seed == 7
    assert "image" not in pipe.kwargs


def test_local_no_seed_no_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    made = install_fakes(monkeypatch)
    LocalDiffusionProvider("m").generate(ImageRequest("x", None))
    assert made["t2i"][0].kwargs["generator"] is None


def test_local_image_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    made = install_fakes(monkeypatch)
    p = LocalDiffusionProvider("m")
    p.generate(ImageRequest("x", png_bytes(), seed=1))
    assert len(made["i2i"]) == 1
    kwargs = made["i2i"][0].kwargs
    assert kwargs["strength"] == 0.6
    assert isinstance(kwargs["image"], Image.Image)
    assert kwargs["generator"].seed == 1
    p.generate(ImageRequest("y", png_bytes()))
    assert len(made["i2i"]) == 1


def test_local_bad_reference_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fakes(monkeypatch)
    with pytest.raises(ProviderError):
        LocalDiffusionProvider("m").generate(ImageRequest("x", b"not a png"))


def test_local_device_auto_order(monkeypatch: pytest.MonkeyPatch) -> None:
    made = install_fakes(monkeypatch, cuda=True, mps=True)
    LocalDiffusionProvider("m")
    assert made["t2i"][0].moved_to == "cuda"
    made = install_fakes(monkeypatch, mps=True)
    LocalDiffusionProvider("m")
    assert made["t2i"][0].moved_to == "mps"
    made = install_fakes(monkeypatch)
    LocalDiffusionProvider("m", device="cpu")
    assert made["t2i"][0].moved_to == "cpu"


def test_make_local_diffusion_is_free(monkeypatch: pytest.MonkeyPatch, channel_file: Path) -> None:
    install_fakes(monkeypatch)
    ch = _channel(channel_file, "local-diffusion", "stabilityai/sdxl-turbo")
    settings = Settings()
    tracker = SpendTracker(Decimal("0"))
    p = make_image_provider(ch, settings, tracker, default_prices(settings))
    assert isinstance(p, TrackedImageProvider)
    assert isinstance(p.inner, LocalDiffusionProvider)
    p.generate(ImageRequest("x", None))
    assert tracker.spent == 0


def test_sources_are_ascii() -> None:
    for module in (factory_module, local_module):
        assert Path(module.__file__ or "").read_bytes().isascii()
