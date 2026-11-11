"""Build the configured image provider, wrapped so every call is budget-checked."""

from __future__ import annotations

import logging

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker
from narratoflow.errors import ChannelConfigError, MissingApiKeyError, ProviderError
from narratoflow.images import GeneratedImage, ImageProvider, ImageRequest
from narratoflow.images.local_diffusion import LocalDiffusionProvider
from narratoflow.images.openai_images import OpenAIImageProvider
from narratoflow.settings import Settings

logger = logging.getLogger(__name__)
KEY_ENV_VAR = "NARRATOFLOW_IMAGE_API_KEY"


class TrackedImageProvider:
    """Wraps a provider: checks the budget before each call, charges after success."""

    def __init__(
        self,
        inner: ImageProvider,
        tracker: SpendTracker,
        prices: PriceTable,
        provider: str,
        model: str,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.tracker = tracker
        self.prices = prices
        self.provider = provider
        self.model_key = model

    def generate(self, req: ImageRequest) -> GeneratedImage:
        what = f"image {self.provider}/{self.model_key}"
        cost = self.prices.image(self.provider, self.model_key)
        self.tracker.ensure_room(cost, what)
        try:
            out = self.inner.generate(req)
        except ProviderError as err:
            logger.warning("image call failed (%s/%s): %s", self.provider, self.model_key, err)
            raise
        self.tracker.charge(cost, what)
        return out


def make_image_provider(
    channel: ChannelConfig, settings: Settings, tracker: SpendTracker, prices: PriceTable
) -> ImageProvider:
    style = channel.style
    provider = style.image_provider.strip().lower()
    inner: ImageProvider
    if provider == "openai":
        key = settings.image_api_key
        if key is None or not key.get_secret_value():
            raise MissingApiKeyError(f"Missing API key environment variable(s): {KEY_ENV_VAR}")
        inner = OpenAIImageProvider(api_key=key, model=style.image_model)
    elif provider == "local-diffusion":
        inner = LocalDiffusionProvider(style.image_model)
    else:
        raise ChannelConfigError(f"unsupported image provider {style.image_provider!r}")
    return TrackedImageProvider(inner, tracker, prices, provider, style.image_model)
