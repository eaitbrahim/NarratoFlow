"""Channel configuration: a named setup with its own sketch style."""

from __future__ import annotations

import hashlib
import json
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from narratoflow.errors import ChannelConfigError
from narratoflow.llm.base import Effort
from narratoflow.models import AspectRatio, Language
from narratoflow.text.spoken import Lexicon


class StyleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    reference_sketch: Path
    instructions: str = Field(min_length=1)
    image_provider: str
    image_model: str
    line_weight_px: float = Field(3.0, gt=0, le=20)
    ink: str = Field("#111111", pattern=r"^#[0-9a-fA-F]{6}$")

    @model_validator(mode="before")
    @classmethod
    def _strip_instructions(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("instructions"), str):
            data = {**data, "instructions": data["instructions"].strip()}
        return data


class VoiceConfig(BaseModel):
    """The primary voice and its free fallback chain (`tts.factory.build_voice_chain`).

    `edge_voice` None means the language default (`tts.edge.DEFAULT_EDGE_VOICES`);
    `piper_model` is resolved relative to the channel TOML by `load_channel`.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: str
    model: str = ""
    voice_id: str
    fallback: list[Literal["edge", "piper"]] = ["edge"]  # pydantic copies the default
    edge_voice: str | None = None
    piper_model: Path | None = None

    @field_validator("provider")
    @classmethod
    def _lower_provider(cls, value: str) -> str:
        return value.strip().lower()


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    provider: str
    model: str
    effort: Effort = "medium"
    max_tokens: int = Field(default=16000, gt=0)
    refusal_fallback: bool = True


class EmbedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    provider: str = "voyage"
    model: str = "voyage-multilingual-2"


VISION_CAPABLE_PROVIDERS: frozenset[str] = frozenset({"anthropic"})
"""LLM providers whose channel model can judge images when `[vision]` is unset."""
VISION_OFF = "none"
"""`[vision] provider = "none"` turns the vision check off."""


class VisionConfig(BaseModel):
    """Vision-check model (see `vision_model`); both unset means the channel's LLM
    when its provider is vision capable, else no vision check."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: str | None = None
    model: str | None = None


class ChannelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str
    slug: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    language: Language
    style: StyleConfig
    voice: VoiceConfig
    llm: LlmConfig
    embed: EmbedConfig = EmbedConfig()
    vision: VisionConfig = VisionConfig()
    outputs: list[AspectRatio] = Field(default_factory=lambda: [AspectRatio.PORTRAIT])
    spending_limit_usd: Decimal | None = Field(None, gt=0)
    approval_list: bool = True
    background_music: bool = False
    part_min_s: int = 60
    part_max_s: int = 90
    riwaya: str = "hafs"
    reciter: str | None = None
    """An allow-listed reciter id; None means the first allow-listed reciter of `riwaya`."""
    diacritizer: Literal["llm", "mishkal"] = "llm"
    pronunciation: dict[str, str] = Field(default_factory=dict)
    """Word to spoken phrase, applied to narration (`text.spoken.Lexicon`)."""

    @field_validator("pronunciation")
    @classmethod
    def _valid_pronunciation(cls, value: dict[str, str]) -> dict[str, str]:
        Lexicon.from_entries(value)  # raises ValueError naming the key, never the phrase
        return value

    @model_validator(mode="after")
    def _part_bounds(self) -> ChannelConfig:
        if self.part_min_s >= self.part_max_s:
            raise ValueError("part_min_s must be less than part_max_s")
        return self

    @model_validator(mode="after")
    def _piper_model_set(self) -> ChannelConfig:
        voice = self.voice
        uses_piper = voice.provider == "piper" or "piper" in voice.fallback
        if uses_piper and voice.piper_model is None:
            raise ValueError("piper needs voice.piper_model")
        return self


def vision_model(channel: ChannelConfig) -> tuple[str, str] | None:
    """`(provider, model)` of the vision check, or None when vision is off.

    - `[vision] provider = "none"`: off;
    - `[vision]` unset: the channel LLM when its provider is in
      `VISION_CAPABLE_PROVIDERS` (a text-only model must not judge images), else off;
    - otherwise `channel.vision`, each unset field taken from the channel LLM.

    The provider is normalized (stripped, lower case) the way the factories use it.
    Off means new sketches are stored UNCHECKED ("vision: not checked").
    """
    vision, llm = channel.vision, channel.llm
    if vision.provider is None and vision.model is None:
        provider = llm.provider.strip().lower()
        return (provider, llm.model) if provider in VISION_CAPABLE_PROVIDERS else None
    provider = (vision.provider or llm.provider).strip().lower()
    if provider == VISION_OFF:
        return None
    return provider, vision.model or llm.model


def load_channel(path: Path) -> ChannelConfig:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        style = raw.get("style")
        if isinstance(style, dict) and "reference_sketch" in style:
            style["reference_sketch"] = str(
                (path.parent / str(style["reference_sketch"])).resolve()
            )
        voice = raw.get("voice")
        if isinstance(voice, dict) and isinstance(voice.get("piper_model"), str):
            voice["piper_model"] = str((path.parent / voice["piper_model"]).resolve())
        return ChannelConfig.model_validate(raw)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError) as e:
        raise ChannelConfigError(f"Invalid channel config {path}: {e}") from e


def style_fingerprint(style: StyleConfig) -> str:
    try:
        reference_sha256 = hashlib.sha256(style.reference_sketch.read_bytes()).hexdigest()
    except OSError as e:
        raise ChannelConfigError(
            f"Cannot read reference sketch {style.reference_sketch}: {e}"
        ) from e
    payload = {
        "instructions": style.instructions,
        "image_provider": style.image_provider,
        "image_model": style.image_model,
        "line_weight_px": style.line_weight_px,
        "ink": style.ink,
        "reference_sha256": reference_sha256,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
