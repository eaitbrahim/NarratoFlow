"""Typed application settings loaded from NARRATOFLOW_* environment variables."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from narratoflow.errors import SettingsError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NARRATOFLOW_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    data_dir: Path = Path("~/.narratoflow")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    image_api_key: SecretStr | None = None
    voice_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    embed_api_key: SecretStr | None = None
    ollama_url: str = "http://localhost:11434"
    ffmpeg: Path | None = None

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("ffmpeg", mode="before")
    @classmethod
    def _blank_ffmpeg_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("ffmpeg")
    @classmethod
    def _expand_ffmpeg(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    prefix = Settings.model_config["env_prefix"]
    known = sorted(prefix + name.upper() for name in Settings.model_fields)
    names = {*os.environ, *dotenv_values(".env")}
    unknown = sorted(n for n in names if n.upper().startswith(prefix) and n.upper() not in known)
    if unknown:
        # Names only, never values: an unknown name may be a mistyped secret.
        raise SettingsError(
            "unknown NarratoFlow setting(s) in the environment or .env: "
            f"{', '.join(unknown)}; known: {', '.join(known)}"
        )
    try:
        return Settings()
    except ValidationError as e:
        # hide_input_in_errors keeps raw values (possibly API keys) out of this message.
        raise SettingsError(f"invalid NarratoFlow settings (environment or .env):\n{e}") from e
