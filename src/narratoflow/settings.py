"""Typed application settings loaded from NARRATOFLOW_* environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        # hide_input_in_errors keeps raw values (possibly API keys) out of this message.
        raise SettingsError(f"invalid NarratoFlow settings (environment or .env):\n{e}") from e
