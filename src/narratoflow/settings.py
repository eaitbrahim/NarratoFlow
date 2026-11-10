"""Typed application settings loaded from NARRATOFLOW_* environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NARRATOFLOW_", env_file=".env")

    data_dir: Path = Path("~/.narratoflow")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    image_api_key: SecretStr | None = None
    voice_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, value: Path) -> Path:
        return value.expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()
