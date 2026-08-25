import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    @definition: Settings for the application, loaded from environment variables.
    @args/params: None
    @return: Settings object containing application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Environment Configuration ---
    # APP_ENV can be 'development' or 'production'
    APP_ENV: Literal["development", "production"] = "development"
    ENGINE_MODE: Literal["remote", "local"] = "remote"

    # --- Model & Log Paths ---
    MODEL_PATH: str = Field(default="models/merged_dpo_final_chsa", env="MODEL_PATH")
    LOG_FILE: Path = Path("logs/triage.log")

    # --- vLLM Engine Configuration ---
    max_model_len: int = 2048
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True

    # --- Computed Properties ---
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def IS_MACOS(self) -> bool:
        # For local dev, we assume macOS. In production, it will be Linux.
        return not self.IS_PRODUCTION


# Debug: Print environment variables
print("DEBUG: All environment variables starting with MODEL:")
for key, value in os.environ.items():
    if key.startswith("MODEL"):
        print(f"DEBUG: {key}={value}")

settings = Settings()
print(f"DEBUG: Settings loaded. MODEL: {settings.MODEL_PATH}, ENV: {settings.APP_ENV}")
