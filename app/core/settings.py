import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    @definition: Settings for the application, loaded from environment variables.
    @args/params: None
    @return: Settings object containing application configuration.
    """
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Environment Configuration ---
    # APP_ENV can be 'development' or 'production'
    APP_ENV: Literal["development", "production"] = "development"

    # --- Model & Log Paths ---
    MODEL_PATH: str = "models/merged_dpo_final_chsa"
    LOG_FILE: Path = Path("logs/triage.log")

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
print(f"DEBUG: Settings loaded. MODEL_PATH: {settings.MODEL_PATH}, APP_ENV: {settings.APP_ENV}")
