from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    @definition: Settings for the application, loaded from environment variables.
    """

    # --- Environment Configuration ---
    # APP_ENV can be 'development' or 'production'
    APP_ENV: Literal["development", "production"] = "development"

    # --- Model & Log Paths ---
    MODEL_PATH: Path = Path("models/merged_dpo_final_chsa")
    LOG_FILE: Path = Path("logs/triage.log")

    # --- Computed Properties ---
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def IS_MACOS(self) -> bool:
        # For local dev, we assume macOS. In production, it will be Linux.
        return not self.IS_PRODUCTION


settings = Settings()
