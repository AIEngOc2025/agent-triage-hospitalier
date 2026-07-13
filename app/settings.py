import platform
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Manages application configuration using Pydantic, loading values from environment
    variables and a .env file.
    """

    # --- Environment Settings ---
    # Set to 'production' in your deployment environment
    APP_ENV: str = "development"

    # --- Model & Path Settings ---
    # The model path can now be overridden by an environment variable.
    # It defaults to a path relative to the project root.
    MODEL_PATH: Path = (
        Path(__file__).resolve().parent.parent / "models/merged_dpo_final_chsa"
    )
    LOG_FILE: Path = Path(__file__).resolve().parent.parent / "logs/triage.log"

    # --- vLLM Parameters ---
    VLLM_MAX_MODEL_LEN: int = 2048
    VLLM_TENSOR_PARALLEL_SIZE: int = 1

    # --- Computed Fields for Environment ---
    @computed_field
    @property
    def IS_PRODUCTION(self) -> bool:
        import os

        return (
            self.APP_ENV.lower() == "production" or "FASTAPI_CLOUD_APP_ID" in os.environ
        )

    @property
    def IS_MACOS(self) -> bool:
        return platform.system() == "Darwin"

    # --- Pydantic-Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from a .env file
        env_file_encoding="utf-8",
        case_sensitive=False,  # Environment variables are case-insensitive
        extra='ignore',       # Ignore extra environment variables
    )



# Create a single, importable instance of the settings
settings = Settings()
