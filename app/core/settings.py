import logging
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    @definition: Configuration globale de l'application chargée depuis l'environnement.
    @args/params: None
    @return: Objet Settings contenant la configuration de l'application.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Configuration d'Environnement ---
    APP_ENV: Literal["development", "production"] = "development"
    ENGINE_MODE: Literal["remote", "local"] = "remote"

    # --- Chemins Modèle & Logs ---
    MODEL_PATH: str = Field(
        default="models/merged_dpo_final_chsa", validation_alias="MODEL_PATH"
    )
    LOG_FILE: Path = Path("logs/triage.log")

    # --- Paramètres Moteur vLLM ---
    max_model_len: int = 2048
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True

    # --- Propriétés Calculées ---
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def IS_MACOS(self) -> bool:
        return not self.IS_PRODUCTION


settings = Settings()
logger.debug(
    "Settings chargés : MODEL=%s, ENV=%s, MODE=%s",
    settings.MODEL_PATH,
    settings.APP_ENV,
    settings.ENGINE_MODE,
)
