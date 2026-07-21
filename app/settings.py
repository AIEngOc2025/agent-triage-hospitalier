from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    @definition : Settings for the application.
    @args/params : None
    @return : None
    """

    MODEL_PATH: Path = Path("models/merged_dpo_final_chsa")
    LOG_FILE: Path = Path("logs/triage.log")
    APP_ENV: str = "development"
    IS_PRODUCTION: bool = True
    IS_MACOS: bool = False


settings = Settings()
