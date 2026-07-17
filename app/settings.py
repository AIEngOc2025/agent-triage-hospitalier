from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    @definition : Settings for the application.
    @args/params : None
    @return : None
    """

    MODEL_PATH: Path = Path("models/merged_dpo_final_chsa")


settings = Settings()

