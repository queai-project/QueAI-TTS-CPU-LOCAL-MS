from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================
    # PROJECT
    # ==========================================
    PROJECT_NAME: str = "TTS Local CPU"
    PROJECT_SLUG: str = "tts_local_cpu"
    VERSION: str = "5.0.0"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["*"]

    # Base pública única para todo el módulo
    BASE_PATH: str = "/api/tts_local_cpu"


    # ==========================================
    # LOG
    # ==========================================
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_DIR: str = "logs"
    LOG_FILENAME: str = "file.log"

    # ==========================================
    # PLUGIN META
    # ==========================================
    DISPLAY_NAME: str = "TTS Local CPU"
    DESCRIPTION: str = (
        "Módulo TTS oficial para QueAI. Texto a voz de alta fidelidad "
        "basado en Kokoro-82M. Ejecución local en CPU, soporte para "
        "Inglés/Español y múltiples voces."
    )
    AUTHOR: str = "Alejandro Fonseca && Juana Iris"
    LICENSE: str = "MIT"
    LOGO: str = "stt_logo.png"

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "development"

    @computed_field
    @property
    def OPENAPI_PATH(self) -> str:
        return f"{self.BASE_PATH}/openapi.json"

    @computed_field
    @property
    def DOCS_PATH(self) -> str:
        return f"{self.BASE_PATH}/docs"

    @computed_field
    @property
    def REDOC_PATH(self) -> str:
        return f"{self.BASE_PATH}/redoc"

    @computed_field
    @property
    def UI_PATH(self) -> str:
        return f"{self.BASE_PATH}/ui"

    @computed_field
    @property
    def HEALTH_PATH(self) -> str:
        return f"{self.BASE_PATH}/health"

    @computed_field
    @property
    def CONFIG_PATH(self) -> str:
        return f"{self.BASE_PATH}/config"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()