"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration."""

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/topicdrift"
    test_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/topicdrift_test"
    )
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
