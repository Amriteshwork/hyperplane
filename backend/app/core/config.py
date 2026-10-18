"""Application settings, read from the environment (and the repo-root `.env` in dev)."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/app/core/config.py → repo root. Missing in containers, which use real env vars.
_ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class Settings(BaseSettings):
    """Every variable is documented in `.env.example`."""

    model_config = SettingsConfigDict(
        env_file=_ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    app_env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    database_url: str = "postgresql+asyncpg://hyperplane:hyperplane@localhost:5432/hyperplane"

    redis_url: str = "redis://localhost:6379/0"
    job_queue_name: str = "default"
    job_timeout_seconds: int = Field(default=300, gt=0)

    storage_backend: str = "local"
    local_storage_root: Path = Path(".data/storage")
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "hyperplane"
    s3_access_key_id: str | None = None
    s3_secret_access_key: SecretStr | None = None

    content_dir: Path = Path("../content")
    datasets_dir: Path = Path("../datasets")

    health_timeout_seconds: float = Field(default=2.0, gt=0)

    # Upload limits (ARCHITECTURE §10) and retention.
    max_upload_mb: int = Field(default=50, gt=0)
    max_rows: int = Field(default=1_000_000, gt=0)
    max_columns: int = Field(default=500, gt=0)
    retention_days: int = Field(default=7, gt=0)

    # Anonymous sessions: cookie for browsers, X-Session-ID header for API clients.
    session_cookie_name: str = "hp_session"
    session_cookie_secure: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """CORS_ORIGINS is a comma-separated string in the environment."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
