from __future__ import annotations

from pydantic_settings import BaseSettings


class SAPSettings(BaseSettings):
    # Postgres
    postgres_dsn: str = "postgresql+asyncpg://sap:sap@localhost:5433/sap"

    # Valkey / Redis
    valkey_url: str = "valkey://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "sap-audio"
    minio_secure: bool = False

    # Audio
    sample_rate: int = 44100

    # General
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = SAPSettings()
