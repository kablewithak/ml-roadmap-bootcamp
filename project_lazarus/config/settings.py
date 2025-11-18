"""Configuration settings for Project Lazarus."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Project Lazarus"
    app_version: str = "0.1.0"
    debug: bool = False

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # PostgreSQL Settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "lazarus"
    postgres_password: str = "lazarus_secret"
    postgres_db: str = "lazarus_db"

    # Exploration Settings
    epsilon: float = Field(default=0.01, description="Exploration probability (1%)")
    initial_loss_budget: float = Field(default=1000.0, description="Initial loss budget in dollars")
    default_ltv: float = Field(default=500.0, description="Lifetime value per good loan")

    # Model Settings
    risk_threshold: float = Field(default=0.5, description="Risk score threshold for approval")
    model_path: str = "models/risk_model.joblib"

    # Feast Settings
    feast_repo_path: str = "features/feature_repo"

    # Prometheus Settings
    prometheus_port: int = 9090

    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        env_prefix = "LAZARUS_"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
