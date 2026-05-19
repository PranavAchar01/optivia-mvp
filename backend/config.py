import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env and override any empty env vars (Claude Code injects ANTHROPIC_API_KEY="")
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    from dotenv import dotenv_values
    for k, v in dotenv_values(_dotenv_path).items():
        if v and not os.environ.get(k):
            os.environ[k] = v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://localhost/optivia"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Voyage AI
    voyage_api_key: str = ""

    # Auth
    jwt_secret: str = "dev-secret"

    # Runtime
    env: str = "development"
    session_token_budget: int = 200_000

    # Optivia API base (for CLI)
    optivia_api_base: str = "http://localhost:8000"

    # Model tiers (Stage 1 heuristic routing)
    model_haiku: str = "gemini-pro"
    model_sonnet: str = "gemini-pro"
    model_opus: str = "gemini-pro"

    # Clarification predicate thresholds (§4.5) — per-tenant configurable
    clarify_kappa_threshold: int = 7
    clarify_sigma_threshold: float = 0.5
    clarify_scope_threshold: float = 0.7
    clarify_confidence_threshold: float = 0.5

    # Semantic cache cosine threshold (§6.2)
    semantic_cache_threshold: float = 0.97

    # Quality scalar thresholds (§4.10)
    quality_halt_threshold: float = 0.5
    quality_reverify_threshold: float = 0.75
    quality_long_clean_n: int = 5


settings = Settings()
