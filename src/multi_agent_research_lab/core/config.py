"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # LLM Provider selection: 'groq', 'openai', or 'auto'
    llm_provider: str = Field(default="auto", validation_alias="LLM_PROVIDER")

    # Groq specific configuration
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str | None = Field(default=None, validation_alias="GROQ_MODEL")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", validation_alias="GROQ_BASE_URL"
    )

    # OpenAI specific configuration / legacy fallback
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # Fallback models for rate limit / quota management (especially for Groq)
    fallback_models: list[str] = Field(
        default=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        validation_alias="FALLBACK_MODELS",
    )

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")
    inter_call_delay_seconds: float = Field(default=1.5, ge=0.0, le=10.0, validation_alias="INTER_CALL_DELAY_SECONDS")

    @property
    def is_groq(self) -> bool:
        """Check if Groq is the active provider."""
        if self.llm_provider.lower() == "groq":
            return True
        if self.groq_api_key:
            return True
        if self.openai_api_key and self.openai_api_key.startswith("gsk_"):
            return True
        return False

    @property
    def effective_api_key(self) -> str | None:
        """Return the active API key (prefers GROQ_API_KEY if Groq configured)."""
        if self.groq_api_key:
            return self.groq_api_key
        return self.openai_api_key

    @property
    def effective_model(self) -> str:
        """Return the active model name."""
        if self.groq_model:
            return self.groq_model
        return self.openai_model

    @property
    def effective_base_url(self) -> str | None:
        """Return the active base URL for API completions."""
        if self.is_groq:
            return self.groq_base_url
        return self.openai_base_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


