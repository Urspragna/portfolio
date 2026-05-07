"""Configuration loaded from environment or .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuration for the portfolio backend."""

    # App settings
    app_name: str = "Pragna AI Portfolio"
    environment: Literal["development", "production"] = "development"
    debug: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )

    # Database
    # TODO: support DATABASE_URL env var for easier production switching
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'portfolio.db'}"

    # LLM provider
    llm_provider: Literal["anthropic", "ollama", "echo"] = "echo"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings
    embedding_provider: Literal["sbert", "hash"] = "sbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RAG
    knowledge_base_path: Path = ROOT / "data" / "knowledge_base.md"
    chunk_min_chars: int = 80
    retrieval_k: int = 4
    max_context_chars: int = 4000

    # Frontend
    public_dir: Path = Field(default_factory=lambda: ROOT)

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton cached settings."""
    return Settings()
