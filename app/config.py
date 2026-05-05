"""Application settings loaded from environment / .env.

Uses Pydantic Settings v2 — the canonical 2026 way to manage Python config.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration for the portfolio backend.

    Read from environment variables and an optional `.env` file at the project root.
    """

    # ─── App ───────────────────────────────────────────────────────────────
    app_name: str = "Pragna AI Portfolio"
    environment: Literal["development", "production"] = "development"
    debug: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )

    # ─── Database ──────────────────────────────────────────────────────────
    # Defaults to local SQLite for zero-setup; flip to Postgres+pgvector in prod.
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'portfolio.db'}"

    # ─── LLM provider ──────────────────────────────────────────────────────
    # "anthropic" (frontier, requires ANTHROPIC_API_KEY)
    # "ollama"    (local, free — needs `ollama serve` running)
    # "echo"      (deterministic stub for offline / no key — answers from retrieved chunks only)
    llm_provider: Literal["anthropic", "ollama", "echo"] = "echo"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ─── Embeddings ────────────────────────────────────────────────────────
    # "sbert"  : sentence-transformers (open source, downloads on first run)
    # "hash"   : deterministic hashing-based fallback for environments
    #            with no model download (offline CI, locked-down sandboxes)
    embedding_provider: Literal["sbert", "hash"] = "sbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ─── RAG ───────────────────────────────────────────────────────────────
    knowledge_base_path: Path = ROOT / "data" / "knowledge_base.md"
    chunk_min_chars: int = 80
    retrieval_k: int = 4
    max_context_chars: int = 4000

    # ─── Static frontend ───────────────────────────────────────────────────
    public_dir: Path = ROOT  # serves index.html / lab.html / assets from project root

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — single instance across the app."""
    return Settings()
