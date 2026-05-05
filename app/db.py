"""Async SQLAlchemy 2.0 setup with typed models.

Demonstrates production-grade DB engineering Pragna brings from her enterprise work:
  - Async session lifecycle
  - Typed Mapped[] declarations (SQLAlchemy 2.0 style)
  - Migration-ready schema (drop-in Alembic later)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

SessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Conversation(Base):
    """Logged chat turn for analytics and debugging.

    Stored anonymously — no IP, no user identity. Just question/answer pairs
    with the retrieval citations and latency metrics.
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str] = mapped_column(Text, default="", nullable=False)  # JSON-encoded list of chunk titles
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), default="echo", nullable=False)


class DemoRun(Base):
    """One execution of an /api/demo/* endpoint — for usage analytics."""

    __tablename__ = "demo_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    demo: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    input_excerpt: Mapped[str] = mapped_column(String(280), default="", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


async def init_db() -> None:
    """Create tables on first start (replace with Alembic for production migrations)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a managed session."""
    async with SessionLocal() as session:
        yield session
