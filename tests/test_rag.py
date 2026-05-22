"""Tests for the RAG pipeline."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.rag import Chunk, Embedder, RAGIndex, build_prompt, chunk_markdown


# ── chunk_markdown ──────────────────────────────────────────────────────────

def test_chunk_markdown_single():
    md = "## CHUNK: Intro\nStudied machine learning and data engineering at university.\n"
    chunks = chunk_markdown(md, min_chars=10)
    assert len(chunks) == 1
    assert chunks[0].title == "Intro"
    assert "machine learning" in chunks[0].text


def test_chunk_markdown_multiple():
    md = (
        "## CHUNK: Education\nStudied computer science and machine learning at university.\n\n"
        "## CHUNK: Experience\nBuilt data pipelines and backend services professionally.\n"
    )
    chunks = chunk_markdown(md, min_chars=10)
    assert len(chunks) == 2
    assert chunks[0].title == "Education"
    assert chunks[1].title == "Experience"


def test_chunk_markdown_no_markers():
    assert chunk_markdown("No CHUNK markers here") == []


def test_chunk_markdown_strips_trailing_hr():
    md = "## CHUNK: Section\nContent long enough to pass the minimum character filter.\n---"
    chunks = chunk_markdown(md, min_chars=10)
    assert not chunks[0].text.endswith("---")


def test_chunk_markdown_min_chars_filter():
    md = (
        "## CHUNK: Short\nToo short.\n"
        "## CHUNK: Long\nThis section is long enough to pass the minimum character filter.\n"
    )
    chunks = chunk_markdown(md, min_chars=50)
    assert len(chunks) == 1
    assert chunks[0].title == "Long"


# ── RAGIndex ─────────────────────────────────────────────────────────────────

def _index(*texts: str) -> RAGIndex:
    settings = Settings(embedding_provider="hash", embedding_dim=384)
    embedder = Embedder(settings)
    chunks = [Chunk(title=f"chunk_{i}", text=t) for i, t in enumerate(texts)]
    return RAGIndex(chunks, embedder)


def test_retrieve_returns_k_hits():
    index = _index(
        "Python machine learning neural networks sklearn",
        "Cooking recipes pasta dinner Italian",
        "FastAPI backend REST API deployment Docker",
    )
    hits = index.retrieve("machine learning", k=2)
    assert len(hits) == 2


def test_retrieve_top_hit_is_relevant():
    index = _index(
        "Python machine learning neural networks data science",
        "Cooking recipes pasta dinner Italian food",
    )
    hits = index.retrieve("machine learning Python", k=1)
    assert hits[0].chunk.title == "chunk_0"


def test_retrieve_scores_in_range():
    index = _index("data science pandas sklearn", "React frontend JavaScript")
    for hit in index.retrieve("data analysis", k=2):
        assert 0.0 <= hit.score <= 1.0


def test_retrieve_empty_index():
    settings = Settings(embedding_provider="hash", embedding_dim=384)
    embedder = Embedder(settings)
    assert RAGIndex([], embedder).retrieve("anything") == []


def test_retrieve_k_larger_than_index():
    index = _index("only one chunk with enough text here")
    hits = index.retrieve("query", k=10)
    assert len(hits) == 1


# ── build_prompt ─────────────────────────────────────────────────────────────

def test_build_prompt_contains_question_and_context():
    index = _index("Worked on NLP projects and data pipelines at Rostock.")
    hits = index.retrieve("NLP experience", k=1)
    prompt = build_prompt("What NLP work have you done?", hits, max_chars=4000)
    assert "What NLP work have you done?" in prompt
    assert "NLP projects" in prompt


def test_build_prompt_no_hits_uses_fallback():
    prompt = build_prompt("Any question?", hits=[], max_chars=4000)
    assert "(no context retrieved)" in prompt


def test_build_prompt_respects_max_chars():
    long_text = "word " * 5000
    index = _index(long_text)
    hits = index.retrieve("word", k=1)
    prompt = build_prompt("question?", hits, max_chars=100)
    assert len(prompt) < len(long_text)
