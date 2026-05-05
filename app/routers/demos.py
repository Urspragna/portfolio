"""Live AI demos — embedding similarity, sentiment, summarisation.

Endpoints are intentionally small and self-contained so each one is a
standalone teaching example readers can study independently.

  POST /api/demo/embed     — encode text, return vector + similarity
  POST /api/demo/sentiment — sentiment analysis (HF transformers pipeline)
  POST /api/demo/tokenise  — show how a tokeniser splits text
"""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.db import DemoRun, SessionLocal

router = APIRouter(prefix="/api/demo", tags=["demos"])


async def _log_demo(**fields) -> None:
    """Best-effort analytics write — never raise."""
    try:
        async with SessionLocal() as s:
            s.add(DemoRun(**fields))
            await s.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[demos] log skipped: {type(exc).__name__}: {exc}")


# ────────────────────────────────────────────────────────────────────────
# Embedding similarity
# ────────────────────────────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    a: str = Field(..., min_length=1, max_length=1500)
    b: str = Field(..., min_length=1, max_length=1500)


@router.post("/embed")
async def embed_similarity(
    body: EmbedRequest,
    request: Request,
) -> dict:
    """Embed two texts and return cosine similarity + a tiny vector preview.

    Teaches: 'similarity' is just the angle between two high-dimensional vectors.
    """
    t0 = time.perf_counter()
    embedder = request.app.state.embedder
    vecs = embedder.encode([body.a, body.b])  # (2, D)
    a, b = vecs[0], vecs[1]
    cos = float(
        np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    await _log_demo(demo="embed", input_excerpt=body.a[:140], latency_ms=latency_ms)

    return {
        "a": body.a,
        "b": body.b,
        "similarity": round(cos, 4),
        "dim": int(a.shape[0]),
        "vector_a_preview": [round(float(x), 4) for x in a[:8]],
        "vector_b_preview": [round(float(x), 4) for x in b[:8]],
        "latency_ms": latency_ms,
    }


# ────────────────────────────────────────────────────────────────────────
# Sentiment (lazy-loaded HF pipeline)
# ────────────────────────────────────────────────────────────────────────


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)


_SENTIMENT_PIPELINE = None  # lazily initialised on first call


def _get_sentiment():
    """Load HF pipeline on demand. Falls back to a tiny lexicon classifier."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE
    try:
        from transformers import pipeline  # type: ignore

        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception:
        _SENTIMENT_PIPELINE = "fallback"
    return _SENTIMENT_PIPELINE


_POS = {
    "good", "great", "excellent", "love", "amazing", "fantastic", "wonderful",
    "happy", "delighted", "awesome", "beautiful", "best", "brilliant",
}
_NEG = {
    "bad", "terrible", "awful", "hate", "worst", "horrible", "sad",
    "disappointing", "poor", "broken", "failure", "ugly",
}


@router.post("/sentiment")
async def sentiment(
    body: SentimentRequest,
    request: Request,
) -> dict:
    t0 = time.perf_counter()
    pipe = _get_sentiment()

    if pipe == "fallback":
        toks = [w for w in body.text.lower().split() if w]
        pos = sum(t in _POS for t in toks)
        neg = sum(t in _NEG for t in toks)
        if pos == neg:
            label, score = "NEUTRAL", 0.5
        else:
            label = "POSITIVE" if pos > neg else "NEGATIVE"
            score = (max(pos, neg)) / (pos + neg + 1)
        out = {"label": label, "score": round(float(score), 4), "engine": "lexicon-fallback"}
    else:
        result = pipe(body.text[:512])[0]
        out = {
            "label": result["label"],
            "score": round(float(result["score"]), 4),
            "engine": "distilbert-sst2",
        }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    out["latency_ms"] = latency_ms

    await _log_demo(demo="sentiment", input_excerpt=body.text[:140], latency_ms=latency_ms)
    return out


# ────────────────────────────────────────────────────────────────────────
# Tokeniser visualiser
# ────────────────────────────────────────────────────────────────────────


class TokeniseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@router.post("/tokenise")
async def tokenise(body: TokeniseRequest) -> dict:
    """Return a simple whitespace + sub-word tokenisation for visualisation.

    Teaches: LLMs see *tokens*, not characters. Tokens cost money and dictate
    how prompts are framed. We return both word-level and a coarse byte-pair
    illustrative split so users can compare.
    """
    text = body.text
    words = text.split()
    # Illustrative BPE-style split: break long tokens into 4-char pieces
    bpe_like: list[str] = []
    for w in words:
        if len(w) <= 5:
            bpe_like.append(w)
        else:
            for i in range(0, len(w), 4):
                bpe_like.append(("##" if i else "") + w[i : i + 4])
    return {
        "char_count": len(text),
        "word_count": len(words),
        "approx_tokens": max(1, int(len(text) / 4)),  # ~4 chars/token rule of thumb
        "words": words[:200],
        "bpe_like": bpe_like[:300],
    }
