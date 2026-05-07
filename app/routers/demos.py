"""AI demos: embeddings, sentiment, tokenisation."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.db import DemoRun, SessionLocal

router = APIRouter(prefix="/api/demo", tags=["demos"])


async def _log_demo(**fields) -> None:
    """Best-effort logging; never raises."""
    try:
        async with SessionLocal() as s:
            s.add(DemoRun(**fields))
            await s.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[demos] log skipped: {type(exc).__name__}: {exc}")


# Embeddings

class EmbedRequest(BaseModel):
    a: str = Field(..., min_length=1, max_length=1500)
    b: str = Field(..., min_length=1, max_length=1500)


@router.post("/embed")
async def embed_similarity(
    body: EmbedRequest,
    request: Request,
) -> dict:
    """Cosine similarity + vector preview."""
    t0 = time.perf_counter()
    embedder = request.app.state.embedder
    vecs = embedder.encode([body.a, body.b])
    a, b = vecs[0], vecs[1]
    
    # Compute cosine similarity
    # Note: vectors should already be normalized if using sbert
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        cos = 0.0  # Handle edge case, though rare
    else:
        cos = float(np.dot(a, b) / (a_norm * b_norm))
    
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


# Sentiment (lazy-loaded)

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)


_SENTIMENT_PIPELINE = None


def _get_sentiment():
    """Load HF pipeline on demand. Falls back to a tiny lexicon classifier."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE
    try:
        from transformers import pipeline  # type: ignore
        # Note: first load can take a while (~2s), might want to do this on startup
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
        # Simple lexicon-based fallback
        toks = [w.strip(".,!?;:") for w in body.text.lower().split() if w]
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
            "label": result["label"].upper(),
            "score": round(float(result["score"]), 4),
            "engine": "distilbert-sst2",
        }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    out["latency_ms"] = latency_ms

    await _log_demo(demo="sentiment", input_excerpt=body.text[:140], latency_ms=latency_ms)
    return out


# ─── Tokeniser visualiser ─────────────────────────────

class TokeniseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@router.post("/tokenise")
async def tokenise(body: TokeniseRequest) -> dict:
    """Return a simple whitespace + sub-word tokenisation for visualisation."""
    text = body.text
    words = text.split()
    
    # Illustrative BPE-style split: break long tokens into 4-char pieces
    # TODO: this is super naive, real BPE is way more sophisticated
    bpe_like: list[str] = []
    for w in words:
        if len(w) <= 5:
            bpe_like.append(w)
        else:
            for i in range(0, len(w), 4):
                piece = ("##" if i else "") + w[i : i + 4]
                bpe_like.append(piece)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "approx_tokens": max(1, int(len(text) / 4)),
        "words": words[:200],
        "bpe_like": bpe_like[:300],
    }
