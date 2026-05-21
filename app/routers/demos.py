"""AI demos: embeddings, sentiment, tokenisation."""
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import DemoRun, SessionLocal
from app.security import limiter, sanitise_visitor_input

router = APIRouter(prefix="/api/demo", tags=["demos"])
_settings = get_settings()


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
@limiter.limit(_settings.rate_limit_demo)
async def embed_similarity(
    request: Request,
    body: Annotated[EmbedRequest, Body()],
) -> dict:
    """Cosine similarity + vector preview."""
    t0 = time.perf_counter()
    embedder = request.app.state.embedder
    a_text = sanitise_visitor_input(body.a, max_chars=_settings.max_input_chars)
    b_text = sanitise_visitor_input(body.b, max_chars=_settings.max_input_chars)
    vecs = embedder.encode([a_text, b_text])
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

    await _log_demo(demo="embed", input_excerpt=a_text[:140], latency_ms=latency_ms)

    return {
        "a": a_text,
        "b": b_text,
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
@limiter.limit(_settings.rate_limit_demo)
async def sentiment(
    request: Request,
    body: Annotated[SentimentRequest, Body()],
) -> dict:
    t0 = time.perf_counter()
    body_text = sanitise_visitor_input(body.text, max_chars=_settings.max_input_chars)
    pipe = _get_sentiment()

    if pipe == "fallback":
        # Simple lexicon-based fallback
        toks = [w.strip(".,!?;:") for w in body_text.lower().split() if w]
        pos = sum(t in _POS for t in toks)
        neg = sum(t in _NEG for t in toks)
        if pos == neg:
            label, score = "NEUTRAL", 0.5
        else:
            label = "POSITIVE" if pos > neg else "NEGATIVE"
            score = (max(pos, neg)) / (pos + neg + 1)
        out = {"label": label, "score": round(float(score), 4), "engine": "lexicon-fallback"}
    else:
        result = pipe(body_text[:512])[0]
        out = {
            "label": result["label"].upper(),
            "score": round(float(result["score"]), 4),
            "engine": "distilbert-sst2",
        }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    out["latency_ms"] = latency_ms

    await _log_demo(demo="sentiment", input_excerpt=body_text[:140], latency_ms=latency_ms)
    return out


# ─── Tokeniser visualiser ─────────────────────────────

class TokeniseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@router.post("/tokenise")
@limiter.limit(_settings.rate_limit_demo)
async def tokenise(request: Request, body: Annotated[TokeniseRequest, Body()]) -> dict:
    """Return a simple whitespace + sub-word tokenisation for visualisation."""
    text = sanitise_visitor_input(body.text, max_chars=_settings.max_input_chars)
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
