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


# Sentiment — loaded once at module import time so the first request is fast.

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)


def _load_sentiment_pipeline():
    """Return (pipeline, engine_name). Never raises — falls back to lexicon."""
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[demos] DistilBERT sentiment pipeline loaded ✓")
        return pipe, "distilbert-sst2"
    except Exception as exc:
        print(f"[demos] transformers unavailable ({exc}); using lexicon fallback")
        return None, "lexicon-fallback"


_SENTIMENT_PIPE, _SENTIMENT_ENGINE = _load_sentiment_pipeline()

_POS = {
    "good", "great", "excellent", "love", "amazing", "fantastic", "wonderful",
    "happy", "delighted", "awesome", "beautiful", "best", "brilliant",
}
_NEG = {
    "bad", "terrible", "awful", "hate", "worst", "horrible", "sad",
    "disappointing", "poor", "broken", "failure", "ugly",
}


def _lexicon_sentiment(text: str) -> dict:
    toks = [w.strip(".,!?;:") for w in text.lower().split() if w]
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    if pos == neg:
        label, score = "NEUTRAL", 0.5
    else:
        label = "POSITIVE" if pos > neg else "NEGATIVE"
        score = max(pos, neg) / (pos + neg + 1)
    return {"label": label, "score": round(float(score), 4)}


@router.post("/sentiment")
@limiter.limit(_settings.rate_limit_demo)
async def sentiment(
    request: Request,
    body: Annotated[SentimentRequest, Body()],
) -> dict:
    t0 = time.perf_counter()
    body_text = sanitise_visitor_input(body.text, max_chars=_settings.max_input_chars)

    if _SENTIMENT_PIPE is not None:
        result = _SENTIMENT_PIPE(body_text[:512])[0]
        out = {
            "label": result["label"].upper(),
            "score": round(float(result["score"]), 4),
            "engine": _SENTIMENT_ENGINE,
        }
    else:
        out = {**_lexicon_sentiment(body_text), "engine": _SENTIMENT_ENGINE}

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
